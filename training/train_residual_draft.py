"""training.train_residual_draft — train the anchor-residual draft on the
EXISTING router-teacher dumps (samplers.dump_router_teacher format). No new
GPU collection is needed: one dense-trajectory dump serves every cache period.

    PYTHONPATH=. python -m training.train_residual_draft \
        --teacher /mnt/HDD_12TB/bam_ki/flux_fill/router_teacher_1024 \
        --out /mnt/HDD_12TB/bam_ki/flux_fill/residual_draft_ckpt \
        --steps 60000 --cache-periods 2 3

Pairs are composed on the fly (training.train_router.TeacherPairs recipe):
step i with anchor a = i - (i mod c), skipping anchor steps and the dense
tail — the exact input distribution the draft sees at inference.

Split is IMAGE-LEVEL by sample_id hash (train / val / calib); every timestep
record of one image stays in one split (no timestep leakage).

Losses (per token, mask-weighted residual; error targets detached):
  L_res   = mean_w || dv_hat - dv* ||^2
  L_cache = smooth_l1( log_e_cache_hat, log(||dv*||^2 + eps) )
  L_draft = smooth_l1( log_e_draft_hat, log(||dv_hat.detach() - dv*||^2 + eps) )

Resume RESTORES the model config from the checkpoint (review fix): CLI arch
flags are ignored on resume with a warning if they differ. Checkpoints are
atomic (tmp+rename), rolling (last 3 + last.pt), and carry model/EMA/
optimizer/scheduler/step/RNG — same discipline as train_router.

Argparse help strings contain no bare percent characters.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from models.drafts.residual_draft import ResidualDraftNet

LOG_EPS = 1e-8


# ------------------------------------------------------------------ split ---
def split_of(sample_id: str, val_frac: float = 0.10,
             calib_frac: float = 0.10) -> str:
    h = int(hashlib.sha256(sample_id.encode()).hexdigest(), 16) % 10000 / 10000
    if h < val_frac:
        return "val"
    if h < val_frac + calib_frac:
        return "calib"
    return "train"


# ------------------------------------------------------------------ pairs ---
class ResidualTeacherPairs:
    """Lazy (anchor, current) pair sampler over dump_router_teacher shards."""

    def __init__(self, root: str, cache_periods, split: str,
                 val_frac: float = 0.10, calib_frac: float = 0.10,
                 dense_tail: int = 4, cache_size: int = 8):
        idx = json.load(open(Path(root) / "index.json"))
        self.steps = idx["steps"]
        self.dense_tail = dense_tail
        self.cache_periods = list(cache_periods)
        self.root = Path(root)
        self.shards = sorted(
            s for s in idx["shards"]
            if split_of(Path(s).stem, val_frac, calib_frac) == split)
        assert self.shards, f"no shards for split={split} in {root}"
        # Valid SPARSE steps per period, precomputed. This closes the boundary
        # bug where i=hi-1 with (hi-1) % c == 0 (e.g. steps=50, tail=4, c=3,
        # i=45) survived the old "i = min(i+1, hi-1)" nudge and produced a
        # degenerate anchor-anchor pair (dz = dv* = dsigma = 0).
        hi = self.steps - self.dense_tail
        self.valid_steps = {c: [i for i in range(1, hi) if i % c != 0]
                            for c in self.cache_periods}
        for c, v in self.valid_steps.items():
            assert v, f"no valid sparse steps for c={c} (steps={self.steps}, " \
                      f"tail={self.dense_tail})"
            assert all(i % c != 0 for i in v)
        self.cache_size = cache_size
        self._cache: dict[str, dict] = {}

    def _shard(self, name):
        if name not in self._cache:
            if len(self._cache) >= self.cache_size:
                self._cache.pop(next(iter(self._cache)))
            self._cache[name] = torch.load(self.root / name,
                                           map_location="cpu",
                                           weights_only=False)
        return self._cache[name]

    def sample(self, rng: random.Random):
        sh = self._shard(rng.choice(self.shards))
        c = rng.choice(self.cache_periods)
        i = rng.choice(self.valid_steps[c])        # never an anchor step
        a = i - (i % c)
        sig_a = sh["sigmas"][a].float()
        return {
            "v_anchor": sh["preds"][a].float(),            # [N, 64]
            "dz": (sh["latents"][i] - sh["latents"][a]).float(),
            "dv_star": (sh["preds"][i] - sh["preds"][a]).float(),
            "mask_tok": sh["mask_tok"].float(),
            "dsigma": (sh["sigmas"][i] - sh["sigmas"][a]).float(),
            "token_hw": tuple(sh["token_hw"]),
            # v2 content inputs (free at a sparse step / from the anchor cache)
            "z_t": sh["latents"][i].float(),
            "x0_anchor": (sh["latents"][a].float()
                          - sig_a * sh["preds"][a].float()),
            "sigma_t": sh["sigmas"][i].float(),
            # debug/test metadata (not stacked by batch())
            "step": i, "anchor_step": a, "cache_period": c,
        }

    BATCH_KEYS = ("v_anchor", "dz", "dv_star", "mask_tok", "dsigma",
                  "z_t", "x0_anchor", "sigma_t")

    def batch(self, bs, rng, device):
        items = [self.sample(rng) for _ in range(bs)]
        out = {k: torch.stack([it[k] for it in items]).to(device)
               for k in self.BATCH_KEYS}
        out["token_hw"] = items[0]["token_hw"]
        return out


# ----------------------------------------------------------------- losses ---
def content_kwargs(net, batch):
    kw = {}
    if net.config.get("use_latent"):
        kw["z_t"] = batch["z_t"]
    if net.config.get("use_anchor_x0"):
        kw["x0_anchor"] = batch["x0_anchor"]
    if net.config.get("use_sigma_t"):
        kw["sigma_t"] = batch["sigma_t"]
    return kw


def compute_losses(net, batch, mask_weight, lambda_cache, lambda_draft):
    v_a, dz, dv_star = batch["v_anchor"], batch["dz"], batch["dv_star"]
    mask_tok, dsig, hw = batch["mask_tok"], batch["dsigma"], batch["token_hw"]
    dv_hat, log_ec, log_ed = net(v_a, dz, mask_tok, dsig, hw,
                                 **content_kwargs(net, batch))
    w = 1.0 + (mask_weight - 1.0) * (mask_tok > 0.5).float()      # [B, N]
    res_err = (dv_hat - dv_star).pow(2).mean(-1)                  # [B, N]
    loss_res = (w * res_err).sum() / w.sum()
    e_cache = dv_star.pow(2).mean(-1).detach()
    e_draft = res_err.detach()
    loss_cache = F.smooth_l1_loss(log_ec, torch.log(e_cache + LOG_EPS))
    loss_draft = F.smooth_l1_loss(log_ed, torch.log(e_draft + LOG_EPS))
    return (loss_res + lambda_cache * loss_cache + lambda_draft * loss_draft,
            loss_res, loss_cache, loss_draft)


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    ra = a.flatten().argsort().argsort().float()
    rb = b.flatten().argsort().argsort().float()
    ra = (ra - ra.mean()) / (ra.std() + 1e-12)
    rb = (rb - rb.mean()) / (rb.std() + 1e-12)
    return float((ra * rb).mean())


@torch.no_grad()
def validate(net, pairs, rng, device, bs, n_batches):
    net.eval()
    sr = sd = sri = sdi = 0.0
    sp_c = sp_d = 0.0
    for _ in range(n_batches):
        b = pairs.batch(bs, rng, device)
        v_a, dz, dv_star, mask_tok = (b["v_anchor"], b["dz"], b["dv_star"],
                                      b["mask_tok"])
        dv_hat, log_ec, log_ed = net(v_a, dz, mask_tok, b["dsigma"],
                                     b["token_hw"], **content_kwargs(net, b))
        e_c = dv_star.pow(2).mean(-1)
        e_d = (dv_hat - dv_star).pow(2).mean(-1)
        sr += float(e_c.sum()); sd += float(e_d.sum())
        m = mask_tok > 0.5
        sri += float(e_c[m].sum()); sdi += float(e_d[m].sum())
        sp_c += spearman(log_ec, torch.log(e_c + LOG_EPS))
        sp_d += spearman(log_ed, torch.log(e_d + LOG_EPS))
    net.train()
    return {"mse_ratio_all": sd / max(sr, 1e-12),        # <1.0: draft beats reuse
            "mse_ratio_in_mask": sdi / max(sri, 1e-12),
            "spearman_cache_head": sp_c / n_batches,
            "spearman_draft_head": sp_d / n_batches}


# ------------------------------------------------------------- checkpoints ---
def save_ckpt(path: Path, net, ema, opt, sched, step):
    tmp = path.with_suffix(".pt.tmp")
    torch.save({"model": net.state_dict(), "ema": ema.state_dict(),
                "model_config": net.config,
                "opt": opt.state_dict(), "sched": sched.state_dict(),
                "step": step,
                "rng": {"torch": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state_all(),
                        "numpy": np.random.get_state(),
                        "python": random.getstate()}}, tmp)
    tmp.rename(path)


def _prune_rolling(out: Path, keep: int = 3):
    cks = sorted(out.glob("step_*.pt"))
    for p in cks[:-keep]:
        p.unlink()


def main():
    ap = argparse.ArgumentParser(
        description="Train the anchor-residual draft on router-teacher dumps")
    ap.add_argument("--teacher", required=True,
                    help="dump_router_teacher output dir (index.json + shards)")
    ap.add_argument("--out", required=True, help="checkpoint directory")
    ap.add_argument("--steps", type=int, default=60000, help="optimizer steps")
    ap.add_argument("--bs", type=int, default=8, help="batch size")
    ap.add_argument("--lr", type=float, default=2e-4, help="peak learning rate")
    ap.add_argument("--cache-periods", type=int, nargs="+", default=[2, 3],
                    help="anchor periods to sample pairs for, e.g. 2 3")
    ap.add_argument("--dense-tail", type=int, default=4,
                    help="final steps excluded from pairs (forced dense at inference)")
    ap.add_argument("--lambda-cache", type=float, default=0.1,
                    help="cache-error head loss weight")
    ap.add_argument("--lambda-draft", type=float, default=0.1,
                    help="draft-error head loss weight")
    ap.add_argument("--mask-weight", type=float, default=2.0,
                    help="in-mask loss multiplier (2.0 doubles in-mask weight)")
    ap.add_argument("--hidden", type=int, default=192, help="CNN width")
    ap.add_argument("--num-blocks", type=int, default=4, help="conv blocks")
    ap.add_argument("--detach-error-trunk", action="store_true",
                    help="stop error-head gradients at the shared trunk")
    ap.add_argument("--use-latent", action=argparse.BooleanOptionalAction,
                    default=True, help="feed current packed latent z_t")
    ap.add_argument("--use-anchor-x0", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="feed anchor clean estimate x0_a (image content)")
    ap.add_argument("--use-sigma-t", action=argparse.BooleanOptionalAction,
                    default=True, help="feed absolute sigma_t plane")
    ap.add_argument("--val-frac", type=float, default=0.10,
                    help="image-level val fraction (same value in eval)")
    ap.add_argument("--calib-frac", type=float, default=0.10,
                    help="image-level calib fraction (same value in eval)")
    ap.add_argument("--ema", type=float, default=0.999, help="EMA decay")
    ap.add_argument("--seed", type=int, default=0, help="random seed")
    ap.add_argument("--log-every", type=int, default=100, help="log interval")
    ap.add_argument("--val-every", type=int, default=2000, help="val interval")
    ap.add_argument("--save-every", type=int, default=2000, help="ckpt interval")
    ap.add_argument("--resume", default="",
                    help="checkpoint to resume (empty = auto last.pt in --out)")
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed)
    torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    train_pairs = ResidualTeacherPairs(a.teacher, a.cache_periods, "train",
                                       a.val_frac, a.calib_frac, a.dense_tail)
    val_pairs = ResidualTeacherPairs(a.teacher, a.cache_periods, "val",
                                     a.val_frac, a.calib_frac, a.dense_tail)
    print(f"[pairs] train shards={len(train_pairs.shards)} "
          f"val shards={len(val_pairs.shards)} steps/traj={train_pairs.steps}")

    # ---- resume-aware model construction (review fix: config from ckpt) ----
    last = out / "last.pt"
    resume_path = Path(a.resume) if a.resume else (last if last.exists() else None)
    if resume_path is not None:
        ck = torch.load(resume_path, map_location="cpu", weights_only=False)
        cfg = ck["model_config"]
        cli_cfg = {"latent_ch": 64, "hidden": a.hidden,
                   "num_blocks": a.num_blocks,
                   "detach_error_trunk": a.detach_error_trunk,
                   "use_latent": a.use_latent,
                   "use_anchor_x0": a.use_anchor_x0,
                   "use_sigma_t": a.use_sigma_t}
        if cli_cfg != cfg:
            print(f"[resume] WARNING: CLI arch {cli_cfg} != checkpoint arch "
                  f"{cfg}; using the CHECKPOINT config")
        net = ResidualDraftNet(**cfg).to(device)
    else:
        ck = None
        net = ResidualDraftNet(hidden=a.hidden, num_blocks=a.num_blocks,
                               detach_error_trunk=a.detach_error_trunk,
                               use_latent=a.use_latent,
                               use_anchor_x0=a.use_anchor_x0,
                               use_sigma_t=a.use_sigma_t).to(device)
    ema = copy.deepcopy(net).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    print(f"[model] {net.num_params() / 1e6:.2f}M params  config={net.config}")

    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.01)
    warm = 1000
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warm if s < warm
        else 0.5 * (1 + math.cos(math.pi * min((s - warm) / max(a.steps - warm, 1), 1))))

    step = 0
    if ck is not None:
        net.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        step = ck["step"]
        rng = ck.get("rng")
        if rng is not None:
            torch.set_rng_state(rng["torch"])
            if torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["cuda"])
            np.random.set_state(rng["numpy"])
            random.setstate(rng["python"])
        print(f"[resume] {resume_path} at step {step}")

    rng_train = random.Random(a.seed + step)
    rng_val = random.Random(a.seed + 10_000)
    t0, run = time.time(), [0.0, 0.0, 0.0]

    while step < a.steps:
        batch = train_pairs.batch(a.bs, rng_train, device)
        loss, l_res, l_c, l_d = compute_losses(
            net, batch, a.mask_weight, a.lambda_cache, a.lambda_draft)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step(); step += 1
        with torch.no_grad():
            for pe, pn in zip(ema.parameters(), net.parameters()):
                pe.lerp_(pn, 1 - a.ema)
            for be, bn in zip(ema.buffers(), net.buffers()):
                be.copy_(bn)
        run[0] += l_res.item(); run[1] += l_c.item(); run[2] += l_d.item()

        if step % a.log_every == 0:
            dt = time.time() - t0
            print(f"step {step:6d}  L_res {run[0] / a.log_every:.4e}  "
                  f"L_cache {run[1] / a.log_every:.4f}  "
                  f"L_draft {run[2] / a.log_every:.4f}  "
                  f"lr {sched.get_last_lr()[0]:.2e}  {a.log_every / dt:.1f} it/s")
            run, t0 = [0.0, 0.0, 0.0], time.time()
        if step % a.val_every == 0:
            m = validate(ema, val_pairs, rng_val, device, a.bs, 16)
            print(f"[val @ {step}] " + "  ".join(f"{k}={v:.4f}"
                                                 for k, v in m.items()))
        if step % a.save_every == 0 or step == a.steps:
            save_ckpt(last, net, ema, opt, sched, step)
            save_ckpt(out / f"step_{step:06d}.pt", net, ema, opt, sched, step)
            _prune_rolling(out)
            print(f"[ckpt] saved at step {step}")


if __name__ == "__main__":
    main()
