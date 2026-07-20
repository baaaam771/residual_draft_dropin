"""samplers.three_tier_flux_fill — reuse / approximate / recompute sampling.

Extends the anchored backend (samplers.cached_flux_fill) with a third tier:
a ~3M anchor-residual draft (models.drafts.residual_draft) that corrects
staleness on tokens where TARGET refresh is overkill but pure reuse is stale.

Per step:
  dense anchor      every c steps + dense head/tail: full transformer,
                    records the same depth-aligned cache as cached_flux_fill
  sparse step       draft net consumes (v_a, z_t - z_a, mask, dsigma) — all
                    free from FluxAnchorCache — and outputs (dv, e_cache_hat,
                    e_draft_hat); token_selectors.three_tier assigns tiers;
                    TARGET tokens go through the EXISTING sparse_forward
                    (dual+K/V levers unchanged); v is composed per tier.

Methods:
  draft_only    Stage 3: CACHE + DRAFT only (r_target=0, boundary forcing
                off) — same target-eval count as `reuse`, so any mask-LPIPS
                gain at ~equal wall-clock is pure profit. Report the MEASURED
                draft cost from the timing breakdown, not the param count.
  three_tier    Stage 4: full routing with sparse TARGET refresh.

Accounting: per-step realized tier ratios and a CUDA-event timing breakdown
(draft_ms / route_ms / sparse_ms) are logged into run.json rows; arms must
run SEQUENTIALLY on the GPU (wall-clock discipline).

    PYTHONPATH=. python -m samplers.three_tier_flux_fill \
        --manifest data/coco_manifest_1024.json --out out/stage_res --tag draft_c2 \
        --method draft_only --cache-period 2 --dense-tail 4 \
        --draft-ckpt /mnt/HDD_12TB/bam_ki/flux_fill/residual_draft_ckpt/last.pt \
        --prompt-cache /mnt/HDD_12TB/bam_ki/flux_fill/prompt_cache --limit 100
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from data.dataset import FluxFillBenchmark
from data.prompt_cache import load_cached
from models.drafts.residual_draft import ResidualDraftNet
from models.flux_cache import FluxAnchorCache
from models.flux_fill_loader import load_flux_fill
from models.flux_sparse_transformer import FluxSparseRunner
from samplers.cached_flux_fill import get_model_provenance
from samplers.dense_flux_fill import (_save_img, decode_latents,
                                      prepare_flux_fill_inputs, scheduler_step)
from token_selectors.mask import mask_score
from token_selectors.three_tier import (ThreeTierConfig, compose_prediction,
                                        hard_indices, route_three_tier)


class _CudaSpan:
    def __init__(self, acc: dict, name: str):
        self.acc, self.name = acc, name

    def __enter__(self):
        if torch.cuda.is_available():
            self.s = torch.cuda.Event(enable_timing=True)
            self.e = torch.cuda.Event(enable_timing=True)
            self.s.record()
        return self

    def __exit__(self, *a):
        if torch.cuda.is_available():
            self.e.record()
            torch.cuda.synchronize()
            self.acc[self.name] = self.acc.get(self.name, 0.0) + \
                self.s.elapsed_time(self.e)


@torch.no_grad()
def sample_one_three_tier(pipe, runner: FluxSparseRunner, state, *,
                          net: ResidualDraftNet, method: str,
                          cache_period: int, cfg: ThreeTierConfig,
                          mask_px: torch.Tensor,
                          dense_head: int = 0, dense_tail: int = 4,
                          kv_cache: bool = False, dual_sparse: bool = False,
                          log: dict | None = None):
    assert method in ("draft_only", "three_tier")
    if method == "draft_only":
        cfg = cfg.stage3()
    grid = state.grid
    hw = grid.token_hw
    cache = FluxAnchorCache()
    mask_tok = mask_score(mask_px, grid).to(state.latents.device)      # [1, N]

    n_anchor = n_sparse = 0
    step_rows, timing = [], {}
    n_steps = len(state.timesteps)
    attn_fracs, mac_ratios = [], []

    for i, t in enumerate(state.timesteps):
        sigma = pipe.scheduler.sigmas[i]
        forced_dense = (i < dense_head) or (i >= n_steps - dense_tail)
        is_anchor = (i % cache_period == 0) or forced_dense

        model_input = torch.cat([state.latents, state.cond], dim=2)
        timestep = t.expand(1).to(state.latents.dtype) / 1000

        if is_anchor:
            v, _ = runner.dense_forward(model_input, state.prompt_embeds,
                                        state.pooled, timestep, state.guidance,
                                        state.img_ids, state.txt_ids,
                                        cache=cache, step_index=i,
                                        record_kv=kv_cache,
                                        record_dual=dual_sparse)
            cache.set_anchor_context(state.latents, sigma)
            n_anchor += 1
        else:
            v_a = cache.final_prediction                               # [1,N,64]
            with _CudaSpan(timing, "draft_ms"):
                dz = (state.latents - cache.anchor_latents)
                dsig = torch.tensor(
                    [float(sigma) - float(cache.anchor_sigma)],
                    device=state.latents.device)
                kw = {}
                if net.config.get("use_latent"):
                    kw["z_t"] = state.latents.float()
                if net.config.get("use_anchor_x0"):
                    kw["x0_anchor"] = cache.anchor_clean_estimate.float()
                if net.config.get("use_sigma_t"):
                    kw["sigma_t"] = torch.tensor(
                        [float(sigma)], device=state.latents.device)
                dv, log_ec, log_ed = net(v_a.float(), dz.float(),
                                         mask_tok.float(), dsig, hw, **kw)
                e_c, e_d = ResidualDraftNet.routing_errors(log_ec, log_ed)
            with _CudaSpan(timing, "route_ms"):
                tier, info = route_three_tier(e_c, e_d, mask_tok, hw, cfg)
                hard_idx = hard_indices(tier)
            v_hard = None
            if hard_idx.shape[1] > 0:
                if method == "draft_only":
                    raise RuntimeError("draft_only routed TARGET tokens — "
                                       "stage3 config broken")
                with _CudaSpan(timing, "sparse_ms"):
                    v_hard, st = runner.sparse_forward(
                        model_input, state.prompt_embeds, state.pooled,
                        timestep, state.guidance, state.img_ids, state.txt_ids,
                        cache, hard_idx, kv_cache=kv_cache,
                        dual_sparse=dual_sparse)
                attn_fracs.append(st.single_attn_fraction)
                mac_ratios.append(st.est_transformer_mac_ratio)
            v = compose_prediction(tier, v_a, dv, hard_idx, v_hard)
            n_sparse += 1
            step_rows.append({"step": i, **info})

        state.latents = scheduler_step(pipe, v, t, state.latents)

    stats = {"anchor_evals": n_anchor, "sparse_steps": n_sparse,
             "timing_ms": timing, "step_rows": step_rows}
    if step_rows:
        for key in ("realized_cache", "realized_draft", "realized_target"):
            stats[f"mean_{key}"] = sum(r[key] for r in step_rows) / len(step_rows)
    if attn_fracs:
        stats["mean_single_attn_fraction"] = sum(attn_fracs) / len(attn_fracs)
        stats["mean_est_mac_ratio"] = sum(mac_ratios) / len(mac_ratios)
    if log is not None:
        log.update(stats)
    return stats


# --------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", choices=["draft_only", "three_tier"],
                    required=True)
    ap.add_argument("--draft-ckpt", required=True,
                    help="training.train_residual_draft checkpoint")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cache-period", type=int, default=2)
    ap.add_argument("--r-target", type=float, default=0.15,
                    help="TARGET budget fraction (three_tier)")
    ap.add_argument("--r-draft", type=float, default=0.35,
                    help="DRAFT budget fraction")
    ap.add_argument("--block", type=int, default=1,
                    help="structured tier assignment window: 1, 2, 4")
    ap.add_argument("--boundary-policy", default="budget",
                    choices=["budget", "all"],
                    help="budget: boundary trimmed to fit r_target; "
                         "all: boundary may exceed (realized ratio logged)")
    ap.add_argument("--no-force-boundary", action="store_true",
                    help="disable boundary-band forced TARGET")
    ap.add_argument("--guidance", type=float, default=30.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--prompt-cache", default="")
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--save-step-rows", choices=["first", "all", "none"],
                    default="first",
                    help="per-step realized tier ratios in run.json: first "
                         "sample only (default), all samples (5-image "
                         "closed-loop diagnostic), or none")
    ap.add_argument("--dense-head", type=int, default=0)
    ap.add_argument("--dense-tail", type=int, default=4)
    ap.add_argument("--kv-cache", action="store_true")
    ap.add_argument("--dual-sparse", action="store_true")
    ap.add_argument("--tag", default="run")
    a = ap.parse_args()

    out = Path(a.out) / a.tag
    out.mkdir(parents=True, exist_ok=True)
    comps = load_flux_fill(keep_text_encoders=not a.prompt_cache)
    _prov_at_load = get_model_provenance(comps.pipe)
    pipe, dev, dtype = comps.pipe, comps.device, comps.dtype
    runner = FluxSparseRunner(pipe.transformer)
    net = ResidualDraftNet.from_checkpoint(a.draft_ckpt).to(dev).float().eval()
    print(f"[draft] {net.num_params() / 1e6:.2f}M params  config={net.config}")

    cfg = ThreeTierConfig(r_target=a.r_target, r_draft=a.r_draft,
                          force_target_boundary=not a.no_force_boundary,
                          boundary_policy=a.boundary_policy, block=a.block)
    ds = FluxFillBenchmark(a.manifest)
    n = len(ds) if a.limit == 0 else min(a.limit, len(ds))

    rows = []
    for i in range(n):
        s = ds[i]
        pe = po = None
        if a.prompt_cache:
            pe, po = load_cached(a.prompt_cache, s["prompt"], dev, dtype)
        state = prepare_flux_fill_inputs(
            pipe, s["image"], s["mask"], s["prompt"],
            s["latent_seed"] + a.seed_offset,
            a.steps, a.guidance, dev, dtype, prompt_embeds=pe, pooled=po)

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        log: dict = {}
        sample_one_three_tier(pipe, runner, state, net=net, method=a.method,
                              cache_period=a.cache_period, cfg=cfg,
                              mask_px=s["mask"].unsqueeze(0).to(dev),
                              dense_head=a.dense_head, dense_tail=a.dense_tail,
                              kv_cache=a.kv_cache, dual_sparse=a.dual_sparse,
                              log=log)
        img = decode_latents(pipe, state)
        torch.cuda.synchronize()
        log.update({"sample_id": s["sample_id"], "bucket": s["bucket"],
                    "mask_type": s["mask_type"], "warmup": i == 0,
                    "wall_s": time.perf_counter() - t0,
                    "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30})
        # step_rows are bulky: every sample keeps its MEAN tier ratios; the
        # per-step detail is kept per --save-step-rows (default: first sample
        # only; use "all" for the 5-image closed-loop diagnostic)
        if a.save_step_rows == "none" or (a.save_step_rows == "first" and i > 0):
            log.pop("step_rows", None)
        rows.append(log)

        stem = Path(s["sample_id"]).stem
        _save_img(img, out / f"{stem}.png")
        m_px = s["mask"].to(img.device, img.dtype)
        inp = s["image"].to(img.device, img.dtype)
        _save_img(m_px * img + (1 - m_px) * inp, out / f"{stem}_pasted.png")
        torch.save(s["mask"], out / f"{stem}_mask.pt")
        print(f"[{a.tag}] {i + 1}/{n} {stem} wall={log['wall_s']:.2f}s "
              f"tiers c/d/t="
              f"{log.get('mean_realized_cache', 1.0):.2f}/"
              f"{log.get('mean_realized_draft', 0.0):.2f}/"
              f"{log.get('mean_realized_target', 0.0):.2f}")

    cfg_d = vars(a)
    cfg_d["route_config"] = cfg.__dict__
    cfg_d["draft_model_config"] = net.config
    import hashlib as _hl
    h = _hl.sha256()
    with open(a.manifest, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    cfg_d["manifest_sha256"] = h.hexdigest()
    cfg_d["resolution"] = json.load(open(a.manifest)).get("resolution")
    _prov_end = get_model_provenance(pipe)
    _prov_at_load["timesteps_sha256"] = _prov_end["timesteps_sha256"]
    _prov_at_load["sigmas_sha256"] = _prov_end["sigmas_sha256"]
    cfg_d["provenance"] = _prov_at_load
    json.dump({"config": cfg_d, "rows": rows}, open(out / "run.json", "w"),
              indent=1)
    print(f"[{a.tag}] {n} samples -> {out}")


if __name__ == "__main__":
    main()
