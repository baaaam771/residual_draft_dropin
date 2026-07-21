"""training.eval_residual_router — offline (teacher-forced) gate before any
GPU-expensive Stage 3/4 sampling. Runs on the held-out val/calib split of the
router-teacher dumps.

    PYTHONPATH=. python -m resdraft.training.eval_residual_router \
        --teacher /mnt/HDD_12TB/bam_ki/flux_fill/router_teacher_1024 \
        --ckpt /mnt/HDD_12TB/bam_ki/flux_fill/residual_draft_ckpt/last.pt

Reports:
  B. residual predictor quality: MSE(v_a+dv_hat, v_t) / MSE(v_a, v_t),
     overall and split by in-mask / boundary / outside
  C. ranking quality of the error heads: Spearman, top-r error-mass capture,
     tier overlap between learned routing and upper-bound routing
  D. teacher-forced routed per-token error under: pure reuse / all draft /
     upper-bound routing / learned routing

NAMING (review fix): the "upper bound" here assumes IDEAL target fallback —
TARGET tokens are scored as error 0. The real dual+K/V sparse refresh carries
priced dual staleness and K/V staleness, so this is a *routing upper bound
under ideal target fallback*, i.e. draft+router headroom — NOT a full system
oracle. The system-level check is the closed-loop diagnostic in
samplers.three_tier_flux_fill (oracle hard tokens through the real
sparse_forward vs the dense output).

GO/NO-GO: if the upper-bound routed error does not clearly beat pure reuse
here, the residual draft lacks headroom — stop before spending GPU time.

Argparse help strings contain no bare percent characters.
"""
from __future__ import annotations

import argparse
import random

import torch

from resdraft.models.residual_draft import ResidualDraftNet, boundary_band_tok
from resdraft.routing.three_tier import (CACHE, DRAFT, TARGET, ThreeTierConfig,
                                        route_three_tier)
from resdraft.training.train_residual_draft import (LOG_EPS, ResidualTeacherPairs,
                                           spearman)


def _content_kwargs(net, it, device):
    kw = {}
    if net.config.get("use_latent"):
        kw["z_t"] = it["z_t"].unsqueeze(0).to(device)
    if net.config.get("use_anchor_x0"):
        kw["x0_anchor"] = it["x0_anchor"].unsqueeze(0).to(device)
    if net.config.get("use_sigma_t"):
        kw["sigma_t"] = it["sigma_t"].view(1).to(device)
    return kw


def top_r_capture(pred: torch.Tensor, true: torch.Tensor, r: float) -> float:
    n = true.numel()
    k = max(int(round(r * n)), 1)
    idx = pred.flatten().argsort(descending=True)[:k]
    return float(true.flatten()[idx].sum() / (true.sum() + 1e-12))


def routed_error(tier: torch.Tensor, e_cache: torch.Tensor,
                 e_draft: torch.Tensor) -> float:
    """Teacher-forced mean per-token error; TARGET scored as 0 (IDEAL fallback
    assumption — see module docstring)."""
    e = torch.zeros_like(e_cache)
    e[tier == CACHE] = e_cache[tier == CACHE]
    e[tier == DRAFT] = e_draft[tier == DRAFT]
    return float(e.mean())


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description="Offline gate for the residual draft")
    ap.add_argument("--teacher", required=True, help="router-teacher dump dir")
    ap.add_argument("--ckpt", required=True, help="trained checkpoint")
    ap.add_argument("--split", default="val", help="val or calib")
    ap.add_argument("--pairs", type=int, default=400, help="pairs to evaluate")
    ap.add_argument("--cache-periods", type=int, nargs="+", default=[2],
                    help="anchor periods to compose pairs for")
    ap.add_argument("--dense-tail", type=int, default=4,
                    help="final steps excluded (forced dense at inference)")
    ap.add_argument("--r-target", type=float, default=0.15, help="target budget")
    ap.add_argument("--r-draft", type=float, default=0.35, help="draft budget")
    ap.add_argument("--val-frac", type=float, default=0.10,
                    help="MUST match training")
    ap.add_argument("--calib-frac", type=float, default=0.10,
                    help="MUST match training")
    ap.add_argument("--seed", type=int, default=0, help="pair sampling seed")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = ResidualDraftNet.from_checkpoint(a.ckpt).to(device).eval()
    pairs = ResidualTeacherPairs(a.teacher, a.cache_periods, a.split,
                                 a.val_frac, a.calib_frac, a.dense_tail)
    cfg = ThreeTierConfig(r_target=a.r_target, r_draft=a.r_draft)
    rng = random.Random(a.seed)

    reuse = [0.0] * 4; draft = [0.0] * 4; cnt = [0] * 4
    sp_c = sp_d = cap = ov_t = ov_d = 0.0
    err_reuse = err_all = err_ub = err_lrn = 0.0

    for _ in range(a.pairs):
        it = pairs.sample(rng)
        v_a = it["v_anchor"].unsqueeze(0).to(device)
        dz = it["dz"].unsqueeze(0).to(device)
        dv_star = it["dv_star"].unsqueeze(0).to(device)
        mask = it["mask_tok"].unsqueeze(0).to(device)
        dsig = it["dsigma"].view(1).to(device)
        hw = it["token_hw"]

        dv_hat, log_ec, log_ed = net(v_a, dz, mask, dsig, hw,
                                     **_content_kwargs(net, it, device))
        e_c = dv_star.pow(2).mean(-1)                       # true, [1, N]
        e_d = (dv_hat - dv_star).pow(2).mean(-1)
        ec_hat, ed_hat = ResidualDraftNet.routing_errors(log_ec, log_ed)

        bnd = boundary_band_tok(mask, hw) > 0.5
        inm = (mask > 0.5) & ~bnd
        out = ~(mask > 0.5) & ~bnd
        for j, sel in enumerate([torch.ones_like(bnd), inm, bnd, out]):
            reuse[j] += float(e_c[sel].sum())
            draft[j] += float(e_d[sel].sum())
            cnt[j] += int(sel.sum())

        sp_c += spearman(ec_hat, torch.log(e_c + LOG_EPS))
        sp_d += spearman(ed_hat, torch.log(e_d + LOG_EPS))
        cap += top_r_capture(ed_hat, e_d, a.r_target)

        tier_ub, _ = route_three_tier(e_c, e_d, mask, hw, cfg)
        tier_ln, _ = route_three_tier(ec_hat, ed_hat, mask, hw, cfg)
        for name, tval in (("t", TARGET), ("d", DRAFT)):
            o, l = (tier_ub == tval), (tier_ln == tval)
            frac = float((o & l).sum()) / max(float(o.sum()), 1.0)
            if name == "t":
                ov_t += frac
            else:
                ov_d += frac

        err_reuse += float(e_c.mean())
        err_all += float(e_d.mean())
        err_ub += routed_error(tier_ub, e_c, e_d)
        err_lrn += routed_error(tier_ln, e_c, e_d)

    nb = a.pairs
    names = ["all", "in_mask", "boundary", "outside"]
    print("== B. residual predictor (MSE ratio draft/reuse; below 1.0 = draft wins) ==")
    for j, nm in enumerate(names):
        print(f"  {nm:9s} ratio={draft[j] / max(reuse[j], 1e-12):.4f}  "
              f"(tokens={cnt[j]})")
    print(f"== C. ranking (mean over {nb} pairs) ==")
    print(f"  spearman cache-head {sp_c / nb:.4f}   draft-head {sp_d / nb:.4f}")
    print(f"  top-r_target error capture (draft head) {cap / nb:.4f}")
    print(f"  tier overlap vs upper bound: TARGET {ov_t / nb:.4f}  "
          f"DRAFT {ov_d / nb:.4f}")
    print("== A/D. teacher-forced routed error (lower is better) ==")
    print(f"  pure reuse              {err_reuse / nb:.4e}")
    print(f"  all draft               {err_all / nb:.4e}")
    print(f"  upper-bound routing     {err_ub / nb:.4e}   "
          f"<- IDEAL target fallback (TARGET err=0); draft+router headroom, "
          f"NOT a system oracle")
    print(f"  learned routing         {err_lrn / nb:.4e}")
    print("GO/NO-GO: upper-bound routing must clearly beat pure reuse before "
          "Stage 3/4 GPU runs. Closed-loop remains the real test.")


if __name__ == "__main__":
    main()
