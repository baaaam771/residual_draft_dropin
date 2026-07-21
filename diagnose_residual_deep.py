"""training.diagnose_residual_deep — why does the residual head fail?

The v1/v2 runs both show flat L_res and mse_ratio ~ 1.0 while the error heads
reach spearman ~0.75. That means staleness MAGNITUDE is predictable but the
residual DIRECTION is not learned. This script separates the competing causes
on the real teacher dumps, WITHOUT training the draft (all estimators are
computed in closed form from teacher pairs), so it runs in minutes.

Estimators (all per mask region: in-mask / boundary / outside):

H1  Regression-to-mean floor.
    Best CONSTANT residual is dv*=0 (=reuse). Best per-(sigma,offset) MEAN
    residual is a group-conditional predictor with NO content input. We report
    mse_ratio of the group-mean predictor: if it is also ~1.0, even the optimal
    content-free predictor cannot win -> the residual has no low-order
    structure a small head could exploit. If it is <1.0 but the trained head
    is ~1.0, the head is underfitting a signal that IS there (training/arch
    problem, not data).

H2  Intrinsic unpredictability (dispersion).
    Signal-to-dispersion ratio  ||E[dv]||^2 / E||dv - E[dv]||^2  within each
    (sigma,offset) group. <<1 means the conditional mean carries almost no
    energy: the residual is dominated by per-token variation that no predictor
    conditioned on group statistics can reduce -> genuine unpredictability.

H2b Neighbour oracle (content-conditional ceiling, kNN in input space).
    For each target token, find its k nearest tokens IN INPUT FEATURE SPACE
    (v_a, dz, x0_a, sigma) from OTHER images and predict their mean dv*. This
    is a nonparametric upper bound on what ANY content-conditioned regressor
    can achieve. mse_ratio of this oracle is the real ceiling: if ~1.0, no
    architecture will help; if <<1.0, our CNN is the bottleneck.

H3  Spatial-frequency structure.
    Fraction of residual energy in the top half of the 2D DFT (high freq) on
    the token grid. High -> 3x3-conv, one-shot regression is a poor prior;
    a smoother target (or a different parameterisation) might work.

H4  Scale domination.
    Gini of per-token ||dv*||^2 and the loss share of the top-1% tokens. If
    the top 1% carry most of the loss, mask-weighting / log-target / Huber on
    the residual could change the picture.

Directional baseline: cos similarity between v_a and v_t (are they even
close in direction?), and between dv* and -v_a (is the residual mostly
"undo the anchor" or something new?).

    PYTHONPATH=. python -m training.diagnose_residual_deep \
        --teacher /mnt/HDD_12TB/bam_ki/flux_fill/router_teacher_1024 \
        --pairs 400 --knn 16

Argparse help strings contain no bare percent characters.
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict

import torch


def _region_masks(mask_tok, boundary):
    inm = (mask_tok > 0.5) & ~boundary
    bnd = boundary
    out = (mask_tok <= 0.5) & ~boundary
    return {"in_mask": inm, "boundary": bnd, "outside": out, "all":
            torch.ones_like(inm)}


def boundary_from_mask(mask_tok, hw):
    import torch.nn.functional as F
    hp, wp = hw
    m = (mask_tok.view(1, 1, hp, wp) > 0.5).float()
    dil = F.max_pool2d(m, 3, 1, 1)
    ero = -F.max_pool2d(-m, 3, 1, 1)
    return (dil - ero).clamp(0, 1).flatten().bool()


def high_freq_fraction(res_grid):
    """res_grid [C, hp, wp] -> fraction of energy above the DFT median radius."""
    C, hp, wp = res_grid.shape
    F = torch.fft.fftshift(torch.fft.fft2(res_grid), dim=(-2, -1))
    p = (F.abs() ** 2).mean(0)                      # [hp, wp]
    cy, cx = hp / 2, wp / 2
    yy, xx = torch.meshgrid(torch.arange(hp).float(), torch.arange(wp).float(),
                            indexing="ij")
    r = ((yy - cy) ** 2 + (xx - cx) ** 2).sqrt()
    hi = r > r.median()
    return float(p[hi].sum() / p.sum().clamp_min(1e-12))


def gini(x):
    x = torch.sort(x.flatten())[0]
    n = x.numel()
    idx = torch.arange(1, n + 1, dtype=x.dtype)
    return float((2 * (idx * x).sum() / (n * x.sum().clamp_min(1e-12))) - (n + 1) / n)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description="Deep diagnostics for the residual head failure")
    ap.add_argument("--teacher", required=True, help="router-teacher dump dir")
    ap.add_argument("--split", default="val", help="val or calib")
    ap.add_argument("--pairs", type=int, default=400, help="pairs to evaluate")
    ap.add_argument("--cache-periods", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--dense-tail", type=int, default=4)
    ap.add_argument("--val-frac", type=float, default=0.10, help="MUST match training")
    ap.add_argument("--calib-frac", type=float, default=0.10, help="MUST match training")
    ap.add_argument("--knn", type=int, default=16, help="neighbours for the content-oracle ceiling")
    ap.add_argument("--knn-pool", type=int, default=4000, help="token pool size for kNN search")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from training.train_residual_draft import ResidualTeacherPairs
    pairs = ResidualTeacherPairs(a.teacher, a.cache_periods, a.split,
                                 a.val_frac, a.calib_frac, a.dense_tail)
    rng = random.Random(a.seed)

    # accumulate group stats for H1/H2 (keyed by (sigma_bin, offset))
    grp_sum = defaultdict(lambda: 0.0)     # sum dv*  (vector)  -> mean
    grp_sq = defaultdict(lambda: 0.0)      # sum ||dv*||^2
    grp_n = defaultdict(int)
    samples = []                            # for kNN + region metrics

    def sbin(s):
        return min(int(s / 0.2), 4)

    for _ in range(a.pairs):
        it = pairs.sample(rng)
        o = it["step"] - it["anchor_step"]
        sb = sbin(float(it["sigma_t"]))
        dv = it["dv_star"]                  # [N, 64]
        key = (sb, o)
        grp_sum[key] = grp_sum[key] + dv.sum(0)
        grp_sq[key] = grp_sq[key] + float(dv.pow(2).sum())
        grp_n[key] += dv.shape[0]
        samples.append({"dv": dv, "v_a": it["v_anchor"],
                        "dz": it["dz"], "x0": it["x0_anchor"],
                        "sig": float(it["sigma_t"]), "o": o,
                        "mask": it["mask_tok"], "hw": it["token_hw"]})

    # group means
    grp_mean = {k: grp_sum[k] / grp_n[k] for k in grp_sum}

    # ---- H1: group-mean predictor mse_ratio, per region ----
    reuse_e = defaultdict(float); grpmean_e = defaultdict(float); cnt = defaultdict(int)
    sig_num = defaultdict(float); disp_num = defaultdict(float)  # H2
    cos_va_vt = []; cos_dv_negva = []
    hf_frac = []; per_tok_sq_all = []

    for s in samples:
        dv, hw = s["dv"], s["hw"]
        bnd = boundary_from_mask(s["mask"], hw)
        regions = _region_masks(s["mask"], bnd)
        key = (sbin(s["sig"]), s["o"])
        gm = grp_mean[key]                          # [64]
        e_reuse = dv.pow(2).mean(-1)                # [N] vs dv*=0
        e_gm = (dv - gm).pow(2).mean(-1)            # [N] vs group mean
        for rn, rmask in regions.items():
            if rmask.any():
                reuse_e[rn] += float(e_reuse[rmask].sum())
                grpmean_e[rn] += float(e_gm[rmask].sum())
                cnt[rn] += int(rmask.sum())
        # H2 dispersion (all tokens, per group aggregated later via region 'all')
        sig_num["all"] += float(gm.pow(2).sum()) * dv.shape[0]  # ||E||^2 * n
        disp_num["all"] += float((dv - gm).pow(2).sum())
        # directional
        v_a = s["v_a"]; v_t = v_a + dv
        cva = torch.nn.functional.cosine_similarity(v_a, v_t, dim=-1)
        cos_va_vt.append(float(cva.mean()))
        cdv = torch.nn.functional.cosine_similarity(dv, -v_a, dim=-1)
        cos_dv_negva.append(float(cdv.mean()))
        # H3 high-freq
        C = dv.shape[-1]
        grid = dv.transpose(0, 1).reshape(C, hw[0], hw[1])
        hf_frac.append(high_freq_fraction(grid))
        per_tok_sq_all.append(dv.pow(2).mean(-1))

    print("== H1: predictor mse_ratio (lower = beats reuse) ==")
    print("   reuse baseline = 1.0 by construction; group-MEAN uses only "
          "(sigma,offset), NO content")
    for rn in ["all", "in_mask", "boundary", "outside"]:
        if cnt[rn]:
            r = grpmean_e[rn] / max(reuse_e[rn], 1e-12)
            print(f"   {rn:9s} group-mean ratio = {r:.4f}")

    print("== H2: signal-to-dispersion  ||E[dv]||^2 / E||dv-E[dv]||^2 (<<1 = "
          "intrinsically unpredictable from group stats) ==")
    ratio = sig_num["all"] / max(disp_num["all"], 1e-12)
    print(f"   all: {ratio:.4e}")

    # ---- H2b: content kNN oracle ceiling ----
    # pool tokens from all samples, build feature = [v_a, dz, x0, sigma]
    feats, targs = [], []
    for s in samples:
        n = s["dv"].shape[0]
        sig_col = torch.full((n, 1), s["sig"])
        f = torch.cat([s["v_a"], s["dz"], s["x0"], sig_col], dim=-1)
        feats.append(f); targs.append(s["dv"])
    F = torch.cat(feats); T = torch.cat(targs)
    P = min(a.knn_pool, F.shape[0])
    sel = torch.randperm(F.shape[0])[:P]
    Fp, Tp = F[sel], T[sel]
    # normalise features
    mu, sd = Fp.mean(0), Fp.std(0).clamp_min(1e-6)
    Fpn = (Fp - mu) / sd
    q_idx = torch.randperm(F.shape[0])[:min(2000, F.shape[0])]
    Fq = (F[q_idx] - mu) / sd
    Tq = T[q_idx]
    # exclude self by large-distance trick: chunked cdist
    knn_e = 0.0; reuse_e2 = 0.0
    CH = 256
    for i in range(0, Fq.shape[0], CH):
        d = torch.cdist(Fq[i:i+CH], Fpn)            # [ch, P]
        nn = d.topk(a.knn + 1, largest=False).indices[:, 1:]  # drop nearest (self-ish)
        pred = Tp[nn].mean(1)                        # [ch, 64]
        knn_e += float((pred - Tq[i:i+CH]).pow(2).mean(-1).sum())
        reuse_e2 += float(Tq[i:i+CH].pow(2).mean(-1).sum())
    print("== H2b: content kNN oracle ceiling (nonparametric upper bound on "
          "ANY content regressor) ==")
    print(f"   kNN(k={a.knn}) mse_ratio = {knn_e / max(reuse_e2, 1e-12):.4f}  "
          f"(pool={P}, queries={Fq.shape[0]})")

    print("== H3: residual high-frequency energy fraction (>0.5 = "
          "dominated by high spatial freq) ==")
    print(f"   mean {sum(hf_frac)/len(hf_frac):.3f}  "
          f"min {min(hf_frac):.3f}  max {max(hf_frac):.3f}")

    print("== H4: scale domination ==")
    allsq = torch.cat(per_tok_sq_all)
    g = gini(allsq)
    k = max(int(0.01 * allsq.numel()), 1)
    top1 = float(allsq.topk(k).values.sum() / allsq.sum().clamp_min(1e-12))
    print(f"   Gini(||dv||^2) = {g:.3f}   top-1pct loss share = {top1:.3f}")

    print("== directional structure ==")
    print(f"   cos(v_a, v_t)   mean {sum(cos_va_vt)/len(cos_va_vt):.4f}  "
          "(near 1 = anchor already points right)")
    print(f"   cos(dv, -v_a)   mean {sum(cos_dv_negva)/len(cos_dv_negva):.4f}  "
          "(near 1 = residual just shrinks the anchor)")

    print("\nDECISION TABLE:")
    print("  group-mean ~1.0 AND kNN ~1.0  -> intrinsic: no predictor wins. "
          "Report as negative result; pivot error-head to router.")
    print("  group-mean ~1.0 BUT kNN <<1.0 -> content IS predictive but needs "
          "capacity/locality our CNN lacks: try deeper/attention or kNN draft.")
    print("  group-mean <1.0 BUT trained head ~1.0 -> underfitting: fix "
          "loss (Huber/log) or lr, not the idea.")
    print("  high-freq >0.6 -> target too rough for 3x3 one-shot; consider "
          "predicting a smoothed/low-rank residual.")


if __name__ == "__main__":
    main()
