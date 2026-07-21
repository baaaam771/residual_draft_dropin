"""training.diagnose_residual — post-mortem for a flat-L_res training run.

Hypothesis under test: with inputs (v_a, dz, mask, dsigma) only, offset-1
pairs (i = a+1; ALL of c=2 and half of c=3) carry ZERO information about the
target residual, because the dense Euler trajectory gives exactly

    z_{a+1} = z_a + (sigma_{a+1} - sigma_a) * v_a   =>   dz = dsigma * v_a.

If that identity holds on the real dumps, the network input is a deterministic
function of the anchor state at offset 1, the best achievable predictor is the
content-independent conditional mean, and mse_ratio ~ 1.0 is an information
limit — not a capacity or training bug. The fix is content inputs (z_t,
anchor x0, sigma_t), not more steps.

    PYTHONPATH=. python -m training.diagnose_residual \
        --teacher /mnt/HDD_12TB/bam_ki/flux_fill/router_teacher_1024 \
        --ckpt /mnt/HDD_12TB/bam_ki/flux_fill/residual_draft_ckpt/last.pt

Reports:
  1. dz-identity check: ||dz - dsigma*v_a|| / ||dz|| per offset (o=1 must be ~0)
  2. mse_ratio split by offset (i-a) and by sigma bin
  3. true-gain statistics per offset: with ratio~1 the DRAFT tier has no
     headroom regardless of routing quality — quantifies the no-go
  4. error-head ranking recap on the same pairs

Argparse help strings contain no bare percent characters.
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict

import torch

from models.drafts.residual_draft import ResidualDraftNet
from training.train_residual_draft import LOG_EPS, ResidualTeacherPairs, spearman


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description="Post-mortem diagnostics for the residual draft")
    ap.add_argument("--teacher", required=True, help="router-teacher dump dir")
    ap.add_argument("--ckpt", default="", help="trained checkpoint (optional; skip model metrics if empty)")
    ap.add_argument("--split", default="val", help="val or calib")
    ap.add_argument("--pairs", type=int, default=300, help="pairs to evaluate")
    ap.add_argument("--cache-periods", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--dense-tail", type=int, default=4)
    ap.add_argument("--val-frac", type=float, default=0.10, help="MUST match training")
    ap.add_argument("--calib-frac", type=float, default=0.10, help="MUST match training")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pairs = ResidualTeacherPairs(a.teacher, a.cache_periods, a.split,
                                 a.val_frac, a.calib_frac, a.dense_tail)
    net = None
    if a.ckpt:
        net = ResidualDraftNet.from_checkpoint(a.ckpt).to(device).eval()
        print(f"[model] config={net.config}")

    rng = random.Random(a.seed)
    dz_rel = defaultdict(list)              # offset -> ||dz - dsig*v_a||/||dz||
    cand_rel = defaultdict(list)            # candidate transition forms (o=1)
    cos_dz = defaultdict(list)
    alpha_ratio, bestfit_rel = [], []
    sel_reuse = defaultdict(float)          # top-K by predicted gain
    sel_draft = defaultdict(float)
    orc_reuse = defaultdict(float)          # top-K by ORACLE gain
    orc_draft = defaultdict(float)
    head_corr = []
    reuse_e = defaultdict(float)            # key -> sum e_cache
    draft_e = defaultdict(float)            # key -> sum e_draft (model)
    gain_pos_frac = defaultdict(list)       # offset -> frac tokens with model gain>0
    true_res = defaultdict(list)            # offset -> mean ||dv*||^2 (staleness scale)
    sp_c = sp_d = 0.0

    for _ in range(a.pairs):
        it = pairs.sample(rng)
        o = it["step"] - it["anchor_step"]
        sig = float(it["sigma_t"]) if "sigma_t" in it else None
        v_a = it["v_anchor"].unsqueeze(0).to(device)
        dz = it["dz"].unsqueeze(0).to(device)
        dv_star = it["dv_star"].unsqueeze(0).to(device)
        mask = it["mask_tok"].unsqueeze(0).to(device)
        dsig = it["dsigma"].view(1).to(device)
        hw = it["token_hw"]

        # 1) transition-convention candidates (offset 1 = one true transition)
        pred_dz = dsig.view(1, 1, 1) * v_a
        rel = float((dz - pred_dz).norm() / dz.norm().clamp_min(1e-12))
        dz_rel[o].append(rel)
        if o == 1:
            v_t = v_a + dv_star
            ds = dsig.view(1, 1, 1)
            for cname, c in {"C1 +ds*v_a": ds * v_a, "C2 -ds*v_a": -ds * v_a,
                             "C3 +ds*v_t": ds * v_t,
                             "C4 trapezoid": ds * 0.5 * (v_a + v_t)}.items():
                cand_rel[cname].append(
                    float((dz - c).norm() / dz.norm().clamp_min(1e-12)))
                cos_dz[cname].append(float(
                    torch.dot(c.flatten(), dz.flatten())
                    / (c.norm() * dz.norm()).clamp_min(1e-12)))
            va = v_a.flatten()
            alpha = float(torch.dot(dz.flatten(), va)
                          / torch.dot(va, va).clamp_min(1e-12))
            dsf = float(dsig)
            alpha_ratio.append(alpha / dsf if dsf != 0 else float("nan"))
            bestfit_rel.append(float((alpha * v_a - dz).norm()
                                     / dz.norm().clamp_min(1e-12)))

        e_c = dv_star.pow(2).mean(-1)                     # [1, N] true reuse error
        true_res[o].append(float(e_c.mean()))

        if net is not None:
            kw = {}
            if net.config.get("use_latent"):
                kw["z_t"] = it["z_t"].unsqueeze(0).to(device)
            if net.config.get("use_anchor_x0"):
                kw["x0_anchor"] = it["x0_anchor"].unsqueeze(0).to(device)
            if net.config.get("use_sigma_t"):
                kw["sigma_t"] = it["sigma_t"].view(1).to(device)
            dv_hat, log_ec, log_ed = net(v_a, dz, mask, dsig, hw, **kw)
            e_d = (dv_hat - dv_star).pow(2).mean(-1)
            for key in (f"o{o}", "all"):
                reuse_e[key] += float(e_c.sum())
                draft_e[key] += float(e_d.sum())
            sig_bin = f"sig[{0.2 * int(min(float(dsig.abs()) * 0 + (sig or 0), 0.999) // 0.2):.1f}]" \
                if sig is not None else "sig[?]"
            reuse_e[sig_bin] += float(e_c.sum())
            draft_e[sig_bin] += float(e_d.sum())
            gain = e_c - e_d
            gain_pos_frac[o].append(float((gain > 0).float().mean()))
            ec_hat, ed_hat = ResidualDraftNet.routing_errors(log_ec, log_ed)
            sp_c += spearman(ec_hat, torch.log(e_c + LOG_EPS))
            sp_d += spearman(ed_hat, torch.log(e_d + LOG_EPS))
            head_corr.append(spearman(ec_hat, ed_hat))
            # SELECTION QUALITY: even with global ratio ~1.0 the draft tier
            # has value iff, within the tokens the router would pick, e_draft
            # beats e_cache. Top-K by PREDICTED gain (deployable) and by
            # ORACLE gain (ceiling).
            gain_hat = (ec_hat - ed_hat).flatten()
            n_tok = gain_hat.numel()
            for frac in (0.10, 0.30):
                k = max(int(frac * n_tok), 1)
                key = f"top{int(frac * 100)}"
                idx = gain_hat.topk(k).indices
                sel_reuse[key] += float(e_c.flatten()[idx].sum())
                sel_draft[key] += float(e_d.flatten()[idx].sum())
                oidx = gain.flatten().topk(k).indices
                orc_reuse[key] += float(e_c.flatten()[oidx].sum())
                orc_draft[key] += float(e_d.flatten()[oidx].sum())

    print("== 1. transition-convention candidates at offset 1 (global rel "
          "err; fp16 floor ~1e-3) ==")
    for cname in sorted(cand_rel):
        rs, cs = cand_rel[cname], cos_dz[cname]
        print(f"  {cname:14s} rel mean {sum(rs)/len(rs):.3e}  max "
              f"{max(rs):.3e}  cos {sum(cs)/len(cs):.4f}")
    if alpha_ratio:
        ar = torch.tensor(alpha_ratio); bf = torch.tensor(bestfit_rel)
        print(f"  best-fit alpha/dsigma: mean {float(ar.mean()):.4f} std "
              f"{float(ar.std()):.4f}   best-fit rel mean "
              f"{float(bf.mean()):.3e}")
    print("  offset rel err vs C1 (all offsets):")
    for o in sorted(dz_rel):
        v = dz_rel[o]
        print(f"    offset {o}: mean {sum(v) / len(v):.3e}  max {max(v):.3e}  "
              f"(n={len(v)})")

    print("== 2. true residual scale per offset (mean ||dv*||^2) ==")
    for o in sorted(true_res):
        v = true_res[o]
        print(f"  offset {o}: {sum(v) / len(v):.4e}  (n={len(v)})")

    if net is not None:
        print("== 3. model mse_ratio (draft/reuse) by group ==")
        for key in sorted(reuse_e):
            r = draft_e[key] / max(reuse_e[key], 1e-12)
            print(f"  {key:8s} ratio={r:.4f}")
        print("== 4. token-level model gain > 0 fraction per offset ==")
        for o in sorted(gain_pos_frac):
            v = gain_pos_frac[o]
            print(f"  offset {o}: {sum(v) / len(v):.3f}")
        print(f"== 5. ranking recap: spearman cache {sp_c / a.pairs:.4f}  "
              f"draft {sp_d / a.pairs:.4f}  head-vs-head corr "
              f"{sum(head_corr)/len(head_corr):.4f} ==")
        print("   (head corr ~1.0 => both heads predict the same staleness "
              "map, e_draft ~= e_cache)")
        print("== 6. SELECTION quality: e_draft/e_cache WITHIN the routed "
              "subset (value exists iff < 1.0 here, even when global = 1.0) ==")
        for key in sorted(sel_reuse):
            rp = sel_draft[key] / max(sel_reuse[key], 1e-12)
            ro = orc_draft[key] / max(orc_reuse[key], 1e-12)
            print(f"   {key:6s} predicted-gain ratio = {rp:.4f}   "
                  f"oracle-gain ratio = {ro:.4f}")
        print("   oracle < 1 but predicted ~ 1  -> router quality problem")
        print("   oracle ~ 1 too                -> no draft-tier value at any "
              "router quality")

    print("READ: if offset-1 identity holds (~1e-3 or below) and offset-1 "
          "ratio ~ 1.0 while offset-2 is (even slightly) below 1.0, the flat "
          "L_res is an INPUT-INFORMATION limit -> retrain with content inputs "
          "(--use-latent --use-anchor-x0 --use-sigma-t), not more steps.")


if __name__ == "__main__":
    main()
