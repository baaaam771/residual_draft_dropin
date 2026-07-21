"""training.diagnose_transition — verify the dump's step-transition convention.

The pair diagnostic found offset-1 identity rel err ~9e-2 against the assumed
form dz = dsigma * v_a. Before any information-bottleneck claim, the stored
(z, v, sigma) convention must be pinned down. This walks CONSECUTIVE
transitions (j -> j+1) inside real shards — no pair sampling, no model — and
tests every candidate form the review lists:

    C1  +dsigma * v_j            (assumed form)
    C2  -dsigma * v_j            (sign flip)
    C3  +dsigma * v_{j+1}        (one-index shift: v stored post-update)
    C4  +dsigma * (v_j+v_{j+1})/2  (trapezoid / higher-order solver)

For each: GLOBAL relative norm error, cosine(dz, candidate), and R^2 against
the mean-dz baseline. Plus the best-fit scalar

    alpha* = <dz, v_j> / <v_j, v_j>,   fit = alpha* v_j

with alpha*/dsigma and the best-fit residual. Interpretation (per review):
    best-fit rel ~ 0 and alpha ~ -dsigma      -> sign error in the diagnostic
    best-fit rel ~ 0 and alpha/dsigma = const -> sigma scaling mismatch
    best-fit rel ~ 0.09 too                   -> not a single-Euler relation
                                                 or an index mismatch
    one of C1-C4 drops to ~1e-3               -> convention identified; fix the
                                                 pair diagnostic accordingly

fp16 note: shards store latents/preds in fp16; the fp16 quantisation floor on
this rel-err metric is ~1e-3, so treat anything at or below that as exact.

    PYTHONPATH=. python -m training.diagnose_transition \
        --teacher /mnt/HDD_12TB/bam_ki/flux_fill/router_teacher_1024 --shards 6

Argparse help strings contain no bare percent characters.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch


def rel_err(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float((pred - target).norm() / target.norm().clamp_min(1e-12))


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten(), b.flatten()
    return float(torch.dot(a, b) /
                 (a.norm() * b.norm()).clamp_min(1e-12))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(
        description="Verify the teacher dump's step-transition convention")
    ap.add_argument("--teacher", required=True, help="router-teacher dump dir")
    ap.add_argument("--shards", type=int, default=6, help="shards to scan")
    ap.add_argument("--max-transitions", type=int, default=200,
                    help="cap on transitions scanned")
    a = ap.parse_args()

    idx = json.load(open(Path(a.teacher) / "index.json"))
    shard_names = sorted(idx["shards"])[: a.shards]

    cand_rel = defaultdict(list)
    cand_cos = defaultdict(list)
    alpha_ratio, bestfit_rel = [], []
    fp16_floor, fp16_excess, dz_over_z = [], [], []
    n_done = 0

    for name in shard_names:
        sh = torch.load(Path(a.teacher) / name, map_location="cpu",
                        weights_only=False)
        lat = sh["latents"].float()          # [S, N, 64]
        v = sh["preds"].float()
        sig = sh["sigmas"].float()
        S = lat.shape[0]
        for j in range(S - 1):
            if n_done >= a.max_transitions:
                break
            dz = lat[j + 1] - lat[j]
            if float(dz.norm()) < 1e-8:
                continue
            ds = float(sig[j + 1] - sig[j])
            cands = {
                "C1 +ds*v_j": ds * v[j],
                "C2 -ds*v_j": -ds * v[j],
                "C3 +ds*v_j1": ds * v[j + 1],
                "C4 trapezoid": ds * 0.5 * (v[j] + v[j + 1]),
            }
            for cname, c in cands.items():
                cand_rel[cname].append(rel_err(c, dz))
                cand_cos[cname].append(cosine(c, dz))
            va = v[j].flatten()
            alpha = float(torch.dot(dz.flatten(), va) /
                          torch.dot(va, va).clamp_min(1e-12))
            alpha_ratio.append(alpha / ds if ds != 0 else float("nan"))
            bestfit_rel.append(rel_err(alpha * v[j], dz))
            # fp16-cancellation attribution: dz is a small difference of two
            # LARGE fp16-stored latents, so quantization noise is amplified by
            # ||z|| / ||dz||. Predicted rel-err floor from storage alone:
            #   sqrt(2) * eps_rms * ||z|| / ||dz||,  eps_rms ~ 2^-10/sqrt(12)
            eps_rms = (2.0 ** -10) / (12 ** 0.5)
            z_rms = float(0.5 * (lat[j].norm() + lat[j + 1].norm()))
            floor = (2 ** 0.5) * eps_rms * z_rms / float(dz.norm())
            fp16_floor.append(floor)
            fp16_excess.append(rel_err(ds * v[j], dz) / max(floor, 1e-12))
            dz_over_z.append(float(dz.norm()) / z_rms)
            n_done += 1
        if n_done >= a.max_transitions:
            break

    print(f"== transition-convention check over {n_done} consecutive "
          f"transitions ({len(shard_names)} shards) ==")
    print(f"{'candidate':14s} {'rel mean':>9s} {'rel max':>9s} {'cos mean':>9s}")
    best = None
    for cname in cand_rel:
        rs, cs = cand_rel[cname], cand_cos[cname]
        m = sum(rs) / len(rs)
        print(f"{cname:14s} {m:9.3e} {max(rs):9.3e} {sum(cs)/len(cs):9.4f}")
        if best is None or m < best[1]:
            best = (cname, m)

    ar = torch.tensor(alpha_ratio)
    bf = torch.tensor(bestfit_rel)
    print("== best-fit scalar (dz ~ alpha* v_j) ==")
    print(f"  alpha*/dsigma: mean {float(ar.mean()):.4f}  std "
          f"{float(ar.std()):.4f}  min {float(ar.min()):.4f}  "
          f"max {float(ar.max()):.4f}")
    print(f"  best-fit rel:  mean {float(bf.mean()):.3e}  "
          f"max {float(bf.max()):.3e}")

    fl = torch.tensor(fp16_floor); ex = torch.tensor(fp16_excess)
    dzz = torch.tensor(dz_over_z)
    print("== fp16-cancellation attribution ==")
    print(f"  ||dz||/||z||:            mean {float(dzz.mean()):.4f}  "
          f"min {float(dzz.min()):.4f}  max {float(dzz.max()):.4f}")
    print(f"  predicted fp16 floor:    mean {float(fl.mean()):.3e}  "
          f"max {float(fl.max()):.3e}")
    print(f"  measured / floor:        mean {float(ex.mean()):.2f}  "
          f"median {float(ex.median()):.2f}  max {float(ex.max()):.2f}")
    # correlation across transitions: pure-storage noise makes rel err track
    # 1/(||dz||/||z||) exactly
    c1 = torch.tensor(cand_rel["C1 +ds*v_j"])
    inv = 1.0 / dzz
    cor = float(torch.corrcoef(torch.stack([c1, inv]))[0, 1])
    print(f"  corr(C1 rel err, ||z||/||dz||): {cor:.3f}")
    print("READ (fp16 attribution):")
    print("  measured/floor ~ 1-2 and corr ~ 1  -> the 8e-2 is STORAGE noise; "
          "convention is exact Euler; dz carries no real extra signal beyond "
          "v_a at offset 1")
    print("  measured/floor >> 2 (e.g. > 5)     -> a real non-Euler component "
          "exists in the stored variables; dz has genuine extra signal")
    print("READ:")
    print(f"  best candidate: {best[0]} (mean rel {best[1]:.3e}; fp16 floor "
          "~1e-3 counts as exact)")
    print("  alpha/dsigma ~ +1 & best-fit ~1e-3  -> C1 correct, earlier 9e-2 "
          "came from elsewhere (check pair-sampler sigma indexing)")
    print("  alpha/dsigma ~ -1                   -> sign convention flipped")
    print("  alpha/dsigma ~ const != +-1         -> sigma scaling (shifted "
          "schedule) mismatch; use that constant")
    print("  best-fit rel also ~9e-2             -> transition is NOT a "
          "single-Euler step in the stored variables (guidance/scheduler "
          "internals); dz then carries real extra signal beyond v_a")


if __name__ == "__main__":
    main()
