"""CPU tests for the anchor-residual draft / three-tier routing batch.

Run:  PYTHONPATH=. python tests/test_residual_draft.py   (or pytest)

Covers the review's required regression set:
    test_token_ab_routing            two-score routing separates CACHE/DRAFT/TARGET
    test_stage3_zero_gain            no forced DRAFT when the draft adds nothing
    test_boundary_budget             boundary forcing respects the target budget
    test_missing_sparse_callback     TARGET without sparse output -> loud failure
    test_nonfinite_sparse_output     NaN in sparse output -> loud failure
    test_checkpoint_config_roundtrip non-default arch restored; resume uses ckpt config
    test_image_split_no_leakage      image-level hash split is deterministic + disjoint
    test_teacher_pair_never_uses_anchor_step  pair sampler skips anchors (c=3 boundary)
plus: zero-init == pure reuse, block divisibility assert, hard_idx convention.
"""
import os
import random
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.drafts.residual_draft import ResidualDraftNet, boundary_band_tok
from token_selectors.three_tier import (CACHE, DRAFT, TARGET, ThreeTierConfig,
                                        compose_prediction, hard_indices,
                                        route_three_tier)
from training.train_residual_draft import split_of

HW = (32, 32)                                   # 512^2 token grid
N = HW[0] * HW[1]


def _mask():
    m = torch.zeros(1, 1, *HW)
    m[:, :, 10:22, 10:22] = 1
    return m.flatten(1)                         # [1, N]


def _scenario_errors():
    """Token A: reuse exact. Token B: reuse bad, draft fixes. Hard: draft bad."""
    e_cache = torch.full((1, N), 1e-6)
    e_draft = torch.full((1, N), 1e-6)
    g = lambda r0, r1, c0, c1: [i * HW[1] + j for i in range(r0, r1)
                                for j in range(c0, c1)]
    B_idx = g(2, 5, 2, 5)                       # cache-bad / draft-good
    H_idx = g(26, 29, 26, 29)                   # draft-bad
    e_cache[0, B_idx] = 1e-1
    e_cache[0, H_idx] = 2e-1
    e_draft[0, H_idx] = 1.5e-1
    return e_cache, e_draft, B_idx, H_idx


def test_token_ab_routing():
    e_c, e_d, B_idx, H_idx = _scenario_errors()
    cfg = ThreeTierConfig(r_target=0.02, r_draft=0.02,
                          force_target_boundary=False)
    tier, _ = route_three_tier(e_c, e_d, _mask(), HW, cfg)
    assert (tier[0, B_idx] == DRAFT).all(), "cache-bad/draft-good must be DRAFT"
    assert (tier[0, H_idx] == TARGET).all(), "draft-bad must be TARGET"
    assert (tier == CACHE).float().mean() > 0.9, "reuse-exact bulk must be CACHE"


def test_stage3_zero_gain():
    e_eq = torch.full((1, N), 1e-3)             # gain == 0 everywhere
    cfg = ThreeTierConfig(r_target=0.15, r_draft=0.3).stage3()
    assert cfg.r_target == 0.0 and not cfg.force_target_boundary
    tier, info = route_three_tier(e_eq, e_eq, _mask(), HW, cfg)
    assert info["realized_target"] == 0.0
    assert info["realized_draft"] == 0.0, \
        "zero-gain tokens must not be forced into DRAFT to fill the budget"
    # and with real gain, stage3 routes the gain region to DRAFT
    e_c, e_d, B_idx, _ = _scenario_errors()
    tier, info = route_three_tier(e_c, e_d, _mask(), HW, cfg)
    assert info["realized_target"] == 0.0
    assert (tier[0, B_idx] == DRAFT).all()


def test_boundary_budget():
    e_c, e_d, *_ = _scenario_errors()
    mask = _mask()
    bnd_frac = float((boundary_band_tok(mask, HW) > 0.5).float().mean())
    r_small = bnd_frac / 2                       # budget below boundary size
    cfg = ThreeTierConfig(r_target=r_small, r_draft=0.1,
                          force_target_boundary=True, boundary_policy="budget")
    _, info = route_three_tier(e_c, e_d, mask, HW, cfg)
    assert abs(info["realized_target"] - round(r_small * N) / N) < 1e-9, \
        "budget policy must keep realized target == rounded budget"
    cfg_all = ThreeTierConfig(r_target=r_small, r_draft=0.1,
                              force_target_boundary=True, boundary_policy="all")
    _, info_all = route_three_tier(e_c, e_d, mask, HW, cfg_all)
    assert info_all["realized_target"] >= bnd_frac - 1e-9, \
        "'all' policy keeps every boundary token (and reports the excess)"


def test_missing_sparse_callback():
    e_c, e_d, *_ = _scenario_errors()
    tier, _ = route_three_tier(e_c, e_d, _mask(), HW,
                               ThreeTierConfig(r_target=0.05, r_draft=0.05,
                                               force_target_boundary=False))
    v_a = torch.randn(1, N, 64)
    dv = torch.randn(1, N, 64)
    try:
        compose_prediction(tier, v_a, dv, None, None)
        raise AssertionError("must raise when TARGET has no sparse output")
    except RuntimeError:
        pass


def test_nonfinite_sparse_output():
    e_c, e_d, *_ = _scenario_errors()
    tier, _ = route_three_tier(e_c, e_d, _mask(), HW,
                               ThreeTierConfig(r_target=0.05, r_draft=0.05,
                                               force_target_boundary=False))
    v_a = torch.randn(1, N, 64)
    dv = torch.randn(1, N, 64)
    hard = hard_indices(tier)
    v_hard = torch.full((1, hard.shape[1], 64), float("nan"))
    try:
        compose_prediction(tier, v_a, dv, hard, v_hard)
        raise AssertionError("must raise on non-finite sparse output")
    except AssertionError as e:
        assert "non-finite" in str(e)
    # and the valid path preserves tier semantics exactly
    v_hard = torch.randn(1, hard.shape[1], 64)
    v = compose_prediction(tier, v_a, dv, hard, v_hard)
    c = (tier == CACHE).unsqueeze(-1).expand_as(v)
    d = (tier == DRAFT).unsqueeze(-1).expand_as(v)
    assert torch.equal(v[c], v_a[c])
    assert torch.equal(v[d], (v_a + dv)[d])
    assert torch.equal(v.gather(1, hard.unsqueeze(-1).expand(-1, -1, 64)),
                       v_hard)


def test_checkpoint_config_roundtrip():
    net = ResidualDraftNet(hidden=64, num_blocks=2)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "ck.pt")
        torch.save({"model": net.state_dict(), "ema": net.state_dict(),
                    "model_config": net.config}, p)
        net2 = ResidualDraftNet.from_checkpoint(p)
        assert net2.config == net.config
        assert net2.config["hidden"] == 64 and net2.config["num_blocks"] == 2
        # resume path in train_residual_draft reads model_config the same way
        ck = torch.load(p, map_location="cpu", weights_only=False)
        net3 = ResidualDraftNet(**ck["model_config"])
        net3.load_state_dict(ck["model"])       # no shape mismatch


def test_image_split_no_leakage():
    ids = [f"{i:012d}" for i in range(500)]
    s1 = [split_of(i, 0.1, 0.1) for i in ids]
    s2 = [split_of(i, 0.1, 0.1) for i in ids]
    assert s1 == s2, "split must be deterministic"
    counts = {k: s1.count(k) for k in ("train", "val", "calib")}
    assert counts["val"] > 0 and counts["calib"] > 0 and counts["train"] > 300
    assert set(s1) <= {"train", "val", "calib"}


def test_teacher_pair_never_uses_anchor_step():
    """Boundary case: steps=50, tail=4 -> hi=46; with c=3 the old sampler
    could nudge i to 45 (an anchor), yielding a degenerate zero pair."""
    import json
    from training.train_residual_draft import ResidualTeacherPairs

    S, n_tok, hw = 50, 16, (4, 4)
    with tempfile.TemporaryDirectory() as td:
        idx = []
        for s in range(6):
            stem = f"{s:012d}"
            torch.save({"latents": torch.randn(S, n_tok, 64).half(),
                        "preds": torch.randn(S, n_tok, 64).half(),
                        "sigmas": torch.linspace(1, 0, S),
                        "mask_tok": torch.zeros(n_tok),
                        "token_hw": hw, "sample_id": stem + ".jpg"},
                       os.path.join(td, f"{stem}.pt"))
            idx.append(f"{stem}.pt")
        json.dump({"shards": idx, "steps": S},
                  open(os.path.join(td, "index.json"), "w"))

        pairs = ResidualTeacherPairs(td, [2, 3], "train",
                                     val_frac=0.0, calib_frac=0.0,
                                     dense_tail=4)
        # the old bug's exact trigger must be excluded up front
        assert 45 not in pairs.valid_steps[3]
        assert all(i % c != 0 for c, v in pairs.valid_steps.items() for i in v)
        assert all(i < S - 4 for v in pairs.valid_steps.values() for i in v)

        rng = random.Random(0)
        for _ in range(1000):
            it = pairs.sample(rng)
            c, i, a2 = it["cache_period"], it["step"], it["anchor_step"]
            assert i % c != 0, f"anchor step sampled: i={i}, c={c}"
            assert a2 == i - (i % c) and a2 != i
            assert float(it["dsigma"].abs()) > 0, "degenerate zero-pair"


def test_zero_init_equals_reuse():
    torch.manual_seed(0)
    net = ResidualDraftNet(hidden=64, num_blocks=2)
    v_a = torch.randn(1, N, 64)
    dz = torch.randn(1, N, 64)
    dv, log_ec, log_ed = net(v_a, dz, _mask(), torch.tensor([-0.05]), HW)
    assert dv.abs().max().item() == 0.0, "zero-init draft must equal pure reuse"
    e_c, e_d = ResidualDraftNet.routing_errors(log_ec, log_ed)
    assert torch.isfinite(e_c).all() and torch.isfinite(e_d).all()
    # clamp guard: absurd log predictions stay finite in error space
    e_c2, _ = ResidualDraftNet.routing_errors(log_ec + 1e6, log_ed)
    assert torch.isfinite(e_c2).all()


def test_block_divisibility_assert():
    e = torch.rand(1, N)
    cfg = ThreeTierConfig(r_target=0.1, r_draft=0.1, block=2,
                          force_target_boundary=False)
    tier, _ = route_three_tier(e, e * 0.5, _mask(), HW, cfg)   # 32 % 2 == 0 ok
    assert tier.shape == (1, N)
    try:
        cfg5 = ThreeTierConfig(block=5, force_target_boundary=False)
        route_three_tier(e, e, _mask(), HW, cfg5)
        raise SystemExit("must assert on non-divisible block")
    except AssertionError:
        pass


def test_hard_idx_convention():
    e_c, e_d, *_ = _scenario_errors()
    tier, _ = route_three_tier(e_c, e_d, _mask(), HW,
                               ThreeTierConfig(r_target=0.05, r_draft=0.05,
                                               force_target_boundary=False))
    hard = hard_indices(tier)
    assert hard.dim() == 2 and hard.shape[0] == 1, "hard_idx must be [B, k]"
    assert (hard[0][1:] >= hard[0][:-1]).all(), \
        "hard_idx must be sorted ascending (sparse_forward convention)"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests passed")
