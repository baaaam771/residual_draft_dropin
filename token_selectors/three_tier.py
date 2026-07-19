"""token_selectors.three_tier — CACHE / DRAFT / TARGET routing from two scores.

Inputs are per-token error estimates in ERROR space (ResidualDraftNet.
routing_errors), or ground-truth errors for the ROUTING UPPER BOUND (same
function, no separate code path):

    e_cache  ||v_a - v_t||^2            reuse error
    e_draft  ||v_a + dv_hat - v_t||^2   remaining draft error
    gain     e_cache - e_draft          draft improvement

Budget mode decision (default):
  1. boundary-band tokens -> TARGET, COUNTING AGAINST the r_target budget
     (policy "budget": boundary exceeding the budget keeps only top-e_draft;
      policy "all": keep all boundary, realized ratio may exceed and is logged)
  2. remaining target budget -> largest e_draft
  3. r_draft budget          -> largest gain, only where gain > 0
     (require_positive_gain; zero-gain tokens stay CACHE even inside budget)
  4. rest -> CACHE

Threshold mode: TARGET if e_draft >= tau_target; DRAFT if gain >= tau_gain.

hard_indices() returns [B, k] SORTED ASCENDING — the convention
FluxSparseRunner.sparse_forward expects (coalesced gathers). block > 1 does
true block-level assignment on the (hp, wp) grid with a divisibility assert.

Always log the realized ratios from `info` next to nominal budgets — with
boundary_policy="all" or require_positive_gain the arm name's nominal r is
not the executed r, and wall-clock comparisons must use the realized value.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from models.drafts.residual_draft import boundary_band_tok

CACHE, DRAFT, TARGET = 0, 1, 2


@dataclass
class ThreeTierConfig:
    budget_mode: bool = True
    r_target: float = 0.15
    r_draft: float = 0.35
    tau_target: float = 1e-3      # threshold mode, error space
    tau_gain: float = 1e-4
    force_target_boundary: bool = True
    boundary_policy: str = "budget"     # "budget" | "all"
    require_positive_gain: bool = True
    block: int = 1

    def stage3(self) -> "ThreeTierConfig":
        """Stage 3 arm (draft_only): no TARGET tier at sparse steps."""
        import dataclasses
        return dataclasses.replace(self, r_target=0.0,
                                   force_target_boundary=False)


def _pool_scores(s: torch.Tensor, token_hw, block: int) -> torch.Tensor:
    if block <= 1:
        return s
    hp, wp = token_hw
    assert hp % block == 0 and wp % block == 0, (
        f"token grid {hp}x{wp} not divisible by block={block}")
    return F.avg_pool2d(s.view(-1, 1, hp, wp), block).flatten(1)


def _expand_tier(tier_b: torch.Tensor, token_hw, block: int) -> torch.Tensor:
    if block <= 1:
        return tier_b
    hp, wp = token_hw
    g = tier_b.view(-1, 1, hp // block, wp // block).float()
    return F.interpolate(g, scale_factor=block, mode="nearest").long().flatten(1)


@torch.no_grad()
def route_three_tier(e_cache: torch.Tensor, e_draft: torch.Tensor,
                     mask_tok: torch.Tensor, token_hw,
                     cfg: ThreeTierConfig):
    """e_cache/e_draft/mask_tok: [1, N] -> (tier [1, N] long, info dict)."""
    assert e_cache.shape[0] == 1, "routing assumes batch size 1"
    b = cfg.block
    ec = _pool_scores(e_cache, token_hw, b).flatten()
    ed = _pool_scores(e_draft, token_hw, b).flatten()
    gain = ec - ed
    n = ec.numel()

    forced = torch.zeros(n, dtype=torch.bool, device=ec.device)
    if cfg.force_target_boundary:
        bnd = boundary_band_tok(mask_tok, token_hw)
        forced = _pool_scores(bnd, token_hw, b).flatten() > 0.0

    tier = torch.full((n,), CACHE, dtype=torch.long, device=ec.device)

    if cfg.budget_mode:
        k_t = int(round(cfg.r_target * n))
        k_d = int(round(cfg.r_draft * n))
        f_idx = forced.nonzero(as_tuple=True)[0]
        if f_idx.numel() > k_t and cfg.boundary_policy == "budget":
            keep = torch.argsort(ed[f_idx], descending=True)[:k_t]
            f_idx = f_idx[keep]
        tier[f_idx] = TARGET
        k_rem = max(k_t - int(f_idx.numel()), 0)
        if k_rem > 0:
            cand = (tier == CACHE).nonzero(as_tuple=True)[0]
            tier[cand[torch.argsort(ed[cand], descending=True)[:k_rem]]] = TARGET
        if k_d > 0:
            cand = (tier == CACHE).nonzero(as_tuple=True)[0]
            sel = cand[torch.argsort(gain[cand], descending=True)[:k_d]]
            if cfg.require_positive_gain:
                sel = sel[gain[sel] > 0]
            tier[sel] = DRAFT
    else:
        tier[gain >= cfg.tau_gain] = DRAFT
        tier[ed >= cfg.tau_target] = TARGET
        if cfg.force_target_boundary:
            tier[forced] = TARGET

    tier = _expand_tier(tier.unsqueeze(0), token_hw, b)

    N = tier.numel()
    info = {
        "realized_cache": float((tier == CACHE).sum()) / N,
        "realized_draft": float((tier == DRAFT).sum()) / N,
        "realized_target": float((tier == TARGET).sum()) / N,
        "nominal_r_target": cfg.r_target if cfg.budget_mode else float("nan"),
        "nominal_r_draft": cfg.r_draft if cfg.budget_mode else float("nan"),
        "forced_boundary_frac": float(forced.float().mean()),
    }
    return tier, info


def hard_indices(tier: torch.Tensor) -> torch.Tensor:
    """tier [1, N] -> TARGET token indices [1, k], sorted ascending
    (FluxSparseRunner convention). k may be 0."""
    idx = (tier[0] == TARGET).nonzero(as_tuple=True)[0]
    return torch.sort(idx)[0].unsqueeze(0)


@torch.no_grad()
def compose_prediction(tier: torch.Tensor, v_anchor: torch.Tensor,
                       dv: torch.Tensor, hard_idx: torch.Tensor | None,
                       v_hard: torch.Tensor | None) -> torch.Tensor:
    """v_i = v_a (CACHE) | v_a + dv (DRAFT) | fresh sparse output (TARGET).
    tier [1,N]; v_anchor/dv [1,N,C]; v_hard [1,k,C] on hard_idx rows.
    Fails loudly if TARGET tokens exist without a sparse output."""
    has_target = bool((tier == TARGET).any())
    if has_target and (v_hard is None or hard_idx is None):
        raise RuntimeError("TARGET tokens routed but no sparse output was "
                           "provided — sparse_forward missing or integration "
                           "broken")
    v = v_anchor.clone()
    d = (tier == DRAFT).unsqueeze(-1)
    v = torch.where(d, v_anchor + dv.to(v.dtype), v)
    if has_target:
        assert v_hard.shape[:2] == hard_idx.shape and \
               v_hard.shape[-1] == v.shape[-1], (
            f"v_hard {tuple(v_hard.shape)} vs hard_idx {tuple(hard_idx.shape)}")
        assert torch.isfinite(v_hard).all(), \
            "non-finite values in sparse TARGET output"
        v.scatter_(1, hard_idx.unsqueeze(-1).expand(-1, -1, v.shape[-1]),
                   v_hard.to(v.dtype))
    return v
