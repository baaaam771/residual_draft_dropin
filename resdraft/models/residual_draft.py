"""models.drafts.residual_draft — anchor-residual draft with two routing scores.

Extends the cache-or-recompute decision of the anchored backend to a
three-tier reuse / approximate / recompute decision. Per image token:

    dv_hat           [B, N, 64]  anchor-to-current velocity residual
                                 (v_draft = cache.final_prediction + dv_hat)
    log_e_cache_hat  [B, N]      log ||v_a - v_t||^2          (staleness error)
    log_e_draft_hat  [B, N]      log ||v_a + dv_hat - v_t||^2 (remaining error)

Both scores are needed: draft error alone cannot separate "reuse already
exact" from "reuse stale but draft fixes it" (both have low draft error);
gain = e_cache - e_draft identifies the second group as DRAFT tokens.

Interface follows models.drafts.cnn_router: packed [B, N, 64] tensors plus
token_hw, everything free at a sparse step (cache.final_prediction,
z_t - cache.anchor_latents, cache.mask coverage, sigma_t - sigma_a).

Error heads regress LOG error (heavy-tailed target; only ranking matters
for budget routing). Residual head is zero-initialized: at init the draft
output equals pure anchored reuse (r=0). That fixes the starting point only
— it does not guarantee the trained draft beats reuse per sample; the CACHE
tier plus require_positive_gain is the runtime fallback and
training.eval_residual_router is the offline gate.

Error-head gradients flow into the shared trunk (only their TARGETS are
detached); set detach_error_trunk=True if L_res degrades when they are on.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_ERR_CLAMP = (-30.0, 20.0)   # exp() guard for routing (closed-loop safety)


class _ConvBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


def _error_head(hidden: int) -> nn.Module:
    return nn.Sequential(nn.Conv2d(hidden, hidden // 2, 1), nn.SiLU(),
                         nn.Conv2d(hidden // 2, 1, 1))


def boundary_band_tok(mask_tok: torch.Tensor, token_hw, kernel: int = 3) -> torch.Tensor:
    """[B, N] coverage -> boundary band [B, N] in {0,1}. Same morphological
    gradient as token_selectors.boundary.boundary_score (kernel=3 == +-16 px)."""
    hp, wp = token_hw
    m = (mask_tok.view(-1, 1, hp, wp) > 0.5).float()
    pad = kernel // 2
    dil = F.max_pool2d(m, kernel, stride=1, padding=pad)
    ero = -F.max_pool2d(-m, kernel, stride=1, padding=pad)
    return (dil - ero).clamp(0, 1).flatten(1)


class ResidualDraftNet(nn.Module):
    """forward(v_anchor, dz, mask_tok, dsigma, token_hw) ->
    (dv [B,N,64], log_e_cache [B,N], log_e_draft [B,N]). ~3M params default."""

    def __init__(self, latent_ch: int = 64, hidden: int = 192,
                 num_blocks: int = 4, detach_error_trunk: bool = False,
                 use_latent: bool = False, use_anchor_x0: bool = False,
                 use_sigma_t: bool = False):
        """Content inputs (v2, all default OFF for v1-checkpoint compat):
        use_latent    += z_t   [B,N,64]  current packed latent (free at a sparse step)
        use_anchor_x0 += x0_a  [B,N,64]  cache.anchor_clean_estimate (image content)
        use_sigma_t   += sigma_t plane   absolute timestep (drift is sigma-dependent)

        v1 lesson: with only (v_a, dz, mask, dsigma), offset-1 pairs are a
        deterministic function of the anchor (dz = dsigma * v_a on the Euler
        trajectory) and carry no image content, so the residual head cannot
        beat pure reuse (mse_ratio ~ 1.0). Content inputs are the fix to test.
        """
        super().__init__()
        self.config = {"latent_ch": latent_ch, "hidden": hidden,
                       "num_blocks": num_blocks,
                       "detach_error_trunk": detach_error_trunk,
                       "use_latent": use_latent,
                       "use_anchor_x0": use_anchor_x0,
                       "use_sigma_t": use_sigma_t}
        self.latent_ch = latent_ch
        self.detach_error_trunk = detach_error_trunk
        self.use_latent = use_latent
        self.use_anchor_x0 = use_anchor_x0
        self.use_sigma_t = use_sigma_t
        in_ch = (2 + use_latent + use_anchor_x0) * latent_ch \
            + 3 + int(use_sigma_t)
        self.stem = nn.Conv2d(in_ch, hidden, 3, padding=1)
        self.blocks = nn.ModuleList(_ConvBlock(hidden) for _ in range(num_blocks))
        self.out_norm = nn.GroupNorm(8, hidden)
        self.residual_head = nn.Conv2d(hidden, latent_ch, 1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        self.cache_err_head = _error_head(hidden)
        self.draft_err_head = _error_head(hidden)

    def forward(self, v_anchor: torch.Tensor, dz: torch.Tensor,
                mask_tok: torch.Tensor, dsigma: torch.Tensor, token_hw,
                z_t: torch.Tensor | None = None,
                x0_anchor: torch.Tensor | None = None,
                sigma_t: torch.Tensor | None = None):
        hp, wp = token_hw
        B, N, C = v_anchor.shape
        assert N == hp * wp and C == self.latent_ch, (N, C, token_hw)
        to_grid = lambda t: t.transpose(1, 2).reshape(B, C, hp, wp)
        bnd = boundary_band_tok(mask_tok, token_hw)
        plane_list = [mask_tok, bnd, dsigma.view(B, 1).expand(B, N)]
        if self.use_sigma_t:
            assert sigma_t is not None, "config use_sigma_t=True needs sigma_t"
            plane_list.append(sigma_t.view(B, 1).expand(B, N))
        planes = torch.stack(plane_list, dim=1)            # [B, 3(+1), N]
        feats = [to_grid(v_anchor.float()), to_grid(dz.float())]
        if self.use_latent:
            assert z_t is not None, "config use_latent=True needs z_t"
            feats.append(to_grid(z_t.float()))
        if self.use_anchor_x0:
            assert x0_anchor is not None, \
                "config use_anchor_x0=True needs x0_anchor"
            feats.append(to_grid(x0_anchor.float()))
        x = torch.cat(feats + [planes.float().reshape(B, -1, hp, wp)], dim=1)
        h = self.stem(x)
        for blk in self.blocks:
            h = blk(h)
        h = F.silu(self.out_norm(h))
        he = h.detach() if self.detach_error_trunk else h
        dv = self.residual_head(h).reshape(B, C, N).transpose(1, 2)     # [B,N,C]
        log_ec = self.cache_err_head(he).flatten(1)                     # [B,N]
        log_ed = self.draft_err_head(he).flatten(1)
        return dv, log_ec, log_ed

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def routing_errors(log_ec: torch.Tensor, log_ed: torch.Tensor):
        """Log heads -> error space with a clamp so early-training or
        distribution-shifted predictions cannot produce inf (review fix)."""
        lo, hi = LOG_ERR_CLAMP
        return log_ec.float().clamp(lo, hi).exp(), \
               log_ed.float().clamp(lo, hi).exp()

    @staticmethod
    def from_checkpoint(path: str, map_location="cpu") -> "ResidualDraftNet":
        ck = torch.load(path, map_location=map_location, weights_only=False)
        if "model_config" not in ck:
            raise KeyError(f"{path} has no model_config — re-save with "
                           "training.train_residual_draft")
        net = ResidualDraftNet(**ck["model_config"])
        net.load_state_dict(ck["ema"] if "ema" in ck else ck["model"])
        return net
