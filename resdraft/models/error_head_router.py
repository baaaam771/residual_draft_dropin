"""models.drafts.error_head_router — the residual-draft error head as a
selector router (the pivot from the residual-draft negative result).

The residual-draft study showed the anchor-to-current velocity residual is
information-inaccessible (kNN ceiling 1.06 vs zero-predictor), but its
MAGNITUDE ranks well: the cache-error head reaches spearman ~0.75 against
log ||v_t - v_a||^2 — the exact quantity the paper's learned router
(models.drafts.cnn_router, ~1M, quantile labels) approximates. This wrapper
exposes that head through the SAME interface as RouterDraft, so it drops into
selectors.combo's eta term with zero sampler changes:

    scores(latents, mask_tok, cache, t, grid) -> [B, N]

Inputs assembled exactly as at a sparse step (all free):
    v_a      = cache.final_prediction
    dz       = latents - cache.anchor_latents
    sigma_t  = t / 1000        (FlowMatchEuler: timesteps = sigmas * 1000)
    dsigma   = sigma_t - cache.anchor_sigma
    x0_a     = cache.anchor_clean_estimate       (if the ckpt was trained with it)

The returned score is the raw log cache-error head output — monotone in the
predicted error, which is all rank_norm needs. Latency is inside the sampled
wall-clock, same accounting as RouterDraft.
"""
from __future__ import annotations

from pathlib import Path

import torch

from resdraft.models.residual_draft import ResidualDraftNet
from models.flux_cache import FluxAnchorCache
from utils.token_mapping import TokenGrid


class ErrorHeadRouter:
    def __init__(self, model: ResidualDraftNet, device):
        self.model = model.to(device).float().eval()
        self.device = device

    @classmethod
    def load(cls, ckpt_path: str, device):
        return cls(ResidualDraftNet.from_checkpoint(str(Path(ckpt_path))), device)

    @torch.no_grad()
    def scores(self, latents: torch.Tensor, mask_tok: torch.Tensor,
               cache: FluxAnchorCache, t: torch.Tensor,
               grid: TokenGrid) -> torch.Tensor:
        """latents [B, N, 64] packed z_t -> [B, N] predicted log reuse error."""
        assert cache.final_prediction is not None and \
            cache.anchor_latents is not None, \
            "ErrorHeadRouter needs an anchored cache (set_anchor_context)"
        v_a = cache.final_prediction.float()
        dz = latents.float() - cache.anchor_latents.float()
        tt = (t if torch.is_tensor(t) else torch.tensor(t)).reshape(-1).float()
        sigma_t = (tt / 1000.0).to(latents.device)       # FlowMatchEuler convention
        dsigma = sigma_t - float(cache.anchor_sigma)
        kw = {}
        cfg = self.model.config
        if cfg.get("use_latent"):
            kw["z_t"] = latents.float()
        if cfg.get("use_anchor_x0"):
            assert cache.anchor_clean_estimate is not None, \
                "ckpt trained with use_anchor_x0 but cache has no anchor x0"
            kw["x0_anchor"] = cache.anchor_clean_estimate.float()
        if cfg.get("use_sigma_t"):
            kw["sigma_t"] = sigma_t
        _, log_ec, _ = self.model(v_a, dz,
                                  mask_tok.to(latents.device).float(),
                                  dsigma, grid.token_hw, **kw)
        return log_ec                                    # monotone; rank_norm follows


def load_draft(ckpt_path: str, device):
    """Factory: auto-detect the draft kind from the checkpoint contents.
    Checkpoints written by training.train_residual_draft carry model_config
    -> ErrorHeadRouter; anything else -> the original RouterDraft."""
    ck = torch.load(Path(ckpt_path), map_location="cpu", weights_only=False)
    if "model_config" in ck:
        return ErrorHeadRouter.load(ckpt_path, device)
    from models.drafts.router_draft import RouterDraft
    return RouterDraft.load(ckpt_path, device)
