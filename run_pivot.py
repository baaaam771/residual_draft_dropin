#!/usr/bin/env python
"""run_pivot.py — run the repo's cached_flux_fill with the error-head router,
WITHOUT editing the repo.

Both repos stay separate git projects:

    export PYTHONPATH=$HOME/residual_draft_dropin:$HOME/Frozen-flux-draft-DiT
    python run_pivot.py --manifest data/coco_manifest_1024.json ... \
        --selector mbd_draft --draft-ckpt $CKPT_V2/last.pt ...

All arguments are passed through unchanged to samplers.cached_flux_fill.main.
Relative paths (data/..., out/...) resolve against the CWD, so run this from
the Frozen-flux-draft-DiT root (or pass absolute paths).

Mechanism: cached_flux_fill.main() does
    from models.drafts.router_draft import RouterDraft
    draft = RouterDraft.load(a.draft_ckpt, dev)
at call time, so replacing RouterDraft.load BEFORE main() runs makes every
--draft-ckpt go through the auto-detect factory: checkpoints written by
resdraft.training.train_residual_draft (carrying model_config) load as
ErrorHeadRouter; legacy CNN-router checkpoints fall back to the ORIGINAL
RouterDraft.load (captured below — no recursion).
"""
from __future__ import annotations

import sys

import torch

from models.drafts.router_draft import RouterDraft          # repo (untouched)
from resdraft.models.error_head_router import ErrorHeadRouter

_orig_load = RouterDraft.load                               # bound classmethod


def _autodetect_load(ckpt_path, device, *args, **kwargs):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model_config" in ck:
        print(f"[run_pivot] {ckpt_path} -> ErrorHeadRouter "
              f"(config={ck['model_config']})")
        return ErrorHeadRouter.load(ckpt_path, device)
    print(f"[run_pivot] {ckpt_path} -> legacy RouterDraft")
    return _orig_load(ckpt_path, device, *args, **kwargs)


RouterDraft.load = _autodetect_load

if __name__ == "__main__":
    from samplers.cached_flux_fill import main              # repo (untouched)
    sys.argv[0] = "cached_flux_fill"                        # argparse prog name
    main()
