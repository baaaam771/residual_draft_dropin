"""samplers.cached_flux_fill — Stage 4–7: FreqSpec-Cache-FLUX sampling loop.

Per denoising step, one of three modes (plan Sec. 1):

  dense anchor step   every c steps: full transformer, record depth-aligned cache
  sparse refresh step selector -> hard tokens -> single-stream selective refresh;
                      easy tokens reuse same-depth cached states as K/V context
                      and the anchor's final prediction as their output
  (r = 0)             anchored prediction reuse: scheduler-only step, zero
                      model calls between anchors (DACE's strongest draft-free
                      baseline; the sampler falls back to this when --ratio 0)

Methods exposed through --method:
  dense           reduced-step dense baseline (use --steps to sweep 50/40/30/...)
  reuse           r = 0 anchored reuse, anchor period --cache-period
  cache_sparse    anchor + selective refresh with --selector
                    {mask, mask_boundary, mask_delta, mask_frequency,
                     mbd, mbfd, mbfd_draft, random, oracle}
  hetero          Q1 diagnostic: dense every step, log per-token temporal change
                  in/out mask (the DACE two-factor deployment test)

All methods consume the frozen manifest (data.dataset) so every method sees the
same image / mask / prompt / latent seed.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from data.dataset import FluxFillBenchmark
from data.prompt_cache import load_cached
from models.flux_cache import FluxAnchorCache
from models.flux_fill_loader import load_flux_fill, unload_text_encoders
from models.flux_sparse_transformer import FluxSparseRunner
from samplers.dense_flux_fill import (FluxFillState, decode_latents,
                                      prepare_flux_fill_inputs, scheduler_step,
                                      transformer_forward)
from token_selectors.boundary import boundary_score
from token_selectors.combo import (PRESETS, combo_score, oracle_score, random_score,
                             select_hard_tokens)
from token_selectors.delta import delta_score
from token_selectors.frequency import frequency_score
from token_selectors.mask import mask_score
from utils.flow_math import clean_estimate
from utils.token_mapping import TokenGrid


# --------------------------------------------------------------- selectors ----
class SelectorState:
    """Precomputes the static priors (M, B) once per sample and evaluates the
    dynamic terms (F, Δ, A^D) per sparse step from the cache."""

    def __init__(self, name: str, mask_px: torch.Tensor, grid: TokenGrid,
                 pipe, freq_source: str = "anchor_x0", draft=None):
        self.name = name
        self.grid = grid
        self.pipe = pipe
        self.freq_source = freq_source
        self.draft = draft
        self.mask_tok = mask_score(mask_px, grid)                       # [B, N]
        self.bnd_tok = boundary_score(self.mask_tok, grid)

    def _unpack(self, packed: torch.Tensor) -> torch.Tensor:
        return self.pipe._unpack_latents(
            packed, self.grid.height, self.grid.width, self.pipe.vae_scale_factor)

    @torch.no_grad()
    def scores(self, latents: torch.Tensor, sigma_t, cache: FluxAnchorCache,
               t: torch.Tensor, generator=None,
               v_dense_now: torch.Tensor | None = None) -> torch.Tensor:
        dev = latents.device
        B, N, _ = latents.shape
        if self.name == "random":
            return random_score(B, N, generator=generator, device=dev)
        if self.name == "oracle":
            assert v_dense_now is not None, "oracle needs the extra dense pass"
            return oracle_score(v_dense_now, cache.final_prediction)

        w = PRESETS[self.name]
        freq = delta = draft_term = None
        if w.gamma != 0.0:
            if self.freq_source == "noisy":
                src = self._unpack(latents)
            elif self.freq_source == "cached_v_current_x0":
                # z_t with the STALE anchor velocity — a mixed estimate, kept
                # only as an explicit ablation arm; this is NOT the anchor x0.
                src = self._unpack(clean_estimate(latents, cache.final_prediction, sigma_t))
            else:  # 'anchor_x0' (default): TRUE anchor clean estimate
                #   x0_hat_a = z_a - sigma_a * v_a, precomputed at the anchor
                assert cache.anchor_clean_estimate is not None, \
                    "anchor_x0 requires cache.set_anchor_context() at anchor steps"
                src = self._unpack(cache.anchor_clean_estimate)
            freq = frequency_score(src, self.grid).to(dev)
        if w.delta != 0.0 and cache.prev_final_prediction is not None:
            delta = delta_score(cache.final_prediction, cache.prev_final_prediction)
        if w.eta != 0.0 and self.draft is not None:
            # router-style draft outputs a per-token difficulty score directly
            draft_term = self.draft.scores(latents, self.mask_tok, cache, t, self.grid)
        return combo_score(w, mask=self.mask_tok.to(dev), boundary=self.bnd_tok.to(dev),
                           frequency=freq, delta=delta, draft=draft_term)


# ------------------------------------------------------------------ sampler ---
@torch.no_grad()
def get_model_provenance(pipe) -> dict:
    """모델·스케줄러·코드 provenance (fix3 #1). paired 비교의 전제인 '같은
    가중치·같은 스케줄러·같은 코드'를 run.json에 기록해 validator가 검증."""
    import hashlib as _hl
    import json as _js
    import subprocess as _sp
    from pathlib import Path as _P

    RUNTIME_KEYS = ("timesteps", "sigmas", "num_inference_steps")

    def _is_base_key(k):
        # '_' 프리픽스는 diffusers 내부 메타데이터 (_use_default_values,
        # _step_index, _begin_index, _class_name 등) — set 유래 list의 순서
        # 비결정성이 있고 모델 설정이 아니므로 base config에서 제외.
        return not str(k).startswith("_") and k not in RUNTIME_KEYS

    def _canonicalize(x):
        """프로세스 간 결정적 직렬화: set/frozenset의 순서 비결정성,
        Path/텐서 스칼라 등 비기본 타입을 재귀적으로 정규화 (fix: config에
        set이 있으면 str(v)가 프로세스마다 다른 hash를 낼 수 있음)."""
        if isinstance(x, dict):
            return {str(k): _canonicalize(v)
                    for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
        if isinstance(x, (set, frozenset)):
            vals = [_canonicalize(v) for v in x]
            return sorted(vals, key=lambda v: _js.dumps(v, sort_keys=True,
                                                        default=str))
        if isinstance(x, (list, tuple)):
            return [_canonicalize(v) for v in x]
        if isinstance(x, _P):
            return str(x)
        if isinstance(x, (str, int, float, bool)) or x is None:
            return x
        if hasattr(x, "item"):
            try:
                return x.item()
            except Exception:
                pass
        return str(x)

    def _cfg_sha(cfg):
        payload = _js.dumps(_canonicalize(dict(cfg)), sort_keys=True,
                            separators=(",", ":")).encode("utf-8")
        return _hl.sha256(payload).hexdigest()

    def _tensor_sha(t):
        if t is None:
            return None
        arr = t.detach().cpu().contiguous().numpy()
        return _hl.sha256(arr.tobytes()).hexdigest()

    def _git(*args):
        try:
            return _sp.check_output(["git", *args], text=True,
                                    stderr=_sp.DEVNULL).strip()
        except Exception:
            return None

    t = pipe.transformer
    status = _git("status", "--porcelain")
    return {
        "pipeline_name_or_path": getattr(pipe, "name_or_path", None),
        "transformer_name_or_path": getattr(t, "name_or_path", None) or
                                    getattr(t.config, "_name_or_path", None),
        "model_revision": getattr(t.config, "_commit_hash", None),
        "transformer_config_sha256": _cfg_sha(t.config),
        "scheduler_class": type(pipe.scheduler).__name__,
        # base config: 실행 중 변하는 runtime 필드 제거 후 hash (전 arm 동일해야)
        "scheduler_base_config": _canonicalize({
            k: v for k, v in dict(pipe.scheduler.config).items()
            if _is_base_key(k)}),
        "scheduler_base_config_sha256": _cfg_sha({
            k: v for k, v in dict(pipe.scheduler.config).items()
            if _is_base_key(k)}),
        # runtime schedule: 같은 step 수 arm끼리만 같아야 (30 vs 50은 달라야 정상)
        "timesteps_sha256": _tensor_sha(getattr(pipe.scheduler, "timesteps", None)),
        "sigmas_sha256": _tensor_sha(getattr(pipe.scheduler, "sigmas", None)),
        "code_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
    }


def _mask_blind_scores(method, n_tokens, grid, generator, device, seed=0):
    """mask-blind 예산 배분 CONTROL 점수 (prior-work baseline 아님; 논문 명시).
    - uniform_grid: 다중 stride 격자 순서로 전 프레임에 균등 확산 (어떤 ratio에서도
      선택 토큰이 공간 전역에 고르게 퍼짐).
    - contiguous_block: latent seed로 위치가 결정되는 직사각형 블록 (좌상단 편향 제거).
    Unknown method는 즉시 에러 — silent misclassification 방지 (#1)."""
    import torch as _t
    hp, wp = grid.token_hw
    assert hp * wp == n_tokens, (hp, wp, n_tokens)
    if method == "uniform_grid":
        # #2: 여러 offset 격자를 stride 4->2->1 순으로 채워 공간 균등 순서를 만든다.
        order = []
        seen = set()
        for stride in (4, 2, 1):
            for ro in range(stride):
                for co in range(stride):
                    for r in range(ro, hp, stride):
                        for c in range(co, wp, stride):
                            idx = r * wp + c
                            if idx not in seen:
                                seen.add(idx)
                                order.append(idx)
        order_t = _t.tensor(order, device=device)
        scores = _t.empty(n_tokens, device=device)
        scores[order_t] = _t.arange(len(order), 0, -1, device=device,
                                    dtype=_t.float32)
        return scores.unsqueeze(0)
    elif method == "contiguous_block":
        # #3: seed로 위치가 정해지는 실제 2D 직사각형 (좌상단 고정 편향 제거).
        # 블록 크기 = ratio를 여기서 모르므로, 프레임의 절반 폭 블록을 기준으로 하고
        # top-k가 필요한 만큼 인접 확장되도록 중심 거리 기반 점수를 준다.
        gen = _t.Generator(device="cpu").manual_seed(seed)
        # 중심을 프레임 안쪽으로 제한해 경계 wrap 없이 하나의 연속 블록이 되게
        cy = int(_t.randint(hp // 4, 3 * hp // 4 + 1, (1,), generator=gen).item())
        cx = int(_t.randint(wp // 4, 3 * wp // 4 + 1, (1,), generator=gen).item())
        rr = _t.arange(hp, device=device).view(hp, 1).float()
        cc = _t.arange(wp, device=device).view(1, wp).float()
        dist = ((rr - cy).abs() + (cc - cx).abs()).view(-1)   # L1 -> 다이아몬드 블록
        return (-dist).unsqueeze(0)          # 가까울수록 높음 -> top-k = 연속 블록
    else:
        raise ValueError(f"Unknown mask-blind control: {method!r} "
                         "(uniform_grid | contiguous_block)")


def sample_one(pipe, runner: FluxSparseRunner, state: FluxFillState, *,
               method: str, cache_period: int, ratio: float, selector: str,
               block: int, mask_px: torch.Tensor, freq_source: str,
               dense_head: int = 0, dense_tail: int = 0, kv_cache: bool = False,
               dual_sparse: bool = False, teacache_thresh: float = 0.15,
               teacache_rel_l1: float = 0.4,
               sample_seed_offset: int = 0, dump_selection: list | None = None,
               draft=None, log: dict | None = None):
    """Runs one image through the chosen method; mutates state.latents; returns
    stats dict (target evals, sparse fraction, per-step records)."""
    grid = state.grid
    cache = FluxAnchorCache()
    sel = SelectorState(selector, mask_px, grid, pipe, freq_source, draft) \
        if method == "cache_sparse" else None
    g = torch.Generator(state.latents.device).manual_seed(0)

    n_anchor = n_sparse = 0
    n_forced_dense = n_thresh_reuse = 0
    last_anchor_sigma = None
    attn_fracs, mac_ratios, actual_ratios = [], [], []
    hetero_rows = []
    v_prev = None
    n_steps = len(state.timesteps)
    tc = {"cnt": 0, "num_steps": n_steps, "rel_l1_thresh": teacache_rel_l1,
          "accumulated": 0.0, "prev_mod": None, "prev_residual": None}

    for i, t in enumerate(state.timesteps):
        sigma = pipe.scheduler.sigmas[i]
        # schedule-aware policy: hetero 측정에서 마지막 ~4 step은 변화가 mask 밖으로
        # 퍼지고(in/out 25x -> 0.4) energy가 튐 -> 그 구간은 무조건 dense(anchor).
        forced_dense = (i < dense_head) or (i >= n_steps - dense_tail)
        if method == "teacache":
            model_input = torch.cat([state.latents, state.cond], dim=2)
            timestep = t.expand(1).to(state.latents.dtype) / 1000
            v, calc = runner.teacache_forward(
                model_input, state.prompt_embeds, state.pooled, timestep,
                state.guidance, state.img_ids, state.txt_ids, tc)
            if calc:
                n_anchor += 1          # full compute
            else:
                n_thresh_reuse += 1    # residual reuse (final head only)
            state.latents = scheduler_step(pipe, v, t, state.latents)
            continue

        _anchored = ("reuse", "cache_sparse", "temporal_thresh",
                     "uniform_grid", "contiguous_block")
        is_anchor = (method in _anchored) and \
                    (i % cache_period == 0 or forced_dense)

        if method in ("dense", "hetero") or is_anchor:
            model_input_cache = cache if is_anchor else None
            model_input = torch.cat([state.latents, state.cond], dim=2)
            timestep = t.expand(1).to(state.latents.dtype) / 1000
            v, _ = runner.dense_forward(model_input, state.prompt_embeds, state.pooled,
                                        timestep, state.guidance, state.img_ids,
                                        state.txt_ids, cache=model_input_cache,
                                        step_index=i,
                                        record_kv=kv_cache and is_anchor,
                                        record_dual=dual_sparse and is_anchor)
            n_anchor += 1
            if is_anchor:
                # Fix 1: precompute the TRUE anchor clean estimate x0_a = z_a - s_a*v_a
                cache.set_anchor_context(state.latents, sigma)
                last_anchor_sigma = float(sigma)
            if method == "hetero" and v_prev is not None:
                d = (v.float() - v_prev.float()).pow(2).mean(-1)        # [B, N]
                m = sel_mask(mask_px, grid, d.device)
                hetero_rows.append(_hetero_row(i, d, m, v))
            v_prev = v
        elif method == "reuse":
            v = cache.final_prediction                                   # r = 0
        elif method == "temporal_thresh":
            # Adapted timestep-threshold reuse (NOT faithful TeaCache): 마지막 dense
            # anchor 이후 sigma 상대변화가 임계 미만이면 전 토큰 재사용, 넘으면 dense.
            # #4: threshold-triggered dense는 새 anchor로 기록 (방식 A — baseline에
            # 유리, "일부러 약화" 공격 회피). 원 방법의 rescaling은 재현 안 함(명시).
            rel = abs(float(sigma) - last_anchor_sigma) / max(abs(last_anchor_sigma), 1e-6)
            if rel < teacache_thresh:
                v = cache.final_prediction
                n_thresh_reuse += 1
            else:
                model_input = torch.cat([state.latents, state.cond], dim=2)
                timestep = t.expand(1).to(state.latents.dtype) / 1000
                v, _ = runner.dense_forward(model_input, state.prompt_embeds,
                                            state.pooled, timestep, state.guidance,
                                            state.img_ids, state.txt_ids,
                                            cache=cache, step_index=i)
                cache.set_anchor_context(state.latents, sigma)
                last_anchor_sigma = float(sigma)
                n_forced_dense += 1
        elif ratio == 0.0:
            v = cache.final_prediction                                   # r = 0
        else:
            v_dense_now = None
            if selector == "oracle":                                     # upper bound
                model_input = torch.cat([state.latents, state.cond], dim=2)
                timestep = t.expand(1).to(state.latents.dtype) / 1000
                v_dense_now, _ = runner.dense_forward(
                    model_input, state.prompt_embeds, state.pooled, timestep,
                    state.guidance, state.img_ids, state.txt_ids)
            if method in ("uniform_grid", "contiguous_block"):
                # CONTROL (not a prior-work baseline): 같은 예산 r을 mask-blind로
                # 배분. uniform_grid=고정 격자 stride, contiguous_block=연속 블록.
                # "동일 anchored backend에서 mask-aware 선택의 이득"만 격리한다.
                scores = _mask_blind_scores(
                    method, state.latents.shape[1], grid, g,
                    state.latents.device, seed=sample_seed_offset + i)
            else:
                scores = sel.scores(state.latents, sigma, cache, t, generator=g,
                                    v_dense_now=v_dense_now)
            hard_idx, _, r_actual = select_hard_tokens(scores, grid, ratio, block=block)
            model_input = torch.cat([state.latents, state.cond], dim=2)
            timestep = t.expand(1).to(state.latents.dtype) / 1000
            v_hard, st = runner.sparse_forward(model_input, state.prompt_embeds,
                                               state.pooled, timestep, state.guidance,
                                               state.img_ids, state.txt_ids,
                                               cache, hard_idx, kv_cache=kv_cache,
                                               dual_sparse=dual_sparse)
            v = runner.merge_prediction(cache, hard_idx, v_hard)
            if dump_selection is not None:
                dump_selection.append({"step": i, "hard_idx": hard_idx[0].cpu()})
            n_sparse += 1
            attn_fracs.append(st.single_attn_fraction)
            mac_ratios.append(st.est_transformer_mac_ratio)
            actual_ratios.append(r_actual)

        state.latents = scheduler_step(pipe, v, t, state.latents)

    stats = {"anchor_evals": n_anchor, "sparse_steps": n_sparse,
             "thresh_dense": n_forced_dense, "thresh_reuse": n_thresh_reuse,
             "mean_single_attn_fraction": (sum(attn_fracs) / len(attn_fracs))
             if attn_fracs else None,
             # Fix 3: with block > 1 the realized refresh ratio != requested ratio
             "mean_actual_ratio": (sum(actual_ratios) / len(actual_ratios))
             if actual_ratios else None,
             # Fix 10: whole-transformer MAC estimate (dense dual + full K/V included)
             "mean_est_transformer_mac_ratio": (sum(mac_ratios) / len(mac_ratios))
             if mac_ratios else None}
    if hetero_rows:
        stats["heterogeneity"] = hetero_rows
    if log is not None:
        log.update(stats)
    return stats


def sel_mask(mask_px, grid, device):
    return mask_score(mask_px, grid).to(device)


def _hetero_row(step, d, m, v_now):
    """Q1 row — Factor A (spatial concentration) AND the per-step half of
    Factor B: E_rel = mean||v_t - v_{t-1}||² / mean||v_t||² (consequence scale).
    The other half of Factor B (step-reduction quality sensitivity) is joined
    in eval.heterogeneity from the dense-step-sweep metrics (Fix 4)."""
    dm = d.flatten()
    k = max(1, int(0.3 * dm.numel()))
    top = torch.topk(dm, k).values.sum() / dm.sum().clamp_min(1e-12)
    inm = (m.flatten() > 0.5)
    return {
        "step": int(step),
        "top30_share": top.item(),
        "cv": (dm.std() / dm.mean().clamp_min(1e-12)).item(),
        "in_mask_mean": dm[inm].mean().item() if inm.any() else 0.0,
        "out_mask_mean": dm[~inm].mean().item() if (~inm).any() else 0.0,
        "energy_ratio": (dm.mean() / v_now.float().pow(2).mean().clamp_min(1e-12)).item(),
    }


# --------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--method",
                    choices=["dense", "reuse", "cache_sparse", "hetero",
                             "temporal_thresh", "uniform_grid", "contiguous_block",
                             "teacache"],
                    required=True)
    ap.add_argument("--selector", default="mask",
                    choices=list(PRESETS) + ["random", "oracle"])
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cache-period", type=int, default=3)
    ap.add_argument("--teacache-thresh", type=float, default=0.15,
                    help="temporal_thresh control: anchor 이후 sigma 상대변화 임계")
    ap.add_argument("--teacache-rel-l1", type=float, default=0.4,
                    help="faithful TeaCache: accumulated rescaled rel-L1 임계 "
                         "(공식 운영점 0.25/0.4/0.6/0.8)")
    ap.add_argument("--ratio", type=float, default=0.3)
    ap.add_argument("--block", type=int, default=1,
                    help="structured selection window in tokens (Stage 7): 1, 2, 4")
    ap.add_argument("--freq-source", default="anchor_x0",
                    choices=["anchor_x0", "cached_v_current_x0", "noisy"])
    ap.add_argument("--guidance", type=float, default=30.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--prompt-cache", default="")
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="added to each sample's manifest latent_seed (Stage 8 multi-seed)")
    ap.add_argument("--draft-ckpt", default="",
                    help="CNN router checkpoint for mbfd_draft (Stage 6)")
    ap.add_argument("--dump-selection", action="store_true",
                    help="step별 hard 토큰 인덱스 저장 (selection map figure용)")
    ap.add_argument("--dual-sparse", action="store_true",
                    help="Lever A: dual stream image 토큰도 sparse (fresh cache에서 exact)")
    ap.add_argument("--kv-cache", action="store_true",
                    help="Lever B: easy 토큰 K/V를 anchor에서 동결 (temb-staleness 근사)")
    ap.add_argument("--dense-head", type=int, default=0,
                    help="처음 K step 강제 dense (anchor)")
    ap.add_argument("--dense-tail", type=int, default=0,
                    help="마지막 K step 강제 dense — hetero 곡선의 말기 붕괴 구간 방어")
    ap.add_argument("--prefetch", action=__import__("argparse").BooleanOptionalAction,
                    default=True, help="background next-sample loading (--no-prefetch로 끔)")
    ap.add_argument("--tag", default="run")
    a = ap.parse_args()

    out = Path(a.out) / a.tag
    out.mkdir(parents=True, exist_ok=True)
    comps = load_flux_fill(keep_text_encoders=not a.prompt_cache)
    # provenance는 반드시 로드 직후(어떤 set_timesteps보다 전) 캡처 —
    # set_timesteps가 scheduler config에 step-수 의존 runtime 항목을 남겨
    # base hash가 arm마다 달라지는 오염을 원천 차단 (N=100 validator 실패 원인).
    _prov_at_load = get_model_provenance(comps.pipe)
    pipe, dev, dtype = comps.pipe, comps.device, comps.dtype
    runner = FluxSparseRunner(pipe.transformer)
    draft = None
    if a.draft_ckpt:
        # auto-detects: residual-draft ckpt (model_config key) -> ErrorHeadRouter
        # (cache-error head as the eta score); legacy ckpt -> CNN RouterDraft
        from models.drafts.error_head_router import load_draft
        draft = load_draft(a.draft_ckpt, dev)
    ds = FluxFillBenchmark(a.manifest)
    n = len(ds) if a.limit == 0 else min(a.limit, len(ds))

    # 다음 sample(이미지 로드 + mask 생성)을 GPU가 도는 동안 백그라운드로 준비.
    # sample당 50-step FLUX(수십 초) 대비 데이터(수십 ms)라 이득은 작지만 공짜이고,
    # rows[i]["data_s"]로 데이터가 GPU를 실제로 막는지 직접 확인 가능.
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(max_workers=1) if a.prefetch else None
    pending = pool.submit(ds.__getitem__, 0) if pool else None

    rows = []
    for i in range(n):
        t_data = time.perf_counter()
        if pool:
            s = pending.result()
            if i + 1 < n:
                pending = pool.submit(ds.__getitem__, i + 1)
        else:
            s = ds[i]
        data_s = time.perf_counter() - t_data
        pe = po = None
        if a.prompt_cache:
            pe, po = load_cached(a.prompt_cache, s["prompt"], dev, dtype)
        state = prepare_flux_fill_inputs(
            pipe, s["image"], s["mask"], s["prompt"],
            s["latent_seed"] + a.seed_offset,
            a.steps, a.guidance, dev, dtype, prompt_embeds=pe, pooled=po)

        # Fix 7: measure each sample from a clean slate; the first sample stays
        # flagged as warm-up (compile/cudnn autotune/allocator growth) and is
        # excluded from wall-clock aggregation in eval.assemble.
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        log: dict = {}
        sample_one(pipe, runner, state, method=a.method,
                   cache_period=a.cache_period, ratio=a.ratio,
                   selector=a.selector, block=a.block,
                   mask_px=s["mask"].unsqueeze(0).to(dev), freq_source=a.freq_source,
                   dense_head=a.dense_head, dense_tail=a.dense_tail,
                   kv_cache=a.kv_cache, dual_sparse=a.dual_sparse,
                   teacache_thresh=a.teacache_thresh,
                   teacache_rel_l1=a.teacache_rel_l1,
                   sample_seed_offset=(s["latent_seed"] + a.seed_offset) * 131 + i,
                   dump_selection=(sel_dump := [] if a.dump_selection else None),
                   draft=draft, log=log)
        img = decode_latents(pipe, state)
        torch.cuda.synchronize()
        log.update({"sample_id": s["sample_id"], "bucket": s["bucket"],
                    "mask_type": s["mask_type"], "warmup": i == 0,
                    "data_s": data_s,
                    "wall_s": time.perf_counter() - t0,
                    "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30})
        rows.append(log)

        from samplers.dense_flux_fill import _save_img
        stem = Path(s["sample_id"]).stem
        _save_img(img, out / f"{stem}.png")                     # raw model output
        # Fix 6: composited output x_paste = M*x_model + (1-M)*x_input — FLUX Fill
        # does not mathematically guarantee known-region identity in raw output.
        m_px = s["mask"].to(img.device, img.dtype)              # [1,H,W]
        inp = s["image"].to(img.device, img.dtype)              # [3,H,W]
        _save_img(m_px * img + (1 - m_px) * inp, out / f"{stem}_pasted.png")
        torch.save(s["mask"], out / f"{stem}_mask.pt")
        if a.dump_selection and sel_dump:
            torch.save({"selection": sel_dump, "token_hw": state.grid.token_hw},
                       out / f"{stem}_selection.pt")

    cfg = vars(a)
    # #3: provenance — 재사용 시 validate_run_compat이 비교할 근거
    import hashlib as _hl
    def _sha(p):
        h = _hl.sha256()
        with open(p, "rb") as f:
            for c in iter(lambda: f.read(1 << 20), b""):
                h.update(c)
        return h.hexdigest()
    cfg["manifest_sha256"] = _sha(a.manifest)
    cfg["resolution"] = json.load(open(a.manifest)).get("resolution")
    import torch as _th, diffusers as _df, transformers as _tf
    cfg["versions"] = {"torch": _th.__version__, "diffusers": _df.__version__,
                       "transformers": _tf.__version__}
    _prov_end = get_model_provenance(pipe)          # schedule hash는 실행 후가 유의미
    _prov_at_load["timesteps_sha256"] = _prov_end["timesteps_sha256"]
    _prov_at_load["sigmas_sha256"] = _prov_end["sigmas_sha256"]
    cfg["provenance"] = _prov_at_load
    json.dump({"config": cfg, "rows": rows}, open(out / "run.json", "w"), indent=1)
    print(f"[{a.tag}] {n} samples -> {out}")


if __name__ == "__main__":
    main()