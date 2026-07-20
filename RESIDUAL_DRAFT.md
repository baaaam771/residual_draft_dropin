# RESIDUAL_DRAFT — reuse / approximate / recompute on the anchored backend

flux_fill_sparse repo에 drop-in되는 파일 5개 + 테스트. 기존 anchored 실행
backend(FluxAnchorCache / FluxSparseRunner)를 그대로 쓰고, 그 위에 anchor-residual
draft tier 하나를 추가한다.

```
models/drafts/residual_draft.py       ~3M CNN: dv_hat + log e_cache + log e_draft
token_selectors/three_tier.py         2-score routing, hard_idx [B,k] 오름차순
training/train_residual_draft.py      기존 router_teacher 덤프로 학습 (재수집 불필요)
training/eval_residual_router.py      offline gate (Stage 3 GPU 실행 전 필수)
training/diagnose_residual.py         v1 정보병목 가설 검증 (offset별 ratio, dz-identity)
samplers/three_tier_flux_fill.py      method: draft_only(Stage 3) / three_tier(Stage 4)
tests/test_residual_draft.py          회귀 테스트 12개 (CPU)
```

## Routing

```
e_cache_i ~ ||v_a - v_t||^2            reuse(staleness) error
e_draft_i ~ ||v_a + dv_hat - v_t||^2   draft 잔여 error
gain_i     = e_cache_i - e_draft_i

TARGET : top r_target by e_draft   (boundary band는 budget 내에서 우선)
DRAFT  : top r_draft  by gain > 0  (require_positive_gain)
CACHE  : 나머지
```

Error head는 log-error 회귀; 추론 시 clamp(-30, 20) 후 exp (inf 방지).
realized tier ratio: 모든 이미지에 sample-level mean이 기록되고, step-level
상세는 `--save-step-rows {first,all,none}` (기본 first = 첫 샘플만; 5-image
closed-loop 진단에서는 `all`). 테이블 비교는 nominal r 대신 realized 사용.

## 데이터 경로 (repo 기존 산출물 재사용)

| 무엇 | 경로 |
|---|---|
| teacher 덤프 1024² (200장, manifest idx 300–499 → 평가 100장과 disjoint) | `/mnt/HDD_12TB/bam_ki/flux_fill/router_teacher_1024` |
| teacher 덤프 512² | `/mnt/HDD_12TB/bam_ki/flux_fill/router_teacher` |
| prompt cache | `/mnt/HDD_12TB/bam_ki/flux_fill/prompt_cache` |
| 평가 manifest | `data/coco_manifest_1024.json` (앞 100장 사용) |
| teacher manifest | `data/coco_manifest_1024_teacher.json` |
| draft ckpt (신규) | `/mnt/HDD_12TB/bam_ki/flux_fill/residual_draft_ckpt` |

**Stage 1(수집)은 이미 완료** — dump_router_teacher 포맷
(`latents [S,N,64], preds [S,N,64], sigmas, mask_tok, token_hw`)을 그대로 읽고,
pair는 어떤 cache period든 on-the-fly로 구성한다 (train_router와 동일 레시피).
Anchor step은 period별 사전 계산된 valid_steps에서 원천 배제 (steps=50, tail=4,
c=3에서 i=45가 anchor-anchor 영(0) pair로 새던 경계 버그 수정; 테스트로 고정).
split은 sample_id SHA-256 해시 기반 image 단위 (train/val/calib = 80/10/10);
한 이미지의 모든 timestep record가 같은 split.

## 실행 순서

```bash
export PYTHONPATH=.
TEACH=/mnt/HDD_12TB/bam_ki/flux_fill/router_teacher_1024
CKPT=/mnt/HDD_12TB/bam_ki/flux_fill/residual_draft_ckpt
PC=/mnt/HDD_12TB/bam_ki/flux_fill/prompt_cache
MAN=data/coco_manifest_1024.json

# 0) CPU 회귀 테스트 (모델 불필요)
python tests/test_residual_draft.py

# 1) 학습 (기존 덤프 재사용; resume은 ckpt의 model_config 자동 복원)
python -m training.train_residual_draft --teacher $TEACH --out $CKPT \
    --steps 60000 --cache-periods 2 3

# 2) OFFLINE GATE — Stage 3 GPU 실행 전 go/no-go
#    "upper-bound routing"이 pure reuse를 확실히 이기지 못하면 중단.
#    (이 upper bound는 TARGET err=0 가정 = IDEAL target fallback;
#     실제 dual+K/V staleness는 미포함 — system oracle이 아님)
python -m training.eval_residual_router --teacher $TEACH --ckpt $CKPT/last.pt

# 3) Stage 3: draft_only — reuse_c2+tail(8.9s, 0.0469)과 동일 target-eval 수.
#    성공 기준: mask-LPIPS→ref < 0.0469 (draft_ms 실측치 함께 보고)
python -m samplers.three_tier_flux_fill --manifest $MAN --out out/stage_res \
    --tag draft_c2 --method draft_only --cache-period 2 --dense-tail 4 \
    --draft-ckpt $CKPT/last.pt --prompt-cache $PC --limit 100

# 4) Stage 4: three_tier sweep — mbd c2 r.15 dual+KV(11.33s, 0.0357) 대비
for RT in 0.10 0.15 0.30; do for RD in 0.20 0.35 0.50; do
python -m samplers.three_tier_flux_fill --manifest $MAN --out out/stage_res \
    --tag tri_c2_rt${RT}_rd${RD} --method three_tier --cache-period 2 \
    --dense-tail 4 --r-target $RT --r-draft $RD --dual-sparse --kv-cache \
    --draft-ckpt $CKPT/last.pt --prompt-cache $PC --limit 100
done; done

# 5) c=5 rescue 테스트 (H3): staleness가 c=5를 off-frontier로 만들었으므로
#    draft tier가 이를 복구하면 최강 결과
python -m samplers.three_tier_flux_fill ... --cache-period 5 --method draft_only
```

GPU job은 **순차 실행** (wall-time 오염 방지). 평가는 기존 eval 파이프라인
(mask-LPIPS→ref 등)을 run.json + png 출력에 그대로 적용.

## v1 60k 결과 판독과 v2 (content inputs)

v1 60k run: L_res 평탄(1.7e-3 → 1.5e-3), mse_ratio ≈ 1.00 전 구간,
error-head spearman 0.65–0.74. **residual head는 실패, error head는 성공.**

원인 (정보 병목): dense Euler에서 `z_{a+1} = z_a + Δσ·v_a` → offset-1 pair
(c=2 전부, c=3 절반)의 `dz = Δσ·v_anchor`는 anchor 상태의 결정론적 함수이고,
v1 입력(v_a, dz, mask, Δσ)에는 이미지 콘텐츠가 전혀 없다. residual의 예측
가능한 성분은 콘텐츠(x̂0)와 절대 σ에 실려 있는데 그 입력이 빠져 있었다.

v2: `--use-latent --use-anchor-x0 --use-sigma-t` (신규 학습 기본 ON) —
z_t는 sparse step에서 공짜, x̂0_a는 cache.anchor_clean_estimate로 이미 존재.
v1 checkpoint는 config에 flag가 없으므로 기본 OFF로 그대로 로드된다 (호환).

순서:
```bash
# 0) 가설 검증 (v1 ckpt, GPU 수 분)
python -m training.diagnose_residual --teacher $TEACH --ckpt $CKPT/last.pt
#   기대: offset-1 dz-identity rel err ~1e-3 이하 (bf16 덤프 오차 수준),
#         offset-1 ratio ≈ 1.0, offset-2도 ≈ 1.0 (콘텐츠 없이는 못 배움)

# 1) v2 재학습 (새 out 디렉토리)
python -m training.train_residual_draft --teacher $TEACH     --out ${CKPT}_v2 --steps 60000 --cache-periods 2 3
#   판정 지점: val mse_ratio_in_mask가 2k step 내에 1.0 아래로 내려가기
#   시작하는가. 60k까지 1.0이면 콘텐츠를 줘도 residual 외삽이 안 되는 것 —
#   그 자체로 측정된 negative result (caching은 되고 extrapolation은 안 된다).
#   그 경우 error head 2개(spearman 0.65+)를 기존 learned router 자리에 넣는
#   router-reliability 기여로 pivot.
```

## 첫 실행 전 5-image closed-loop 진단 (필수)

1. sigma / anchor 위치가 collection(dump)과 inference에서 일치 —
   dump는 매 step latents/preds 저장이므로 pair (i, a=i−i%c)가 sampler의
   anchor 규칙과 동일한지 step index로 확인
2. hard_idx 규약: `[1, k]` 오름차순 (tests/test_residual_draft.py가 검증;
   sparse_forward gather와 일치)
3. teacher-forced vs closed-loop: 같은 5장의 step별 tier map / e_draft 분포 비교
   — `--save-step-rows all --limit 5`로 실행해 5장 전부 step-level 기록
4. step별 realized cache/draft/target ratio가 run.json에 기록되는지
5. timing_ms(draft_ms/route_ms/sparse_ms) 실측 — "draft는 공짜" 주장 금지,
   측정치로 보고

## 정직성 규칙 (repo 기존 규칙 승계)

- zero-init은 시작점이 reuse와 같다는 뜻일 뿐, 학습 후 per-sample 우위 보장이
  아님 — CACHE tier + require_positive_gain이 runtime fallback
- exactness gate(B0/B1/B2)는 CACHE/TARGET tier에만 적용; DRAFT tier는 명시적
  근사 tier
- boundary_policy="all" 또는 require_positive_gain 때문에 nominal r ≠ realized
  r일 수 있음 — 테이블에는 realized 기재
