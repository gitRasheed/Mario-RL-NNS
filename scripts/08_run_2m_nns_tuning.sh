#!/usr/bin/env bash
set -euo pipefail

N_ENVS="${N_ENVS:-64}"
N_STEPS="${N_STEPS:-256}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
EVAL_MAX_EPISODE_STEPS="${EVAL_MAX_EPISODE_STEPS:-20000}"
EVAL_PREFIX_STEPS="${EVAL_PREFIX_STEPS:-0,5,10,15,20,25,30}"
RUN_INDEX="${RUN_INDEX:-003}"
SWEEP_NAME="${SWEEP_NAME:-rtx4090_2m_nns_tuning}"
HOURLY_PRICE="${HOURLY_PRICE:-0.69}"
RUN_PREFIX="${RUN_INDEX}_${SWEEP_NAME}"

runs=(
  "configs/nns_lpm_soft_ppo.yaml ${RUN_PREFIX}_soft_n${N_ENVS}_s0"
  "configs/nns_lpm_very_soft_ppo.yaml ${RUN_PREFIX}_very_soft_n${N_ENVS}_s0"
  "configs/nns_lpm_no_stuck_ppo.yaml ${RUN_PREFIX}_no_stuck_n${N_ENVS}_s0"
  "configs/nns_lpm_low_target_ppo.yaml ${RUN_PREFIX}_low_target_n${N_ENVS}_s0"
  "configs/nns_lpm_warmup_soft_ppo.yaml ${RUN_PREFIX}_warmup_soft_n${N_ENVS}_s0"
  "configs/nns_lpm_clip_small_ppo.yaml ${RUN_PREFIX}_clip_small_n${N_ENVS}_s0"
)

run_dirs=()
for item in "${runs[@]}"; do
  read -r config run_id <<<"$item"
  run_dirs+=("--run-dir" "artifacts/runs/$run_id")
  uv run mario-train-sb3 \
    --config "$config" \
    --run-id "$run_id" \
    --total-timesteps 2000000 \
    --n-envs "$N_ENVS" \
    --n-steps "$N_STEPS" \
    --batch-size "$BATCH_SIZE" \
    --seed 0 \
    --device cuda
  uv run mario-evaluate \
    --run-dir "artifacts/runs/$run_id" \
    --episodes "$EVAL_EPISODES" \
    --max-episode-steps "$EVAL_MAX_EPISODE_STEPS" \
    --eval-prefix-steps "$EVAL_PREFIX_STEPS"
done

uv run mario-summarize-runs \
  "${run_dirs[@]}" \
  --output-csv "artifacts/runs/${RUN_PREFIX}_summary.csv" \
  --output-md "artifacts/runs/${RUN_PREFIX}_summary.md" \
  --hourly-price "$HOURLY_PRICE"
