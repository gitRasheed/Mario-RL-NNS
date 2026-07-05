#!/usr/bin/env bash
set -euo pipefail

N_ENVS="${N_ENVS:-64}"
N_STEPS="${N_STEPS:-256}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
RUN_INDEX="${RUN_INDEX:-001}"
SWEEP_NAME="${SWEEP_NAME:-rtx4090_1m_diagnostic}"
HOURLY_PRICE="${HOURLY_PRICE:-0.69}"
RUN_PREFIX="${RUN_INDEX}_${SWEEP_NAME}"

runs=(
  "configs/baseline_ppo.yaml ${RUN_PREFIX}_baseline_n${N_ENVS}_s0"
  "configs/flat_penalty_ppo.yaml ${RUN_PREFIX}_flat_n${N_ENVS}_s0"
  "configs/nns_lpm_ppo.yaml ${RUN_PREFIX}_nns_lpm_n${N_ENVS}_s0"
)

run_dirs=()
for item in "${runs[@]}"; do
  read -r config run_id <<<"$item"
  run_dirs+=("--run-dir" "artifacts/runs/$run_id")
  uv run mario-train-sb3 \
    --config "$config" \
    --run-id "$run_id" \
    --total-timesteps 1000000 \
    --n-envs "$N_ENVS" \
    --n-steps "$N_STEPS" \
    --batch-size "$BATCH_SIZE" \
    --seed 0 \
    --device cuda
  uv run mario-evaluate --run-dir "artifacts/runs/$run_id" --episodes "$EVAL_EPISODES"
done

uv run mario-summarize-runs \
  "${run_dirs[@]}" \
  --output-csv "artifacts/runs/${RUN_PREFIX}_summary.csv" \
  --output-md "artifacts/runs/${RUN_PREFIX}_summary.md" \
  --hourly-price "$HOURLY_PRICE"
