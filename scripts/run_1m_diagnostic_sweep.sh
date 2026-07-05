#!/usr/bin/env bash
set -euo pipefail

runs=(
  "configs/baseline_ppo.yaml rtx4090_baseline_1m_n16_s0"
  "configs/flat_penalty_ppo.yaml rtx4090_flat_1m_n16_s0"
  "configs/nns_lpm_ppo.yaml rtx4090_nns_lpm_1m_n16_s0"
)

for item in "${runs[@]}"; do
  read -r config run_id <<<"$item"
  uv run mario-train-sb3 \
    --config "$config" \
    --run-id "$run_id" \
    --total-timesteps 1000000 \
    --n-envs "${N_ENVS:-64}" \
    --n-steps "${N_STEPS:-256}" \
    --batch-size "${BATCH_SIZE:-256}" \
    --seed 0 \
    --device cuda
  uv run mario-evaluate --run-dir "artifacts/runs/$run_id" --episodes "${EVAL_EPISODES:-20}"
done

uv run mario-summarize-runs \
  --run-dir artifacts/runs/rtx4090_baseline_1m_n16_s0 \
  --run-dir artifacts/runs/rtx4090_flat_1m_n16_s0 \
  --run-dir artifacts/runs/rtx4090_nns_lpm_1m_n16_s0 \
  --output-csv artifacts/runs/rtx4090_1m_diagnostic_summary.csv \
  --output-md artifacts/runs/rtx4090_1m_diagnostic_summary.md
