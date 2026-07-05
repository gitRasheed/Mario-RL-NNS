#!/usr/bin/env bash
set -euo pipefail

N_ENVS="${N_ENVS:-64}"
N_STEPS="${N_STEPS:-256}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
EVAL_MAX_EPISODE_STEPS="${EVAL_MAX_EPISODE_STEPS:-20000}"
EVAL_PREFIX_STEPS="${EVAL_PREFIX_STEPS:-0,5,10,15,20,25,30}"
RUN_INDEX="${RUN_INDEX:-004}"
SWEEP_NAME="${SWEEP_NAME:-rtx4090_5m_selected_nns}"
HOURLY_PRICE="${HOURLY_PRICE:-0.69}"
RUN_PREFIX="${RUN_INDEX}_${SWEEP_NAME}"

runs=(
  "configs/nns_lpm_clip_small_ppo.yaml ${RUN_PREFIX}_clip_small_n${N_ENVS}_s0"
  "configs/nns_lpm_no_stuck_ppo.yaml ${RUN_PREFIX}_no_stuck_n${N_ENVS}_s0"
)

run_dirs=()
for item in "${runs[@]}"; do
  read -r config run_id <<<"$item"
  run_dirs+=("--run-dir" "artifacts/runs/$run_id")
  uv run mario-train-sb3 \
    --config "$config" \
    --run-id "$run_id" \
    --total-timesteps 5000000 \
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

RUN_PREFIX="$RUN_PREFIX" uv run python - <<'PY'
import csv
import json
import os
from pathlib import Path

fields = [
    "run_id",
    "timestep",
    "wall_time_s",
    "progress_mean",
    "progress_max_mean",
    "stuck_mean",
    "reward_base_mean",
    "reward_train_mean",
    "extra_reward_mean",
    "lpm_speed_mean",
    "fraction_below_target_speed",
]
prefix = os.environ["RUN_PREFIX"]
rows = []
for path in sorted(Path("artifacts/runs").glob(f"{prefix}_*_n*_s0/metrics/rollout_metrics.jsonl")):
    run_id = path.parents[1].name
    for line in path.read_text().splitlines():
        row = json.loads(line)
        rows.append({"run_id": run_id, **{field: row.get(field) for field in fields[1:]}})

out = Path(f"artifacts/runs/{prefix}_diagnostics.csv")
with out.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print(out)
PY
