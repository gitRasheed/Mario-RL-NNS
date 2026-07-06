#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${CHECKPOINT:-/tmp/mario_checkpoint_candidates/yumouwei/models/pre-trained-1.zip}"
N_ENVS="${N_ENVS:-64}"
N_STEPS="${N_STEPS:-256}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
EVAL_MAX_EPISODE_STEPS="${EVAL_MAX_EPISODE_STEPS:-5000}"
HOURLY_PRICE="${HOURLY_PRICE:-0.69}"
RUN_INDEX="${RUN_INDEX:-010}"
RUN_PREFIX="${RUN_PREFIX:-${RUN_INDEX}_high_confidence_time_tail}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-250000,500000,1000000}"
SEEDS="${SEEDS:-0 1 2 3 4}"

echo "ETA estimate: 10M train steps + 30 evals, roughly 4-6 hours on the 4090 lane."
echo "Estimated cost at \$${HOURLY_PRICE}/hr: roughly \$3-\$5."

for seed in $SEEDS; do
  for variant in baseline baseline_plus_time_tail; do
    if [[ "$variant" == "baseline" ]]; then
      train_variant="baseline"
      nns_args=()
    else
      train_variant="nns"
      nns_args=(--nns-config baseline_plus_time_tail)
    fi

    run_id="${RUN_PREFIX}_${variant}_s${seed}"
    run_dir="artifacts/runs/${run_id}"
    uv run mario-external-checkpoint finetune \
      --checkpoint "$CHECKPOINT" \
      --run-id "$run_id" \
      --variant "$train_variant" \
      --total-timesteps 1000000 \
      --n-envs "$N_ENVS" \
      --seed "$seed" \
      --device cpu \
      --learning-rate 0.00001 \
      --n-steps "$N_STEPS" \
      --batch-size "$BATCH_SIZE" \
      --checkpoint-steps "$CHECKPOINT_STEPS" \
      "${nns_args[@]}"

    IFS=',' read -r -a steps <<< "$CHECKPOINT_STEPS"
    for step in "${steps[@]}"; do
      uv run mario-external-checkpoint eval \
        --checkpoint "${run_dir}/checkpoints/step_${step}.zip" \
        --run-id "${run_id}/eval_${step}" \
        --episodes "$EVAL_EPISODES" \
        --max-episode-steps "$EVAL_MAX_EPISODE_STEPS" \
        --seed "$seed" \
        --device cpu
    done
  done
done

RUN_PREFIX="$RUN_PREFIX" HOURLY_PRICE="$HOURLY_PRICE" uv run python - <<'PY'
import csv
import json
import os
from pathlib import Path

prefix = os.environ["RUN_PREFIX"]
hourly_price = float(os.environ["HOURLY_PRICE"])
out_dir = Path("artifacts/runs")
summary_csv = out_dir / f"{prefix}_summary.csv"
summary_md = out_dir / f"{prefix}_summary.md"
deltas_csv = out_dir / f"{prefix}_paired_deltas.csv"

fields = [
    "run_id",
    "variant",
    "seed",
    "checkpoint_step",
    "clear_rate",
    "eval_cap_rate",
    "completion_time_mean",
    "completion_time_p50",
    "completion_time_p90",
    "death_rate",
    "stuck_fraction",
    "mean_progress",
    "p10_progress",
    "p50_progress",
    "p90_progress",
    "base_reward_mean",
    "shaped_reward_mean",
    "extra_reward_mean",
    "unique_trajectory_count",
    "train_wall_time_s",
    "eval_wall_time_s",
    "cost",
]

def load_json(path):
    return json.loads(path.read_text()) if path.exists() else {}

def last_jsonl(path):
    if not path.exists() or not path.read_text().strip():
        return {}
    return json.loads(path.read_text().strip().splitlines()[-1])

rows = []
for train_dir in sorted(out_dir.glob(f"{prefix}_*_s*")):
    name = train_dir.name
    if "_baseline_plus_time_tail_s" in name:
        variant = "baseline_plus_time_tail"
        seed = int(name.rsplit("_s", 1)[1])
    elif "_baseline_s" in name:
        variant = "baseline"
        seed = int(name.rsplit("_s", 1)[1])
    else:
        continue
    train = load_json(train_dir / "train_summary.json")
    rollout = last_jsonl(train_dir / "metrics/rollout_metrics.jsonl")
    for eval_dir in sorted(train_dir.glob("eval_*")):
        step = int(eval_dir.name.split("_", 1)[1])
        eval_summary = load_json(eval_dir / "eval_summary.json")
        train_wall = float(train.get("wall_time_s", 0.0) or 0.0)
        eval_wall = float(eval_summary.get("wall_time_s", 0.0) or 0.0)
        rows.append({
            "run_id": train_dir.name,
            "variant": variant,
            "seed": seed,
            "checkpoint_step": step,
            "clear_rate": eval_summary.get("clear_rate"),
            "eval_cap_rate": eval_summary.get("eval_cap_rate"),
            "completion_time_mean": eval_summary.get("completion_time_mean"),
            "completion_time_p50": eval_summary.get("completion_time_p50"),
            "completion_time_p90": eval_summary.get("completion_time_p90"),
            "death_rate": eval_summary.get("death_rate"),
            "stuck_fraction": eval_summary.get("stuck_fraction"),
            "mean_progress": eval_summary.get("mean_progress"),
            "p10_progress": eval_summary.get("p10_progress"),
            "p50_progress": eval_summary.get("p50_progress"),
            "p90_progress": eval_summary.get("p90_progress"),
            "base_reward_mean": eval_summary.get("base_reward_mean"),
            "shaped_reward_mean": rollout.get("reward_train_mean"),
            "extra_reward_mean": rollout.get("extra_reward_mean"),
            "unique_trajectory_count": eval_summary.get("unique_trajectory_count"),
            "train_wall_time_s": train_wall,
            "eval_wall_time_s": eval_wall,
            "cost": hourly_price * (train_wall + eval_wall) / 3600,
        })

with summary_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

delta_fields = [
    "seed",
    "checkpoint_step",
    "clear_delta",
    "cap_delta",
    "completion_delta",
    "completion_p90_delta",
    "stuck_delta",
]
deltas = []
by_key = {(row["seed"], row["checkpoint_step"], row["variant"]): row for row in rows}
for seed in sorted({row["seed"] for row in rows}):
    for step in sorted({row["checkpoint_step"] for row in rows}):
        base = by_key.get((seed, step, "baseline"))
        tail = by_key.get((seed, step, "baseline_plus_time_tail"))
        if not base or not tail:
            continue
        deltas.append({
            "seed": seed,
            "checkpoint_step": step,
            "clear_delta": tail["clear_rate"] - base["clear_rate"],
            "cap_delta": tail["eval_cap_rate"] - base["eval_cap_rate"],
            "completion_delta": (
                None if tail["completion_time_mean"] is None or base["completion_time_mean"] is None
                else tail["completion_time_mean"] - base["completion_time_mean"]
            ),
            "completion_p90_delta": (
                None if tail["completion_time_p90"] is None or base["completion_time_p90"] is None
                else tail["completion_time_p90"] - base["completion_time_p90"]
            ),
            "stuck_delta": tail["stuck_fraction"] - base["stuck_fraction"],
        })

with deltas_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=delta_fields)
    writer.writeheader()
    writer.writerows(deltas)

def markdown(rows, fields):
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return "\n".join(lines) + "\n"

wins = {
    "clear_better_or_equal": sum(d["clear_delta"] >= 0 for d in deltas),
    "cap_lower": sum(d["cap_delta"] < 0 for d in deltas),
    "completion_lower": sum(d["completion_delta"] is not None and d["completion_delta"] < 0 for d in deltas),
    "completion_p90_lower": sum(d["completion_p90_delta"] is not None and d["completion_p90_delta"] < 0 for d in deltas),
    "stuck_lower": sum(d["stuck_delta"] < 0 for d in deltas),
    "pairs": len(deltas),
}

summary_md.write_text(
    "# 010 High-Confidence Time-Tail\n\n"
    "## Runs\n\n"
    + markdown(rows, fields)
    + "\n## Paired Deltas\n\n"
    + markdown(deltas, delta_fields)
    + "\n## Win Counts\n\n```json\n"
    + json.dumps(wins, indent=2, sort_keys=True)
    + "\n```\n",
    encoding="utf-8",
)

print(summary_csv)
print(deltas_csv)
print(summary_md)
PY
