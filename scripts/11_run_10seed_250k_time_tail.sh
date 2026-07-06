#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${CHECKPOINT:-/tmp/mario_checkpoint_candidates/yumouwei/models/pre-trained-1.zip}"
N_ENVS="${N_ENVS:-64}"
N_STEPS="${N_STEPS:-256}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
EVAL_MAX_EPISODE_STEPS="${EVAL_MAX_EPISODE_STEPS:-5000}"
HOURLY_PRICE="${HOURLY_PRICE:-0.69}"
SEEDS="${SEEDS:-5 6 7 8 9}"
RUN_PREFIX="${RUN_PREFIX:-011_10seed_250k_time_tail}"
OLD_PREFIX="${OLD_PREFIX:-010_high_confidence_time_tail}"
EVAL_STEP=250000
EVAL_CAP_STEPS=5000
FIXED_PENALTY=1000

echo "ETA estimate: 5M train steps + 10 evals, roughly 2-3 hours on the 4090 lane."
echo "Reusing ${OLD_PREFIX} seeds 0-4 at ${EVAL_STEP}; running only seeds: ${SEEDS}."

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
      --total-timesteps "$EVAL_STEP" \
      --n-envs "$N_ENVS" \
      --seed "$seed" \
      --device cpu \
      --learning-rate 0.00001 \
      --n-steps "$N_STEPS" \
      --batch-size "$BATCH_SIZE" \
      "${nns_args[@]}"

    uv run mario-external-checkpoint eval \
      --checkpoint "${run_dir}/model.zip" \
      --run-id "${run_id}/eval_${EVAL_STEP}" \
      --episodes "$EVAL_EPISODES" \
      --max-episode-steps "$EVAL_MAX_EPISODE_STEPS" \
      --seed "$seed" \
      --device cpu

    cat > "${run_dir}/RUN.md" <<EOF
# ${run_id}

Purpose: 011 250k 10-seed confirmation.

- variant: ${variant}
- seed: ${seed}
- timesteps: ${EVAL_STEP}
- eval: ${run_dir}/eval_${EVAL_STEP}/eval_summary.json
- source checkpoint: ${CHECKPOINT}
EOF
  done
done

RUN_PREFIX="$RUN_PREFIX" OLD_PREFIX="$OLD_PREFIX" HOURLY_PRICE="$HOURLY_PRICE" EVAL_STEP="$EVAL_STEP" EVAL_CAP_STEPS="$EVAL_CAP_STEPS" FIXED_PENALTY="$FIXED_PENALTY" uv run python - <<'PY'
import csv
import json
import os
from pathlib import Path

run_prefix = os.environ["RUN_PREFIX"]
old_prefix = os.environ["OLD_PREFIX"]
hourly_price = float(os.environ["HOURLY_PRICE"])
eval_step = int(os.environ["EVAL_STEP"])
eval_cap_steps = float(os.environ["EVAL_CAP_STEPS"])
fixed_penalty = float(os.environ["FIXED_PENALTY"])
out_dir = Path("artifacts/runs")

fields = [
    "run_id",
    "source",
    "variant",
    "seed",
    "checkpoint_step",
    "clear_rate",
    "eval_cap_rate",
    "penalized_completion_cap",
    "penalized_completion_cap_plus_penalty",
    "completion_time_given_clear",
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


def variant_seed(name):
    if "_baseline_plus_time_tail_s" in name:
        return "baseline_plus_time_tail", int(name.rsplit("_s", 1)[1])
    if "_baseline_s" in name:
        return "baseline", int(name.rsplit("_s", 1)[1])
    return None, None


rows = []
for source, prefix, seeds in [("010", old_prefix, range(5)), ("011", run_prefix, range(5, 10))]:
    for train_dir in sorted(out_dir.glob(f"{prefix}_*_s*")):
        variant, seed = variant_seed(train_dir.name)
        if variant is None or seed not in seeds:
            continue
        eval_summary = load_json(train_dir / f"eval_{eval_step}/eval_summary.json")
        if not eval_summary:
            continue
        train_summary = load_json(train_dir / "train_summary.json")
        rollout = last_jsonl(train_dir / "metrics/rollout_metrics.jsonl")
        clear = float(eval_summary.get("clear_rate") or 0.0)
        completion = eval_summary.get("completion_time_mean")
        completion = float(completion) if completion is not None else eval_cap_steps
        train_wall = float(train_summary.get("wall_time_s", 0.0) or 0.0)
        eval_wall = float(eval_summary.get("wall_time_s", 0.0) or 0.0)
        rows.append({
            "run_id": train_dir.name,
            "source": source,
            "variant": variant,
            "seed": seed,
            "checkpoint_step": eval_step,
            "clear_rate": eval_summary.get("clear_rate"),
            "eval_cap_rate": eval_summary.get("eval_cap_rate"),
            "penalized_completion_cap": clear * completion + (1.0 - clear) * eval_cap_steps,
            "penalized_completion_cap_plus_penalty": clear * completion + (1.0 - clear) * (eval_cap_steps + fixed_penalty),
            "completion_time_given_clear": eval_summary.get("completion_time_mean"),
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
            "cost": hourly_price * (train_wall + eval_wall) / 3600 if train_wall or eval_wall else 0.0,
        })

rows.sort(key=lambda row: (row["seed"], row["variant"]))
summary_csv = out_dir / f"{run_prefix}_summary.csv"
with summary_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

delta_fields = [
    "seed",
    "clear_delta",
    "cap_delta",
    "penalized_completion_delta",
    "penalized_completion_cap_plus_penalty_delta",
    "stuck_delta",
    "progress_delta",
]
deltas = []
by_key = {(row["seed"], row["variant"]): row for row in rows}
for seed in sorted({row["seed"] for row in rows}):
    base = by_key.get((seed, "baseline"))
    tail = by_key.get((seed, "baseline_plus_time_tail"))
    if not base or not tail:
        continue
    deltas.append({
        "seed": seed,
        "clear_delta": tail["clear_rate"] - base["clear_rate"],
        "cap_delta": tail["eval_cap_rate"] - base["eval_cap_rate"],
        "penalized_completion_delta": tail["penalized_completion_cap"] - base["penalized_completion_cap"],
        "penalized_completion_cap_plus_penalty_delta": tail["penalized_completion_cap_plus_penalty"] - base["penalized_completion_cap_plus_penalty"],
        "stuck_delta": tail["stuck_fraction"] - base["stuck_fraction"],
        "progress_delta": tail["mean_progress"] - base["mean_progress"],
    })

deltas_csv = out_dir / f"{run_prefix}_paired_deltas.csv"
with deltas_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=delta_fields)
    writer.writeheader()
    writer.writerows(deltas)


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def md_table(table_rows, table_fields):
    lines = ["| " + " | ".join(table_fields) + " |", "| " + " | ".join(["---"] * len(table_fields)) + " |"]
    for row in table_rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in table_fields) + " |")
    return "\n".join(lines)


aggregate_fields = ["variant", "clear_rate", "eval_cap_rate", "penalized_completion_cap", "penalized_completion_cap_plus_penalty", "completion_time_given_clear", "stuck_fraction", "mean_progress", "cost"]
aggregate_rows = []
for variant in ["baseline", "baseline_plus_time_tail"]:
    group = [row for row in rows if row["variant"] == variant]
    aggregate_rows.append({"variant": variant, **{field: round(mean(float(row[field] or 0.0) for row in group), 6) for field in aggregate_fields[1:]}})

wins = {
    "clear_ge_baseline": sum(d["clear_delta"] >= 0 for d in deltas),
    "cap_le_baseline": sum(d["cap_delta"] <= 0 for d in deltas),
    "penalized_completion_le_baseline": sum(d["penalized_completion_delta"] <= 0 for d in deltas),
    "stuck_le_baseline": sum(d["stuck_delta"] <= 0 for d in deltas),
    "progress_ge_baseline": sum(d["progress_delta"] >= 0 for d in deltas),
    "pairs": len(deltas),
}

summary_md = out_dir / f"{run_prefix}_summary.md"
summary_md.write_text(
    "# 011 10-Seed 250k Time-Tail Confirmation\n\n"
    f"Reuses 010 seeds 0-4 and adds 011 seeds 5-9. Non-clears use eval cap `{int(eval_cap_steps)}`; cap+penalty adds `{int(fixed_penalty)}`.\n\n"
    "## Aggregate Means\n\n"
    + md_table(aggregate_rows, aggregate_fields)
    + "\n\n## Paired Deltas\n\n"
    + md_table(deltas, delta_fields)
    + "\n\n## Win Counts\n\n```json\n"
    + json.dumps(wins, indent=2, sort_keys=True)
    + "\n```\n",
    encoding="utf-8",
)

print(summary_csv)
print(deltas_csv)
print(summary_md)
PY
