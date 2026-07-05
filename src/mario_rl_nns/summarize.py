from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path

FIELDS = [
    "run_id",
    "wall_time_s",
    "final_fps",
    "hourly_price",
    "cost_usd",
    "cost_per_1m_train_steps",
    "clear_rate",
    "mean_max_progress",
    "p10_max_progress",
    "p50_max_progress",
    "p90_max_progress",
    "death_rate",
    "stuck_fraction",
    "base_reward_mean",
    "shaped_reward_mean",
    "extra_reward_mean",
    "lpm_speed_below_target",
    "fraction_below_target_speed",
]


def summarize(
    run_dirs: list[Path],
    output_csv: Path,
    output_md: Path,
    hourly_price: float | None = None,
) -> list[dict[str, object]]:
    rows = [_row(run_dir, hourly_price) for run_dir in run_dirs]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    output_md.write_text(_markdown(rows), encoding="utf-8")
    _write_docs(run_dirs, rows)
    return rows


def _row(run_dir: Path, hourly_price: float | None) -> dict[str, object]:
    train_summary = _json(run_dir / "train_summary.json")
    rollout = _last_jsonl(run_dir / "metrics" / "rollout_metrics.jsonl")
    train = _last_jsonl(run_dir / "metrics" / "train_metrics.jsonl")
    eval_summary = _json(run_dir / "metrics" / "eval_summary.json")
    fps = train.get("time_fps")
    if fps is None and train_summary.get("wall_time_s"):
        fps = float(train_summary.get("total_timesteps", 0)) / float(train_summary["wall_time_s"])
    wall_time_s = train_summary.get("wall_time_s")
    timesteps = float(train_summary.get("total_timesteps", 0) or 0)
    cost_usd = None
    cost_per_1m = None
    if hourly_price is not None and wall_time_s:
        cost_usd = hourly_price * float(wall_time_s) / 3600
        if timesteps:
            cost_per_1m = cost_usd / (timesteps / 1_000_000)
    return {
        "run_id": train_summary.get("run_id", run_dir.name),
        "wall_time_s": _round(wall_time_s),
        "final_fps": _round(fps),
        "hourly_price": _round(hourly_price),
        "cost_usd": _round(cost_usd),
        "cost_per_1m_train_steps": _round(cost_per_1m),
        "clear_rate": _round(eval_summary.get("clear_rate")),
        "mean_max_progress": _round(eval_summary.get("max_progress_mean")),
        "p10_max_progress": _round(eval_summary.get("max_progress_p10")),
        "p50_max_progress": _round(eval_summary.get("max_progress_p50")),
        "p90_max_progress": _round(eval_summary.get("max_progress_p90")),
        "death_rate": _round(eval_summary.get("death_rate")),
        "stuck_fraction": _round(rollout.get("stuck_mean")),
        "base_reward_mean": _round(rollout.get("reward_base_mean")),
        "shaped_reward_mean": _round(rollout.get("reward_train_mean")),
        "extra_reward_mean": _round(rollout.get("extra_reward_mean")),
        "lpm_speed_below_target": _round(rollout.get("lpm_speed_mean")),
        "fraction_below_target_speed": _round(rollout.get("fraction_below_target_speed")),
    }


def _write_docs(run_dirs: list[Path], rows: list[dict[str, object]]) -> None:
    docs = Path("docs")
    docs.mkdir(exist_ok=True)
    all_runs = docs / "ALL_RUNS.md"
    all_runs.write_text(_markdown(rows), encoding="utf-8")
    for run_dir, row in zip(run_dirs, rows, strict=False):
        run_dir.joinpath("RUN.md").write_text(_run_md(run_dir, row), encoding="utf-8")


def _run_md(run_dir: Path, row: dict[str, object]) -> str:
    config = _json(run_dir / "config_resolved.json")
    return "\n".join(
        [
            f"# {row['run_id']}",
            "",
            "Purpose: 1M seed-0 diagnostic PPO run.",
            "",
            "| Field | Value |",
            "|---|---:|",
            f"| env | {config.get('env_id')} |",
            f"| action_space | {config.get('action_space')} |",
            f"| variant | {config.get('variant')} |",
            f"| seed | {config.get('seed')} |",
            f"| n_envs | {config.get('n_envs')} |",
            f"| timesteps | {config.get('total_timesteps')} |",
            "",
            _markdown([row]),
            "",
            "Interpretation: compare task metrics, not shaped reward alone.",
            "",
            "Artifacts:",
            f"- `{run_dir / 'model.zip'}`",
            f"- `{run_dir / 'train_summary.json'}`",
            f"- `{run_dir / 'metrics'}`",
        ]
    )


def _markdown(rows: list[dict[str, object]]) -> str:
    lines = ["| " + " | ".join(FIELDS) + " |", "| " + " | ".join(["---"] * len(FIELDS)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in FIELDS) + " |")
    return "\n".join(lines) + "\n"


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text()) if path.exists() else {}


def _last_jsonl(path: Path) -> dict[str, object]:
    if not path.exists() or not path.read_text().strip():
        return {}
    return json.loads(path.read_text().strip().splitlines()[-1])


def _round(value):
    return None if value is None else round(float(value), 4)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--hourly-price", type=float)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    rows = summarize(args.run_dir, args.output_csv, args.output_md, args.hourly_price)
    print(_markdown(rows))


if __name__ == "__main__":
    main()
