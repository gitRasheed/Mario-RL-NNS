from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

from mario_rl_nns.nns_rewards import NNSRewardConfig
from mario_rl_nns.ram_adapter import RAM_OBSERVATION_SPACE, make_ram_vec_env
from mario_rl_nns.reward_wrappers import RewardShapingConfig
from mario_rl_nns.train_sb3 import (
    BaselineMetricsCallback,
    ShapingMetricsCallback,
    env_info,
    git_info,
)

SIMPLE_ACTION_COUNT = 7
DEFAULT_PREFIXES = [0, 5, 10, 15, 20, 25, 30]
DEFAULT_CHECKPOINT = Path("/tmp/mario_checkpoint_candidates/yumouwei/models/pre-trained-1.zip")
NNS_PRESETS = (
    "conservative",
    "speed_tail_tiny",
    "speed_tail_clear_gated",
    "completion_time_lpm",
    "baseline_plus_time_tail",
)


def load_external_ppo(
    checkpoint: Path,
    device: str,
    learning_rate: float = 0.0,
    env=None,
    n_steps: int | None = None,
    batch_size: int | None = None,
) -> PPO:
    sys.modules.setdefault("gym", gym)
    sys.modules.setdefault("gym.spaces", gym.spaces)
    custom_objects = {
        "observation_space": RAM_OBSERVATION_SPACE,
        "action_space": spaces.Discrete(SIMPLE_ACTION_COUNT),
        "learning_rate": learning_rate,
        "lr_schedule": lambda _: learning_rate,
        "clip_range": lambda _: 0.2,
    }
    if n_steps is not None:
        custom_objects["n_steps"] = n_steps
    if batch_size is not None:
        custom_objects["batch_size"] = batch_size
    return PPO.load(
        checkpoint,
        env=env,
        device=device,
        custom_objects=custom_objects,
        print_system_info=False,
    )


def nns_preset(name: str, n_envs: int) -> RewardShapingConfig:
    presets = {
        "conservative": dict(
            window=64,
            target_speed=1.0,
            lambda_down=0.02,
            lambda_death=0.1,
            clip_min=-1.0,
            clip_max=0.25,
        ),
        "speed_tail_tiny": dict(
            window=64,
            target_speed=2.95,
            lambda_down=0.002,
            clip_min=-0.05,
            clip_max=0.0,
        ),
        "speed_tail_clear_gated": dict(
            window=64,
            target_speed=2.95,
            lambda_down=0.004,
            clip_min=-0.08,
            clip_max=0.0,
            global_warmup_steps=50_000,
        ),
        "completion_time_lpm": dict(
            window=64,
            target_speed=0.0,
            lambda_down=0.0,
            clip_min=-0.25,
            clip_max=0.0,
            target_clear_steps=1077,
            lambda_slow_clear=0.0005,
        ),
        "baseline_plus_time_tail": dict(
            window=64,
            target_speed=0.0,
            lambda_down=0.0,
            clip_min=-0.1,
            clip_max=0.0,
            target_clear_steps=1077,
            lambda_slow_clear=0.0001,
        ),
    }
    if name not in presets:
        raise ValueError(f"unknown NNS preset {name!r}")
    values = presets[name]
    return RewardShapingConfig(
        variant="nns_lpm",
        nns=NNSRewardConfig(
            variant="nns_lpm",
            window=int(values["window"]),
            target_speed=float(values["target_speed"]),
            d_down=2.0,
            lambda_down=float(values["lambda_down"]),
            lambda_stuck=0.0,
            lambda_death=float(values.get("lambda_death", 0.0)),
            lambda_timeout=0.0,
            clip_min=float(values["clip_min"]),
            clip_max=float(values["clip_max"]),
            global_warmup_steps=int(values.get("global_warmup_steps", 0)),
            n_envs=n_envs,
            target_clear_steps=int(values.get("target_clear_steps", 0)),
            lambda_slow_clear=float(values.get("lambda_slow_clear", 0.0)),
        ),
    )


def evaluate(
    checkpoint: Path,
    run_dir: Path,
    episodes: int = 50,
    max_episode_steps: int = 5000,
    seed: int = 0,
    device: str = "cpu",
) -> dict[str, object]:
    model = load_external_ppo(checkpoint, device=device)
    rows = []
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with (metrics_dir / "eval_metrics.jsonl").open("w", encoding="utf-8") as f:
        for episode in range(episodes):
            env = make_ram_vec_env(seed=seed + 10_000 + episode, n_envs=1)
            try:
                obs = env.reset()
                prefix = DEFAULT_PREFIXES[episode % len(DEFAULT_PREFIXES)]
                for _ in range(prefix):
                    obs, _reward, dones, _infos = env.step([0])
                    if dones[0]:
                        obs = env.reset()
                row = _episode_row(episode, seed + 10_000 + episode, prefix)
                trajectory = hashlib.sha1()
                progress_history: list[float] = []
                stuck_steps = 0
                while row["episode_length"] < max_episode_steps:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, rewards, dones, infos = env.step(action)
                    info = infos[0]
                    progress = float(info.get("progress", info.get("x_pos", 0.0)))
                    x_pos = float(info.get("x_pos", progress))
                    row["episode_return_base"] += float(rewards[0])
                    row["episode_length"] += 1
                    row["max_progress"] = max(row["max_progress"], progress)
                    row["max_x_pos"] = max(row["max_x_pos"], x_pos)
                    row["clear"] = row["clear"] or bool(
                        info.get("clear", info.get("flag_get", False))
                    )
                    row["death"] = row["death"] or bool(
                        info.get("death", info.get("is_dead", False))
                    )
                    row["timeout"] = row["timeout"] or bool(info.get("timeout", False))
                    progress_history.append(progress)
                    if len(progress_history) > 32 and progress <= progress_history[-32]:
                        stuck_steps += 1
                    trajectory.update(f"{progress:.0f},{x_pos:.0f},{int(action[0])};".encode())
                    if row["clear"] and row["completion_time"] is None:
                        row["completion_time"] = row["episode_length"]
                    if dones[0]:
                        break
                row["eval_cap"] = row["episode_length"] >= max_episode_steps
                row["stuck_fraction"] = stuck_steps / row["episode_length"]
                row["trajectory_hash"] = trajectory.hexdigest()
                rows.append(row)
                f.write(json.dumps(row, sort_keys=True) + "\n")
            finally:
                env.close()
    summary = _summary(rows, wall_time_s=time.perf_counter() - started)
    _write_json(run_dir / "eval_summary.json", summary)
    _write_json(metrics_dir / "eval_summary.json", summary)
    _write_run_md(run_dir, "External checkpoint evaluation", summary)
    return summary


def finetune(
    checkpoint: Path,
    run_dir: Path,
    variant: str,
    total_timesteps: int,
    n_envs: int,
    seed: int,
    device: str,
    learning_rate: float,
    n_steps: int,
    batch_size: int,
    nns_config: str = "conservative",
) -> dict[str, object]:
    reward_shaping = nns_preset(nns_config, n_envs) if variant == "nns" else None
    env = make_ram_vec_env(seed=seed, n_envs=n_envs, reward_shaping=reward_shaping)
    metrics_dir = run_dir / "metrics"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "variant": f"external_checkpoint_{variant}_finetune",
        "checkpoint": str(checkpoint),
        "total_timesteps": total_timesteps,
        "n_envs": n_envs,
        "seed": seed,
        "device": device,
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "nns_config": nns_config if variant == "nns" else None,
    }
    if reward_shaping is not None:
        config["nns"] = asdict(reward_shaping.nns)
    _write_json(run_dir / "config_resolved.json", config)
    _write_json(run_dir / "git_info.json", git_info())
    _write_json(run_dir / "env_info.json", env_info())
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    try:
        model = load_external_ppo(
            checkpoint,
            device=device,
            learning_rate=learning_rate,
            env=env,
            n_steps=n_steps,
            batch_size=batch_size,
        )
        started = time.perf_counter()
        callback = (
            ShapingMetricsCallback(metrics_dir, started)
            if reward_shaping is not None
            else BaselineMetricsCallback(metrics_dir, started)
        )
        model.learn(total_timesteps=total_timesteps, callback=callback, reset_num_timesteps=False)
        model.save(run_dir / "model.zip")
        summary = {
            "run_id": run_dir.name,
            "variant": variant,
            "total_timesteps": total_timesteps,
            "n_envs": n_envs,
            "seed": seed,
            "device": device,
            "learning_rate": learning_rate,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "wall_time_s": time.perf_counter() - started,
            "model_path": str(run_dir / "model.zip"),
        }
        _write_json(run_dir / "train_summary.json", summary)
        return summary
    finally:
        env.close()


def summarize(run_dirs: list[Path], output_csv: Path, output_md: Path, hourly_price: float) -> None:
    fields = [
        "run_id",
        "clear_rate",
        "eval_cap_rate",
        "completion_time_mean",
        "completion_time_p50",
        "completion_time_p90",
        "mean_progress",
        "p10_progress",
        "p50_progress",
        "p90_progress",
        "death_rate",
        "stuck_fraction",
        "base_reward_mean",
        "shaped_reward_mean",
        "extra_reward_mean",
        "unique_trajectory_count",
        "wall_time_s",
        "cost",
    ]
    rows = []
    for run_dir in run_dirs:
        eval_summary = _json(run_dir / "eval_summary.json") or _json(
            run_dir / "metrics/eval_summary.json"
        )
        train_summary = _json(run_dir / "train_summary.json")
        rollout = _last_jsonl(run_dir / "metrics/rollout_metrics.jsonl")
        wall_time = train_summary.get("wall_time_s", eval_summary.get("wall_time_s"))
        rows.append(
            {
                "run_id": run_dir.name,
                "clear_rate": eval_summary.get("clear_rate"),
                "eval_cap_rate": eval_summary.get("eval_cap_rate"),
                "completion_time_mean": eval_summary.get("completion_time_mean"),
                "completion_time_p50": eval_summary.get("completion_time_p50"),
                "completion_time_p90": eval_summary.get("completion_time_p90"),
                "mean_progress": eval_summary.get("mean_progress"),
                "p10_progress": eval_summary.get("p10_progress"),
                "p50_progress": eval_summary.get("p50_progress"),
                "p90_progress": eval_summary.get("p90_progress"),
                "death_rate": eval_summary.get("death_rate"),
                "stuck_fraction": eval_summary.get("stuck_fraction"),
                "base_reward_mean": rollout.get(
                    "reward_base_mean",
                    eval_summary.get("base_reward_mean"),
                ),
                "shaped_reward_mean": rollout.get("reward_train_mean"),
                "extra_reward_mean": rollout.get("extra_reward_mean"),
                "unique_trajectory_count": eval_summary.get("unique_trajectory_count"),
                "wall_time_s": wall_time,
                "cost": hourly_price * float(wall_time) / 3600 if wall_time else None,
            }
        )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    output_md.write_text(_markdown(rows, fields), encoding="utf-8")


def _episode_row(index: int, seed: int, prefix: int) -> dict[str, object]:
    return {
        "episode_index": index,
        "eval_seed": seed,
        "eval_prefix_steps": prefix,
        "episode_return_base": 0.0,
        "episode_length": 0,
        "max_progress": 0.0,
        "max_x_pos": 0.0,
        "clear": False,
        "death": False,
        "timeout": False,
        "eval_cap": False,
        "completion_time": None,
        "stuck_fraction": 0.0,
        "trajectory_hash": "",
    }


def _summary(rows: list[dict[str, object]], wall_time_s: float) -> dict[str, object]:
    progress = [float(row["max_progress"]) for row in rows]
    clear_times = [
        float(row["completion_time"]) for row in rows if row["completion_time"] is not None
    ]
    return {
        "episodes": len(rows),
        "wall_time_s": wall_time_s,
        "clear_rate": _mean(row["clear"] for row in rows),
        "death_rate": _mean(row["death"] for row in rows),
        "timeout_rate": _mean(row["timeout"] for row in rows),
        "eval_cap_rate": _mean(row["eval_cap"] for row in rows),
        "unique_trajectory_count": len({str(row["trajectory_hash"]) for row in rows}),
        "mean_progress": _mean(progress),
        "p10_progress": _percentile(progress, 10),
        "p50_progress": _percentile(progress, 50),
        "p90_progress": _percentile(progress, 90),
        "stuck_fraction": _mean(row["stuck_fraction"] for row in rows),
        "base_reward_mean": _mean(row["episode_return_base"] for row in rows),
        "episode_length_mean": _mean(row["episode_length"] for row in rows),
        "completion_time_mean": _mean(clear_times),
        "completion_time_p50": _percentile(clear_times, 50),
        "completion_time_p90": _percentile(clear_times, 90),
    }


def _mean(values) -> float | None:
    vals = [float(value) for value in values]
    return sum(vals) / len(vals) if vals else None


def _percentile(values, pct: float) -> float | None:
    vals = sorted(float(value) for value in values)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * pct / 100
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def _write_run_md(run_dir: Path, purpose: str, summary: dict[str, object]) -> None:
    run_dir.joinpath("RUN.md").write_text(
        "\n".join(
            [
                f"# {run_dir.name}",
                "",
                f"Purpose: {purpose}.",
                "",
                "Source: yumouwei/super-mario-bros-reinforcement-learning `pre-trained-1.zip`.",
                "Provenance: external GPL project used as checkpoint reference; "
                "wrapper code here is local.",
                "",
                "```json",
                json.dumps(summary, indent=2, sort_keys=True),
                "```",
            ]
        ),
        encoding="utf-8",
    )


def _markdown(rows: list[dict[str, object]], fields: list[str]) -> str:
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return "\n".join(lines) + "\n"


def _write_json(path: Path, data: dict[str, object]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text()) if path.exists() else {}


def _last_jsonl(path: Path) -> dict[str, object]:
    if not path.exists() or not path.read_text().strip():
        return {}
    return json.loads(path.read_text().strip().splitlines()[-1])


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    eval_p = sub.add_parser("eval")
    eval_p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    eval_p.add_argument("--run-id", default="007_clean_adapter_checkpoint_eval")
    eval_p.add_argument("--output-dir", type=Path, default=Path("artifacts/runs"))
    eval_p.add_argument("--episodes", type=int, default=50)
    eval_p.add_argument("--max-episode-steps", type=int, default=5000)
    eval_p.add_argument("--seed", type=int, default=0)
    eval_p.add_argument("--device", default="cpu")

    train_p = sub.add_parser("finetune")
    train_p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    train_p.add_argument("--run-id", required=True)
    train_p.add_argument("--variant", choices=["baseline", "nns"], required=True)
    train_p.add_argument("--output-dir", type=Path, default=Path("artifacts/runs"))
    train_p.add_argument("--total-timesteps", type=int, default=1_000_000)
    train_p.add_argument("--n-envs", type=int, default=16)
    train_p.add_argument("--seed", type=int, default=0)
    train_p.add_argument("--device", default="cpu")
    train_p.add_argument("--learning-rate", type=float, default=1e-5)
    train_p.add_argument("--n-steps", type=int, default=256)
    train_p.add_argument("--batch-size", type=int, default=256)
    train_p.add_argument("--nns-config", choices=NNS_PRESETS, default="conservative")

    sum_p = sub.add_parser("summary")
    sum_p.add_argument("--run-dir", action="append", type=Path, required=True)
    sum_p.add_argument("--output-csv", type=Path, required=True)
    sum_p.add_argument("--output-md", type=Path, required=True)
    sum_p.add_argument("--hourly-price", type=float, default=0.69)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "eval":
        summary = evaluate(
            args.checkpoint,
            args.output_dir / args.run_id,
            episodes=args.episodes,
            max_episode_steps=args.max_episode_steps,
            seed=args.seed,
            device=args.device,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
    elif args.command == "finetune":
        summary = finetune(
            args.checkpoint,
            args.output_dir / args.run_id,
            args.variant,
            args.total_timesteps,
            args.n_envs,
            args.seed,
            args.device,
            args.learning_rate,
            args.n_steps,
            args.batch_size,
            args.nns_config,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
    elif args.command == "summary":
        summarize(args.run_dir, args.output_csv, args.output_md, args.hourly_price)


if __name__ == "__main__":
    main()
