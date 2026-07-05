from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from mario_rl_nns.reward_wrappers import reward_shaping_from_config
from mario_rl_nns.wrappers import make_vec_env


class ShapingMetricsCallback(BaseCallback):
    def __init__(self, metrics_dir: Path, started: float):
        super().__init__()
        self.metrics_dir = metrics_dir
        self.started = started
        self.rows: list[dict[str, float]] = []
        self.episodes: list[dict[str, float]] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            episode = info.get("episode")
            if isinstance(episode, dict):
                self.episodes.append(
                    {
                        key: float(value)
                        for key, value in episode.items()
                        if isinstance(value, int | float)
                    }
                )
            shaping = info.get("shaping")
            if isinstance(shaping, dict):
                self.rows.append(
                    {
                        key: float(value)
                        for key, value in shaping.items()
                        if isinstance(value, int | float)
                    }
                )
        return True

    def _on_rollout_end(self) -> None:
        if not self.rows:
            return
        keys = sorted({key for row in self.rows for key in row})
        out = {
            "timestep": self.num_timesteps,
            "wall_time_s": time.perf_counter() - self.started,
            "samples": len(self.rows),
        }
        for key in keys:
            values = [row[key] for row in self.rows if key in row]
            out[f"{key}_mean"] = sum(values) / len(values)
            out[f"{key}_min"] = min(values)
            out[f"{key}_max"] = max(values)
            if key in {"death", "timeout", "stuck"}:
                out[f"{key}_count"] = sum(values)
        if "actual_speed" in keys and "target_speed" in keys:
            below = [
                row["actual_speed"] < row["target_speed"]
                for row in self.rows
                if "actual_speed" in row and "target_speed" in row
            ]
            if below:
                out["fraction_below_target_speed"] = sum(below) / len(below)
        if self.episodes:
            lengths = [row["l"] for row in self.episodes if "l" in row]
            returns = [row["r"] for row in self.episodes if "r" in row]
            if lengths:
                out["episode_length_mean"] = sum(lengths) / len(lengths)
            if returns:
                out["episode_return_train_mean"] = sum(returns) / len(returns)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        for name in ("rollout_metrics.jsonl", "shaping_metrics.jsonl"):
            with (self.metrics_dir / name).open("a", encoding="utf-8") as f:
                f.write(json.dumps(out, sort_keys=True) + "\n")
        train = {"timestep": self.num_timesteps, "wall_time_s": out["wall_time_s"]}
        for key, value in self.model.logger.name_to_value.items():
            if isinstance(value, int | float):
                train[key.replace("/", "_")] = float(value)
        with (self.metrics_dir / "train_metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(train, sort_keys=True) + "\n")
        self.rows.clear()
        self.episodes.clear()


class BaselineMetricsCallback(BaseCallback):
    def __init__(self, metrics_dir: Path, started: float):
        super().__init__()
        self.metrics_dir = metrics_dir
        self.started = started
        self.rows: list[dict[str, float]] = []
        self.episodes: list[dict[str, float]] = []

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", [])
        for reward, info in zip(rewards, self.locals.get("infos", []), strict=False):
            progress = float(info.get("progress", info.get("x_pos", 0.0)))
            row = {
                "reward_base": float(reward),
                "reward_train": float(reward),
                "extra_reward": 0.0,
                "progress": progress,
                "progress_max": float(info.get("progress_max", info.get("x_pos_max", progress))),
                "progress_delta": float(info.get("reward_components", {}).get("progress", 0.0)),
                "clear": float(bool(info.get("clear", info.get("flag_get", False)))),
                "death": float(bool(info.get("death", info.get("is_dead", False)))),
                "timeout": float(bool(info.get("timeout", False))),
                "stuck": 0.0,
            }
            self.rows.append(row)
            episode = info.get("episode")
            if isinstance(episode, dict):
                self.episodes.append(
                    {
                        key: float(value)
                        for key, value in episode.items()
                        if isinstance(value, int | float)
                    }
                )
        return True

    def _on_rollout_end(self) -> None:
        if not self.rows:
            return
        keys = sorted({key for row in self.rows for key in row})
        out = {
            "timestep": self.num_timesteps,
            "wall_time_s": time.perf_counter() - self.started,
            "samples": len(self.rows),
        }
        for key in keys:
            values = [row[key] for row in self.rows if key in row]
            out[f"{key}_mean"] = sum(values) / len(values)
            out[f"{key}_min"] = min(values)
            out[f"{key}_max"] = max(values)
            if key in {"death", "timeout", "stuck"}:
                out[f"{key}_count"] = sum(values)
        if self.episodes:
            lengths = [row["l"] for row in self.episodes if "l" in row]
            returns = [row["r"] for row in self.episodes if "r" in row]
            if lengths:
                out["episode_length_mean"] = sum(lengths) / len(lengths)
            if returns:
                out["episode_return_train_mean"] = sum(returns) / len(returns)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        with (self.metrics_dir / "rollout_metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(out, sort_keys=True) + "\n")
        train = {"timestep": self.num_timesteps, "wall_time_s": out["wall_time_s"]}
        for key, value in self.model.logger.name_to_value.items():
            if isinstance(value, int | float):
                train[key.replace("/", "_")] = float(value)
        with (self.metrics_dir / "train_metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(train, sort_keys=True) + "\n")
        self.rows.clear()
        self.episodes.clear()


def load_config(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def override(config: dict[str, object], key: str, value: object | None) -> None:
    if value is not None:
        config[key] = value


def git_info() -> dict[str, object]:
    def run(*args: str) -> str:
        return subprocess.run(args, check=False, capture_output=True, text=True).stdout.strip()

    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "short_commit": run("git", "rev-parse", "--short", "HEAD"),
        "branch": run("git", "branch", "--show-current"),
        "remote": run("git", "remote", "get-url", "origin"),
        "dirty": bool(run("git", "status", "--porcelain")),
    }


def env_info() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cwd": os.getcwd(),
    }


def train(config: dict[str, object], run_id: str, output_dir: Path) -> Path:
    env_id = str(config.get("env_id", "SuperMarioBros-1-1-v0"))
    action_space = str(config.get("action_space", "RIGHT_ONLY"))
    seed = int(config.get("seed", 0))
    n_envs = int(config.get("n_envs", 1))
    total_timesteps = int(config.get("total_timesteps", 50_000))
    device = str(config.get("device", "auto"))

    run_dir = output_dir / run_id
    metrics_dir = run_dir / "metrics"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (run_dir / "config_resolved.json").write_text(
        json.dumps(config, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (run_dir / "git_info.json").write_text(json.dumps(git_info(), indent=2), encoding="utf-8")
    (run_dir / "env_info.json").write_text(json.dumps(env_info(), indent=2), encoding="utf-8")

    reward_shaping = reward_shaping_from_config(config)
    env = make_vec_env(
        env_id=env_id,
        action_space=action_space,
        seed=seed,
        n_envs=n_envs,
        reward_shaping=reward_shaping,
    )
    try:
        model = PPO(
            "CnnPolicy",
            env,
            seed=seed,
            device=device,
            verbose=1,
            tensorboard_log=str(run_dir / "tb"),
            n_steps=int(config.get("n_steps", 128)),
            batch_size=int(config.get("batch_size", 64)),
        )
        started = time.perf_counter()
        callback = (
            ShapingMetricsCallback(metrics_dir, started)
            if reward_shaping is not None
            else BaselineMetricsCallback(metrics_dir, started)
        )
        model.learn(total_timesteps=total_timesteps, tb_log_name=run_id, callback=callback)
        model_path = run_dir / "model.zip"
        model.save(model_path)
        metadata = {
            "run_id": run_id,
            "env_id": env_id,
            "action_space": action_space,
            "seed": seed,
            "n_envs": n_envs,
            "total_timesteps": total_timesteps,
            "wall_time_s": time.perf_counter() - started,
            "model_path": str(model_path),
            "metrics_dir": str(metrics_dir),
        }
        (run_dir / "train_summary.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return model_path
    finally:
        env.close()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/baseline_ppo.yaml"))
    parser.add_argument("--run-id")
    parser.add_argument("--env-id")
    parser.add_argument("--action-space")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--total-timesteps", type=int)
    parser.add_argument("--n-envs", type=int)
    parser.add_argument("--n-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/runs"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    override(config, "env_id", args.env_id)
    override(config, "action_space", args.action_space)
    override(config, "seed", args.seed)
    override(config, "total_timesteps", args.total_timesteps)
    override(config, "n_envs", args.n_envs)
    override(config, "n_steps", args.n_steps)
    override(config, "batch_size", args.batch_size)
    override(config, "device", args.device)
    run_id = args.run_id or f"{config.get('variant', 'ppo')}_s{config.get('seed', 0)}"
    print(train(config, run_id=run_id, output_dir=args.output_dir))


if __name__ == "__main__":
    main()
