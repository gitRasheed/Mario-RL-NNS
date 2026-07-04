from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from mario_rl_nns.reward_wrappers import reward_shaping_from_config
from mario_rl_nns.wrappers import make_vec_env


class ShapingMetricsCallback(BaseCallback):
    def __init__(self, path: Path):
        super().__init__()
        self.path = path
        self.rows: list[dict[str, float]] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
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
        out = {"timestep": self.num_timesteps, "samples": len(self.rows)}
        for key in keys:
            values = [row[key] for row in self.rows if key in row]
            out[f"{key}_mean"] = sum(values) / len(values)
            if key in {"death", "timeout", "stuck"}:
                out[f"{key}_count"] = sum(values)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(out, sort_keys=True) + "\n")
        self.rows.clear()


def load_config(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def override(config: dict[str, object], key: str, value: object | None) -> None:
    if value is not None:
        config[key] = value


def train(config: dict[str, object], run_id: str, output_dir: Path) -> Path:
    env_id = str(config.get("env_id", "SuperMarioBros-1-1-v0"))
    action_space = str(config.get("action_space", "RIGHT_ONLY"))
    seed = int(config.get("seed", 0))
    n_envs = int(config.get("n_envs", 1))
    total_timesteps = int(config.get("total_timesteps", 50_000))
    device = str(config.get("device", "auto"))

    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True),
        encoding="utf-8",
    )

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
        metrics_path = run_dir / "shaping_metrics.jsonl"
        callback = ShapingMetricsCallback(metrics_path)
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
            "shaping_metrics_path": str(metrics_path),
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
    override(config, "device", args.device)
    run_id = args.run_id or f"{config.get('variant', 'ppo')}_s{config.get('seed', 0)}"
    print(train(config, run_id=run_id, output_dir=args.output_dir))


if __name__ == "__main__":
    main()
