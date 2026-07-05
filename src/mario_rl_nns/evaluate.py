from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from stable_baselines3 import PPO

from mario_rl_nns.train_sb3 import load_config
from mario_rl_nns.wrappers import make_vec_env


def evaluate(
    model_path: Path,
    config: dict[str, object],
    episodes: int,
    output_dir: Path,
    max_episode_steps: int | None,
) -> dict[str, object]:
    env_id = str(config.get("env_id", "SuperMarioBros-1-1-v0"))
    action_space = str(config.get("action_space", "RIGHT_ONLY"))
    seed = int(config.get("seed", 0)) + 10_000
    env = make_vec_env(env_id=env_id, action_space=action_space, seed=seed, n_envs=1)
    model = PPO.load(model_path, env=env, device=str(config.get("device", "auto")))
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "eval_metrics.jsonl"
    rows = []
    obs = env.reset()
    current = _new_episode()
    try:
        with metrics_path.open("w", encoding="utf-8") as f:
            while len(rows) < episodes:
                action, _ = model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = env.step(action)
                info = infos[0]
                current["episode_return_train"] += float(rewards[0])
                current["episode_length"] += 1
                progress = float(info.get("progress", info.get("x_pos", 0.0)))
                current["max_progress"] = max(current["max_progress"], progress)
                current["max_x_pos"] = max(current["max_x_pos"], float(info.get("x_pos", progress)))
                current["death"] = current["death"] or bool(
                    info.get("death", info.get("is_dead", False))
                )
                current["timeout"] = current["timeout"] or bool(info.get("timeout", False))
                current["clear"] = current["clear"] or bool(
                    info.get("clear", info.get("flag_get", False))
                )
                capped = (
                    max_episode_steps is not None
                    and current["episode_length"] >= max_episode_steps
                )
                if dones[0] or capped:
                    current["eval_cap"] = capped
                    rows.append(current)
                    f.write(json.dumps(current, sort_keys=True) + "\n")
                    f.flush()
                    current = _new_episode()
                    if capped:
                        obs = env.reset()
        summary = _summary(rows)
        (output_dir / "eval_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        return summary
    finally:
        env.close()


def _new_episode() -> dict[str, object]:
    return {
        "episode_return_train": 0.0,
        "episode_length": 0,
        "max_progress": 0.0,
        "max_x_pos": 0.0,
        "clear": False,
        "death": False,
        "timeout": False,
        "eval_cap": False,
    }


def _summary(rows: list[dict[str, object]]) -> dict[str, object]:
    progress = sorted(float(row["max_progress"]) for row in rows)
    lengths = [float(row["episode_length"]) for row in rows]
    returns = [float(row["episode_return_train"]) for row in rows]
    return {
        "episodes": len(rows),
        "clear_rate": _mean(bool(row["clear"]) for row in rows),
        "death_rate": _mean(bool(row["death"]) for row in rows),
        "timeout_rate": _mean(bool(row["timeout"]) for row in rows),
        "eval_cap_rate": _mean(bool(row["eval_cap"]) for row in rows),
        "max_progress_mean": sum(progress) / len(progress),
        "max_progress_p10": _percentile(progress, 10),
        "max_progress_p50": _percentile(progress, 50),
        "max_progress_p90": _percentile(progress, 90),
        "episode_length_mean": sum(lengths) / len(lengths),
        "episode_return_train_mean": sum(returns) / len(returns),
    }


def _mean(values) -> float:
    vals = [float(value) for value in values]
    return sum(vals) / len(vals)


def _percentile(sorted_values: list[float], pct: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * pct / 100
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-episode-steps", type=int, default=20_000)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config_path = args.config or args.run_dir / "config_resolved.json"
    config = load_config(config_path)
    summary = evaluate(
        args.run_dir / "model.zip",
        config,
        args.episodes,
        args.run_dir / "metrics",
        args.max_episode_steps,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
