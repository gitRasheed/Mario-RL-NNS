from __future__ import annotations

import argparse
import hashlib
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
    eval_seeds: list[int] | None = None,
    eval_prefix_steps: list[int] | None = None,
    eval_prefix_action: int = 0,
) -> dict[str, object]:
    env_id = str(config.get("env_id", "SuperMarioBros-1-1-v0"))
    action_space = str(config.get("action_space", "RIGHT_ONLY"))
    base_seed = int(config.get("seed", 0)) + 10_000
    seeds = eval_seeds or [base_seed + i for i in range(episodes)]
    prefixes = eval_prefix_steps or [0, 5, 10, 15, 20, 25, 30]
    model = PPO.load(model_path, device=str(config.get("device", "auto")))
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "eval_metrics.jsonl"
    per_episode_path = output_dir / "per_episode_eval.jsonl"
    rows = []
    with (
        metrics_path.open("w", encoding="utf-8") as metrics_f,
        per_episode_path.open("w", encoding="utf-8") as episode_f,
    ):
        while len(rows) < episodes:
            episode_index = len(rows)
            seed = seeds[episode_index % len(seeds)]
            prefix_steps = prefixes[episode_index % len(prefixes)]
            env = make_vec_env(env_id=env_id, action_space=action_space, seed=seed, n_envs=1)
            current = _new_episode(episode_index, seed, prefix_steps, eval_prefix_action)
            trajectory = hashlib.sha1()
            try:
                obs = env.reset()
                for _ in range(prefix_steps):
                    obs, _rewards, dones, _infos = env.step([eval_prefix_action])
                    if dones[0]:
                        obs = env.reset()
                action, _ = model.predict(obs, deterministic=True)
                while True:
                    obs, rewards, dones, infos = env.step(action)
                    info = infos[0]
                    current["episode_return_train"] += float(rewards[0])
                    current["episode_length"] += 1
                    progress = float(info.get("progress", info.get("x_pos", 0.0)))
                    x_pos = float(info.get("x_pos", progress))
                    trajectory.update(f"{progress:.1f},{x_pos:.1f},{int(action[0])};".encode())
                    current["max_progress"] = max(current["max_progress"], progress)
                    current["max_x_pos"] = max(current["max_x_pos"], x_pos)
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
                        current["trajectory_hash"] = trajectory.hexdigest()
                        rows.append(current)
                        line = json.dumps(current, sort_keys=True) + "\n"
                        metrics_f.write(line)
                        episode_f.write(line)
                        metrics_f.flush()
                        episode_f.flush()
                        break
                    action, _ = model.predict(obs, deterministic=True)
            finally:
                env.close()
    summary = _summary(rows)
    (output_dir / "eval_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def _new_episode(
    episode_index: int = 0,
    eval_seed: int = 0,
    eval_prefix_steps: int = 0,
    eval_prefix_action: int = 0,
) -> dict[str, object]:
    return {
        "episode_index": episode_index,
        "eval_seed": eval_seed,
        "eval_prefix_steps": eval_prefix_steps,
        "eval_prefix_action": eval_prefix_action,
        "episode_return_train": 0.0,
        "episode_length": 0,
        "max_progress": 0.0,
        "max_x_pos": 0.0,
        "clear": False,
        "death": False,
        "timeout": False,
        "eval_cap": False,
        "trajectory_hash": "",
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
        "unique_trajectory_count": len({str(row["trajectory_hash"]) for row in rows}),
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


def _parse_int_list(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-episode-steps", type=int, default=20_000)
    parser.add_argument("--eval-seeds")
    parser.add_argument("--eval-prefix-steps", default="0,5,10,15,20,25,30")
    parser.add_argument("--eval-prefix-action", type=int, default=0)
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
        _parse_int_list(args.eval_seeds),
        _parse_int_list(args.eval_prefix_steps),
        args.eval_prefix_action,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
