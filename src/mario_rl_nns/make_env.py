from __future__ import annotations

import argparse
from collections.abc import Sequence

import gym_super_mario_bros  # noqa: F401  # registers Mario envs with Gymnasium
import gymnasium as gym
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, RIGHT_ONLY, SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

ACTION_SPACES = {
    "RIGHT_ONLY": RIGHT_ONLY,
    "SIMPLE_MOVEMENT": SIMPLE_MOVEMENT,
    "COMPLEX_MOVEMENT": COMPLEX_MOVEMENT,
}

USEFUL_INFO_KEYS = (
    "x_pos",
    "progress",
    "progress_max",
    "clear",
    "death",
    "timeout",
)


def make_mario_env(
    env_id: str = "SuperMarioBros-1-1-v0",
    action_space: str = "RIGHT_ONLY",
    render_mode: str | None = None,
):
    if action_space not in ACTION_SPACES:
        choices = ", ".join(ACTION_SPACES)
        raise ValueError(f"unknown action_space {action_space!r}; expected one of {choices}")
    env = gym.make(env_id, render_mode=render_mode)
    return JoypadSpace(env, ACTION_SPACES[action_space])


def smoke_env(
    env_id: str = "SuperMarioBros-1-1-v0",
    action_space: str = "RIGHT_ONLY",
    steps: int = 1000,
    seed: int = 0,
) -> dict[str, object]:
    env = make_mario_env(env_id=env_id, action_space=action_space)
    try:
        obs, info = env.reset(seed=seed)
        last_reward = 0.0
        terminated = truncated = False
        for _ in range(steps):
            obs, last_reward, terminated, truncated, info = env.step(env.action_space.sample())
            if terminated or truncated:
                obs, info = env.reset()
        return {
            "env_id": env_id,
            "action_space": action_space,
            "steps": steps,
            "obs_shape": getattr(obs, "shape", None),
            "last_reward": float(last_reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "info_keys": sorted(info),
            "useful_info": {key: info.get(key) for key in USEFUL_INFO_KEYS if key in info},
        }
    finally:
        env.close()


def print_smoke_result(result: dict[str, object]) -> None:
    for key, value in result.items():
        print(f"{key}: {value}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="SuperMarioBros-1-1-v0")
    parser.add_argument("--action-space", default="RIGHT_ONLY", choices=ACTION_SPACES)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    print_smoke_result(
        smoke_env(
            env_id=args.env_id,
            action_space=args.action_space,
            steps=args.steps,
            seed=args.seed,
        )
    )


if __name__ == "__main__":
    main()

