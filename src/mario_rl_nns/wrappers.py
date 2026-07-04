from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence

from gymnasium.wrappers import GrayscaleObservation, ResizeObservation
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecFrameStack,
    VecTransposeImage,
)

from mario_rl_nns.make_env import ACTION_SPACES, make_mario_env


def make_preprocessed_env(
    env_id: str = "SuperMarioBros-1-1-v0",
    action_space: str = "RIGHT_ONLY",
    seed: int = 0,
    rank: int = 0,
) -> Callable[[], Monitor]:
    def thunk() -> Monitor:
        env = make_mario_env(env_id=env_id, action_space=action_space)
        env = ResizeObservation(env, (84, 84))
        env = GrayscaleObservation(env, keep_dim=True)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    return thunk


def make_vec_env(
    env_id: str = "SuperMarioBros-1-1-v0",
    action_space: str = "RIGHT_ONLY",
    seed: int = 0,
    n_envs: int = 1,
):
    env_fns = [
        make_preprocessed_env(env_id=env_id, action_space=action_space, seed=seed, rank=rank)
        for rank in range(n_envs)
    ]
    vec_env = DummyVecEnv(env_fns) if n_envs == 1 else SubprocVecEnv(env_fns, start_method="fork")
    vec_env = VecFrameStack(vec_env, n_stack=4, channels_order="last")
    return VecTransposeImage(vec_env)


def smoke_train_env(
    env_id: str = "SuperMarioBros-1-1-v0",
    action_space: str = "RIGHT_ONLY",
    seed: int = 0,
    n_envs: int = 1,
    steps: int = 10,
) -> dict[str, object]:
    env = make_vec_env(env_id=env_id, action_space=action_space, seed=seed, n_envs=n_envs)
    try:
        obs = env.reset()
        rewards = None
        dones = None
        infos = None
        for _ in range(steps):
            actions = [env.action_space.sample() for _ in range(n_envs)]
            obs, rewards, dones, infos = env.step(actions)
        return {
            "env_id": env_id,
            "action_space": action_space,
            "n_envs": n_envs,
            "obs_shape": obs.shape,
            "reward_sample": rewards.tolist() if rewards is not None else None,
            "done_sample": dones.tolist() if dones is not None else None,
            "info_keys": sorted(infos[0]) if infos else [],
        }
    finally:
        env.close()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="SuperMarioBros-1-1-v0")
    parser.add_argument("--action-space", default="RIGHT_ONLY", choices=ACTION_SPACES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    for key, value in smoke_train_env(
        env_id=args.env_id,
        action_space=args.action_space,
        seed=args.seed,
        n_envs=args.n_envs,
        steps=args.steps,
    ).items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
