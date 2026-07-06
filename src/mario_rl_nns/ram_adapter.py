from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gym_super_mario_bros  # noqa: F401
import gymnasium as gym
import numpy as np
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gymnasium import spaces
from nes_py.wrappers import JoypadSpace
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from mario_rl_nns.reward_wrappers import RewardShapingConfig, RewardShapingWrapper

RAM_OBSERVATION_SPACE = spaces.Box(low=-1.0, high=2.0, shape=(13, 16, 4), dtype=np.float32)


def ram_grid(ram: np.ndarray) -> np.ndarray:
    mario_level_x = int(ram[0x6D]) * 256 + int(ram[0x86])
    mario_x = int(ram[0x3AD])
    mario_y = int(ram[0x3B8]) + 16
    x_start = mario_level_x - mario_x
    screen = np.zeros((13, 16), dtype=np.float32)
    screen_start = int(np.rint(x_start / 16))

    for x in range(16):
        for y in range(13):
            tile_x = (screen_start + x) % 32
            page = tile_x // 16
            address = 0x500 + (tile_x % 16) + (page * 13 + y) * 16
            if int(ram[address]) != 0:
                screen[y, x] = 1.0

    mario_tile_x = (mario_x + 8) // 16
    mario_tile_y = (mario_y - 32) // 16
    if 0 <= mario_tile_x < 16 and 0 <= mario_tile_y < 13:
        screen[mario_tile_y, mario_tile_x] = 2.0

    for slot in range(5):
        if int(ram[0xF + slot]) != 1:
            continue
        enemy_x = int(ram[0x6E + slot]) * 256 + int(ram[0x87 + slot]) - x_start
        enemy_y = int(ram[0xCF + slot])
        enemy_tile_x = (enemy_x + 8) // 16
        enemy_tile_y = (enemy_y + 8 - 32) // 16
        if 0 <= enemy_tile_x < 16 and 0 <= enemy_tile_y < 13:
            screen[enemy_tile_y, enemy_tile_x] = -1.0

    return screen


class RamObservationWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, n_stack: int = 4, n_skip: int = 4):
        super().__init__(env)
        if n_stack != 4 or n_skip != 4:
            raise ValueError("the imported checkpoint expects n_stack=4 and n_skip=4")
        self.n_stack = n_stack
        self.n_skip = n_skip
        self.observation_space = RAM_OBSERVATION_SPACE
        self._frames = np.zeros((13, 16, (n_stack - 1) * n_skip + 1), dtype=np.float32)

    def reset(self, **kwargs: Any):
        _obs, info = self.env.reset(**kwargs)
        frame = ram_grid(self.unwrapped.ram)
        self._frames[:] = frame[:, :, None]
        return self._obs(), info

    def step(self, action: Any):
        _obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames[:, :, 1:] = self._frames[:, :, :-1]
        self._frames[:, :, 0] = ram_grid(self.unwrapped.ram)
        return self._obs(), reward, terminated, truncated, info

    def _obs(self) -> np.ndarray:
        return self._frames[:, :, :: self.n_skip].astype(np.float32, copy=False)


def make_ram_env(
    env_id: str = "SuperMarioBros-1-1-v0",
    seed: int = 0,
    rank: int = 0,
    reward_shaping: RewardShapingConfig | None = None,
) -> Callable[[], Monitor]:
    def thunk() -> Monitor:
        env = JoypadSpace(gym.make(env_id), SIMPLE_MOVEMENT)
        if reward_shaping is not None:
            env = RewardShapingWrapper(env, reward_shaping)
        env = RamObservationWrapper(env)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    return thunk


def make_ram_vec_env(
    env_id: str = "SuperMarioBros-1-1-v0",
    seed: int = 0,
    n_envs: int = 1,
    reward_shaping: RewardShapingConfig | None = None,
):
    env_fns = [
        make_ram_env(env_id=env_id, seed=seed, rank=rank, reward_shaping=reward_shaping)
        for rank in range(n_envs)
    ]
    return DummyVecEnv(env_fns) if n_envs == 1 else SubprocVecEnv(env_fns, start_method="fork")
