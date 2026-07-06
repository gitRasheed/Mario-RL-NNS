import gymnasium as gym
import numpy as np
from gymnasium import spaces

from mario_rl_nns.ram_adapter import RamObservationWrapper, ram_grid


class RamEnv(gym.Env):
    action_space = spaces.Discrete(2)
    observation_space = spaces.Box(0, 255, shape=(1,), dtype=np.uint8)

    def __init__(self):
        self.ram = np.zeros(2048, dtype=np.uint8)

    def reset(self, **kwargs):
        self.ram[0x3AD] = 40
        self.ram[0x3B8] = 79
        return np.zeros(1, dtype=np.uint8), {"progress": 40}

    def step(self, action):
        self.ram[0x86] += 1
        return np.zeros(1, dtype=np.uint8), 1.0, False, False, {"progress": int(self.ram[0x86])}


def test_ram_grid_marks_mario() -> None:
    env = RamEnv()
    env.reset()
    grid = ram_grid(env.ram)
    assert grid.shape == (13, 16)
    assert 2.0 in grid


def test_ram_observation_wrapper_shape_and_dtype() -> None:
    env = RamObservationWrapper(RamEnv())
    obs, _info = env.reset()
    assert obs.shape == (13, 16, 4)
    assert obs.dtype == np.float32
    obs, *_ = env.step(0)
    assert obs.shape == (13, 16, 4)
