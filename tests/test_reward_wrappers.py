import gymnasium as gym
import numpy as np
from gymnasium import spaces

from mario_rl_nns.reward_wrappers import (
    FlatPenaltyConfig,
    RewardShapingConfig,
    RewardShapingWrapper,
)


class TinyEnv(gym.Env):
    action_space = spaces.Discrete(2)
    observation_space = spaces.Box(0, 255, shape=(2, 2, 1), dtype=np.uint8)

    def __init__(self):
        self.step_count = 0

    def reset(self, **kwargs):
        self.step_count = 0
        return np.zeros((2, 2, 1), dtype=np.uint8), {"progress": 0}

    def step(self, action):
        self.step_count += 1
        return (
            np.zeros((2, 2, 1), dtype=np.uint8),
            1.0,
            False,
            False,
            {"progress": 0, "death": self.step_count == 2, "timeout": False},
        )


def test_flat_penalty_keeps_base_and_train_reward_separate() -> None:
    env = RewardShapingWrapper(
        TinyEnv(),
        RewardShapingConfig(
            variant="flat_penalty",
            flat=FlatPenaltyConfig(lambda_death=0.5, lambda_stuck=0.25, stuck_window=1),
        ),
    )
    env.reset()
    _, reward, *_rest, info = env.step(0)
    assert reward == 0.75
    assert info["shaping"]["reward_base"] == 1.0
    assert info["shaping"]["reward_train"] == 0.75
    assert info["shaping"]["stuck"] == 1.0

