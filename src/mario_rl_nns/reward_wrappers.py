from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym

from mario_rl_nns.nns_rewards import NNSRewardConfig, NNSRewardState


@dataclass(slots=True)
class FlatPenaltyConfig:
    lambda_death: float = 0.5
    lambda_timeout: float = 0.1
    lambda_stuck: float = 0.1
    stuck_window: int = 32


@dataclass(slots=True)
class RewardShapingConfig:
    variant: str
    flat: FlatPenaltyConfig | None = None
    nns: NNSRewardConfig | None = None


@dataclass(slots=True)
class RewardShapingState:
    stuck_window: int
    progress: deque[float] = field(init=False)

    def __post_init__(self) -> None:
        if self.stuck_window < 1:
            raise ValueError("stuck_window must be >= 1")
        self.progress = deque(maxlen=self.stuck_window + 1)

    def reset(self, progress: float) -> None:
        self.progress.clear()
        self.progress.append(float(progress))

    def push(self, progress: float) -> tuple[float, float]:
        prev = self.progress[-1] if self.progress else float(progress)
        self.progress.append(float(progress))
        window_delta = self.progress[-1] - self.progress[0]
        stuck = len(self.progress) == self.progress.maxlen and window_delta <= 0.0
        return float(progress) - prev, float(stuck)


class RewardShapingWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, config: RewardShapingConfig):
        super().__init__(env)
        self.config = config
        self.episode_steps = 0
        if config.variant == "flat_penalty" and config.flat is not None:
            self.state = RewardShapingState(config.flat.stuck_window)
        elif config.nns is not None:
            self.state = RewardShapingState(config.nns.window)
            self.nns_state = NNSRewardState(config.nns)
        else:
            raise ValueError("reward shaping config must define flat or nns settings")

    def reset(self, **kwargs: Any):
        obs, info = self.env.reset(**kwargs)
        self.episode_steps = 0
        progress = _progress(info)
        self.state.reset(progress)
        if hasattr(self, "nns_state"):
            self.nns_state.reset(progress)
        return obs, info

    def step(self, action: Any):
        obs, reward_base, terminated, truncated, info = self.env.step(action)
        self.episode_steps += 1
        progress = _progress(info)
        progress_delta, stuck = self.state.push(progress)
        death = bool(info.get("death", info.get("is_dead", False)))
        timeout = bool(info.get("timeout", False))
        clear = bool(info.get("clear", info.get("flag_get", False)))

        if self.config.variant == "flat_penalty" and self.config.flat is not None:
            shaping = _flat_shape(self.config.flat, reward_base, death, timeout, stuck)
        elif self.config.nns is not None:
            shaping = self.nns_state.shape(
                reward_base,
                progress,
                death=death,
                timeout=timeout,
                clear=clear,
                episode_step=self.episode_steps,
            )
        else:
            raise RuntimeError("invalid reward shaping config")

        shaping.update(
            {
                "clear": float(clear),
                "variant": self.config.variant,
                "progress": progress,
                "progress_max": float(info.get("progress_max", info.get("x_pos_max", progress))),
                "x_pos": float(info.get("x_pos", progress)),
                "x_pos_max": float(info.get("x_pos_max", info.get("progress_max", progress))),
                "progress_delta": progress_delta,
                "stuck": stuck,
                "death": float(death),
                "timeout": float(timeout),
            }
        )
        info["shaping"] = shaping
        return obs, shaping["reward_train"], terminated, truncated, info


def reward_shaping_from_config(config: dict[str, Any]) -> RewardShapingConfig | None:
    variant = str(config.get("variant", "ppo_baseline"))
    if variant == "ppo_flat_penalty":
        return RewardShapingConfig(
            variant="flat_penalty",
            flat=FlatPenaltyConfig(
                lambda_death=float(config.get("flat_death_penalty", 0.5)),
                lambda_timeout=float(config.get("flat_timeout_penalty", 0.1)),
                lambda_stuck=float(config.get("flat_stuck_penalty", 0.1)),
                stuck_window=int(config.get("stuck_window", 32)),
            ),
        )
    nns = config.get("nns")
    if isinstance(nns, dict):
        nns = {**nns, "n_envs": int(config.get("n_envs", 1))}
        return RewardShapingConfig(
            variant=str(nns.get("variant", "nns_lpm")),
            nns=NNSRewardConfig(**nns),
        )
    return None


def _flat_shape(
    config: FlatPenaltyConfig,
    reward_base: float,
    death: bool,
    timeout: bool,
    stuck: float,
) -> dict[str, float]:
    extra = 0.0
    extra -= config.lambda_death * float(death)
    extra -= config.lambda_timeout * float(timeout)
    extra -= config.lambda_stuck * stuck
    return {
        "reward_base": float(reward_base),
        "reward_train": float(reward_base + extra),
        "flat_extra_reward": float(extra),
        "extra_reward": float(extra),
        "death": float(death),
        "timeout": float(timeout),
        "stuck": stuck,
    }


def _progress(info: dict[str, Any]) -> float:
    return float(info.get("progress", info.get("x_pos", 0.0)))
