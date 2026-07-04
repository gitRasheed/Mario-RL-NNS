from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


def lpm(value: float, target: float = 0.0, degree: float = 1.0) -> float:
    if degree < 0:
        raise ValueError("degree must be non-negative")
    return max(target - value, 0.0) ** degree


def upm(value: float, target: float = 0.0, degree: float = 1.0) -> float:
    if degree < 0:
        raise ValueError("degree must be non-negative")
    return max(value - target, 0.0) ** degree


@dataclass(slots=True)
class NNSRewardConfig:
    variant: str = "nns_lpm"
    window: int = 32
    target_speed: float = 0.5
    d_down: float = 2.0
    d_up: float = 1.0
    lambda_down: float = 0.1
    lambda_up: float = 0.02
    lambda_stuck: float = 0.1
    lambda_death: float = 0.5
    clip_min: float = -5.0
    clip_max: float = 5.0


@dataclass(slots=True)
class NNSRewardState:
    config: NNSRewardConfig
    progress: deque[float] = field(init=False)

    def __post_init__(self) -> None:
        if self.config.window < 1:
            raise ValueError("window must be >= 1")
        self.progress = deque(maxlen=self.config.window + 1)

    def reset(self, progress: float = 0.0) -> None:
        self.progress.clear()
        self.progress.append(float(progress))

    def shape(self, reward_base: float, progress: float, death: bool = False) -> dict[str, float]:
        if not self.progress:
            self.reset(progress)
        self.progress.append(float(progress))
        progress_window = self.progress[-1] - self.progress[0]
        actual_speed = progress_window / max(len(self.progress) - 1, 1)
        down = lpm(actual_speed, self.config.target_speed, self.config.d_down)
        up = upm(actual_speed, self.config.target_speed, self.config.d_up)
        stuck = float(len(self.progress) == self.progress.maxlen and progress_window <= 0.0)

        extra = -self.config.lambda_down * down
        if self.config.variant == "nns_upm_lpm":
            extra += self.config.lambda_up * up
        elif self.config.variant != "nns_lpm":
            raise ValueError("variant must be 'nns_lpm' or 'nns_upm_lpm'")

        extra -= self.config.lambda_stuck * stuck
        extra -= self.config.lambda_death * float(death)
        extra = min(max(extra, self.config.clip_min), self.config.clip_max)

        return {
            "reward_base": float(reward_base),
            "reward_train": float(reward_base + extra),
            "nns_extra_reward": float(extra),
            "lpm": float(down),
            "upm": float(up),
            "target_speed": float(self.config.target_speed),
            "actual_speed": float(actual_speed),
            "speed_margin": float(actual_speed - self.config.target_speed),
            "stuck": stuck,
            "death": float(death),
        }


def main() -> None:
    assert lpm(0.25, target=0.5, degree=2.0) == 0.0625
    assert upm(0.75, target=0.5, degree=1.0) == 0.25
    state = NNSRewardState(NNSRewardConfig(window=2, target_speed=1.0))
    state.reset(0.0)
    first = state.shape(1.0, progress=0.0)
    second = state.shape(1.0, progress=0.0)
    assert first["reward_train"] < first["reward_base"]
    assert second["stuck"] == 1.0
    print("nns reward checks passed")


if __name__ == "__main__":
    main()

