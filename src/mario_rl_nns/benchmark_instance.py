from __future__ import annotations

import argparse
import json
import platform
import threading
import time
from collections.abc import Sequence
from pathlib import Path

import psutil
import torch
from stable_baselines3 import PPO

from mario_rl_nns.wrappers import make_vec_env


class UtilizationSampler:
    def __init__(self, interval_s: float = 0.5):
        self.interval_s = interval_s
        self.rows: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        psutil.cpu_percent(interval=None)
        self._thread.start()
        return self

    def __exit__(self, *args):
        self._stop.set()
        self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            gpu = gpu_info()
            self.rows.append(
                {
                    "cpu_percent": psutil.cpu_percent(interval=None),
                    "gpu_util_percent": float(gpu.get("gpu_util_percent", 0.0)),
                    "gpu_memory_util_percent": float(gpu.get("gpu_memory_util_percent", 0.0)),
                    "vram_used_mb": float(gpu.get("vram_used_mb", 0.0)),
                }
            )

    def summary(self) -> dict[str, float]:
        if not self.rows:
            return {}
        out = {}
        for key in self.rows[0]:
            values = [row[key] for row in self.rows]
            out[f"avg_{key}"] = sum(values) / len(values)
            out[f"max_{key}"] = max(values)
        return out


def gpu_info() -> dict[str, object]:
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        name = pynvml.nvmlDeviceGetName(handle)
        return {
            "gpu_name": name.decode() if isinstance(name, bytes) else name,
            "vram_total_mb": memory.total / 1024**2,
            "vram_used_mb": memory.used / 1024**2,
            "gpu_util_percent": util.gpu,
            "gpu_memory_util_percent": util.memory,
        }
    except Exception as exc:
        return {"gpu_error": str(exc)}


def system_info() -> dict[str, object]:
    ram = psutil.virtual_memory()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "ram_total_mb": ram.total / 1024**2,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        **gpu_info(),
    }


def env_steps_per_sec(
    env_id: str,
    action_space: str,
    seed: int,
    n_envs: int,
    steps: int,
) -> float:
    env = make_vec_env(env_id=env_id, action_space=action_space, seed=seed, n_envs=n_envs)
    try:
        env.reset()
        started = time.perf_counter()
        for _ in range(max(steps // n_envs, 1)):
            env.step([env.action_space.sample() for _ in range(n_envs)])
        elapsed = time.perf_counter() - started
        return steps / elapsed
    finally:
        env.close()


def train_steps_per_sec(
    env_id: str,
    action_space: str,
    seed: int,
    n_envs: int,
    device: str,
    timesteps: int,
    n_steps: int,
    batch_size: int,
) -> float:
    env = make_vec_env(env_id=env_id, action_space=action_space, seed=seed, n_envs=n_envs)
    try:
        model = PPO(
            "CnnPolicy",
            env,
            seed=seed,
            device=device,
            verbose=0,
            n_steps=n_steps,
            batch_size=batch_size,
        )
        started = time.perf_counter()
        model.learn(total_timesteps=timesteps)
        elapsed = time.perf_counter() - started
        return timesteps / elapsed
    finally:
        env.close()


def benchmark(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    with UtilizationSampler() as sampler:
        env_sps = env_steps_per_sec(
            args.env_id,
            args.action_space,
            args.seed,
            args.n_envs,
            args.env_steps,
        )
        train_sps = train_steps_per_sec(
            args.env_id,
            args.action_space,
            args.seed,
            args.n_envs,
            args.device,
            args.train_timesteps,
            args.n_steps,
            args.batch_size,
        )
    hourly_price = args.hourly_price
    cost_per_1m_train = (
        hourly_price / (train_sps * 3600 / 1_000_000) if hourly_price is not None else None
    )
    cost_per_1m_env = (
        hourly_price / (env_sps * 3600 / 1_000_000) if hourly_price is not None else None
    )
    return {
        **system_info(),
        "env_id": args.env_id,
        "action_space": args.action_space,
        "seed": args.seed,
        "n_envs": args.n_envs,
        "device": args.device,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "env_benchmark_steps": args.env_steps,
        "train_timesteps": args.train_timesteps,
        "env_steps_per_sec": env_sps,
        "train_steps_per_sec": train_sps,
        "hourly_price": hourly_price,
        "cost_per_1m_env_steps": cost_per_1m_env,
        "cost_per_1m_train_steps": cost_per_1m_train,
        "cpu_percent": psutil.cpu_percent(interval=1.0),
        "ram_used_mb": psutil.virtual_memory().used / 1024**2,
        "wall_time_s": time.perf_counter() - started,
        **sampler.summary(),
        **{f"end_{key}": value for key, value in gpu_info().items()},
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="SuperMarioBros-1-1-v0")
    parser.add_argument("--action-space", default="RIGHT_ONLY")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--env-steps", type=int, default=2048)
    parser.add_argument("--train-timesteps", type=int, default=2048)
    parser.add_argument("--hourly-price", type=float)
    parser.add_argument("--output", type=Path, default=Path("artifacts/runs/throughput.json"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
