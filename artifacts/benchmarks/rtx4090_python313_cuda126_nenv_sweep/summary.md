# RTX 4090 Python 3.13 CUDA 12.6 Throughput

Winner: `n_envs=16`.

```text
GPU: NVIDIA GeForce RTX 4090
Python: 3.13.14
Torch: 2.12.1+cu126
Hourly price: $0.69/hr
Env: SuperMarioBros-1-1-v0
Action space: RIGHT_ONLY
Policy: Stable-Baselines3 PPO CnnPolicy
```

Best quick-benchmark row:

```text
n_envs: 16
env steps/sec: 1814.2
train steps/sec: 446.2
$/1M env steps: 0.106
$/1M train steps: 0.430
```

Use train steps/sec for PPO ETA.

Observed 50k baseline sanity run:

```text
run_id: rtx4090_baseline_50k_n16
requested timesteps: 50000
actual PPO timesteps: 51200
wall time: 104.2s
final SB3 fps: about 504
```

