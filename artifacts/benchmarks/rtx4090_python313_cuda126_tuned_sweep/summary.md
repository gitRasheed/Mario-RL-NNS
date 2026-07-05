# RTX 4090 Tuned PPO Throughput

Winner: `n_envs=64`, `n_steps=256`, `batch_size=256`.

```text
train steps/sec: 1447.4
env steps/sec: 3722.2
$/1M train steps: 0.132
avg GPU util: 7.4%
max GPU util: 24.0%
avg CPU util: 9.5% across 128 logical CPUs
max CPU util: 27.8%
```

The GPU remains underutilized because Mario PPO is still env/orchestration heavy, but the tuned setting is about 2.6x faster than the interrupted `n_envs=16` 1M run.
