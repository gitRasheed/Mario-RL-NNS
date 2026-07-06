# Mario RL NNS

Research harness for testing whether NNS / partial-moment reward shaping improves bad-tail outcomes in Super Mario Bros RL.

The first rule for this repo: keep training reward, base environment reward, and task metrics separate. Shaped reward is not the result.

## Current Finding

The strongest result is an early fine-tuning reliability result from a competent imported Mario PPO checkpoint, not a from-scratch PPO win.

At 250k continued-training steps over 10 paired seeds, `baseline_plus_time_tail` improved reliability relative to normal PPO continuation:

| metric | baseline | time-tail |
| --- | ---: | ---: |
| Clear rate | 0.758 | 0.958 |
| Cap rate | 0.242 | 0.042 |
| Penalized completion | 2044.48 | 1263.26 |
| Completion given clear | 1099.93 | 1099.96 |
| Stuck fraction | 0.192 | 0.040 |
| Mean progress | 3090.38 | 3058.60 |

Interpretation: time-tail/NNS shaping improves early reliability and reduces capped/stuck failures. It does not make successful clears faster, and it should not be described as a better asymptotic Mario policy.

Research-note materials:

- `012_writeup/paper_outline.md`
- `012_writeup/methods.md`
- `012_writeup/results_table.md`
- `docs/ALL_RUNS.md`

Final confirmation artifacts:

- `artifacts/runs/011_10seed_250k_time_tail_summary.csv`
- `artifacts/runs/011_10seed_250k_time_tail_paired_deltas.csv`
- `012_writeup/paired_delta_stats.csv`

## Setup

Current `gym-super-mario-bros` requires Python 3.13+, so use UV with 3.13:

```bash
uv sync --python 3.13
```

On a fresh RunPod:

```bash
git clone https://github.com/gitRasheed/Mario-RL-NNS.git
cd Mario-RL-NNS
uv sync
scripts/01_smoke_env.sh --steps 100
scripts/03_train_smoke.sh
```

Quick throughput benchmark:

```bash
scripts/04_bench_instance.sh --n-envs 4 --device cuda --train-timesteps 2048
```

Serious sweep artifacts are prefixed with `RUN_INDEX` so runs sort chronologically:

```bash
RUN_INDEX=001 HOURLY_PRICE=0.69 scripts/06_run_1m_diagnostic_sweep.sh
```

Next pilot after a promising 1M signal:

```bash
RUN_INDEX=002 HOURLY_PRICE=0.69 scripts/07_run_5m_pilot.sh
```

Small NNS tuning sweep:

```bash
RUN_INDEX=003 HOURLY_PRICE=0.69 scripts/08_run_2m_nns_tuning.sh
```

Scale selected NNS tuning winners:

```bash
RUN_INDEX=004 HOURLY_PRICE=0.69 scripts/09_run_5m_selected_nns.sh
```

Final checkpoint-continuation sweeps:

```bash
scripts/10_run_high_confidence_time_tail.sh
scripts/11_run_10seed_250k_time_tail.sh
```

## First Gates

Smoke-test the Mario environment:

```bash
uv run mario-smoke-env --steps 1000
```

Check the partial-moment reward math:

```bash
uv run mario-check-nns
```

Smoke-test the SB3/CNN preprocessing path:

```bash
uv run mario-smoke-train-env --steps 10
```

Run a tiny PPO baseline sanity check:

```bash
scripts/03_train_smoke.sh
```

## Project Arc

1. Built a reproducible Mario RL harness with separate base reward, shaped reward, and task metrics.
2. Tested from-scratch PPO plus generic NNS/progress shaping. Early signals did not survive 5M checks.
3. Pivoted to a stronger test: continue training from a competent external PPO checkpoint using a clean local RAM adapter.
4. Found that generic conservative NNS did not beat normal checkpoint continuation at 1M.
5. Tested a smaller time-tail term focused on slow-clear tails.
6. Confirmed the useful effect at 250k: early reliability/stability improves, while conditional speed does not.

The current write-up claim is intentionally narrow: partial-moment time-tail shaping improves early fine-tuning reliability from an already-capable Mario policy.
