# Methods

## Environment and Policy

- Environment: `SuperMarioBros-1-1-v0`
- Action space: `SIMPLE_MOVEMENT`
- Observation adapter: local clean RAM adapter matching the imported checkpoint shape
- Starting policy: external PPO checkpoint `pre-trained-1.zip` from `yumouwei/super-mario-bros-reinforcement-learning`
- Training framework: Stable-Baselines3 PPO
- Python: 3.13
- Hardware lane: RTX 4090 RunPod

The external checkpoint is used as a model artifact/reference. The local RAM adapter is our own implementation; GPL wrapper source from the external repository is not vendored into this repo.

## Experimental Design

The final confirmation experiment compares two continued-training variants from the same checkpoint:

- `baseline`: ordinary PPO continuation with the original environment reward.
- `baseline_plus_time_tail`: PPO continuation with a small terminal slow-clear time-tail shaping term.

Final confirmation:

- Budget: 250,000 continued-training steps.
- Seeds: 0-9.
- Seeds 0-4 reused compatible 250k checkpoint evaluations from run 010.
- Seeds 5-9 were added in run 011.
- Evaluation: deterministic policy, 50 episodes per seed/variant.
- Non-clear episodes are counted explicitly through cap and penalized-completion metrics.

## Metrics

Primary task metrics:

- `clear_rate`: fraction of evaluation episodes that cleared the level.
- `eval_cap_rate`: fraction of episodes reaching the evaluation cap.
- `stuck_fraction`: fraction of evaluation steps classified as stuck.
- `mean_progress`: average maximum progress.
- `completion_time_given_clear`: mean completion time over cleared episodes only.

Penalized completion metrics:

- `penalized_completion_cap`: clear episodes use completion time; non-clears use the 5000-step evaluation cap.
- `penalized_completion_cap_plus_penalty`: clear episodes use completion time; non-clears use 6000 steps.

The penalized metrics prevent conditional completion time from hiding failed/capped episodes.

## Statistical Summaries

The write-up reports:

- Mean and median metrics by variant.
- Paired seed deltas.
- Win counts over 10 paired seeds.
- Bootstrap 95% confidence intervals for paired mean deltas.

The bootstrap intervals are descriptive, not a claim of final statistical proof. The sample size is small and several effects are heavy-tailed.

## Reproducibility Pointers

Run history:

- `docs/ALL_RUNS.md`

Final run artifacts:

- `artifacts/runs/011_10seed_250k_time_tail_summary.csv`
- `artifacts/runs/011_10seed_250k_time_tail_paired_deltas.csv`
- `012_writeup/results_table.csv`
- `012_writeup/paired_delta_stats.csv`

Core scripts:

- `scripts/10_run_high_confidence_time_tail.sh`
- `scripts/11_run_10seed_250k_time_tail.sh`
