# Mario RL NNS

Research harness for testing whether NNS / partial-moment reward shaping improves bad-tail outcomes in Super Mario Bros RL.

The first rule for this repo: keep training reward, base environment reward, and task metrics separate. Shaped reward is not the result.

## Setup

Current `gym-super-mario-bros` requires Python 3.13+, so use UV with 3.13:

```bash
uv sync --python 3.13
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
scripts/train_smoke.sh
```

## Early Plan

1. Environment smoke test: import, wrap, reset, step, inspect `info`.
2. Preprocessing: grayscale, 84x84 resize, 4-frame stack.
3. PPO baseline: `RIGHT_ONLY`, `SuperMarioBros-1-1-v0`, short local run.
4. Flat penalty wrapper: death/stuck penalty.
5. NNS LPM wrapper: bounded extra reward, logged separately from base reward.

RunPod benchmarking and multi-seed sweeps come after the local smoke and first PPO baseline work.
