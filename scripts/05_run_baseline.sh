#!/usr/bin/env bash
set -euo pipefail
uv run mario-train-sb3 --config configs/baseline_ppo.yaml "$@"

