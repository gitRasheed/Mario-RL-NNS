#!/usr/bin/env bash
set -euo pipefail
uv run mario-train-sb3 --total-timesteps 128 --n-envs 1 --device cpu --run-id local_smoke "$@"

