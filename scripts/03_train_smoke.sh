#!/usr/bin/env bash
set -euo pipefail
RUN_INDEX="${RUN_INDEX:-000}"
uv run mario-train-sb3 --total-timesteps 128 --n-envs 1 --device cpu --run-id "${RUN_INDEX}_local_smoke" "$@"
