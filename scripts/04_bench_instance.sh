#!/usr/bin/env bash
set -euo pipefail
has_output=0
for arg in "$@"; do
  if [[ "$arg" == "--output" || "$arg" == --output=* ]]; then
    has_output=1
  fi
done

output_args=()
if [[ "$has_output" == 0 ]]; then
  output_args=(--output "artifacts/benchmarks/${RUN_INDEX:-000}_throughput.json")
fi

uv run mario-benchmark-instance "$@" "${output_args[@]}"
