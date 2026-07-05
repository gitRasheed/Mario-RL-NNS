# All Runs

| run_index | run | variants | timesteps | seed | verdict |
| --- | --- | --- | ---: | ---: | --- |
| 001 | rtx4090_1m_diagnostic | baseline / flat / nns_lpm | 1,000,000 | 0 | Promising seed-0 signal: NNS reached 594 mean progress vs baseline 314 and flat 249 at similar wall-clock/cost. Not evidence yet. |
| 002 | rtx4090_5m_pilot | baseline / flat / nns_lpm | 5,000,000 | 0 | Current NNS did not beat vanilla baseline at 5M; tune NNS before seed sweep. |
