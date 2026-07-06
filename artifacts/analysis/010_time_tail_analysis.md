# 010 Time-Tail Analysis

Derived from `artifacts/runs/010_high_confidence_time_tail_summary.csv`.

Penalized completion definitions:
- `completion_time_given_clear`: mean completion time over cleared episodes only.
- `penalized_completion_time_cap`: clear episodes use completion time; non-clears use the eval cap `5000`.
- `penalized_completion_time_cap_plus_penalty`: non-clears use `5000 + 1000`.

## Aggregate Table

| step | variant | clear | cap | completion given clear | penalized cap | penalized cap+penalty | p50 | p90 | stuck | progress |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 250000 | baseline | 0.716 +/- 0.389 | 0.284 +/- 0.389 | 1097.096 +/- 15.823 | 2206.184 +/- 1516.993 | 2490.184 +/- 1905.933 | 1097.400 | 1107.400 | 0.227 +/- 0.311 | 3056.212 +/- 166.229 |
| 250000 | baseline_plus_time_tail | 1.000 +/- 0.000 | 0.000 +/- 0.000 | 1096.232 +/- 18.485 | 1096.232 +/- 18.485 | 1096.232 +/- 18.485 | 1098.000 | 1099.200 | 0.000 +/- 0.000 | 3161.000 +/- 0.000 |
| 500000 | baseline | 1.000 +/- 0.000 | 0.000 +/- 0.000 | 1096.320 +/- 15.921 | 1096.320 +/- 15.921 | 1096.320 +/- 15.921 | 1095.200 | 1104.000 | 0.000 +/- 0.000 | 3161.000 +/- 0.000 |
| 500000 | baseline_plus_time_tail | 0.916 +/- 0.188 | 0.084 +/- 0.188 | 1078.145 +/- 2.483 | 1407.208 +/- 738.289 | 1491.208 +/- 926.119 | 1077.200 | 1081.200 | 0.066 +/- 0.148 | 3147.056 +/- 31.180 |
| 1000000 | baseline | 0.944 +/- 0.125 | 0.056 +/- 0.125 | 1077.484 +/- 0.447 | 1297.172 +/- 490.967 | 1353.172 +/- 616.187 | 1077.600 | 1077.600 | 0.046 +/- 0.103 | 3116.816 +/- 98.798 |
| 1000000 | baseline_plus_time_tail | 1.000 +/- 0.000 | 0.000 +/- 0.000 | 1077.772 +/- 0.360 | 1077.772 +/- 0.360 | 1077.772 +/- 0.360 | 1077.800 | 1078.000 | 0.000 +/- 0.000 | 3161.000 +/- 0.000 |

## Paired Delta Means

| step | clear delta | cap delta | completion delta | penalized cap delta | penalized cap+penalty delta | p90 delta | stuck delta | progress delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 250000 | 0.284 | -0.284 | -0.864 | -1109.952 | -1393.952 | -8.200 | -0.227 | 104.788 |
| 500000 | -0.084 | 0.084 | -18.175 | 310.888 | 394.888 | -22.800 | 0.066 | -13.944 |
| 1000000 | 0.056 | -0.056 | 0.288 | -219.400 | -275.400 | 0.400 | -0.046 | 44.184 |

## Result Narrative

Main finding: at 250k, time-tail improves early fine-tuning stability: clear 1.00 vs 0.716, cap 0.00 vs 0.284, stuck 0.000 vs 0.227, and slightly faster conditional completion.

500k: time-tail has faster conditional completion, but lower clear rate than baseline. Penalized completion does not favor time-tail, so this is a speed/reliability tradeoff rather than a clean win.

1M: baseline mostly catches up. Time-tail has slightly better clear/cap/stuck, but completion is effectively tied and penalized completion is only marginally better.

Recommendation: run 10-seed confirmation at 250k only if we want a stronger early-stability claim. Treat 500k as mixed unless retuned; do not spend more on 1M repeats yet because it saturates.
