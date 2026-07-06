# Results Tables

Primary 10-seed 250k result. Non-clears are assigned the 5000-step eval cap for penalized completion; cap+penalty uses 6000.

| metric | baseline mean | time-tail mean | baseline median | time-tail median | interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| Clear rate | 0.758 | 0.958 | 1.000 | 1.000 | higher is better |
| Cap rate | 0.242 | 0.042 | 0.000 | 0.000 | lower is better |
| Penalized completion | 2044.478 | 1263.258 | 1110.960 | 1104.020 | lower is better |
| Penalized completion cap+penalty | 2286.478 | 1305.258 | 1110.960 | 1104.020 | lower is better |
| Completion time given clear | 1099.927 | 1099.962 | 1100.100 | 1104.020 | lower is better; conditional metric |
| Completion p90 given clear | 1109.333 | 1103.000 | 1111.000 | 1112.000 | lower is better; conditional metric |
| Stuck fraction | 0.192 | 0.040 | 0.000 | 0.000 | lower is better |
| Mean progress | 3090.378 | 3058.604 | 3161.000 | 3161.000 | higher is better |

## Paired Delta Statistics

Positive clear/progress deltas are good; negative cap/completion/stuck deltas are good.

| paired delta | mean | median | bootstrap 95% CI |
| --- | ---: | ---: | ---: |
| clear_delta | 0.200 | 0.000 | [-0.042, 0.480] |
| cap_delta | -0.200 | 0.000 | [-0.480, 0.054] |
| penalized_completion_delta | -781.220 | -1.640 | [-1878.866, 199.568] |
| penalized_completion_cap_plus_penalty_delta | -981.220 | -1.640 | [-2359.186, 253.568] |
| stuck_delta | -0.153 | 0.000 | [-0.379, 0.062] |
| progress_delta | -31.774 | 0.000 | [-278.532, 142.854] |

## Win Counts

| criterion | wins / 10 |
| --- | ---: |
| clear >= baseline | 9 / 10 |
| cap <= baseline | 9 / 10 |
| penalized completion <= baseline | 5 / 10 |
| stuck <= baseline | 9 / 10 |
| progress >= baseline | 9 / 10 |

## Figures

- `figures/011_scatter_clear_rate.png`
- `figures/011_scatter_cap_rate.png`
- `figures/011_scatter_stuck_fraction.png`
