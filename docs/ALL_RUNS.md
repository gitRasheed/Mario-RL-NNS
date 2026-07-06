# All Runs

| run_index | run | variants | timesteps | seed | verdict |
| --- | --- | --- | ---: | ---: | --- |
| 001 | rtx4090_1m_diagnostic | baseline / flat / nns_lpm | 1,000,000 | 0 | Promising seed-0 signal: NNS reached 594 mean progress vs baseline 314 and flat 249 at similar wall-clock/cost. Not evidence yet. |
| 002 | rtx4090_5m_pilot | baseline / flat / nns_lpm | 5,000,000 | 0 | Current NNS did not beat vanilla baseline at 5M; tune NNS before seed sweep. |
| 003 | rtx4090_2m_nns_tuning | six NNS tuning variants | 2,000,000 | 0 | Winners: clip_small for upside, no_stuck for robust tail. Scale only those to 5M. |
| 004 | rtx4090_5m_selected_nns | clip_small / no_stuck | 5,000,000 | 0 | Tuned NNS configs did not beat vanilla baseline at 5M; do not scale these configs to 3 seeds. |
| 006 | external_checkpoint_eval_yumouwei_pretrained1 | external PPO checkpoint eval | n/a | eval only | Reproduced a clearing external SB3 PPO checkpoint: 50-episode clear rate 0.58, mean progress 3006.3. Use this as the next fine-tune starting point after adding a clean RAM adapter. |
| 007 | clean_adapter_checkpoint_eval | external PPO checkpoint eval | n/a | eval only | Clean local RAM adapter reproduced the external checkpoint: clear rate 0.58, mean progress 3006.3, no deaths. |
| 008 | speed_tail_nns_sweep | baseline / speed-tail NNS presets | 250,000 | 0 | `baseline_plus_time_tail` preserved 100% clear and reached 1077-step completion mean at 250k; extend across 500k/1M before seeds. |
