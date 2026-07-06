# 006 External Checkpoint Audit

Goal: find a working Mario RL checkpoint that can be evaluated first, then used for baseline-vs-NNS continued training from the same starting policy.

| source | algorithm | framework | env version | action space | observation | checkpoint | can load | can evaluate | can continue train | license | compatibility risk | recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| https://github.com/yumouwei/super-mario-bros-reinforcement-learning | PPO | Stable-Baselines3 | Gym 0.21 / gym-super-mario-bros 7.3.0 reported | SIMPLE_MOVEMENT | RAM grid, 13x16xN stack | yes, `models/pre-trained-*.zip` | yes, with SB3 2.x custom object overrides | yes, `pre-trained-1.zip` reproduced | likely, after adding a clean RAM env adapter | GPL-3.0 | old Gym/SB3/Torch metadata; custom RAM wrapper; do not copy GPL code into our source | best candidate |
| https://github.com/andreaconti/supermario-dqn | DQN | custom PyTorch | unpinned current `gym-super-mario-bros` in requirements | custom simple action set | pixels preprocessed to 4x30x56 | yes, `trained/train_1_1/model.pt` | yes, Torch state dict loads | not yet tested | possible through custom DQN trainer, but not PPO/SB3 | unclear in clone | custom trainer and model format; less aligned with current harness | backup only |
| https://github.com/tooichitake/gymnasium-mario | PPO | Gymnasium + SB3 | modern Gymnasium fork | configurable | 84x84x4 pixels | no checkpoint found in shallow clone | n/a | n/a | yes if training from scratch | MIT | useful implementation reference, not a checkpoint source | reference for baseline-v2 |
| PyTorch Mario DDQN tutorial | DDQN | PyTorch tutorial | gym-super-mario-bros style | tutorial-defined | pixels | no usable checkpoint found | n/a | n/a | tutorial can be adapted | PyTorch docs terms | reference implementation only | reference only |

## Chosen Candidate

Use `yumouwei/super-mario-bros-reinforcement-learning`, checkpoint `models/pre-trained-1.zip`.

Why:
- It is an actual SB3 PPO checkpoint, not just a video.
- The README reports a World 1-1 clear and provides pretrained model files.
- The checkpoint loads under this repo's Python 3.13/SB3 2.x environment with explicit compatibility overrides:
  - `observation_space = Box(-1, 2, (13, 16, 4), float32)`
  - `action_space = Discrete(7)`
  - legacy schedule objects replaced by inert load-time callables

## Reproduction Result

Run: `artifacts/runs/006_external_checkpoint_eval_yumouwei_pretrained1`

| metric | value |
| --- | ---: |
| episodes | 50 |
| clear_rate | 0.58 |
| mean_progress | 3006.3 |
| p10_progress | 2386.0 |
| p50_progress | 3161.0 |
| p90_progress | 3161.0 |
| death_rate | 0.0 |
| eval_cap_rate | 0.42 |
| unique_trajectory_count | 5 |
| completion_time_mean | 1111.0 |
| wall_time_s | 220.0 |

## Fine-Tune Status

Not started.

Reason: the checkpoint is reproducible, but continued training needs a proper local RAM-observation adapter. Do not copy the external GPL wrapper into our MIT/public repo. Implement the minimal compatible adapter ourselves, then run:

```text
external_checkpoint_baseline_finetune_1M
external_checkpoint_nns_finetune_1M
```

Both must start from the same `pre-trained-1.zip` weights and use the same RAM observation/action wrapper.
