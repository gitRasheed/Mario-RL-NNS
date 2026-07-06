# Partial-Moment Time-Tail Shaping Improves Early Fine-Tuning Reliability in Super Mario Bros

## Abstract

We study whether a small partial-moment-inspired time-tail reward shaping term improves continued reinforcement-learning training from a competent Super Mario Bros PPO checkpoint. The project began with from-scratch PPO/NNS experiments, but those results did not survive longer 5M-step checks. We therefore pivoted to a stronger test: start baseline PPO continuation and NNS/time-tail continuation from the same already-capable checkpoint, then compare actual task metrics rather than shaped reward.

In the final 10-seed, 250k-step continued-training comparison, time-tail increased mean clear rate from 0.758 to 0.958, reduced capped episodes from 0.242 to 0.042, reduced stuck fraction from 0.192 to 0.040, and improved all-episode penalized completion from 2044.48 to 1263.26. Conditional completion time among clears was not better: baseline and time-tail were effectively tied on the mean, with baseline slightly better on the median. We interpret the result as early fine-tuning reliability and sample-efficiency improvement, not faster clearing, speedrunning, or superior asymptotic performance.

## Research Question

Given a competent Mario PPO policy, does partial-moment-style time-tail reward shaping improve early continued-training reliability compared with ordinary PPO continuation from the same checkpoint?

## Main Claim

At 250k continued-training steps, the time-tail/NNS variant substantially improves reliability: higher clear rate, lower cap rate, lower stuck fraction, and better penalized completion. It does not improve conditional speed among successful clears.

## Evidence Summary

| metric | baseline | time-tail | interpretation |
| --- | ---: | ---: | --- |
| Clear rate | 0.758 | 0.958 | Strong reliability gain |
| Cap rate | 0.242 | 0.042 | Fewer capped/non-clear episodes |
| Penalized completion | 2044.48 | 1263.26 | Better all-episode outcome |
| Completion given clear | 1099.93 | 1099.96 | No speed gain |
| Stuck fraction | 0.192 | 0.040 | More stable rollouts |
| Mean progress | 3090.38 | 3058.60 | Similar, slight baseline mean edge |

Paired win counts over 10 seeds:

| criterion | wins / 10 |
| --- | ---: |
| clear >= baseline | 9 / 10 |
| cap <= baseline | 9 / 10 |
| penalized completion <= baseline | 5 / 10 |
| stuck <= baseline | 9 / 10 |
| progress >= baseline | 9 / 10 |

## Nuance

The reliability result is stronger than the completion-speed result. Mean penalized completion improves because time-tail avoids capped episodes, but the paired median penalized-completion delta is near zero and the bootstrap confidence interval crosses zero. The defensible claim is therefore reliability/stability, not universal per-seed completion improvement.

## Negative and Mixed Results

- From-scratch PPO/NNS reward shaping was not enough: early promise at 1M did not survive 5M checks.
- The original generic NNS LPM shaping and selected 5M tuned variants did not beat vanilla PPO.
- At 500k from the competent checkpoint, time-tail showed faster conditional completion but lower clear rate, so it was a speed/reliability tradeoff rather than a clean win.
- At 1M from the competent checkpoint, baseline mostly caught up; the effect saturated.

## Conclusion

Time-tail shaping is worth writing up as an early fine-tuning regularizer for an already-capable Mario PPO policy. The best next experiment is not another long 1M sweep, but either a compact write-up or a targeted retune that preserves the 250k reliability gain while improving conditional speed.
