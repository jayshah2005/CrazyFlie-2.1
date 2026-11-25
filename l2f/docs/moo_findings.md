# MOO on the drone controller: what we found (2026-11-23, updated 2026-11-25)

This document summarises the reward and multi-objective (MOO) studies on branch `feat/moo`: why
the gradient aggregators first appeared not to help, how that was traced to the shared MOO critic,
and how **separate critics per group let SVD-PGA beat the sum and match variant E** (sections
12-13). It is
written to be read on its own. The detailed record of every study, with all tables and settings, is
[reward_studies.md](reward_studies.md); the earlier time-to-target study that started this work is
[time_to_target_study.md](time_to_target_study.md).

## Contents

1. [Summary](#1-summary)
2. [Background](#2-background)
3. [How the studies were run](#3-how-the-studies-were-run)
4. [Finding 1: the reward paid for crashing](#4-finding-1-the-reward-paid-for-crashing)
5. [Finding 2: what fixed it, and what did not](#5-finding-2-what-fixed-it-and-what-did-not)
6. [Finding 3: waypoint training](#6-finding-3-waypoint-training)
7. [Finding 4: the measured conflict](#7-finding-4-the-measured-conflict)
8. [Finding 5: no aggregator beats plain summing](#8-finding-5-no-aggregator-beats-plain-summing)
9. [Why the aggregators do not help](#9-why-the-aggregators-do-not-help)
10. [Finding 6: MOO sum trailed TD3, fixed by head scaling](#10-finding-6-moo-sum-trailed-td3-fixed-by-head-scaling)
11. [Finding 7: untuned, the aggregators still do not help](#11-finding-7-untuned-the-aggregators-still-do-not-help)
12. [Finding 8: the shared critic collapsed the groups; separate critics fix it](#12-finding-8-the-shared-critic-collapsed-the-groups-separate-critics-fix-it)
13. [Finding 9: with separate critics, the aggregators beat sum and SVD-PGA matches variant E](#13-finding-9-with-separate-critics-the-aggregators-beat-sum-and-svd-pga-matches-variant-e)
14. [Caveats](#14-caveats)
15. [What this means for the paper](#15-what-this-means-for-the-paper)
16. [Open questions and next steps](#16-open-questions-and-next-steps)
17. [Lessons about the process](#17-lessons-about-the-process)
18. [Reproducing the results](#18-reproducing-the-results)
19. [Glossary](#19-glossary)

---

## 1. Summary

- **The reward paid for crashing.** With the crash penalty of 200 and no survival bonus, recovering
  from a bad state can cost more than crashing. A reward check found this in 8-22% of free-flight
  states for typical policies, and in 69-84% of waypoint states. It explains the collapsed training
  seeds and the failure of the earlier time-to-target study.
- **A crash penalty of 400 is the new base.** It halves the crash incentive and doubles the number
  of crash-free checkpoints, for about 1.2 cm of circle precision. The progress reward, which looked
  ideal on paper, failed in training.
- **Waypoint training is possible but not yet useful.** With crash 400, a Huber position cap, short
  jumps (0.3-0.5 m) and a time cost of 0.5, policies learn waypoints without collapsing. But
  policies trained only on the mission fly waypoints about as well.
- **Update 2026-11-25, the main result: with one critic per group, PGA beats the sum and matches
  variant E (sections 12-13).** The baseline is the original variant E (TD3, E's tuned reward).
  - **The shared 4-output critic hid the conflict.** Measured directly, the head-scaled critic's 4
    group values were copies of one value (correlation 0.99-1.00); unscaled, the values were
    distinct but the summed value was unstable. Neither was usable for MOO.
  - **Separate critics per group (`--split-critic`)** learn distinct, well-fitted group values (R²
    0.98-1.00) and show real conflict: precision against efficiency in 82% of updates, precision
    against safety 76%, time against precision 68%.
  - **On that critic, 8 seeds:** every reweighting aggregator crashes less than the sum (1.7-2.7%
    against 4.5%); SVD-PGA crashes least, and against variant E
    tracks the circle better (4.9 against 6.0 cm), reaches more waypoints (83.5 / 76 against 79.5 /
    69% within 10 cm), and is level on crashes (1.7 against 1.4%) and uneven motors (10.4 against
    9.7 cm). It gets there by giving smoothing (efficiency) only 6% of the step.
  - **Not yet shown:** that aggregators replace reward tuning (each group still holds E's tuned
    weights); the untuned test is next.
- **Earlier results, measured with the shared critic** (kept as the record; see the corrections in
  sections 7, 11 and 12):
  - No aggregator beat plain summing (6 aggregators, 4-8 seeds, two start times).
  - MOO sum trailed TD3: the 4 output heads made the summed value move 2.3 times as far per update.
    Head scaling (`--head-scale`) fixed the total but, as found later, collapsed the groups.
  - "Most of the measured conflict was an artifact of the critic" was wrong: head scaling hid it.
  - Untuned (equal group shares, crash 600), untuned TD3 was much worse than tuned (10.1 against
    4.9 cm, 6.0 against 0.8% crashes) and no aggregator helped; the gradients looked parallel
    (+0.98) because of the collapsed critic. To be repeated with separate critics.

## 2. Background

### The controller

The policy (variant E) maps 25 inputs (position error to the target, orientation, velocity, angular
velocity, the 4 measured rotor speeds, a leaky running sum of the position error) to 4 rotor-speed
commands, 100 times a second. It is trained with TD3 in simulation (`l2f/py/train.py`, the PyTorch
trainer on the C++ simulator) and flown on a Crazyflie 2.1 Brushless.

### The reward

Every step costs a weighted sum of: position error, tilt, velocity error, action size, action rate
(change of the command), angular acceleration, and landing speed. A crash ends the episode with a
fixed penalty. There is no reward for staying alive (no survival bonus), so every reward is a cost.
Weights rise during training (the weight curriculum): position 5 to 40, velocity and action 0.01 to 1.

### Training episodes

Each training episode is one of: free flight from a random state (about 30%), take-off, landing,
circle, line, and, when enabled, waypoints (the target jumps, then moves again).

### Multi-objective TD3 (framing A)

The reward parts are grouped into objectives. With 4 groups:

| Group | Reward parts |
| --- | --- |
| time | 1 per step until the drone has arrived (within 5 cm and below 0.2 m/s) |
| precision | position, velocity |
| efficiency | action, action rate, angular acceleration |
| safety | the crash penalty, tilt, landing speed |

The critic learns one value per group. The actor gets one gradient per group; an **aggregator**
(from TorchJD) combines the 4 gradients into one update. With **sum**, the update is the gradient of
the total value, which is what plain TD3 does.

### Why this work started

The earlier time-to-target study trained waypoint policies with many changes at once (waypoints,
four new reward terms, five simulator options, seven aggregators, 3M steps). Every method collapsed
into crashing. That study found the cause (section 4) but could not separate the effects of its
changes. This work went back to a known-good recipe and changed one thing at a time.

## 3. How the studies were run

### One change at a time

Every study has a reference arm and arms that each add one change. Every arm is tested under the
same conditions, so the effect of training with a change is not mixed up with testing under it.

### A reward check before training

`l2f/py/reward_check.py` flies a trained policy through the training episode mix with the **final**
reward weights and, for every state, compares the discounted return of flying on with the crash
penalty. It reports the share of states where crashing would pay more ("crash-better"), per episode
type. It takes seconds. New reward options were only trained if this check passed.

### Tests

- **Mission test:** the real flight profile in simulation (take off, hover, circle of 0.6 m in 10 s,
  land), 100 episodes per checkpoint, clean and with uneven motors, at 6 and 12 ms estimate delay.
  Reported: circle tracking error, crashes, jitter (motor-command spread in hover, lower is
  smoother), and how many checkpoints never crashed.
- **Waypoint test:** the target jumps 0.3-0.5 m (or up to 0.8 m), then moves on; 3.5 s per leg.
  Reported: crashes, arrival within 5 cm and within 10 cm (both below 0.2 m/s), time to arrive,
  final error.

Results are medians over seeds and over the late checkpoints (1.2M-2.0M steps), with per-seed values
checked for collapse.

### The base recipe

Variant E, old (truth) observation path, PyTorch TD3, 2M steps; old3's options (the best flown
policy's recipe) plus the simulator options below. All arms use seeds 70-73, and 74-77 where a result
needed confirming. Training ran on a MacBook (about 10-17 minutes per run, 8-12 at a time).

## 4. Finding 1: the reward paid for crashing

TD3 maximises the discounted return, about 1 s ahead (gamma 0.99). With every reward a cost and a
crash ending the episode at -200, crashing is the better choice from any state where the costs still
ahead add up to more than 200.

| Policy and settings | Free flight | Take-off | Landing | Circle | Waypoint |
| --- | --- | --- | --- | --- | --- |
| old3 (best flown policy) | 8.6% of states | 2.5% | 0.7% | 3.6% | - |
| a healthy TD3 seed | 22.4% | 2.4% | 0.6% | 4.1% | - |
| old3, the time-to-target study's waypoint settings | 18.2% | 2.9% | 1.2% | 5.2% | 83.6% |

- **Free flight** starts from random, often tilted states away from the target. Recovering costs
  about as much as the crash penalty (old3's survivors: median -182 against -200).
- **Waypoints:** a perfectly flown leg costs -612 (1 m) to -9200 (3 m) with the squared position cost,
  so crashing is cheaper than flying.
- **Consequence:** every training run is pushed toward crashing. Most seeds resist; one seed in eight
  (base seed 71) did not: its training return *improved* while its episodes shrank to 0.2-0.3 s.

## 5. Finding 2: what fixed it, and what did not

| Change | Reward check (free flight, old3 / a TD3 seed) | Training result |
| --- | --- | --- |
| none | 8.6% / 22.4% | reference |
| simulator options (drag, motor lag range, gain wander) | - | no precision cost, slightly more robust: **kept** |
| crash penalty 400 | 0.8% / 5.8% | crash-free checkpoints 11/36 to 22/36, circle +1.2 cm: **new base** |
| progress reward (20 per metre closed, flat cost far away) | 1.2% / 8.0% | **failed**: 10-20 cm steady offset, 46% crashes |
| progress + crash 400 | 0.1% / 1.2% | **failed**: 14 cm, 26% crashes |
| Huber position cap 0.2 m | 4.8% / 16.6% | slightly better circle, less reliable |

The progress reward is the textbook fix (potential-based shaping) and passed the reward check easily,
yet training failed. Lesson: the check finds reward problems, but it cannot predict whether a policy
can learn the reward.

## 6. Finding 3: waypoint training

### What worked

- **Short jumps.** With crash 400, jumps up to 0.8 m still made crashing pay in 3-19% of waypoint
  states; 0.3-0.5 m jumps passed. A new option keeps the second leg no longer than the first.
- **The Huber cap is needed** for waypoints (without it the waypoint reward failed the check even
  with no time cost).
- **A time cost of 0.5** (larger values failed the check). It mainly pays for closing the last
  10 cm, because "arrived" means within 5 cm.

### Results (8 seeds; mission 12 ms; waypoints 3.5 s legs, 0.3-0.5 m)

| Arm | Circle | Crashes | 5 cm | 10 cm | Final error |
| --- | --- | --- | --- | --- | --- |
| crash 400, no waypoint training (4 seeds) | 5.3 cm | 0.4% | 27% | 81.5% | 8.7 cm |
| + waypoints, no time cost | 7.1 cm | 3.3% | 23% | 68% | 10.2 cm |
| + time cost 0.2 | 7.5 cm | 3.9% | 16.5% | 61.5% | 11.4 cm |
| + time cost 0.5 | 6.0 cm | 1.4% | 30% | 79.5% | 8.9 cm |

- Time cost 0.5 is the best waypoint-trained arm. Time cost 0.2 is consistently worse than none (not
  explained).
- **Mission-only policies fly waypoints about as well**, and are better at the mission. Waypoint
  training only helps on larger jumps (0.8 m: 69% against 62.5% within 10 cm).

### Why every policy stops 9 cm off after a jump

It looked like a fixed offset; it is not. Diagnostics in the waypoint test:

| Test | Within 5 cm | Final error | Along the leg (> 0: past the target) |
| --- | --- | --- | --- |
| 3.5 s legs, uneven motors | 16-44% | 7-13 cm | +3 to +10 cm |
| 3.5 s legs, clean motors | 18-67% | 5-12 cm | +1.5 to +7 cm |
| clean, position-error integral reset at each jump | 3-50% | 5-15 cm | +2 to +6 cm |
| 6 s legs, clean motors | 58-97% | 3.6-8.7 cm | about 0 |

The drones overshoot slightly and creep back over several seconds; uneven motors add 1-3 cm. The
integral input is not the cause; it helps (resetting it makes every policy worse). The original
test's short legs made this look like an offset.

## 7. Finding 4: the measured conflict

MOO sum on the time 0.5 recipe, 4 seeds; cosine between group gradients (negative: they want
opposite changes):

| Pair | 0-0.8M steps | 0.8-1.4M | 1.4-2.0M |
| --- | --- | --- | --- |
| time-precision | +0.94 | +0.70 | +0.15 (39% of updates opposed) |
| time-efficiency | +0.88 | +0.64 | +0.75 |
| time-safety | -0.11 (63%) | +0.86 | +0.81 |
| precision-efficiency | +0.89 | +0.71 | +0.03 (47%) |
| precision-safety | -0.12 (64%) | +0.45 | +0.05 (46%) |
| efficiency-safety | -0.34 (80%) | +0.31 | +0.51 |
| gradient norms time / precision / efficiency / safety | 0.20 / 0.29 / 0.60 / 2.31 | 0.18 / 0.89 / 0.39 / 0.93 | 0.30 / 1.69 / 0.86 / 0.82 |

- **Early** (while the curriculum ramps up): safety, the crash term, has about 10 times the gradient
  of the others and opposes them.
- **Late:** precision is roughly perpendicular to the other three, and about half of the updates are
  opposed. Time agrees with efficiency and safety. The trade-off is not "fast against safe" but
  "hold the exact target against everything else".

**Update: most of this conflict came from the critic.** The table above used the unscaled 4-head
critic (section 10). With the head-scaled critic (7 healthy seeds):

| | Unscaled critic | Head-scaled critic |
| --- | --- | --- |
| updates where the summed step hurts some group, 0-0.8M / 1.4-2.0M | 47% / 22% | 8% / 6% |
| pairwise cosines, 0-0.8M | -0.34 to +0.94 | +0.61 to +0.97 |
| pairwise cosines, 1.4-2.0M | +0.03 to +0.81 | +0.36 to +0.93 |
| gradient norms time / precision / efficiency / safety, 0-0.8M | 0.20 / 0.29 / 0.60 / 2.31 | 0.17 / 0.18 / 0.19 / 0.75 |

The noisy, badly scaled group values of the unscaled critic produced opposed gradients and made the
crash term look 10 times larger than the rest (about 4 times with the fixed critic). With the tuned
reward, the 4 groups mostly agree on this task.

**Correction (2026-11-25): this update was wrong.** The head-scaled critic showed little conflict
because it collapsed the 4 group values into copies of one (section 12). With separate critics per
group the conflict is larger than with the unscaled critic (60-80% of updates in most pairs).

## 8. Finding 5: no aggregator beats plain summing

All arms: the time 0.5 recipe, MOO on 4 groups. Mission 12 ms, waypoints 3.5 s legs 0.3-0.5 m.

| Aggregator | Seeds | Circle | Crashes | 10 cm arrival | What happened |
| --- | --- | --- | --- | --- | --- |
| plain TD3 (reference) | 8 | 6.0 cm | 1.4% | 79.5% | |
| MOO sum | 8 | 6.8 cm | 2.9% | 63% | |
| DualProj | 8 | 6.4 cm | 6.6% | 60% | looked better on 4 seeds; not on 8 |
| UPGrad | 4 | 18.4 cm | 50.6% | 17.5% | 2 of 4 seeds collapsed |
| Aligned-MTL | 4 | 9.6 cm | 13.5% | 45.5% | |
| SVD-PGA | 4 | 12.0 cm | 14.4% | 26.5% | |
| OS-NMF-PGA | 4 | 10.4 cm | 13.6% | 35.5% | |
| DualProj from 1.2M (sum before) | 4 | 7.3 cm | 5.4% | 59% | |
| UPGrad from 1.2M (sum before) | 4 | 7.3 cm | 2.5% | 60.5% | no collapse |

(4-seed rows compare with MOO sum's 4-seed values: 7.4 cm, 2.5%, 61%.)

## 9. Why the aggregators do not help

These explanations describe what happened in the runs of section 8, which all used the unscaled
critic: part of what the aggregators reacted to (the early conflict, the 10x scale gap) was critic
noise (section 7, update). The mechanisms stand; the premise that the task itself has strong conflict
does not.

### 9.1 The groups' scales encode our priorities, and some aggregators discard them

The groups' gradients differ by about 10x (section 7). That is not an accident of units: it comes from
the reward weights we chose (crash penalty 400, position weight 40, and so on), which say how much
each objective matters.

**Aligned-MTL, SVD-PGA and OS-NMF-PGA** rescale the groups so each contributes equally. Here that
gives time, the smallest gradient, 16-36 times its weight, and gives precision and safety **negative
weights** (from the gradient logs):

| Aggregator | Step shares early (time / precision / efficiency / safety) | Late |
| --- | --- | --- |
| sum | 6 / 8 / 18 / 68% | 8 / 46 / 23 / 22% |
| Aligned-MTL, SVD-PGA, OS-NMF-PGA | time x16-36; precision, efficiency, safety negative | the same |

A negative weight on safety means the update pushes the policy *against* crash avoidance: 13-14%
crashes. These methods are designed for objectives whose scales are arbitrary (different units,
different losses); they hurt when the scales are deliberate.

### 9.2 Most of the conflict is the healthy one of learning not to crash

In the first 0.8M steps the policy learns to stop crashing, and the crash term opposes the small
costs in 63-80% of updates. That conflict is the point of the phase: the crash term should win.

**UPGrad** projects each group's gradient so that no group gets worse, and keeps boosting the
smaller groups: efficiency up to 6.3 times early and 2.5 times late, safety too. It changed the
update in 82-97% of updates, often where there was no conflict at all (82% of late updates against
53% that conflicted). Two of four seeds collapsed (70% and 82% crashes).

**The decisive test:** the same UPGrad, started at 1.2M steps with plain sum before, did not collapse
(2.5% crashes, the same as sum). The damage comes from intervening in the crash-learning phase.

### 9.3 The late conflict is a trade-off, not interference

Aggregators are designed for **interference**: one objective's update damages another because they
share network parameters, when a better update for both exists. Late in training our conflict is
different: holding the exact target needs constant small corrections that cost effort and
smoothness. That trade-off lives in the reward itself; even a perfect optimiser must settle it
somewhere, and the reward weights already say where.

Where no update improves every objective, an aggregator can only reweight the objectives, which
moves the result to a different point on the trade-off than the one the weights were tuned for. It
cannot make the trade-off go away.

**DualProj** shows this most clearly. It is the most conservative aggregator: it changes the update
only when the plain sum would make some group worse, and then only raises that group's weight just
enough (its weights never drop below 1, so the natural scales are kept):

| Phase | Updates where the sum would hurt a group | Updates DualProj changed |
| --- | --- | --- |
| 0-0.8M | 83% | 83% (mostly efficiency raised) |
| 0.8-1.4M | 29% | 29% |
| 1.4-2.0M | 17% | 17% |

Late in training it barely acts, and over 8 seeds it is no better than sum (and crashes more, 6.6%
against 2.9%). Started only at 1.2M it is no better either (5.4% crashes).

### 9.4 The MOO baseline starts behind TD3

Even MOO sum, whose actor update equals TD3's in expectation, trains worse than TD3 (section 10).
Every aggregator is built on that weaker base, so even a neutral aggregator cannot win.

### 9.5 Adam reduces what reweighting can do

The actor is updated with Adam, which scales each parameter's step by its own running gradient size.
Changing the overall size of the update has little effect; only its direction matters. Aggregators
that mainly rescale (rather than redirect) therefore change less than their weights suggest, while
those that redirect (negative weights) change a lot.

## 10. Finding 6: MOO sum trailed TD3, fixed by head scaling

| Arm | Seeds | Circle | Crashes | 10 cm arrival |
| --- | --- | --- | --- | --- |
| TD3 | 8 | 6.0 cm | 1.4% | 79.5% |
| MOO sum, critic loss on total and groups (`--total 1`) | 8 | 6.8 cm | 2.9% | 63% |
| TD3 | 4 (70-73) | 4.9 cm | 0.8% | 84% |
| MOO sum `--total 1` | 4 | 7.4 cm | 2.5% | 61% |
| MOO sum `--total 0.1` | 4 | 6.7 cm | 3.5% | 65% |
| MOO sum `--total 0` (total-value loss only, as TD3) | 4 | 7.4 cm | 3.1% | 67.5% |

- MOO sum's actor update is TD3's in expectation; the difference is the critic, which learns 4 group
  values.
- **Not the per-group critic loss:** removing it (`--total 0`) did not close the gap.
- **The cause: the 4 output heads.** The total value is the sum of 4 heads. When the total is wrong,
  all 4 heads get the same correction, and Adam moves each about as far as TD3 moves its single head,
  so the total moves further per update; 4 random heads also start more spread out. Measured on 20
  freshly built critics:

| Critic | Starting spread of the total value | Distance the total moves per update |
| --- | --- | --- |
| TD3, 1 head | 0.137 | 0.0285 |
| MOO, 4 heads | 0.281 (2.1x) | 0.0646 (2.3x) |
| MOO, 4 heads, `--head-scale` | 0.137 | 0.0266 |

- **The fix (`train.py --head-scale`):** the critic outputs are multiplied by 1/k and the output
  layer's initial weights and biases by sqrt(k) (k groups), for the critics and their targets. The
  summed value then starts and moves like TD3's. The group gradients are all scaled by the same 1/k,
  which Adam cancels in the actor. Off by default; with it off, training is unchanged.
- **Result** (`ablation_headscale.txt`, `ablation_headscale_confirm.txt`; mission 12 ms plain world,
  waypoints 3.5 s legs 0.3-0.5 m):

| Arm | Seeds | Circle | Crashes (median of seeds / pooled) | Crash-free checkpoints | 10 cm arrival (time) |
| --- | --- | --- | --- | --- | --- |
| TD3 | 8 | 6.0 cm | ~1.0% / 1.4% | 18/72 | 79.5% (1.73 s) |
| MOO, 1 group | 4 | 5.0 cm (TD3 4.9) | 1.0% (TD3 0.8%) | 9/36 | 84% (as TD3) |
| MOO sum, unscaled | 8 | 6.8 cm | 2.9% pooled | 18/72 | 63% |
| MOO sum `--total 0`, head scale | 4 | 6.3 cm | 4.8% pooled | 9/36 | 73.5% |
| **MOO sum `--total 1`, head scale** | 8 | **5.0 cm** | ~0.9% / 5.6% | **32/72** | 71% (1.52 s) |

  - MOO with 1 group matches TD3 (waypoints identical to the digit): the MOO code path is sound.
  - Head-scaled MOO sum is level with TD3: better circle, crash-free twice as often, faster arrival;
    worse with uneven motors (11.2 against 9.7 cm) and at 10 cm arrival. One seed in 8 (s70) failed
    (39% crashes); the other seven crash 0.4-1.2%. Possibly each group value now learning at a
    quarter of its former per-head speed; not tested.
  - ~~Use `--head-scale` for every MOO run.~~ **Superseded (section 12):** head scaling collapses
    the group values; use `--split-critic`. All MOO results before it (sections 7-9) used the
    unscaled critic.

## 11. Finding 7: untuned, the aggregators still do not help

Hand-tuned reward weights settle the trade-off before MOO starts, so they are an oracle, not the
baseline. This study removed them.

### The untuned reward

- **Equal shares.** `train.py --group-scale` multiplies each group's weights so that every group has
  the same mean cost per step, with the same total as the tuned reward. `reward_check.py` measures
  the group costs and prints the factors. Measured with 4 trained TD3 policies of the tuned time 0.5
  recipe (median factors):

| Group | Mean cost per step, tuned | Factor | Untuned final weights |
| --- | --- | --- | --- |
| time | 0.13-0.14 | 2.3 | time 1.15 |
| precision | 0.52-0.73 | 0.53 | position 21.2, velocity 0.53 |
| efficiency | 0.37-0.60 | 0.68 | action 0.68, action rate 0.68, angular acceleration 0.000136 |
| safety (crash excluded) | 0.03-0.11 | 7.3 | tilt 36.5, landing speed 36.5 |

  The tuned reward gives precision and efficiency about 85% of the cost; the untuned one gives every
  group 25%. The relative weights inside each group are kept (a remaining hand-chosen part).
- **The crash penalty is the task's definition of failure, not a trade-off weight.** It is fixed for
  every arm at the smallest value that passes the reward check under this scale: 400 failed (free
  flight 7-8% of states for 3 of 4 policies; the boosted tilt cost makes recovering from tilted
  starts expensive), 500 passed with up to 2.7%, **600** with at most 0.8% in any episode type.
- **Task:** the time 0.5 recipe (mission plus 0.3-0.5 m waypoints, Huber 0.2, time cost, sim
  options). MOO arms use the head-scaled critic and `--total 1`.

### Results (`ablation_untuned.txt`, seeds 70-73; mission 12 ms plain world, waypoints 3.5 s legs 0.3-0.5 m)

| Arm | Circle | Crashes | Crash-free checkpoints | Jitter | 10 cm arrival | Final error |
| --- | --- | --- | --- | --- | --- | --- |
| **TD3, untuned** (baseline) | 10.1 cm | 6.0% | 0/36 | 0.148 | 29% | 17.2 cm |
| MOO sum | 9.6 cm | 7.5% | 0/36 | 0.136 | 36% | 15.8 cm |
| DualProj | 9.3 cm | 8.2% | 0/36 | 0.140 | 39% | 15.2 cm |
| UPGrad | 8.8 cm | 8.1% | 0/36 | 0.154 | 38% | 15.4 cm |
| Aligned-MTL | 12.8 cm | 11.1% | 0/36 | 0.142 | 15% | 20.6 cm |
| SVD-PGA | 15.8 cm | 17.3% | 0/36 | 0.127 | 12% | 20.9 cm |
| OS-NMF-PGA | 19.5 cm | 20.9% | 0/36 | 0.144 | 2% | 25.9 cm |
| *tuned TD3 (crash 400; reference)* | *4.9 cm* | *0.8%* | *12/36* | *0.124* | *84%* | *8.1 cm* |

### What the gradient logs show

| Arm | Summed step hurts a group (0-0.8M / 1.4-2.0M) | Lowest pair cosine | Median weights time / precision / efficiency / safety |
| --- | --- | --- | --- |
| sum | 0% / 4% | +0.98 / +0.31 | 1 / 1 / 1 / 1 |
| DualProj | 0% / 4% | +0.98 / +0.33 | 1 / 1 / 1 / 1 |
| UPGrad | 0% / 5% | +0.98 / +0.30 | 1 / 1 / 1 / 1 |
| Aligned-MTL | 0% / 4% | +0.99 / +0.67 | 0.07 / -0.00 / 0.08 / -0.06 |
| SVD-PGA | 0% / 0% | +0.99 / +0.95 | 151 / -143 / 77 / -53 |
| OS-NMF-PGA | 0% / 1% | +0.99 / +0.81 | 106 / -121 / 67 / -32 |

- **Hand-tuning was worth a lot:** untuned TD3 doubles the circle error, crashes about 7 times as
  often, never has a crash-free checkpoint and reaches a third as many waypoints. Likely cause:
  equal shares halve precision's weight and multiply the tilt cost by 7.3; a quadrotor must tilt to
  move, so the policy tracks sluggishly.
- **The groups still do not conflict.** Their gradients are nearly parallel early (+0.98) and agree
  late; the summed step hurts a group in at most 5% of updates. DualProj and UPGrad, which act only
  on conflict, kept every weight at 1: they are MOO sum. MOO sum's small differences from untuned TD3
  (better circle and waypoints, slightly more crashes) are within seed noise; all three MOO arms share
  seed 72's high crash rate (14-15%).
- **Aggregators resolve conflicts; they cannot choose priorities.** With equal shares, every group
  agrees on how to improve the (poor) reward. Nothing in gradient aggregation moves the policy to a
  better trade-off; that needs stated priorities (preference vectors) or tuning.
- **Balancing aggregators are unstable when objectives are aligned.** When the group gradients are
  nearly parallel, their differences are mostly noise. Aligned-MTL, SVD-PGA and OS-NMF-PGA equalise
  the groups by amplifying those differences: the PGA variants used weights of +-100 to +-270 with
  alternating signs, so the update is built mostly from noise (11-21% crashes). This is a different
  failure from the tuned case (section 9.1, discarding deliberate scales).

**Correction (2026-11-25):** this study used the head-scaled shared critic, which turned out to
collapse the 4 group values into copies of one value (section 12). The near-parallel gradients
(+0.98) came from that critic, not from the task. The untuned result therefore does not show that
the groups agree; it needs repeating with separate critics.

## 12. Finding 8: the shared critic collapsed the groups; separate critics fix it

### The baseline

From here on the baseline every MOO arm must beat is **the original variant E**: plain TD3 (one
critic output) with E's hand-tuned reward, the `td3` arm of every study on the time 0.5 recipe.

### What the critic learned

The critic was measured directly. A copy of the training (seed 70, `train.py` unchanged, results
identical to the study logs) took 1,024 buffer states at 200k and 1.2M steps and recorded the 4
group values at the policy's action, their correlations, the share of their variance along one
direction ("top share"; 1.0 = the 4 values are copies of one), and how well each group fits its own
TD target (R²; 1.0 = perfect).

| Critic | Step | Correlation between group values | Top share | Fit R² time / precision / efficiency / safety |
| --- | --- | --- | --- | --- |
| shared, unscaled | 200k | -0.19 to +0.98 | 0.994 | 0.94 / 0.90 / 0.77 / 1.00 |
| shared, unscaled | 1.2M | +0.67 to +0.98 | 0.893 | 1.00 / 1.00 / 0.98 / 1.00 |
| shared, head-scaled | 200k | **+0.99 to +1.00** | 0.996 | 0.80 / 0.80 / 0.81 / 0.73 |
| shared, head-scaled | 1.2M | +0.78 to +0.98 | 0.946 | 0.87 / 0.78 / 0.77 / 0.91 |
| shared, head-scaled, untuned reward | 1.2M | **+1.00** | **1.000** | 0.92 / 0.88 / 0.65 / **0.49** |
| **separate critics** (`--split-critic`) | 200k | -0.77 to +0.53 | 0.994 | 0.84 / 0.93 / 0.95 / 1.00 |
| **separate critics** | 1.2M | **+0.29 to +0.68** | **0.859** | **0.98 / 1.00 / 0.98 / 0.99** |

The groups' own per-step rewards are nearly uncorrelated (-0.39 to +0.16) in every run, so the
values have no reason to be copies of each other.

- **Head scaling collapsed the groups.** It divides every output by 4, so each group's own output
  learns at a quarter of its former speed while the shared hidden layers learn at full speed. The
  shared features take over and the 4 outputs become scaled copies of one value. The untuned reward
  (safety x 7.3, crash 600) makes the collapse total. The original variant E has one output, so this
  cannot happen there; the 4-output critic is our addition (not in the l2f repository).
- **Neither shared critic was right.** Unscaled: distinct, well-fitted groups but an unstable total
  (section 10). Head-scaled: a TD3-like total but collapsed groups.
- **Consequences for earlier sections.** The "update" in section 7 (most conflict was an artifact)
  is probably wrong: the unscaled critic's conflict was largely real. Sections 10-11's aggregator
  results ran on collapsed gradients and do not show whether aggregators help.

### The fix: one critic network per group

`train.py --split-critic` replaces each 4-output critic with 4 separate 64-64 networks, one per
group, with no shared layers. Everything else is TD3 as before: two twins (each now 4 networks),
target copies with Polyak 0.995, the twin with the smaller summed value supplying all 4 targets
y_k = r_k + gamma Q'_k (so the summed target equals variant E's), and only twin 1's 4 networks giving
the actor its 4 gradients. In total 1 actor and 8 critic networks (plus their targets) against
variant E's 1 actor and 2 critics.

### Result (`ablation_split.txt`, seeds 70-73; mission 12 ms plain world, waypoints 3.5 s legs 0.3-0.5 m)

| Arm | Circle | Uneven motors | Crashed | Jitter | 10 cm arrival 0.5 / 0.8 m |
| --- | --- | --- | --- | --- | --- |
| **td3 (variant E, baseline)** | 4.9 cm | **8.6 cm** | **0.8%** | 0.124 | **84 / 74%** |
| sum, shared unscaled | 7.4 cm | 11.9 cm | 2.5% | 0.105 | 61 / 54% |
| sum, shared head-scaled | 4.9 cm | 11.1 cm | 10.1% (s70 39%) | 0.121 | 75 / 70% |
| sum, separate critics, `--total 0` | 7.4 cm | 11.4 cm | 1.9% | 0.114 | 67 / 57% |
| **sum, separate critics, `--total 1`** | **4.9 cm** | 11.3 cm | 3.3% | **0.065** | 78 / 71% |

With `--total 0` the networks learn only their own groups and the summed value is less accurate;
`--total 1` stays.

### The real conflict

Share of actor updates in which two groups' gradients point in opposite directions (negative
cosine), last 0.6M steps, MOO sum:

| Pair | Shared, head-scaled | Shared, unscaled | **Separate critics** |
| --- | --- | --- | --- |
| precision-efficiency | 2% | 47% | **82%** (mean cosine -0.41) |
| precision-safety | 16% | 46% | **76%** (-0.30) |
| time-precision | 1% | 38% | **68%** (-0.19) |
| time-safety | 10% | 2% | **57%** (-0.08) |
| efficiency-safety | 41% | 17% | 43% (+0.08) |
| time-efficiency | 15% | 6% | 39% (+0.12) |

With separate critics most pairs conflict in most updates, from the start of training to the end:
tracking costs motor effort (precision-efficiency), and holding the target fights speed
(time-precision) and caution (precision-safety). Noise from separate networks alone would average to
a cosine near 0, not to consistently negative values; the snapshot shows distinct, well-fitted group
values. **This is the first setup in which the aggregators have real conflict to resolve.**

## 13. Finding 9: with separate critics, the aggregators beat sum and SVD-PGA matches variant E

All arms: separate critics, `--total 1`, the time 0.5 recipe with E's tuned weights. Mission test
12 ms plain world; waypoints 3.5 s legs, 0.3-0.5 m and 0.5-0.8 m jumps.

### 4 seeds (`ablation_split_agg.txt`, seeds 70-73)

| Arm | Circle | Uneven motors | Crashed | Crash-free checkpoints | Jitter | 10 cm arrival 0.5 / 0.8 m |
| --- | --- | --- | --- | --- | --- | --- |
| **td3 (variant E)** | 4.9 cm | **8.6 cm** | **0.8%** | **12/36** | 0.124 | 84.0 / 73.5% |
| sum | 4.9 cm | 11.3 cm | 3.3% | 2/36 | **0.065** | 78.0 / 71.0% |
| DualProj | 5.7 cm | 11.3 cm | 4.9% | 4/36 | 0.068 | 76.5 / 65.5% |
| UPGrad | 5.1 cm | 11.2 cm | 3.8% | 1/36 | 0.067 | 82.0 / 71.5% |
| Aligned-MTL | 5.0 cm | 10.6 cm | 2.6% | 5/36 | 0.073 | 80.5 / 70.0% |
| **SVD-PGA** | 4.7 cm | 10.6 cm | 1.7% | 5/36 | 0.098 | **84.5 / 77.5%** |
| OS-NMF-PGA | **4.5 cm** | 10.5 cm | 2.5% | 8/36 | 0.078 | 75.0 / 69.0% |

No seed collapsed in any arm. DualProj's median weights are exactly 1 (it mostly takes the sum).

### 8 seeds (`ablation_split_agg.txt` and `_confirm.txt`, seeds 70-77, every arm)

| Arm | Circle | Uneven motors | Crashed | Crash-free checkpoints | Jitter | 10 cm arrival 0.5 / 0.8 m (time) |
| --- | --- | --- | --- | --- | --- | --- |
| **td3 (variant E)** | 6.0 cm | **9.7 cm** | **1.4%** | **18/72** | 0.121 | 79.5 / 69.0% (1.73 s) |
| sum | 5.0 cm | 11.2 cm | 4.5% | 7/72 | **0.065** | 78.0 / 71.5% (1.55 s) |
| DualProj | 5.3 cm | 11.3 cm | 3.9% | 9/72 | 0.066 | 77.0 / 68.0% (1.60 s) |
| UPGrad | 5.0 cm | 10.1 cm | 2.7% | 7/72 | 0.069 | **84.0 / 76.5%** (1.54 s) |
| Aligned-MTL | 5.0 cm | 10.4 cm | 2.4% | 17/72 | 0.079 | 82.5 / 76.0% (**1.48 s**) |
| **SVD-PGA** | 4.9 cm | 10.4 cm | 1.7% | 14/72 | 0.106 | 83.5 / 76.0% (1.53 s) |
| OS-NMF-PGA | **4.5 cm** | 10.2 cm | 2.5% | 16/72 | 0.081 | 79.5 / 73.0% (**1.48 s**) |

Crashed per seed (70-77): td3 0.5, 0.6, 1.9, 0.3, **4.1**, 1.2, 0.8, 2.3; sum 2.0, 3.1, 4.9, 3.0, 1.6,
**9.7, 9.7**, 1.9; DualProj 2.2, **12.6**, 2.6, 2.2, 3.1, 3.9, 1.2, 3.5; UPGrad 4.1, 2.5, 3.6, 4.8, 1.6, 1.8,
2.1, 1.1; Aligned-MTL 5.7, 1.9, 1.7, 1.3, 2.1, 2.1, 2.4, 2.4; SVD-PGA 1.2, 3.4, 1.0, 1.2, 1.3, 1.4, 2.0, 1.9;
OS-NMF-PGA 1.4, 3.8, 2.8, 1.9, 1.7, 2.7, 2.8, 2.7. With the simulator options at 6 ms
the picture is the same (SVD-PGA 1.2% crashes against td3 1.3%, circle 4.8 against 6.3 cm).

- **The aggregators beat the sum on the same critic and reward:** every aggregator that reweights the
  groups crashes less than the sum (SVD-PGA 1.7%, Aligned-MTL 2.4%, OS-NMF-PGA 2.5%, UPGrad 2.7%,
  against 4.5%) and tracks uneven motors better (10.1-10.4 against 11.2 cm). Only the aggregator
  differs. DualProj, whose weights stay at 1, is the exception (3.9%, one seed at 12.6%). On seeds 70-73
  UPGrad looked no better than the sum (3.8 against 3.3%) and Aligned-MTL only slightly better; both
  improved on 74-77: 4 seeds misled again. SVD-PGA is the steadiest (worst seed 3.4%); UPGrad reaches the most waypoints (84.0 / 76.5%).
- **Against variant E, SVD-PGA is better on the circle (4.9 against 6.0 cm) and at waypoints (83.5 /
  76 against 79.5 / 69%, 0.2 s faster), level on crashes (1.7 against 1.4%; within seed spread) and
  uneven motors (10.4 against 9.7 cm), and mixed on smoothness (jitter 0.106 against 0.121 at 12 ms,
  0.098 against 0.088 with the simulator options at 6 ms; sum is the smoothest, 0.065).** Variant E's lead on seeds 70-73 shrank on 74-77,
  where TD3 itself did worse (seed 74: 8.8 cm, 4.1%); SVD-PGA is steadier across seeds.
- **How SVD-PGA gets there.** Over the last 0.6M steps it gives efficiency about 6% of the step
  (sum: 24%; efficiency's weight is negative in 43% of updates), and time, precision and safety about
  a quarter each:

| Aggregator | Median weights time / precision / efficiency / safety | Share of the step |
| --- | --- | --- |
| DualProj | 1 / 1 / 1 / 1 | 0.16 / 0.36 / 0.24 / 0.21 |
| UPGrad | 1.46 / 1.84 / 1.22 / 3.75 | 0.17 / 0.32 / 0.22 / 0.29 |
| Aligned-MTL | 0.55 / 0.38 / 0.16 / 0.95 | 0.27 / 0.28 / 0.15 / 0.30 |
| SVD-PGA | 0.36 / 0.21 / 0.03 / 0.46 | 0.28 / 0.24 / 0.06 / 0.26 |
| OS-NMF-PGA | 0.37 / 0.22 / 0.06 / 0.52 | 0.28 / 0.29 / 0.10 / 0.28 |

  The trade-off it found from the conflict (less smoothing, faster and safer flight) was not tuned
  by hand; it also explains why SVD-PGA is rougher than sum.
- **Not yet shown:** that the aggregator replaces reward tuning. Each group still holds E's tuned
  weights inside it. The untuned comparison (section 11) needs repeating with separate critics.

## 14. Caveats

- **Seed variance is large.** TD3 on this task varies by several cm between seeds. DualProj's 4-seed
  lead disappeared at 8 seeds; time cost 0.5's lead halved. Differences of 1-2 cm or a few percent
  need 8 or more seeds.
- **The group gradients come from a learned critic.** A shared multi-output critic can hide the
  conflict (head-scaled: the groups collapse into one value) or distort the total (unscaled). Check
  the critic itself (group-value correlations, fit per group) before trusting a conflict
  measurement; use separate critics (section 12).
- **The aggregator results are 8 seeds and medians,** not a significance test. The circle and waypoint gaps
  exceed the seed spread; crashes and uneven motors do not.
- **One task, one recipe.** These results are for the mission plus 0.3-0.5 m waypoints with this
  reward. They do not show that aggregators are useless in general.
- **Simulation only.** Nothing here was flown.

## 15. What this means for the paper

- **The positive result (sections 12-13).** A shared multi-output critic hides the conflict between
  objectives; one critic per objective reveals it (60-80% of updates opposed in most pairs). With
  that conflict visible, the PGA aggregators beat linear scalarization on the same critic and reward
  (crashes 1.7-2.7% against 4.5% for every reweighting aggregator), and SVD-PGA matches or beats the hand-tuned variant E (better circle and
  waypoints, level crashes), with rougher motor commands than the sum.
- **The earlier negative results below were measured with the shared critic** and are kept as the
  record of how the problem was found. On a task with a carefully tuned scalar reward:
  - scale-normalising aggregators fail because the scales encode priorities;
  - projection aggregators fail during the safety-learning phase, and delaying them fixes the
    failure but adds nothing;
  - the most conservative aggregator cannot beat a tuned sum when the conflict is a real trade-off;
  - the decomposed critic must match TD3 before any aggregator comparison is fair.
- **Untuned (section 11):** aggregation does not replace reward tuning on this task. The flight
  objectives do not conflict at the gradient level, tuned or untuned, so conflict-resolving
  aggregators reduce to the sum; balancing aggregators (PGA included) become unstable when the
  objectives are aligned.
- **Where aggregators could still matter:** genuine interference through shared parameters
  (situations that need opposite actions in similar states), for example a real target jump against
  a Lighthouse glitch of the same size, or camera tasks whose objectives have no natural common
  scale. Run a conflict check (one sum run with the head-scaled critic) before any aggregator study.
- **Stating priorities:** preference vectors are MOO's tool for choosing the trade-off; comparing
  them with a grid of TD3 weights (the Pareto front) is the experiment that tests MOO's claim to
  replace tuning.

## 16. Open questions and next steps

0. **Repeat the untuned comparison with separate critics:** TD3 untuned, sum and SVD-PGA, against
   variant E. This tests whether the aggregator removes the need for tuning.
1. **Earlier: the untuned comparison** (section 11, shared head-scaled critic; to be repeated). Follow-ups: preference vectors against a grid of TD3
   weights (the Pareto front); a fully untuned variant (every reward term equalised, not just the
   groups).
2. **A conflict-by-design task** where the right actions are opposite in similar states (target jump
   against a Lighthouse glitch; later camera tasks, where objectives have no natural common scale).
3. **Why time cost 0.2 is worse than none** (consistent over 8 seeds).
4. **Why the progress reward failed** (steady 10-20 cm offset).
5. **Slow settling after a jump** (3-6 s to get within 5 cm): the real target for a time objective.
6. **The free-flight crash incentive** is reduced but not gone (3-7% of states for policies trained
   with crash 400).

## 17. Lessons about the process

- **Change one thing at a time** against a reference, and test every arm under the same conditions.
- **Gate reward changes with the reward check**, with the final weights. It finds crash incentives in
  seconds. It is necessary but not sufficient: the progress reward passed and still failed.
- **Look per seed.** One collapsed seed turned a pooled crash rate into 25.6%.
- **Use 8 seeds before a decision.** Four seeds misled twice.
- **Check what a test measures.** The waypoint test's short legs made slow settling look like an
  offset; its 5 cm arrival measured the test as much as the policy.
- **Verify new options are bit-identical when off,** by training. A zero column in the replay buffer
  changed float32 rounding.
- **Check analysis tools against the real code.** A missing weight in the candidate scoring made two
  rewards look 40 times cheaper until the C++ reward disagreed.

## 18. Reproducing the results

Build (old observation path):

```
source l2f/sim/platform.sh
R="-DL2F_BRUSHLESS -DL2F_REAL_NOISE -DL2F_ANG_ACC_WEIGHT=2e-4 -DL2F_LANDING_SPEED_WEIGHT=5 -DL2F_POS_INTEGRAL
   -DL2F_MISSION -DL2F_STEP_LIMIT=2000001 -DL2F_REPLAY_CAP=700000 -DL2F_RPM_OBS -DL2F_HISTORY=0
   -DL2F_CRASH_PENALTY=200 -DL2F_SURVIVAL_BONUS=0 -DL2F_ACTION_RATE_WEIGHT=1"
cmake -S l2f/learning_to_fly -B l2f/learning_to_fly/build_py_E_nb200 -DCMAKE_BUILD_TYPE=Release \
  "${CMAKE_PLATFORM[@]}" -DL2F_BUILD_PYTHON=ON -DPython_EXECUTABLE=$(which python3) -DCMAKE_CXX_FLAGS="$R"
cmake --build l2f/learning_to_fly/build_py_E_nb200 -j --target l2f_sim evaluate_robustness h5_to_header
```

Run a study (trains every arm, tests every run, prints medians and per-seed results):

```
python l2f/sim/ablation.py --study l2f/sim/ablation_<name>.txt \
  --build l2f/learning_to_fly/build_py_E_nb200/src --seeds 70 71 72 73 --jobs 8
```

Check a reward before training:

```
python l2f/py/reward_check.py --build l2f/learning_to_fly/build_py_E_nb200/src --actor <actor.h5> \
  --env L2F_OMEGA_CRASH=34.9 ... [--candidate crash=400 ...]
```

| Study file | Question |
| --- | --- |
| `l2f/sim/ablation_simopts.txt` | the simulator options |
| `l2f/sim/ablation_progress.txt` | progress reward, crash penalty 400 |
| `l2f/sim/ablation_waypoint.txt` | Huber cap, waypoints without a time cost |
| `l2f/sim/ablation_time.txt`, `ablation_time_confirm.txt` | time cost 0.2 and 0.5, 8 seeds |
| `l2f/sim/ablation_conflict.txt` | gradient conflict (MOO sum) |
| `l2f/sim/ablation_aggregators.txt` | 6 aggregators |
| `l2f/sim/ablation_moo2.txt`, `ablation_moo2_confirm.txt` | DualProj on 8 seeds, the critic loss, delayed aggregators |
| `l2f/sim/ablation_headscale.txt`, `ablation_headscale_confirm.txt` | the 4-head critic: MOO with 1 group, head scaling |
| `l2f/sim/ablation_untuned.txt` | the untuned comparison: TD3 and 6 aggregators, equal group shares, crash 600 |
| `l2f/sim/ablation_split.txt` | separate critics per group (`--split-critic`), `--total 0` and `1` |
| `l2f/sim/ablation_split_agg.txt`, `ablation_split_agg_confirm.txt` | 5 aggregators and sum on separate critics, 8 seeds each (run `ablation_split_agg.txt` with `--seeds 70 ... 77`) |

Raw outputs (`l2f/learning_to_fly/ablation_*`) and checkpoints are not in git; they are on the Mac
that ran them. Gradient logs: `logs/train_moo_*.csv` (per-group norms, pairwise cosines, aggregator
weights every 50 actor updates).

## 19. Glossary

| Term | Meaning |
| --- | --- |
| TD3 | the reinforcement-learning algorithm used to train the policy (actor and two critics) |
| critic | the network that estimates how good an action is (the value) |
| aggregator | combines the per-group actor gradients into one update (TorchJD) |
| sum | adds the group gradients: plain TD3's update |
| DualProj | changes the summed update only when it would make some group worse |
| UPGrad | projects every group's gradient so that no group gets worse, then averages |
| Aligned-MTL, SVD-PGA, OS-NMF-PGA | rescale the groups so each contributes equally |
| `--total` | weight of the per-group critic loss next to the total-value loss |
| `--agg-start` | plain sum until this training step, then the aggregator |
| `--group-scale` | factors on the reward groups' weights; equal shares = the untuned reward |
| `--head-scale` | MOO critic outputs x 1/k and initial output layer x sqrt(k): the summed value starts and moves like TD3's, but the group values collapse (section 12) |
| `--split-critic` | one separate critic network per reward group, no shared hidden layers |
| variant E | the baseline: plain TD3 with E's hand-tuned reward (the `td3` arm) |
| crash-better | share of states where crashing at once would pay more than flying on |
| free flight | training episodes that start from a random state, often tilted and away from the target |
| Huber cap | position cost quadratic near the target and linear beyond 0.2 m |
| jitter | spread of the motor commands in hover (lower is smoother) |
| seed | one independent training run; results vary between seeds |
