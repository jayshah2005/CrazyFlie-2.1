# MOO on the drone controller: what we found (2026-11-23)

This document summarises everything learned in one day of reward and multi-objective (MOO) studies
on branch `feat/moo`, and explains **why the gradient aggregators do not help on this task**. It is
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
7. [Finding 4: the objectives do conflict](#7-finding-4-the-objectives-do-conflict)
8. [Finding 5: no aggregator beats plain summing](#8-finding-5-no-aggregator-beats-plain-summing)
9. [Why the aggregators do not help](#9-why-the-aggregators-do-not-help)
10. [Finding 6: MOO sum trains worse than TD3](#10-finding-6-moo-sum-trains-worse-than-td3)
11. [Caveats](#11-caveats)
12. [What this means for the paper](#12-what-this-means-for-the-paper)
13. [Open questions and next steps](#13-open-questions-and-next-steps)
14. [Lessons about the process](#14-lessons-about-the-process)
15. [Reproducing the results](#15-reproducing-the-results)
16. [Glossary](#16-glossary)

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
- **The objectives conflict.** Early in training the crash term opposes everything else; late in
  training precision is roughly perpendicular to time, efficiency and safety.
- **No aggregator beats plain summing, and MOO itself trails TD3.** Over 6 aggregators, 4-8 seeds
  each, and two start times, nothing beat MOO sum, and MOO sum is worse than plain TD3.
- **Why:** the objectives' gradient scales encode the tuned priorities and some aggregators discard
  them; the main conflict is the healthy one of learning not to crash; the late conflict is a true
  trade-off that reweighting can only move, not remove; and the MOO baseline starts behind.

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

## 7. Finding 4: the objectives do conflict

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

## 10. Finding 6: MOO sum trains worse than TD3

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
- **Hypothesis (not yet tested): the 4 output heads.** The total value is the sum of 4 heads. When
  the total is wrong, all 4 heads get the same correction, and Adam moves each about as far as TD3
  moves its single head, so the total moves about 4 times as far per update: in effect a 4x learning
  rate on the critic's output layer. The 4 randomly initialised heads also start with about twice the
  output spread. Both make the critic noisier, which makes the actor's updates noisier.
- **Test:** MOO with 1 group (must match TD3), and MOO with 4 groups with the output layer's step
  scaled by 1/4 and the initial head weights by 1/2.

## 11. Caveats

- **Seed variance is large.** TD3 on this task varies by several cm between seeds. DualProj's 4-seed
  lead disappeared at 8 seeds; time cost 0.5's lead halved. Differences of 1-2 cm or a few percent
  need 8 or more seeds.
- **The group gradients come from a learned critic.** Part of the measured conflict may be critic
  error rather than true conflict between objectives. This was not measured on this task (an
  earlier C++ study found the per-objective gradients consistent across batches).
- **One task, one recipe.** These results are for the mission plus 0.3-0.5 m waypoints with this
  reward. They do not show that aggregators are useless in general.
- **Simulation only.** Nothing here was flown.

## 12. What this means for the paper

- **A clear, explained negative result.** On a task with a carefully tuned scalar reward:
  - scale-normalising aggregators fail because the scales encode priorities;
  - projection aggregators fail during the safety-learning phase, and delaying them fixes the
    failure but adds nothing;
  - the most conservative aggregator cannot beat a tuned sum when the conflict is a real trade-off;
  - the decomposed critic must match TD3 before any aggregator comparison is fair.
- **Where aggregators could still matter:** objectives with arbitrary scales and no tuned weights;
  genuine interference through shared parameters (situations that need opposite actions in similar
  states); a task designed with that conflict, for example a real target jump against a Lighthouse
  glitch of the same size.

## 13. Open questions and next steps

1. **Close the MOO-TD3 gap** (the 4-head hypothesis, section 10). Then repeat sum against DualProj
   on the fixed critic.
2. **A conflict-by-design task** where the right actions are opposite in similar states.
3. **Why time cost 0.2 is worse than none** (consistent over 8 seeds).
4. **Why the progress reward failed** (steady 10-20 cm offset).
5. **Slow settling after a jump** (3-6 s to get within 5 cm): the real target for a time objective.
6. **The free-flight crash incentive** is reduced but not gone (3-7% of states for policies trained
   with crash 400).

## 14. Lessons about the process

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

## 15. Reproducing the results

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

Raw outputs (`l2f/learning_to_fly/ablation_*`) and checkpoints are not in git; they are on the Mac
that ran them. Gradient logs: `logs/train_moo_*.csv` (per-group norms, pairwise cosines, aggregator
weights every 50 actor updates).

## 16. Glossary

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
| crash-better | share of states where crashing at once would pay more than flying on |
| free flight | training episodes that start from a random state, often tilted and away from the target |
| Huber cap | position cost quadratic near the target and linear beyond 0.2 m |
| jitter | spread of the motor commands in hover (lower is smoother) |
| seed | one independent training run; results vary between seeds |
