# Reward studies after the time-to-target work (2026-11-23)

Follow-up to [time_to_target_study.md](time_to_target_study.md), which found that in waypoint
episodes crashing cost less than flying. This round went back to the mission recipe and changed one
thing at a time, with a reward check before each training study.

Machine: MacBook Pro M6 (Accelerate, `-O3`), PyTorch trainer `l2f/py/train.py`, plain TD3, variant E,
old (truth) observation path, build `build_py_E_nb200` (old3's build flags: brushless, real noise,
angular-acceleration weight 2e-4, landing-speed weight 5, position integral, mission, RPM observation,
no history, crash penalty 200, no survival bonus, action rate 1). 2M steps, seeds 70-73, 12 runs at a
time (about 14 min each).

## Contents

1. [Tools](#1-tools)
2. [Check: the Mac build reproduces old3](#2-check-the-mac-build-reproduces-old3)
3. [Study 1: sim-to-real options](#3-study-1-sim-to-real-options)
4. [Reward check: the base recipe pays for crashing in free flight](#4-reward-check-the-base-recipe-pays-for-crashing-in-free-flight)
5. [Study 2: progress reward and crash penalty 400](#5-study-2-progress-reward-and-crash-penalty-400)
6. [Study 3: waypoints without a time cost](#6-study-3-waypoints-without-a-time-cost)
7. [Study 4: time cost](#7-study-4-time-cost)
8. [Why the policies settle 9 cm off after a jump](#8-why-the-policies-settle-9-cm-off-after-a-jump)
9. [Gradient conflict on the time 0.5 recipe](#9-gradient-conflict-on-the-time-05-recipe)
10. [Lessons](#10-lessons)
11. [Next steps](#11-next-steps)
12. [Options added](#12-options-added)

## 1. Tools

| Tool | What it does |
| --- | --- |
| `l2f/sim/ablation.py` | Runs a study file: trains every arm over the seeds, tests every run under every test condition (mission test at 6 and 12 ms; optional waypoint tests), prints medians per arm and per seed, and flags seeds that crash 100%. Interrupted studies continue where they stopped. |
| `l2f/sim/ablation_*.txt` | The study files of this round (arms, test conditions, the reasoning) |
| `l2f/py/reward_check.py` | Rolls a policy through the training episode mix with the **final** reward weights and reports, per episode type, the share of states where crashing at once pays more than flying on (`G_t < -P - K d_t`), plus a sign check (every part but survival and progress is a cost). `--candidate` scores other rewards on the same flights. Takes seconds. |

Verdict of the reward check per episode type: FAIL above 5% of states, WARN above 0%. It warns when
a policy crashes in more than 20% of a type's episodes: then only the easy episodes are left among
the survivors and the numbers look better than the reward is.

## 2. Check: the Mac build reproduces old3

old3 (`policies/2026-11-22_old3_moosum3_s40_1800k`), mission test, 100 episodes, clean, 6 ms: crash 0,
jitter 0.0489, the same as its `test100.txt`. Training is deterministic run to run.

## 3. Study 1: sim-to-real options

`ablation_simopts.txt`. Arms: `base` (old3's runtime options) and `simopts` (+ `L2F_DRAG_MAX=0.03
L2F_MOTOR_TAU_MIN=0.045 L2F_MOTOR_TAU_MAX=0.09 L2F_GAIN_WALK=0.03`, the options of the time-to-target
runs 4-6). Every run tested with and without the options.

| Arm | Test world | Circle | Uneven motors | Crashed | Jitter | Crash-free checkpoints |
| --- | --- | --- | --- | --- | --- | --- |
| base (3 seeds; s71 collapsed) | plain, 12 ms | 3.5 cm | 7.3 cm | 0.8% | 0.110 | 8/27 |
| simopts | plain, 12 ms | 4.0 cm | 7.9 cm | 0.7% | 0.104 | 11/36 |
| base (3 seeds) | options on, 12 ms | 3.5 cm | 7.5 cm | 0.8% | 0.113 | 9/27 |
| simopts | options on, 12 ms | 4.1 cm | 8.5 cm | 0.5% | 0.106 | 17/36 |

The options cost no measurable precision (seed spread 3.2-5.4 cm) and make the policies slightly more
robust. They did not cause the mission crashes of the time-to-target study. **Kept.**

Base seed 71 crashed in 100% of tests from 1.0M on. Its training return improved (-757 to -347) while
its episodes shrank to 0.2-0.3 s: it learned to crash.

## 4. Reward check: the base recipe pays for crashing in free flight

Share of states where crashing pays more (crash penalty 200):

| Policy | Free flight | Take-off | Landing | Circle | Line | Waypoint |
| --- | --- | --- | --- | --- | --- | --- |
| old3 | 8.6% FAIL | 2.5% | 0.7% | 3.6% | 0.8% | - |
| simopts s72 | 22.4% FAIL | 2.4% | 0.6% | 4.1% | 1.2% | - |
| old3, the time-to-target study's waypoint settings | 18.2% FAIL | 2.9% | 1.2% | 5.2% | 2.5% | 83.6% FAIL |

Free-flight episodes start from random states away from the target; recovering costs about as much
as the crash penalty (old3's survivors: median -182 against -200). Every seed is under steady
pressure to learn to crash; base seed 71 gave in. This is not only a waypoint problem.

Candidates scored on the same flights (old3 / simopts s72, free flight; other types lower):

| Candidate | old3 | simopts s72 |
| --- | --- | --- |
| as built | 8.6% | 22.4% |
| Huber cap 0.2 | 4.8% | 16.6% |
| survival bonus 1 | 2.9% | 11.0% |
| crash penalty 400 | 0.8% | 5.8% |
| progress K = 20 | 1.2% | 8.0% |
| progress 20 + crash 400 | 0.1% | 1.2% |

Perfect flight of one waypoint leg (accelerate 5 m/s², cruise 1.5 m/s, brake 2 m/s²; return over 3 s
from the jump, position cost weight 40, time cost 0.5):

| Reward | 0.5 m | 1.0 m | 1.5 m | 3.0 m |
| --- | --- | --- | --- | --- |
| squared | -122 | -612 | -1652 | -9200 |
| Huber 0.2 | -92 | -273 | -514 | -1461 |
| progress 20 | -32 | -41 | -43 | -33 |

## 5. Study 2: progress reward and crash penalty 400

`ablation_progress.txt`, all arms on top of simopts. Mission test, plain world, 12 ms:

| Arm | Circle | Uneven motors | Crashed | Crash-free checkpoints | Jitter | Collapsed seeds |
| --- | --- | --- | --- | --- | --- | --- |
| simopts | 4.0 cm | 7.9 cm | 0.7% | 11/36 | 0.104 | 0 |
| **crash400** | 5.3 cm | 9.1 cm | **0.4%** | **22/36** | 0.111 | 0 |
| prog20 | 16.9 cm | 31.9 cm | 46% | 0/36 | 0.075 | 1 |
| prog20 + crash400 | 13.7 cm | 22.7 cm | 26% | 0/36 | 0.098 | 0 |

- **crash400** doubles the crash-free checkpoints for about 1.2 cm of circle precision (seeds 4.3-6.3
  cm against 3.8-5.3 cm). **New base.**
- **The progress reward failed** although it passed the reward check: the policies hold a steady
  10-20 cm offset in every phase (hover 9-21 cm against 3 cm) and crash often. Not diagnosed;
  likely the critic has to learn the extra -20 d term (plus noise from moving targets) and the flat
  cost beyond 0.2 m leaves little slope there. The option stays in the code, off.
- The crash400 policies, checked under their own reward, still show 3-7% free-flight states where
  crashing pays (they fly with higher dense costs); in tests they crash less than simopts.

## 6. Study 3: waypoints without a time cost

`ablation_waypoint.txt`, all arms with crash 400. Gate first: with the crash400 policies, jumps up to
0.8 m failed for 3 of 4 policies, up to 0.6 m for 2 of 4, up to 0.5 m passed for 3 of 4 (0.4-5.4%).
So: jumps 0.3-0.5 m, second leg 0.5-1x the first (`L2F_WAYPOINT_S_MIN=0 L2F_WAYPOINT_S_MAX=0.5`),
20% of episodes, no time cost.

Mission test, plain world, 12 ms:

| Arm | Circle | Uneven motors | Crashed | Crash-free checkpoints | Jitter |
| --- | --- | --- | --- | --- | --- |
| crash400 | 5.3 cm | 9.1 cm | 0.4% | 22/36 | 0.111 |
| + Huber 0.2 | 4.8 cm | 9.1 cm | 1.3% | 17/36 | 0.099 |
| + Huber + waypoints | 6.9 cm | 10.9 cm | 4.3% | 8/36 | 0.120 |

Waypoint test, 50 episodes per checkpoint from 1.2M (crashed / arrived within 5 cm and below 0.2 m/s
/ final error / peak speed; time to target about 1.5 s for all):

| Arm | Jumps 0.3-0.5 m | Jumps 0.3-0.8 m |
| --- | --- | --- |
| crash400 | 0% / 12.5% / 13.3 cm / 0.65 m/s | 0% / 8.5% / 15.2 cm / 0.89 m/s |
| + Huber | 0% / 13.0% / 14.0 cm / 0.69 m/s | 0% / 6.5% / 16.7 cm / 0.92 m/s |
| + Huber + waypoints | 0% / 12.0% / 14.2 cm / 0.76 m/s | ~1% / 8.5% / 15.8 cm / 1.02 m/s |

- Mission-only crash400 policies already fly jumps up to 0.8 m without crashing, slowly.
- Waypoint training made them slightly faster but no more accurate, and cost the mission everywhere.
  Without a time cost nothing pays for arriving sooner, so the slow approach is close to optimal.
- Huber alone: slightly better circle and jitter, less reliable. Not kept.
- Re-gate with the waypoint-trained policies: jumps up to 0.5 m pass for 4 of 4 (0.2-3.9%), up to
  0.8 m for 3 of 4 (s72 10.3%).
- The arrival numbers are partly the test's: legs end 1.5-2.5 s after the jump, before slow policies
  settle (time_to_target_study.md 6.4).

Re-scored with 3.5 s legs (`L2F_EVAL_LEG_S=3.5`, test only), arrival within 10 cm added:

| Arm | 0.3-0.5 m: 5 cm / 10 cm (time) / final error | 0.3-0.8 m: 5 cm / 10 cm / final error |
| --- | --- | --- |
| crash400 | 27.0% / 81.5% (1.73 s) / 8.7 cm | 20.5% / 62.5% / 11.2 cm |
| + Huber | 33.0% / 80.5% (1.73 s) / 8.8 cm | 21.0% / 62.0% / 10.7 cm |
| + Huber + waypoints | 25.5% / 73.5% (1.66 s) / 9.9 cm | 18.5% / 58.5% / 11.2 cm |

With time to settle most legs end within 10 cm; waypoint training still does not help.

## 7. Study 4: time cost

Gate (`reward_check.py`, 8 policies, crash 400, waypoints 0.3-0.5 m at 20%):

| Setting | Free flight | Waypoint |
| --- | --- | --- |
| no Huber, no time cost | 5.3-10.7% FAIL | 1.4-18.8% (5 of 8 FAIL) |
| no Huber, time 0.5 / 1 / 2 | +1-2 / +3-5 / +6-10 points | +1.5-4 / +3-5 / +6-10 points |
| Huber 0.2, no time cost | 1.5-4.0% | 0.3-3.9% |
| Huber 0.2, time 0.5 | 2.2-5.0% | 0.4-4.6% |
| Huber 0.2, time 2 | 5.3-10.4% FAIL | 1.3-9.3% |

Without Huber the waypoint reward fails even with no time cost, so the time arms keep Huber 0.2. A
time cost of 0.5 (0.25 per step) is small far from the target and about the position cost 10 cm
out; "arrived" means within 5 cm and below 0.2 m/s, so it mainly pays for the last 10 cm.

`ablation_time.txt` (seeds 70-73) and `ablation_time_confirm.txt` (seeds 74-77), on crash 400 +
Huber 0.2 + waypoints 0.3-0.5 m. Medians over 8 seeds (crash400: 4 seeds); mission test plain world
12 ms; waypoint test 3.5 s legs, 0.3-0.5 m:

| Arm | Circle | Crashed | Crash-free checkpoints | 5 cm | 10 cm | Final error | 0.3-0.8 m: 10 cm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crash400 (no waypoints, 4 seeds) | 5.3 cm | 0.4% | 22/36 | 27.0% | 81.5% | 8.7 cm | 62.5% |
| waypoints, no time cost | 7.1 cm | 3.3% | 16/72 | 23.0% | 68.0% | 10.2 cm | 57.0% |
| + time 0.2 | 7.5 cm | 3.9% | 11/72 | 16.5% | 61.5% | 11.4 cm | 49.0% |
| + time 0.5 | 6.0 cm | 1.4% | 18/72 | 30.0% | 79.5% | 8.9 cm | 69.0% |

- Time 0.5 is the best waypoint-trained arm on every measure; time 0.2 is worse than no time cost
  (consistent over 8 seeds, not explained).
- On seeds 70-73 alone time 0.5 looked better (4.9 cm, 0.8%, 37.5% within 5 cm); seeds 74-77 were
  weaker (circle 5.8-8.8 cm). Four seeds are not enough for these differences.
- **Waypoint training does not beat the mission-only crash400 policies at waypoints** (10 cm: 79.5%
  against 81.5%), and costs some mission precision; it helps only on larger jumps (0.8 m: 69% against
  62.5%). Arrival time to 10 cm is about 1.7 s in every arm.

## 8. Why the policies settle 9 cm off after a jump

Diagnostics in the waypoint test (crash400 and time 0.5, seeds 70-73, 2.0M, 100 episodes each):

| Test | 5 cm | Final error | Along the leg (> 0: past the target) |
| --- | --- | --- | --- |
| 3.5 s legs, uneven motors (as above) | 16-44% | 7.0-13.2 cm | +3.4 to +9.6 cm |
| 3.5 s legs, clean motors | 18-67% | 5.0-12.3 cm | +1.5 to +7.0 cm |
| 3.5 s legs, clean, integral reset at every jump | 3-50% | 5.4-15.0 cm | +1.9 to +6.3 cm |
| 6 s legs, clean motors | 58-97% | 3.6-8.7 cm | -1.2 to -0.1 cm |

No fixed offset: the drones overshoot a little and creep back over several seconds, and uneven
motors add 1-3 cm. The position-error integral is not the cause; resetting it at the jump makes
every policy worse (it carries the compensation for uneven motors and helps the approach). The time
objective's real target is this slow settling.

## 9. Gradient conflict on the time 0.5 recipe

`ablation_conflict.txt`: MOO sum on 4 groups (time; precision = position, velocity; efficiency =
action, action rate, angular acceleration; safety = survival, orientation, landing speed) with
`--total 1`, seeds 70-73. Pairwise cosines of the group gradients (4 seeds, 1970 log points each):

| Pair | 0-0.8M | 0.8-1.4M | 1.4-2.0M |
| --- | --- | --- | --- |
| time-precision | +0.94 | +0.70 | +0.15 (39% negative) |
| time-efficiency | +0.88 | +0.64 | +0.75 |
| time-safety | -0.11 (63% negative) | +0.86 | +0.81 |
| precision-efficiency | +0.89 | +0.71 | +0.03 (47% negative) |
| precision-safety | -0.12 (64% negative) | +0.45 | +0.05 (46% negative) |
| efficiency-safety | -0.34 (80% negative) | +0.31 | +0.51 |
| norms time / precision / efficiency / safety | 0.20 / 0.29 / 0.60 / 2.31 | 0.18 / 0.89 / 0.39 / 0.93 | 0.30 / 1.69 / 0.86 / 0.82 |

- Early (curriculum ramp): safety (the crash term, norm about 10x the others) against the rest.
- Late: precision is roughly orthogonal to the other three (about half of the updates opposed; seeds
  71 and 73 negative, 70 and 72 positive); time agrees with efficiency and safety. The trade-off is
  not fast against safe but holding the exact target against the rest.
- MOO sum trained worse than its TD3 twin on these seeds (circle 7.4 cm, 2.5% crashes, 10 cm arrival
  61%; TD3 4.9 cm, 0.8%, 84%), within the spread of TD3 seeds 74-77. Aggregators are compared against
  MOO sum (same decomposed critic), not TD3.

## 10. Lessons

- **Gate every reward change with `reward_check.py`**, with the final weights. It found the
  free-flight crash incentive and the waypoint leg cost in seconds.
- **The gate is necessary, not sufficient**: the progress reward passed it and still failed in
  training. Every reward change needs a training study.
- **Look per seed**: one collapsed seed made a pooled crash rate of 25.6%.
- **Score candidate rewards against the C++ reward** before trusting them: a missing weight in the
  candidate code made progress and Huber look 40x cheaper until the C++ run disagreed.
- **New options must leave old recipes bit-identical**, checked by training: an extra all-zero reward
  column in the replay buffer changed float32 rounding and the runs (fixed in `train.py`).
- **Four seeds are not enough** for differences of 1-2 cm or a few percent: time 0.5's lead halved
  when seeds 74-77 were added.
- **Check what a test measures**: the 5 cm arrival and the "9 cm offset" were mostly the 1.5-2.5 s
  legs and uneven motors, not the policies.

## 11. Next steps

1. Aggregators on the time 0.5 recipe, against MOO sum: DualProj, UPGrad, Aligned-MTL, SVD-PGA,
   OS-NMF-PGA (4 seeds each), with per-seed collapse checks; projection aggregators boosted the small
   objectives against the crash term in the time-to-target study.
2. More seeds for any difference that decides something.

## 12. Options added

All off by default; with none set, earlier recipes reproduce bit for bit (simulator checked against
the previous build over 64,000 steps, training checked at 100k and 200k steps).

| Option | Effect |
| --- | --- |
| `L2F_PROGRESS_K`, `L2F_PROGRESS_CAP` (0.2), `L2F_PROGRESS_GAMMA` (0.99) | Progress reward: flat position cost and no velocity cost beyond the cap, K (d_t - gamma d_t+1) per step, -K d_t on a crash. New reward part `progress` (position group in every MOO grouping). |
| `L2F_CRASH_PENALTY_OVERRIDE=P` | Crash penalty P instead of the build's `-DL2F_CRASH_PENALTY` (Python binding) |
| `L2F_WAYPOINT_S_MIN`, `L2F_WAYPOINT_S_MAX` (-1, 0.5) | Range of the second waypoint leg: it goes to s x the jump |
| `L2F_EVAL_LEG_S` (test only) | Waypoint test: both legs last this long; the episode runs past the 5 s step limit |
| `L2F_EVAL_RESET_INTEGRAL` (test only) | Waypoint test diagnostic: zero the position-error integral whenever the target moves |

The waypoint test also prints arrival within 10 cm with its time, and the signed final error along
the leg; its first two lines are unchanged.

Raw study outputs (`l2f/learning_to_fly/ablation_*`, ignored like `compare_*`) and checkpoints are on
the Mac that ran them.
