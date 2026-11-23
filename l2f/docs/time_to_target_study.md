# Time-to-target and objective studies (2026-11-22 to 2026-11-23)

This file records what was built, what was trained, what was observed, and what went wrong in
the time-to-target multi-objective (MOO) work on branch `feat/moo`. No policy from these studies
is good enough to fly. The main finding is a defect in the waypoint reward (section 5): flying a
waypoint leg costs more than crashing, so every method learned to crash.

All training ran on the lab computer (14 cores, Linux, PyTorch trainer `l2f/py/train.py`, build
`build_py_E_ttt`). Study scripts and diagnostics are in `~/l2f-runs/` on the lab computer, and
outputs are in `~/crazyflie/l2f/learning_to_fly/compare_ttt*` and `compare_obj` (section 8).

## Contents

1. [Goal](#1-goal)
2. [What was built](#2-what-was-built)
3. [Common setup and metrics](#3-common-setup-and-metrics)
4. [Training rounds and observations](#4-training-rounds-and-observations)
5. [Root cause: in waypoint episodes a crash is cheaper than flying](#5-root-cause-in-waypoint-episodes-a-crash-is-cheaper-than-flying)
6. [Other issues found in the review](#6-other-issues-found-in-the-review)
7. [Recommendations](#7-recommendations)
8. [Where everything is](#8-where-everything-is)

---

## 1. Goal

Before this work, the policy was a hover / circle / landing controller (variant E, 25 inputs,
TD3). The best reference is **old3**: `policies/2026-11-22_old3_moosum3_s40_1800k`, MOO sum on 3
groups, old (truth) observation path.

The new goal was a policy that flies **from point to point as fast as it can while staying
precise, efficient and safe**. The plan had four parts:

- Train with four objectives: **time**, **precision**, **efficiency**, **safety**.
- Combine their gradients with TorchJD aggregators (sum, DualProj, UPGrad, PCGrad, Aligned-MTL,
  SVD-PGA, OS-NMF-PGA), and compare with plain TD3.
- Add sim-to-real options that came from the 2026-11-22 flights: drag, motor lag range, gain
  wander, ESC idle, actuator dead time.
- Check whether the objectives conflict enough to need an aggregator at all.

## 2. What was built

Commits on `feat/moo`:

| Commit | What |
| --- | --- |
| `f3cb6db` | Waypoint episodes, time-to-target reward terms, sim-to-real options |
| `3f8dd03` | Waypoint mission mode in `rl_trigger.py` (`--waypoints`) |
| `68f386f` | Fix: power cost clamped at ≥ 0; Huber position cap |
| `9a3a53d` | Ungrouped MOO mode (`--parts 0`): one objective per reward part |
| `d0df32d` | Objective-study options: energy per leg, braking-distance speed limit, input clips (A), jump curriculum (B), regrouping |

Every option is off by default. With none set, older recipes reproduce exactly.

### 2.1 Waypoint episode

Episode type 9, `traj_type` 6, `operations_generic.h`. It is drawn with probability
`L2F_P_WAYPOINT` (0.2 in all studies). The target is the origin of the observed position, so the
observed position p is the drone's error to the target.

1. Hover at the target for 0.3–1.0 s (`traj[3]`).
2. The target **jumps** by d₀ ~ U(min, min + k·(max − min)) m, with min 0.3 and max 1.5
   (`L2F_WAYPOINT_MIN/MAX`), and k = the mission curriculum progress (or the B ramp). The heading
   is uniform; the climb slope is U(−0.25, 0.25).
3. 1.5–2.5 s later (`traj[4]`) the target moves to traj[5] × the jump, with traj[5] ~ U(−1, 0.5).
   So the **second leg** is |1 − traj[5]| · d₀, which is between 0.5 d₀ and 2 d₀ (up to 3 m).
4. The episode ends at 500 steps (5 s). The position limit for a crash is widened to
   0.6 m + 2·|traj_i| per axis.

### 2.2 New reward terms

From `squared.h`, `time_to_target_terms`. Here *static target* means the target velocity is 0.

| Part | Cost per step | Option |
| --- | --- | --- |
| time | 1 while a static target is not reached (not within `L2F_ARRIVE_POS` 0.05 m and slower than `L2F_ARRIVE_SPEED` 0.2 m/s) | `L2F_W_TIME` |
| power | max(0, Σᵢ ρᵢ³ − 4h³), with ρᵢ = rpmᵢ / 24000 and h = hover rpm / 24000 = 0.6445, so 4h³ = 1.07 | `L2F_W_POWER` |
| speed_limit | (v − v_allowed)² when the speed v is above v_allowed = min(speed_cap, max(0.3, √(2·a_b·d))), with speed_cap 1.5 m/s, a_b = `L2F_BRAKE_ACC`, d = distance to the target | `L2F_W_SPEED`, `L2F_SPEED_CAP`, `L2F_BRAKE_ACC` |
| energy | Σᵢ ρᵢ³ (hover included) while a static target is not reached ("energy per leg") | `L2F_W_ENERGY` |
| position (changed) | Huber with c = `L2F_POS_CAP`: d² for d ≤ c, c² + 2c(d − c) beyond; velocity cost 0 beyond c | `L2F_POS_CAP` |

`L2F_SPEED_CRASH` (2.5 m/s) ends the episode as a crash when the speed is above it.

### 2.3 Sim-to-real options

| Option | Effect |
| --- | --- |
| `L2F_DRAG_MAX` | linear drag U(0, max) N·s/m per episode |
| `L2F_ACT_DELAY_P` | probability of a 10 ms actuator dead time |
| `L2F_MOTOR_TAU_MIN/MAX` | absolute motor time-constant range (s) |
| `L2F_GAIN_WALK` | AR(1) common motor gain wander (std; correlation time 0.3 s) |
| `L2F_MOTOR_IDLE_RPM`, `L2F_MOTOR_GAIN_ALL` | ESC idle and common motor gain (from earlier work) |
| `L2F_OBS_POS_CLIP`, `L2F_OBS_VEL_CLIP` | **A**: the firmware's input clips (`rlt.pdlp` 0.5 m, `rlt.vdlp` 2 m/s); the integral adds up the clipped error, as the firmware does |
| `L2F_WAYPOINT_RAMP` | **B**: the jump size grows over this many steps (`train.py` calls `set_waypoint_progress`) |

### 2.4 MOO groupings

From `l2f/py/moo.py`.

| `--parts` / `--grouping` | Objectives |
| --- | --- |
| 4 (default) | time; precision (position, velocity); efficiency (action, action rate, angular acceleration, power, energy); safety (survival, orientation, landing speed, speed limit) |
| 0 (ungrouped) | one objective per weighted reward part (11 objectives) |
| 4 `regroup` | crash (survival); progress (position, velocity, time, energy); effort (action, action rate, angular acceleration, power); limits (orientation, speed limit, landing speed) |
| 5 `regroup` | as 4 `regroup`, with energy as a fifth objective |

"Survival" is the terminal reward: the crash penalty (−200), or the landing payout.

### 2.5 Evaluation added

- `evaluate_robustness <actor> N waypoint` runs waypoint episodes only. It reports:
  - crashes;
  - arrival per leg (within 5 cm and below 0.2 m/s);
  - time to target;
  - overshoot along the leg;
  - final error (mean over the last 0.3 s of each leg);
  - peak speed;
  - mean rotor power above hover.
- The mission test (`… mission`) is unchanged: circle error, crashes, jitter.

## 3. Common setup and metrics

Every study used:

- Variant E, old (truth) observation path.
- Build flags as old3: brushless, real noise, angular-acceleration weight 2e-4, landing-speed
  weight 5, position integral, mission, RPM observation, no history, crash penalty 200, survival
  bonus 0, action rate 1.
- Runtime options: `L2F_OMEGA_CRASH=34.9 L2F_GYRO_CLIP=34.9 L2F_MOTOR_GAIN_ALL=0.1
  L2F_MOTOR_IDLE_RPM=2500 L2F_DELAY_MAX_MS=20`.
- 3M training steps (the validated recipe is 2M).
- Seeds 50–53 (runs 1–6) or 60–63 (objective study).
- Waypoint and mission tests of the last 6 checkpoints (2.5M–3.0M), 100 episodes each.

The tests used the training options plus the mission-test settings (0.75 m, 0.6 m circle in
10 s, randomization, per-motor mismatch, 12 ms delay).

Table columns:

| Column | Meaning |
| --- | --- |
| wp crash | % of waypoint episodes that crashed |
| arr | % of legs that arrived (5 cm, 0.2 m/s) |
| fin | final error of a leg (cm) |
| mi crash | % of mission episodes that crashed |
| circle | mission circle tracking error (cm) |
| jitter | motor-command spread in hover (lower is smoother) |

Medians are over seeds × checkpoints.

**old3 under the same tests** (measured 2026-11-23 with the objective study's exact settings):
waypoint crash 56%, arrival 5%, final error 30 cm, **mission crash 0%, circle 7.3 cm, jitter
0.058**. old3 was never trained on waypoints.

## 4. Training rounds and observations

### Run 1 (`compare_ttt`): every method collapsed

Settings:
- `L2F_W_TIME=1 L2F_W_POWER=2 L2F_W_SPEED=20 L2F_POS_CAP=0.2` (flat cap) `L2F_SPEED_CRASH=2.5`;
- dead time on (`L2F_ACT_DELAY_P=1`), motor lag 0.045–0.15 s, drag 0.03, gain walk 0.03.

Methods: td3, sum, DualProj, PCGrad, Aligned-MTL, SVD-PGA, OS-NMF-PGA.

**Result:** 100% crashes in both tests for every method and checkpoint; circle error 184–249 cm.

**Cause:** the power cost was Σρ³ − 4h³ without a lower bound. At idle (ρ ≈ 0.1) that is about
−1.07, so at weight 2 and scale 0.5 the reward was **+1.07 per step for cutting the motors**.
That was larger than the time cost (−0.5 per step). Falling was paid for, and the crash came
after the payment.

**Fix** (`68f386f`): power cost = max(0, …). `L2F_W_TIME` lowered to 0.2.

### Run 2 (`compare_ttt2`): still collapsing; stopped

Leave-one-out rollouts with old3 traced it to the dead time (`ACT_DELAY_P=1`) together with motor
lag up to 0.15 s. Motor lag max lowered to 0.09 s.

### Run 3 (`compare_ttt3`): still collapsing; stopped

With the dead time on, old3's mean reward per step in the training environment was −6.7, against
−0.22 without it. The angular-acceleration cost was about 20× larger. The dead time was removed
from the recipe.

> Caveat: these per-step numbers came from `diag/rewards.py`, which used the **initial**
> curriculum weights (position 5, velocity 0.01, action 0.01), not the final ones (40, 1, 1).
> See section 6.1. The ratio still points at the dead time, but the absolute values are too low.

### Run 4 (`compare_ttt4`): first healthy run; flat position cap

Settings: run 1 without dead time, motor lag ≤ 0.09 s, `W_TIME=0.2`, **flat** cap (position cost
= min(d², 0.04)).

| Method | wp crash | arr | fin | mi crash | circle | jitter |
| --- | --- | --- | --- | --- | --- | --- |
| sum | 28.7 | 1.0 | 36.2 | 20.7 | 23.9 | 0.091 |
| td3 | 31.6 | 1.3 | 32.4 | 19.0 | 19.5 | 0.106 |
| DualProj | 62.1 | 0.8 | 30.1 | 31.7 | 64.6 | 0.109 |
| SVD-PGA | 62.2 | 1.1 | 30.9 | 45.6 | 68.2 | 0.127 |
| OS-NMF-PGA | 75.4 | 0.6 | 32.1 | 55.8 | 70.1 | 0.144 |
| Aligned-MTL | 89.2 | 1.1 | 21.5 | 72.1 | 124.8 | 0.158 |
| PCGrad | 100 | 0 | – | 100 | 197.7 | 0.128 |

Observation: the fewest crashes so far, but **the drones stopped about 30 cm from the target**.
A flat cap gives no gradient beyond 0.2 m, so nothing pulls the drone in from there. (In
hindsight this run crashed least because a leg was cheap; see section 5.)

### Run 5 (`compare_ttt5`): no cap; unstable

Plain d² position cost.

| Method | wp crash | mi crash | circle |
| --- | --- | --- | --- |
| td3 | 88.8 | 63.2 | 85.0 |
| SVD-PGA | 92.3 | 84.4 | 211.1 |
| sum | 94.9 | 58.6 | 85.4 |
| DualProj | 100 | 99.7 | 195.4 |

A 1.5 m jump costs 2.25 × 40 × 0.5 = 45 per step without a cap. Everything got worse.

### Run 6 (`compare_ttt6`): Huber cap

Position cost quadratic within 0.2 m and linear beyond it (`68f386f`); `W_TIME=0.5`.

| Method | wp crash | arr | fin | mi crash | circle | jitter |
| --- | --- | --- | --- | --- | --- | --- |
| td3 | 71.6 | 3.0 | 21.7 | 12.1 | 20.5 | 0.154 |
| sum | 80.3 | 0.9 | 24.4 | 26.0 | 34.9 | 0.156 |
| SVD-PGA | 85.0 | 2.0 | 24.7 | 38.2 | 43.3 | 0.151 |
| DualProj | 94.9 | 0.4 | 9.2 | 74.8 | 149.7 | 0.152 |

Best single checkpoint: td3 s51 2.6M, with mission crash 0%, circle 10.3 cm, jitter 0.139. That
is still behind old3 (7.3 cm, 0.058).

Observation: the pull toward the target is back, and **waypoint crashes rose from about 30% to
72%**.

### Crash causes (run 6 policies, `diag/wp_crash.py`)

85–100% of waypoint crashes are **speed above 2.5 m/s**:

- about 1–1.5 s after the jump;
- about 1 m from the target;
- with the drone tilted 64–130°.

old3 crashes differently, from spin and the position limit.

### Conflict study (`compare_conflict`): are the objectives in conflict?

Ungrouped (11 objectives), `--agg sum`, run-6 settings, seeds 50–53, `diag/conflict.py`, rows
after 1M steps.

Two measures:
- cancellation = ‖Σg‖ / Σ‖g‖ (1 = all gradients agree, 0 = they cancel);
- cosine = pairwise cosine between objective gradients.

| Pair | Mean cosine | % of updates negative |
| --- | --- | --- |
| survival – action_rate | −0.66 | 92% |
| survival – time | −0.54 | 85% |
| survival – power | −0.54 | 86% |
| survival – angular_acceleration | −0.49 | 83% |
| action_rate – speed_limit | −0.43 | 79% |
| survival – position | −0.40 | 79% |

- Cancellation: 0.47. Gradient norms: survival 2.84, position 1.9, the others 0.2–0.6.
- Grouped (run 6 sum, 4 groups): cancellation 0.49; precision–safety −0.35; time–safety −0.22;
  **time–precision +0.86**, so time added nothing that precision did not already give.
- old3's recipe: cancellation 0.62, every pair positive on average.

Observation: **the real conflict is crash avoidance against the dense costs.** The 4-group
split hid it inside "safety". Eval of this run: waypoint crash 85%, mission crash 35%.

### Objective study (`compare_obj`): regrouping, energy, A, B

Settings: run-6 settings plus the braking-distance speed limit (`L2F_BRAKE_ACC=2`). Seeds 60–63,
13 arms, finished 12:10.

Arm settings:
- TP = `W_TIME=0.5 W_POWER=2`;
- EN = `W_ENERGY=0.5` (energy per leg replaces time and power);
- A = input clips 0.5 m / 2 m/s;
- B = jump ramp over 2M steps.

All MOO arms use A + B.

| Arm | wp crash | arr | fin | mi crash | circle | jitter |
| --- | --- | --- | --- | --- | --- | --- |
| td3, none (TP) | 72.2 | 3.5 | 25.1 | **12.6** | **24.6** | 0.147 |
| td3 + A | 71.8 | 1.5 | 31.7 | 23.8 | 35.3 | 0.173 |
| td3 + B | 66.5 | 1.4 | 26.1 | 18.7 | 31.8 | 0.155 |
| td3 + A + B | **65.0** | 0.8 | 30.6 | 31.8 | 37.4 | 0.161 |
| ungrouped sum | 77.0 | 0.5 | 33.5 | 39.2 | 47.4 | 0.175 |
| ungrouped DualProj | 100 | 0 | – | 99.0 | 235.8 | 0.133 |
| ungrouped UPGrad | 100 | 0 | – | 100 | 240.9 | 0.125 |
| regroup sum | 82.1 | 1.8 | 26.8 | 39.4 | 59.2 | 0.151 |
| regroup DualProj | 84.2 | 1.6 | 30.6 | 36.7 | 50.2 | 0.127 |
| regroup UPGrad | 94.8 | 2.5 | 12.7 | 77.2 | 195.7 | 0.139 |
| energy sum | 85.1 | 2.5 | 29.3 | 26.4 | 37.3 | 0.164 |
| energy DualProj | 86.9 | 1.1 | 27.3 | 25.2 | 36.2 | 0.149 |
| energy UPGrad | 93.9 | 0.1 | 14.7 | 79.1 | 227.3 | 0.124 |
| **old3 (reference)** | 56.0 | 5.0 | 30.4 | **0.0** | **7.3** | 0.058 |

Best single checkpoints:

| Checkpoint | wp crash | mi crash | circle | jitter |
| --- | --- | --- | --- | --- |
| regroup sum s61 2.8M | 19% | 1% | 14.7 cm | 0.207 |
| td3 + A + B s61 2.6M | 22% | 9% | 15.1 cm | – |
| td3 none s62 3.0M | 51% | 0% | 10.9 cm | – |

Observations:

- **Ungrouped + UPGrad / DualProj collapse to 100% crashes.** Under UPGrad, survival against
  action rate has cosine −0.95 (negative in 100% of updates). Action rate points against the
  combined step in 100% of updates. The survival gradient norm rises to 10.4, against 2.84 under
  sum. The aggregator scales up small costs that oppose the crash penalty, which matches the
  earlier PCGrad / UPGrad findings in `results.md`.
- **Regrouped:** one real conflict, **crash vs. progress** (cosine −0.52 under sum, −0.65 under
  DualProj, negative in 86–93% of updates). DualProj only ties with sum; UPGrad collapses.
- **Energy per leg: no real gain.** Compared with regroup sum, mission crashes fell (39% → 26%)
  but waypoint crashes rose slightly (82% → 85%). Energy conflicts with crash (cosine −0.55
  under DualProj). Every MOO arm is behind plain td3.
- **A and B:** B removes about 6 points of waypoint crashes but adds mission crashes. A does not
  help waypoints, roughly doubles mission crashes, and adds about 6 cm of final error. Neither is
  a clear win.
- **Crash causes are unchanged** in the best checkpoints. 75–95% are speed above 2.5 m/s at about
  1.0–1.5 s, about 1 m out, tilted 80–110°. The braking-distance speed cost did not stop it.
- **Every waypoint-trained arm crashes more in waypoint tests than old3 (56%)**, which never saw
  a waypoint. Waypoint training made the policies worse at waypoints.

### Gradient-explosion check (`diag/gradnorm.py`)

All 36 MOO runs of the objective study, actor gradient every 50 updates, 0–3M steps:

| Arm | Median actor step per 500k window | Largest step | Median largest objective norm |
| --- | --- | --- | --- |
| sum | 1.2–4.8 | about 15 | 1.5–9.6 |
| DualProj | 0.9–6.7 | about 18 | 2–21 |
| UPGrad | 2.3–7.3 | about 20 | 2.7–25 |

There were no non-finite values in about 60,000 logged updates. The norms rise about 2× while the
weight curriculum raises the weights, then flatten. **The actor gradients are not exploding.**
Adam also normalizes each weight's step, so a large gradient changes the direction, not the step
size. UPGrad's larger norms come from its reweighting, not from instability.

Not yet measured: the critic's loss and Q values are not logged, so critic divergence is not
ruled out. Steady returns and episode lengths in the sum and td3 arms make it unlikely.

## 5. Root cause: in waypoint episodes a crash is cheaper than flying

### 5.1 How the reward values a crash

The reward of each step (`brushless.h`, `squared.h`) is

  rₜ = −s · Σₖ wₖ · costₖ(sₜ, aₜ),  s = 0.5

There is **no survival bonus** (`L2F_SURVIVAL_BONUS=0`), so every rₜ ≤ 0. On a crash step the
reward is −P with P = 200 (`L2F_CRASH_PENALTY`), and the episode ends: no further costs. TD3
maximizes G = Σₜ γᵗ rₜ with γ = 0.99, a horizon of about 1/(1 − γ) = 100 steps = 1 s.

From any state the policy compares:

  keep flying: G_fly = Σ γᵗ rₜ  (all costs still ahead)
  crash now:  G_crash ≈ −200

**Crashing is optimal whenever G_fly < −200**, that is, whenever the discounted costs still ahead
add up to more than the crash penalty. Being alive is worth nothing in itself; the −200 is the
only reason to survive.

In missions this never happens. Hovering 5 cm off target costs 0.5 × 40 × 0.05² = 0.05 per step,
about 5 over the horizon.

### 5.2 The position cost far from the target

With c = 0.2 m (`L2F_POS_CAP`) and d = the distance to the target (m), the Huber cost is

  cost_pos(d) = d²         (d ≤ c)
  cost_pos(d) = c² + 2c(d − c)  (d > c; slope 2c = 0.4 per metre)

The position weight w_pos starts at 5 and is multiplied by 1.2 every 100k steps, up to 40
(`td3.py` `weight_caps`; target value in `brushless.h`). It reaches 40 at about 1.2M steps and
stays there for the rest of the 3M-step run. For d > 0.2 m, the reward from position alone is

  r_pos = −0.5 × 40 × (0.4 d − 0.04) = −(8d − 0.8) per step

| d | Reward per step | Over 1 s at that distance |
| --- | --- | --- |
| 0.05 m (hover) | −0.05 | −5 |
| 0.5 m | −3.2 | −320 |
| 1.0 m | −7.2 | −720 |
| 1.5 m | −11.2 | −1120 |
| 3.0 m | −23.2 | −2320 |

When the target jumps, the drone is suddenly 0.3–1.5 m away, and every step costs 2–11 until it
arrives.

### 5.3 The cost of one leg, flown perfectly

The flight profile:
- accelerate at 5 m/s²;
- cruise at the speed cap, 1.5 m/s;
- brake at 2 m/s² along v = √(2 · 2 · d), the fastest flight with no speed-limit penalty.

| Leg D | Flight time | Position cost, w = 40 | Position cost, w = 5 | Position cost, no cap | Time cost (0.25/step) |
| --- | --- | --- | --- | --- | --- |
| 0.3 m | 0.55 s | −30 | −4 | −32 | −11 |
| 0.5 m | 0.74 s | −81 | −10 | −111 | −13 |
| 1.0 m | 1.09 s | **−258** | −32 | −596 | −17 |
| 1.5 m | 1.42 s | **−496** | −62 | −1633 | −19 |
| 2.0 m | 1.76 s | −780 | −98 | −3353 | −21 |
| 3.0 m | 2.42 s | **−1438** | −180 | −9177 | −23 |

At the final weight, **a leg longer than about 0.9 m costs more than a crash even when flown
perfectly**. Speed limit, angular acceleration, action, action rate and power add more on top.

Each episode also has a second leg of up to 3 m, 1.5–2.5 s after the first. Discounted by about
γ²⁰⁰ ≈ 0.13, a 3 m second leg still adds about −190 to the value right after the first jump.

### 5.4 Measured

`diag/leg_cost.py`: trained policies, final weights (position 40, velocity 1, action 1), full jump
sizes. G is the discounted return from the jump to the end of the episode.

| | td3 none s62 3.0M | regroup sum s61 2.8M |
| --- | --- | --- |
| survived: episodes, mean jump | 225, 0.66 m | 170, 0.57 m |
| **survived: G** | **−429** (p10 −635, p90 −227) | **−441** (p10 −657, p90 −270) |
| of which position | −320 | −282 |
| survived, jump < 0.6 m | −316 | −359 |
| survived, jump 0.6–1.0 m | −471 | −545 |
| survived, jump ≥ 1.0 m | −676 | −620 |
| crashed: G (crash included) | −520 | −508 |
| crash penalty | −200 | −200 |

**Surviving a waypoint episode costs more than twice the crash penalty**, even for short jumps.
The crashing policies are not failing at the task; they are doing what the reward asks.

### 5.5 How this explains every observation

- **Speed crash, about 1 m out, tilted 80–110°.** Diving at the target lowers the position cost
  on the way. The 2.5 m/s speed limit is the nearest exit: a hard tilt reaches it in about 1 s,
  while the position limit is 0.6 m + 2|jump| away. At 1 m out and 2.5 m/s, braking would still
  cost the rest of leg 1 plus all of leg 2, more than 200. The tilt is deliberate acceleration,
  not a loss of control.
- **Run 4 (flat cap) crashed least.** Each step cost at most 0.5 × 40 × 0.04 = 0.8, so a leg cost
  well under 200. With no pull beyond 0.2 m, though, the drones stayed about 30 cm off.
- **Run 5 (no cap) was worst.** A 1.5 m leg costs about −1600.
- **Run 6 and the objective study (Huber).** The pull came back, and with it a leg cost of
  −250 to −500: waypoint crashes rose to 65–100%.
- **The crashes appear late in training.** At w_pos = 5 a 1.5 m leg costs −62 (below 200), so the
  policy first learns to fly legs. As the weight climbs toward 40 (by 1.2M steps), crashing
  becomes optimal, and episodes drift shorter.
- **Mission performance gets worse too.** The actor and critic are shared across episode types.
  Mission crashes rose from 0% (old3) to 12–40%.
- **No aggregator helps.** The reward's optimum itself is a crash. Sum, DualProj and UPGrad all
  climb toward it. UPGrad, which boosts the objectives in conflict, only amplifies the tension.
  The measured crash-vs-progress conflict (cosine −0.5 to −0.65) is this incentive seen in the
  gradients.

### 5.6 The underlying tension

With a cost on distance, one number, the slope, sets two things:

- the **pull** toward the target (what a metre closer is worth), and
- the **total cost of a leg** ≈ slope × mean distance × flight time.

A strong pull therefore means an expensive leg. The flat cap gave a cheap leg with no pull; the
Huber cap gave pull with an expensive leg. No slope gives both.

A **progress reward** (potential-based shaping) separates them. With Φ(s) = −k·d:

  r_prog = γ Φ(sₜ₊₁) − Φ(sₜ) = k (dₜ − γ dₜ₊₁)

- Each metre of approach is worth k, so the pull is k.
- Over a leg the sum telescopes to about k·d₀, **however long the flight takes**. With
  k = 20 /m, even a 3 m leg costs at most about 60.
- This kind of shaping leaves the optimal policy unchanged (Ng, Harada & Russell, 1999).
- The time cost (0.25 per step) then provides the pressure to arrive quickly.
- The quadratic cost within 0.2 m stays, for precision at the target.
- On a crash step, −k·d must be added. Otherwise ending the episode refunds the remaining
  distance, which would be a new reward for crashing.

## 6. Other issues found in the review

### 6.1 The crash-incentive diagnostic used the wrong weights

`~/l2f-runs/diag/rewards.py` measured the per-step reward with a new `Sim`'s **initial** weights
(position 5, linear velocity 0.01, action 0.01) instead of the final ones (40, 1, 1). That
understates position costs by 8×. The "healthy per-step reward of −0.2 to −0.5" used as the
crash-incentive check during runs 2–6 was therefore too optimistic, which is why the check never
caught the problem in section 5. (Its printed per-part weights were also wrong: it read the
weight array as a dictionary and printed 1 for every part.)

### 6.2 Confounded comparison with old3

td3 none (12.6% mission crashes, 24.6 cm) and old3 (0%, 7.3 cm) differ in about ten things:

- waypoint episodes;
- drag, motor lag 0.045–0.09 s, gain wander, braking speed limit;
- the time and power terms;
- 3M steps instead of 2M.

No run measures how much circle precision the sim-to-real options cost on their own.

### 6.3 3M steps on a 2M schedule

Exploration and target noise are multiplied by 0.9 every 100k steps from 500k. At 3M the
exploration noise is 0.1 × 0.9²⁶ ≈ 0.007. The recipe was validated at 2M only.

### 6.4 The waypoint test cannot measure precision

- The first leg ends 1.5–2.5 s after the jump, which is about the time a 1.5 m leg needs at the
  braking-limited speed.
- Arrival needs 5 cm and 0.2 m/s.

So "arrival 0–5%" and "final error 25–30 cm" partly measure the test design: old3 also scores
30 cm final error.

### 6.5 Reproducibility

- The diagnostics `rollout.py`, `rewards.py`, `wp_crash.py`, `conflict.py`, `summ.py`,
  `check_new.py`, `leg_cost.py` and `gradnorm.py` exist only on the lab computer
  (`~/l2f-runs/diag/`). They are not in the repository and do not use `run_log`.
- The study scripts (`~/l2f-runs/run_*.sh`) are also lab-only.

### 6.6 Waypoint flights

`rl_trigger.py --waypoints` sets `rlt.pdlp` to 2.0. That matches policies trained without A. A
policy trained with A needs 0.5.

## 7. Recommendations

In order:

1. **Fix the waypoint reward first.** Replace the far-field position cost with the progress term
   of section 5.6, including the −k·d on crash steps. A smaller alternative is a lower Huber slope
   beyond the cap plus a second leg of at most about 1 m. Add both as options, off by default.
2. **Gate every study on the leg cost.** Before training, run `leg_cost.py` (final weights) for
   every episode type. The surviving discounted return must be well above −200.
3. **Fix the tools.** Make `rewards.py` use the final weights. Move the diagnostics into the
   repository with `run_log`. Lengthen the waypoint test legs to 3–4 s and add a 10 cm arrival
   threshold.
4. **One ablation study** (td3 only, 4 seeds, 2M steps), to separate the cost of the sim options
   from the cost of the waypoints:
   - (a) old3's recipe plus the sim-to-real options, no waypoints;
   - (b) (a) plus the fixed waypoints.
5. **Pause the aggregator search** until step 4 is done, then measure conflict once on the fixed
   reward. If crash vs. progress is no longer strongly negative, MOO adds nothing for this task.
   The finding so far still stands: projection-based aggregators (UPGrad, PCGrad) amplify small
   objectives that oppose a large one.
6. **Fallback for flights:** don't send step targets. Let `rl_trigger.py` move the target smoothly
   at a chosen speed. old3 already tracks moving targets (0% crashes, 7.3 cm), so time to target
   becomes a choice of path speed.
7. Optional: log the critic's loss, mean Q against the actual return, and the share of saturated
   actor outputs, to rule out critic divergence.

## 8. Where everything is

On the lab computer (`ssh lab`):

| What | Path |
| --- | --- |
| Study scripts | `~/l2f-runs/run_ttt.sh` (run 1), `run_ttt2.sh` … `run_ttt6.sh`, `run_conflict.sh`, `run_obj.sh`, `run_old3.sh` |
| Diagnostics | `~/l2f-runs/diag/`: `rollout.py`, `rewards.py`, `wp_crash.py`, `conflict.py`, `summ.py`, `check_new.py`, `leg_cost.py`, `gradnorm.py` |
| Training output and evals | `~/crazyflie/l2f/learning_to_fly/compare_ttt`, `compare_ttt2` … `compare_ttt6`, `compare_conflict`, `compare_obj` (each with `eval/`) |
| Checkpoints | `~/crazyflie/l2f/learning_to_fly/checkpoints/multirotor_td3/<run>/actor_*.h5` (run name in the first `run …` line of each `<arm>_s<seed>.txt`) |
| MOO gradient logs | `~/crazyflie/logs/train_moo_py_E_moo{0,4,4r,5r}_…_s{50..53,60..63}_*.csv` |
| Build | `~/crazyflie/l2f/learning_to_fly/build_py_E_ttt` |

Commands used:

```
python ~/l2f-runs/diag/summ.py l2f/learning_to_fly/compare_obj/eval        # eval table
python ~/l2f-runs/diag/conflict.py "logs/train_moo_py_E_moo4r_sum_*s6[0-3]_*.csv"
PYTHONPATH=l2f/learning_to_fly/build_py_E_ttt/src env <training L2F_* options> \
  python ~/l2f-runs/diag/wp_crash.py <actor.h5>                           # crash causes
PYTHONPATH=… env <options> python ~/l2f-runs/diag/leg_cost.py <actor.h5>  # section 5.4
python ~/l2f-runs/diag/gradnorm.py "logs/train_moo_py_E_moo0_*s6[0-3]_*.csv"
```
