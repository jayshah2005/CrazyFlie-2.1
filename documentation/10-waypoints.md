<!-- group: Research | reviewed: 2026-11-26 -->
# Waypoints and time to target

The mission only ever moves the target smoothly (climb, circle at 0.38 m/s, descend). A
**waypoint** task asks for more: the target jumps, and the drone should get there fast, without
overshooting, without wasting energy, and without crashing. This is a natural multi-objective task
(time against precision, efficiency and safety), and it is where the project learned the most
about rewards.

## The waypoint episode

Episode type 9 (`L2F_P_WAYPOINT`, 20% of episodes when on):

1. Hover at the target for 0.3-1.0 s.
2. The target **jumps** $d_0 \sim U(\text{min}, \text{max})$ in a mostly horizontal direction
   (climb slope ±0.25). The far end of the range grows with the curriculum.
3. 1.5-2.5 s later it moves again, to $s \times$ the jump with $s \sim U(s_{\min}, s_{\max})$, so
   the second leg is $\lvert 1 - s \rvert\, d_0$ long.
4. The episode ends at 5 s. The crash limit on position is widened by twice the jump.

| Setting | First study (2026-11-23) | Current recipe |
| --- | --- | --- |
| Jump $d_0$ | 0.3-1.5 m (`L2F_WAYPOINT_MIN/MAX`) | **0.3-0.5 m** |
| Second leg | $s \in [-1, 0.5]$: 0.5-2 times the jump, up to 3 m | $s \in [0, 0.5]$: at most the first leg |
| Position cost | Huber 0.2 (after a flat cap and no cap) | Huber 0.2 |
| Time cost | 0.2-1.0 | 0.5 |
| Crash penalty | 200 | 400 |

Reward terms added for this task: time, power, speed limit (with a braking-distance curve), energy
per leg, progress ([Reward](06-reward.md#every-cost-term)).

## The first study: everything collapsed

The time-to-target study ran on the lab computer: waypoint episodes, four new reward terms, five
simulator options (drag, motor dead time, motor lag range, gain wander, ESC idle), seven
aggregators and 3M-step runs, all at once. Record:
[`time_to_target_study.md`](../l2f/docs/time_to_target_study.md).

| Run | Change | Result |
| --- | --- | --- |
| 1 | everything on | **every method and checkpoint crashed 100%**: the power cost had no floor and paid +1.07 per step for cutting the motors |
| 2 | power cost clamped at 0 | still collapsing: the 10 ms dead time with motor lag up to 0.15 s |
| 3 | lag ≤ 0.09 s | still collapsing: dead time made angular-acceleration costs about 20 times larger |
| 4 | no dead time; flat position cap 0.2 m | first healthy run (td3 32% waypoint crashes) but drones stopped about 30 cm off: no pull beyond the cap |
| 5 | no cap | worse (89-100% crashes) |
| 6 | Huber cap | pull restored, **waypoint crashes 72% (td3) to 95%** |
| objective study | 13 arms: ungrouped, regrouped, energy, input clips, jump ramp | no gain; ungrouped UPGrad and DualProj 100% crashes; every waypoint-trained arm crashed more at waypoints (65-100%) than **old3, never trained on waypoints (56%)** |

The crashes looked like loss of control: 75-95% were speed above 2.5 m/s (the study's speed
limit), about 1-1.5 s after the jump, about 1 m from the target, tilted 80-110°.

## Root cause: a leg cost more than a crash

The policies were not failing; they were doing what the reward asked. With the position weight at
its final 40 and the Huber cap, a target 1.5 m away costs $0.5 \cdot 40 \cdot (0.04 + 0.4 \cdot 1.3) = 11.2$
per step. A **perfectly flown** 1 m leg already costs more than the crash penalty of 200
(try it in the [crash-incentive calculator](06-reward.md#the-crash-incentive)), and the second leg
could be up to 3 m. Measured with the trained policies at the final weights, the discounted return
after a jump was:

| Policy | Episodes that survived | Return from the jump | Of which position | Crash penalty |
| --- | --- | --- | --- | --- |
| td3, seed 62, 3.0M | 225 | **−429** | −320 | −200 |
| regrouped sum, seed 61, 2.8M | 170 | **−441** | −282 | −200 |

Surviving cost more than twice as much as crashing, even for jumps under 0.6 m. Diving at the
target lowers the position cost on the way, and the speed limit was the nearest exit: the hard
tilt was deliberate acceleration into it. The runs make sense in this light: the flat cap (run 4)
made legs cheap and crashed least; no cap (run 5) made them expensive; the Huber cap (run 6)
restored the pull and the incentive. The curriculum hid it early: at weight 5 a 1.5 m leg costs
about −62, so policies first learned to fly legs and then unlearned it as the weight reached 40 at
1.2M steps. **No aggregator can fix this**: the reward's optimum itself is a crash.

The gradient norms were also checked: they stayed within about 2× over training and had no
non-finite values in about 60,000 logged updates, so exploding gradients were ruled out.

## What went wrong with the diagnosis

- **The crash-incentive check used the wrong weights.** The diagnostic script measured the
  per-step reward with a new simulator's *starting* weights (position 5), understating position
  costs by 8 times; "healthy per-step reward of −0.2 to −0.5" looked safe. `reward_check.py` now
  always uses the final weights.
- **Too many changes at once.** About ten differences separated the study's TD3 arm from old3,
  so no result could be attributed. The follow-up studies changed one thing at a time.
- **3M steps on a 2M schedule.** Exploration noise was down to about 0.007 by the end.
- **The test could not measure precision.** Legs ended 1.5-2.5 s after the jump, before slow
  policies settle, and arrival needed 5 cm.

## The follow-up: one change at a time

The reward studies of 2026-11-23 to 25 (Mac, 8 seeds, a reward check before each study) found a
waypoint task that trains ([reward_studies.md, Studies 3-4](../l2f/docs/reward_studies.md#6-study-3-waypoints-without-a-time-cost)):

- **Crash penalty 400** as the base (it halves the free-flight crash incentive).
- **Short jumps.** With crash 400, jumps up to 0.8 m still made crashing pay in 3-19% of waypoint
  states; 0.3-0.5 m passed. A new option keeps the second leg no longer than the first.
- **The Huber cap is needed:** without it the waypoint reward failed the check even with no
  time cost.

## The time cost

A cost of 1 per step until arrival (weight 0.5, so 0.25 per step after the scale), on top of crash
400, Huber 0.2 and 0.3-0.5 m waypoints. Medians over 8 seeds; mission 12 ms; waypoint legs of
3.5 s:

| Arm | Circle | Crashed | Crash-free checkpoints | Within 5 cm | Within 10 cm | Final error | 0.3-0.8 m jumps: 10 cm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crash 400, no waypoint training (4 seeds) | 5.3 cm | 0.4% | 22/36 | 27.0% | 81.5% | 8.7 cm | 62.5% |
| waypoints, no time cost | 7.1 cm | 3.3% | 16/72 | 23.0% | 68.0% | 10.2 cm | 57.0% |
| + time 0.2 | 7.5 cm | 3.9% | 11/72 | 16.5% | 61.5% | 11.4 cm | 49.0% |
| **+ time 0.5** | 6.0 cm | 1.4% | 18/72 | 30.0% | 79.5% | 8.9 cm | **69.0%** |

- Time 0.5 is the best waypoint-trained arm; time 0.2 is consistently worse than none (not
  explained).
- **Mission-only policies fly short waypoints about as well** and are better at the mission.
  Waypoint training helps only on larger jumps (0.8 m: 69% against 62.5% within 10 cm).
- On seeds 70-73 alone time 0.5 looked clearly better; seeds 74-77 halved the lead. Four seeds
  are not enough for these differences.
- Time 0.5 became the base recipe of the MOO studies ("the time 0.5 recipe").

## Why every policy stops about 9 cm off

It looked like a fixed offset. Diagnostics in the waypoint test (crash 400 and time 0.5 policies,
seeds 70-73, 2.0M):

| Test | Within 5 cm | Final error | Along the leg (> 0: past the target) |
| --- | --- | --- | --- |
| 3.5 s legs, uneven motors | 16-44% | 7.0-13.2 cm | +3.4 to +9.6 cm |
| 3.5 s legs, clean motors | 18-67% | 5.0-12.3 cm | +1.5 to +7.0 cm |
| clean, integral reset at each jump | 3-50% | 5.4-15.0 cm | +1.9 to +6.3 cm |
| 6 s legs, clean motors | 58-97% | 3.6-8.7 cm | about 0 |

The drones overshoot a little and creep back over several seconds; uneven motors add 1-3 cm. It
is **slow settling**, not an offset, and it is the real target for a time objective. The
position-error integral is not the cause: resetting it makes every policy worse.

## Real waypoint flights (2026-11-25)

`rl_trigger.py --mode mission --waypoints "0.4,0;0.1,0" --dwell 3.5` (0.5 m test) and
`"0.7,0;0.2,0"` (0.8 m test). On the 0.7 m jump variant E never got within 10 cm (overshot to
0.86 m, then oscillated ±15 cm at about 1 Hz); every MOO pick reached 10 cm in 1.3-1.5 s, SVD-PGA
settled closest (about 8 cm) and OS-NMF-PGA smoothest (12-18 cm long, no oscillation). Waypoint
missions raise the onboard position-error clip `rlt.pdlp` from 0.5 to 2.0 m, so a 0.7 m jump
reaches the policy unclipped, as in training; the flight notes list this as a possible, untested
link to that day's telemetry drop-outs ([Flights](11-flight.md#2026-11-25-seven-methods-28-flights)).

## Open questions

- Settling takes 3-6 s to reach 5 cm: what reward or input would make it faster?
- Why is a time cost of 0.2 worse than none?
- Larger jumps need a reward whose leg cost stays well below the crash penalty; the progress
  reward did that on paper but failed in training ([Reward](06-reward.md#progress-reward-potential-based-shaping)).
- A safer alternative for flights: send a smoothly moving target at a chosen speed instead of a
  jump; the policies already track moving targets well.
