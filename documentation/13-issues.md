<!-- group: Results | reviewed: 2026-11-26 -->
# Issues and open questions

What is broken, unexplained, untested or easy to get wrong, as of 2026-11-26. Each item says why
it matters, the evidence, and a suggested next step. Status words: **open** (no fix yet),
**partly fixed**, **untested** (a hypothesis), **watch** (fixed, but easy to break again).

## At a glance

| ID | Area | Issue | Status |
| --- | --- | --- | --- |
| I01 | evidence | Only the jitter gain survives a multiple-comparison correction | open |
| I02 | MOO | Aggregation replacing reward tuning is not shown | untested |
| I03 | MOO | SVD-PGA's gain may be just a reweighting | untested |
| I04 | MOO | Split critics have 4× the critic parameters | untested |
| I05 | MOO | The sum crashes more on split critics than TD3 | open |
| I06 | MOO | A shared critic with per-head normalization was never tried | untested |
| I07 | reward | Free-flight crash incentive is reduced, not gone (3-7%) | partly fixed |
| I08 | reward | The progress reward failed; not diagnosed | open |
| I09 | reward | Time cost 0.2 is worse than none | open |
| I10 | waypoints | Settling after a jump takes 3-6 s | open |
| I11 | sim-to-real | Real jitter is about 2.5× the simulated one for most policies | open |
| I12 | sim-to-real | Sensor-path policies shook; calibration incomplete | partly fixed |
| I13 | sim-to-real | Drag and ground effect not identified | open |
| I14 | flight | Aligned-MTL crashed in flight | open |
| I15 | flight | Telemetry drop-outs | partly fixed |
| I16 | flight | Waypoint missions use a different input clip (`pdlp` 2.0) | watch |
| I17 | flight | Motor 3 intermittently fails the idle arming check | watch |
| I18 | training | 3M-step runs keep the 2M schedule | watch |
| I19 | training | PyTorch I1 has more collapsed late checkpoints than C++ | open |
| I20 | training | Runs differ between machines | watch |
| I21 | tools | Some first-study diagnostics exist only on the lab computer | open |
| I22 | repository | `main` is 30 commits behind `feat/moo` | open |

## Evidence and statistics

**I01: only the jitter gain survives correction.** With 8 seeds per arm, the per-seed Mann-Whitney
tests with Holm correction over 48 comparisons support only "sum, DualProj, UPGrad and Aligned-MTL
command the motors more smoothly than variant E". Faster waypoints (0.15-0.37 s), SVD-PGA's lower
crash rate against the sum and every circle difference are significant before correction only or
not at all; the 95% interval on circle error is about ±2 cm. *Next:* fix the primary metrics
before running, then 12-16 seeds for claims of 1-2 cm ([study_stats.py](../l2f/sim/study_stats.py),
[next_steps.md](../l2f/docs/next_steps.md)).

**Seed variance** is large in general: four seeds misled twice (DualProj's lead, time 0.5's lead),
and one collapsed seed turned a pooled crash rate into 25.6%. Always look per seed.

## Multi-objective training

**I02: tuning not replaced.** Every split-critic arm uses E's tuned weights inside each group. The
untuned comparison (equal group shares) ran only on the head-scaled critic, which had collapsed
the groups, so it says nothing about aggregation. *Next:* untuned TD3, untuned sum and untuned
SVD-PGA on split critics, against tuned variant E.

**I03: just a reweighting?** SVD-PGA gives efficiency about 6% of the step against the sum's 24%.
A tuned scalar reward with a smaller smoothing weight might reproduce its result. *Next:* one TD3
arm with weights matched to SVD-PGA's effective shares; better, a grid of TD3 weights (a Pareto
front) against preference vectors.

**I04: critic size.** Split critics have 8 networks against TD3's 2. *Next:* TD3 with a wider
critic, or split critics of matched total size.

**I05: sum crashes more on split critics** (4.5% against TD3's 1.4%). A likely cause is the summed
value moving faster per update, as with the unscaled shared critic, since the `--total 1` term
corrects all four networks at once. *Next:* measure the total's step size on split critics.

**I06: shared critic alternatives untested.** The collapse came from the head scaling; per-head
target normalization (PopArt style) or small private layers per head might keep a shared body
usable at lower cost. The literature review recommends this control for any paper
([`reports/`](../reports/MOO%20quadrotor%20RL%20novelty%20review.md)).

Measure the critic, not only its gradients: `l2f/sim/critic_snapshot.py` (group-value correlation,
top share, fit) before trusting any conflict number.

## Reward

**I07: free-flight crash incentive.** Crash 400 lowered the crash-better share from 8.6-22.4% to
0.8-5.8% of free-flight states, but policies trained with it still show 3-7% under their own
reward. Free-flight starts (tilted up to 90°) are the hardest episodes. *Next:* check every new
recipe with `reward_check.py`; consider gentler free-flight starts or a state-dependent penalty.

**I08: the progress reward.** Potential-based shaping, which should not change the optimal policy,
gave a steady 10-20 cm offset in every phase and many crashes. Hypotheses: the critic must learn a
term whose sign depends on the motion, plus noise from moving targets; the flat cost beyond 0.2 m
leaves little slope. *Next:* train with the progress term on static targets only, and check the
critic's fit of the progress part.

**I09: time cost 0.2 worse than none,** consistently over 8 seeds, while 0.5 is the best arm. Not
explained.

## Waypoints

**I10: slow settling.** After a jump the drones overshoot a little and creep back over 3-6 s
(within 5 cm in 58-97% of 6 s legs, 16-44% of 3.5 s legs); uneven motors add 1-3 cm. This, not
reaching the target, is what a time objective should improve. Larger jumps (0.8 m and more) still
make crashing pay unless the reward's leg cost is kept below the crash penalty.

## Sim-to-real

**I11: real jitter about 2.5× the simulator's** for variant E, SVD-PGA and the sum; OS-NMF-PGA, UPGrad
and DualProj kept their low simulated jitter. Selecting on simulated jitter helps; the cause of the
gap is not known (gyro noise correlation on the old path, ESC behaviour, frame vibration?).

**I12: the sensor path.** Its policies shook in flight (2026-11-22). Calibration 1 (correlated gyro
noise) and 0-20 ms delay training followed; the calibration-2 candidate's jitter stayed high. The
survey's `heights`, `slide` and `logs` modes were not run, so the range sensor below 4 cm, the flow
model and the Lighthouse drop-outs keep prior values. The estimator reads the training's random
force as tilt. Current work uses the old path.

**I13: unidentified physics.** Drag and ground effect are randomized because they were never
measured on this drone; impulse disturbances (a sharp poke flipped policy C) are not in training.

## Hardware and flying

**I14: Aligned-MTL crashed in flight.** It rocked ±40° with motors slamming between 0 and full,
then a Lighthouse drop-out and a flip. Its simulated jitter (0.111) was the highest of the picks.

**I15: telemetry drop-outs.** In 3 of 6 waypoint flights before the fixes, the log stream stopped
while the radio stayed up (once after `no LogEntry id=255`). `--slow-telemetry` and an automatic
log-block restart before blind landings followed; one more drop before the restart existed, none
since.

**I16: `pdlp` differs by mission.** Waypoint missions set the onboard position-error clip to 2.0 m
(circles 0.5 m) so jumps reach the policy as in training; a policy trained with the firmware's
clip (`L2F_OBS_POS_CLIP=0.5`) would need 0.5. Possible, untested link to I15.

**I17: motor 3 arming.** It intermittently reads about 230 rpm at idle instead of about 2,500 and
fails the arming check, then flies normally once armed. `rl_trigger.py` waits for the "can fly"
bit and names the motor.

Also: the flown filter's range update asserts when the drone rests on the floor with an almost
level estimate (a firmware rounding edge, about 1 in 10,000 simulated episodes); the same code runs
on the drone.

## Training and tools

**I18: schedule length.** The curriculum and noise decay were designed for 2M steps; at 3M the
exploration noise is about 0.007. Use 2M unless the schedule is changed.

**I19: PyTorch I1 tail.** 4 collapsed late checkpoints in 3 of 6 PyTorch I1 seeds against none in
C++ (Fisher p ≈ 0.18). Irrelevant for variant E, which matches; pick I1 checkpoints by testing.

**I20: machine differences.** OpenBLAS (Linux) and Accelerate (macOS) round differently; runs are
deterministic per machine only. Compare runs from one machine. On the Mac, `test_td3.py` shows a
drift of 1.9e-4 after 1000 rounds against a 1e-4 tolerance, from Accelerate, not from a change.

**I21: lab-only diagnostics.** The first waypoint study's scripts (`~/l2f-runs/diag/` on the lab
computer: rollout, crash classification, conflict, leg cost, gradient norms) are not in the
repository; `reward_check.py`, `moo_report.py` and `critic_snapshot.py` cover most of them.

## Repository

**I22: branches.** `main` stops at 2026-11-22; `feat/moo` holds all later work. *Next:* merge or
fast-forward when the MOO work is ready.

## Process lessons

- **Change one thing at a time** against a reference arm, and test every arm under the same
  conditions.
- **Gate every reward change** with the reward check at the final weights. It is necessary, not
  sufficient.
- **Use 8 seeds before a decision**; look at every seed.
- **Check what a test measures**: short waypoint legs made slow settling look like an offset.
- **New options must leave old recipes bit-identical**, checked by training, not by reading code.
- **Check analysis tools against the real code**: a missing weight in a candidate scoring made two
  rewards look 40 times cheaper.
- **Measure the critic** before interpreting gradients.
- **Report smoothness next to error** for real flights.

## Next steps

From [next_steps.md](../l2f/docs/next_steps.md) and the items above, roughly in order:

1. Score every final policy on the four group returns and plot the seeds in objective space: does
   PGA dominate variant E? (evaluation only)
2. The untuned comparison on split critics (I02).
3. The reweighting and critic-size controls (I03, I04).
4. A tuned linear-scalarization sweep (a Pareto front) against preference vectors.
5. Agility: fast circles and the paper's figure-eight at three speeds; robustness sweeps (noise,
   0-40 ms delay, force, motor mismatch, glitches); training-speed comparison from saved
   checkpoints.
6. Real flights by the paper's protocol (success out of 10, tracking error).
7. A task with designed interference (a target jump against a Lighthouse glitch of the same size).
