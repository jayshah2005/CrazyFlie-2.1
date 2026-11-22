# Learning to Fly on the Crazyflie 2.1 Brushless

Port of [Learning to Fly in Seconds](https://arxiv.org/abs/2311.13081)
([arplaboratory/learning-to-fly](https://github.com/arplaboratory/learning-to-fly))
to our **Crazyflie 2.1 Brushless** (`cf21bl`) with Lighthouse + Flow deck.
First stable learned hover: 2026-11-01, 35 s, ~4 cm position error.
First full mission with no PID (take-off from the floor, 1.2 m wide circle,
landing): 2026-11-04, at 0.6 / 0.75 / 1.0 / 1.5 m, landing within 2-4 cm of the
start (see [Mission](#mission)).

Training runs in **WSL Ubuntu 24.04** or on **macOS** (the build settings per
platform are in `sim/platform.sh`); the firmware build runs in WSL. Flying runs on
Windows from the repo root with `rl_trigger.py`.

| Path | What |
| --- | --- |
| `learning_to_fly/` | The training code: learning-to-fly `d07592d` (MIT, Jonas Eschmann) with our changes (brushless dynamics, reward terms, mission extension, measured motor speeds, multi-objective critic, `evaluate_robustness`); `external/rl_tools` and `controller` are submodules at the upstream commits |
| `sim/train_brushless.sh` | Train the hover policy (default 1M steps), run the robustness check |
| `sim/train_mission.sh` | Train the mission policy (recipe of the flown "I1") and compare all checkpoints |
| `firmware/patch_firmware.py` | Hooks into crazyflie-firmware 2026.08 (learned-controller packet, motor path, PID boot) |
| `firmware/build_firmware.sh` | Out-of-tree controller build for `cf21bl` (fixes below), output `~/l2f_2026/build/cf21bl.bin`; target velocity and position-error integral for mission policies |
| `py/` | PyTorch + TorchJD training on the C++ simulator (plan Step 3): simulator module, networks, checkpoint export, TD3, multi-objective TD3, checks ([PyTorch training](#pytorch-training-plan-step-3)) |
| `estimator/` | The flown Kalman estimator compiled from the firmware tree as a library, for the simulator's sensor path (plan Step 2, [Sensor path](#sensor-path-plan-step-2)) |
| `pytorch_training_plan.html` | Plan for moving training to PyTorch + TorchJD and simulating the sensors through the firmware's Kalman filter (Steps 0-3 implemented; Steps 4-6 need the survey); open it in a browser |

## Start here

**What is going on.** A small neural network (the *policy* or *actor*) reads the drone's
state (position and velocity error to a target, orientation, angular velocity, and its recent
motor commands or the measured motor speeds) 100 times a second and outputs the four motor
commands. It is trained in a fast C++ simulator of this drone with TD3, an actor-critic
reinforcement-learning algorithm: a second network (the *critic*) learns how good each
action is, and the actor is pushed towards actions the critic rates highly. Training runs on
a CPU (about 4 minutes for the 1M-step hover policy). The trained actor is exported as a C header, compiled into the
Crazyflie firmware, and runs on the drone itself.

**The order things were done in**, and the section that covers each:

1. Measure the real drone and build its simulator model ([Brushless model](#brushless-model)).
2. Hover with the learned controller, fix the firmware and the motor chatter
   ([Firmware fixes](#firmware-fixes-needed-on-top-of-the-papers-controller),
   [Motor chatter](#motor-chatter)).
3. One policy for the whole flight: take-off, circle, landing ([Mission](#mission)).
4. Give the policy the measured motor speeds ([Measured motor speeds](#measured-motor-speeds-as-input)).
5. Multi-objective training, which did not beat ordinary training
   ([Multi-objective critic](#multi-objective-critic-moo1)).

**Words used below:**

| Term | Meaning |
| --- | --- |
| checkpoint | The actor saved every 100k training steps (`actor_<step>.h5` / `.h`). TD3 is noisy late in training, so every checkpoint is evaluated and the best one is flown. |
| I1 | The first flown mission policy (checkpoint 1.4M of `train_mission.sh`, default recipe). |
| variant A-E | Which inputs the actor gets (`VARIANT=` in `train_mission.sh`): I1 and A use the last 32 motor commands, C adds the measured motor speeds and keeps 4 commands, E has the speeds and no commands. |
| jitter | The paper's smoothness metric of the motor commands; lower is smoother. |
| domain randomization (DR) | Each training episode varies mass, inertia, motor constants and so on, so the policy copes with model error. |
| MOO | Multi-objective optimisation: the reward is split into parts (or tasks) and the actor gets one gradient per part, combined by an *aggregator* (`AGG=`). |
| framing A / B | MOO objectives are the reward parts (A) or the mission episode types (B, `TASKS=`). |

**Where the code is.** `learning_to_fly/` is upstream learning-to-fly with our changes. The
changes sit behind `L2F_*` switches, so with no switches the code trains like the original.
There are two kinds:

- compile-time flags (`-DL2F_MISSION`, `-DL2F_RPM_OBS`, `-DL2F_MOO`, ...), set in
  `sim/train_*.sh`. They decide the model, inputs, reward and training algorithm.
- runtime environment variables (`L2F_DR`, `L2F_GLITCH_P`, `L2F_EVAL_*`, ...). They set
  randomization, the curriculum and the evaluation conditions.

`grep -rn L2F_ learning_to_fly/src learning_to_fly/include` finds every one; each has a
comment where it is read. The files we changed:

| File (under `learning_to_fly/`) | Our changes |
| --- | --- |
| `include/.../parameters/dynamics/crazyflie_brushless.h` | Measured model of this drone |
| `include/.../parameters/reward_functions/brushless.h` | Reward terms (action rate, angular acceleration, landing speed, ...) |
| `include/.../simulator/multirotor.h` | Mission state, randomization and glitch settings (runtime variables) |
| `include/.../simulator/operations_generic.h` | Mission episodes, floor contact, ground effect, delay, measured-speed inputs |
| `src/config/parameters.h` | Which model, reward, noise and inputs a build uses |
| `src/config/config.h`, `actor_and_critic.h` | Step limit, replay-buffer cap, critic width and outputs |
| `src/training.h`, `training.cpp` | Swaps in the MOO training step; `L2F_SEED` |
| `src/steps/curriculum.h` | Our mission ramp (mission episodes, delay and randomization ramped in over `L2F_MISSION_RAMP`), plus the upstream reward-weight curriculum with our untuned-reward option (`REWARD=uniform`) |
| `src/evaluate_robustness.cpp` | Offline evaluation, including the flight script's mission (`mission` mode) |
| `src/moo/` | Multi-objective TD3 step and gradient aggregators |

## Brushless model

From two stock-PID flights (`sysid_hover.py` + `sysid_analyze.py`) and IDSIA
(Busetto et al., arXiv 2512.14450):

- mass 41.3 g; inertia 2.20e-5 / 2.97e-5 kg m² (IDSIA, mass-scaled); arm 35.35 mm
- thrust 3.86e-8 N/(rad/s)²; torque/thrust 0.0021 m; motor lag 0.06 s
- ESC map `rpm = 5400 * (ratio * vbat) + 3750` (inverted onboard, so commands compensate battery voltage)

## Steps

1. Toolchain (once): `sudo apt install cmake build-essential libhdf5-dev libopenblas-dev libboost-all-dev protobuf-compiler libprotobuf-dev gcc-arm-none-eabi`
2. Submodules (once, from the repo root):
   `git submodule update --init l2f/learning_to_fly/external/rl_tools l2f/learning_to_fly/controller`, then
   `git -C l2f/learning_to_fly/external/rl_tools submodule update --init external/cli11 external/highfive external/json external/tensorboard`
   and, for firmware builds, `git -C l2f/learning_to_fly/controller submodule update --init external/rl_tools`.
   The scripts build in `l2f/learning_to_fly` (override with `L2F=<path>`; a copy
   on the WSL file system compiles faster).
3. `bash l2f/sim/train_brushless.sh` (~4 min for 1M steps).
4. `ACTOR=<checkpoint>.h bash l2f/firmware/build_firmware.sh`
5. Put `cf21bl.bin` into the official `firmware-cf21bl-2026.08.zip` in place of
   `cf21bl-2026.08.bin` (or pass `OFFICIAL_ZIP=<that zip>` in step 4, which writes
   `build/firmware-cf21bl-2026.08-l2f.zip`), flash with cfclient **Bootloader → Cold boot →
   From file** or `flash_cold.py`, then power-cycle. Never flash this drone with `cfloader -w`
   (warm boot). The build also runs on macOS (Homebrew's `arm-none-eabi-gcc`, 2026-11-22): there
   it sets the out-of-tree controller option itself, because Bitcraze's `merge_config.sh` needs
   GNU `sed`, `cp` and `readlink`; on every platform it stops if that option is missing.
6. Fly: `python rl_trigger.py --mode takeoff_and_switch`.

## Firmware fixes needed on top of the paper's controller

- **Platform:** build `cf21bl`, not `cf2`; `cf2` images do not boot on the Brushless.
- **Boot on PID:** the learned controller only runs when `rl_trigger.py` selects it.
  Running it from boot crashed the radio task.
- **Policy at 100 Hz** (the simulator's step): at ~0.7 ms per pass, 500 Hz starved the radio task.
- **Motor commands every 1 kHz tick:** the DShot driver re-arms its DMA in
  `motorsSetRatio()`, and M1/M2 share a DMA request. Writing only on policy
  ticks killed M1/M2 in flight.
- **Handover:** the action history starts from the measured motor RPM, the same way training episodes start.
- **Supervisor watchdog:** each learned-controller packet also sends a
  zero-thrust setpoint, so the 2 s commander watchdog does not cut the motors.

## Motor chatter

The first flown policy (paper noise and reward) chattered between ~0% and ~100%
motor command (gyro ~75 °/s vs ~3 °/s under PID). The paper's gyro noise
(0.002 rad/s) is ~35x below ours (0.069 rad/s); with measured noise
(`L2F_REAL_NOISE`) the simulator reproduces the chatter. Retraining with that
noise plus an angular-acceleration penalty (`L2F_ANG_ACC_WEIGHT=2e-4`, the
`train_brushless.sh` defaults) halves command jumpiness in simulation; 1e-3 is
too strong and the policy stops recovering from tilts.

## Landing

`rl_trigger.py --mode land` needs firmware with target-offset packets (the
learned-controller packet carries a 12-byte x/y/z offset from the activation
point). It hovers, lowers the target at `--descent-rate`, and cuts the motors
below `--cut-height`.

## Mission

One policy flies everything, no PID: it sits on the floor, climbs to
`--height`, hovers, flies a circle (`--radius`, `--period`), returns, and lands
(`rl_trigger.py --mode mission`). The flight script moves the target; the
learned-controller packet carries the target offset and velocity (24 bytes).

```
python rl_trigger.py --mode mission --height 1.0 --radius 0.6 --period 10 --descent-rate 0.25 --ki 0
```

Simulator extension (`L2F_MISSION`, all in `learning_to_fly/`):

- floor contact; take-off, landing, circle (up to 1.2 m, 1 m/s) and line episodes
  with a moving target; observations and reward use velocity relative to the target
- curriculum: mission episodes, randomization and delay ramp in over `L2F_MISSION_RAMP`
  steps. Without it, training never learns to fly
- 0-12 ms state-estimate delay: the first flown policies collapsed above ~5 ms,
  which explained most of their real jitter
- domain randomization (`L2F_DR`, `L2F_DR_SCALE`, 1 = brushless-modelling paper
  ranges; full ranges prevent learning, 0.3 works), ground effect
- position-estimate jumps with velocity spikes (`L2F_GLITCH_P`), like the
  Lighthouse jumps measured in flight
- position-error integral as 3 extra inputs (`L2F_POS_INTEGRAL`, 149 inputs; the
  firmware computes the same leaky integral), so the policy removes its own offset
- reward: action rate (`L2F_ACTION_RATE_WEIGHT`), landing speed
  (`L2F_LANDING_SPEED_WEIGHT`, no measurable effect at 5), angular velocity
- `evaluate_robustness <actor.h5> N mission`: the flight script's mission in
  simulation, with per-phase tracking error, crashes, jitter (paper metric) and
  landing (env `L2F_MISSION_H/R/P`, `L2F_EVAL_DR`, `L2F_EVAL_DELAY_MS`,
  `L2F_EVAL_GLITCH`, `L2F_EVAL_KI`)

Flown policy I1 (checkpoint 1.4M of `train_mission.sh`), real drone, `--ki 0`:

| Mission | Hover | Circle | Landing | Jitter (paper metric) |
| --- | --- | --- | --- | --- |
| 0.6 m, 0.3 m circle | 3.8-4.0 cm | 5.7-6.4 cm | 3.4-3.8 cm | 0.047-0.050 |
| 0.75 m, 0.6 m circle | 4.4 cm | 8.4 cm | 3.9 cm | 0.060 |
| 1.0 m, 0.6 m circle | 5.0 cm | 6.7 cm | 4.1 cm | 0.066 |
| 1.5 m, 0.6 m circle | 3.2 cm | 10.9 cm (one 36 cm Lighthouse jump) | 5.8 cm | 0.072 |

For comparison on the same drone model (brushless-modelling paper, ICRA 2026):
network 4.3 cm / 0.020, PID 2.8 cm / 0.023, Mellinger 5.5 cm / 0.039 (hover).

Safety in mission mode: releasing Enter in the air, a battery below `--land-vbat`
under load, or lost telemetry starts an automatic landing (packets keep
flowing); Ctrl+C cuts the motors; Q only quits on the floor.

Lessons:

- TD3 returns swing late in training: evaluate every 100k checkpoint, never just the last.
- In-flight Lighthouse drop-outs cause 10-37 cm estimate jumps; one flipped a
  policy trained without jumps. Base stations only 15-24 deg down (ours) make
  it worse, especially high up. `lighthouse_check.py` checks the deck,
  calibration, geometry and reception.
- If cfclient cannot reach the bootloader, `flash_cold.py <zip>` does the same
  cold-boot flash with cflib.

## Measured motor speeds as input

The brushless-modelling paper gives the network past actions because the drone
"cannot measure" motor speed. Ours can: bidirectional DShot reports each
motor's mechanical RPM (`motor.m*_rpm`). `rpm_quality.py` (repo root) checks it
from existing logs: ~0.1% invalid reads (65535, single samples), ~14% of 100 Hz
samples repeat the previous value, <= ~100 rpm noise, no delay beyond the
0.06 s motor lag the simulator already models.

Simulator (`learning_to_fly/`): `L2F_RPM_OBS` gives the actor the 4 speeds after the
angular velocity (normalized `rpm / 12000 - 1`, Gaussian noise `L2F_RPM_NOISE`
0.01, repeated value with probability `L2F_RPM_HOLD` 0.15); `L2F_HISTORY` sets the
action-history length; per-motor mismatch (`L2F_MOTOR_GAIN`, `L2F_MOTOR_OFFSET`,
with `L2F_DR`) makes the measured speeds carry information the history does
not. The critic is unchanged. Without these flags the I1 recipe trains
bit-identically. `train_mission.sh` has `VARIANT=I1|A|B|C|D` and `SEED`.

Simulation, 3 seeds each, median over checkpoints 1.2M-2.0M (mission test;
"harder" adds the per-motor mismatch):

| Variant | Inputs | Circle clean | Circle harder | Jitter | Crashed episodes |
| --- | --- | --- | --- | --- | --- |
| A: I1 inputs (32 history) | 149 | 7.1 cm | 12.0 cm | 0.063 | 5.1% |
| C: speeds + 4 history | 41 | 7.2 cm | 9.6 cm | 0.067 | 1.8% |
| D: 4 history, no speeds | 37 | 9.0 cm | 10.2 cm | 0.086 | 1.7% |

B (speeds + 32 history, one seed) was no better than A. Candidate for flight:
C seed 2, checkpoint 1.5M (3.1 / 5.8 cm, jitter 0.057; I1 on the same test:
4.2 / 7.7 cm, 0.050).

Firmware: `HISTORY=4 RPM_OBS=1 ACTOR=<checkpoint>.h bash firmware/build_firmware.sh`.
The controller reads `motorsGetRPM` at 100 Hz, keeps the last valid value on an
invalid read and logs what the policy sees as `rltr.r1-r4` (`rl_trigger.py`
columns `rin1-rin4`). HISTORY / RPM_OBS must match the training variant: the
build only checks the total input count, and 41 inputs also fit a 5-step
history without speeds.

Variant E (speeds, no action history, 25 inputs; `train_mission.sh VARIANT=E`,
firmware `HISTORY=0 RPM_OBS=1`) trains too: best simulated circle when clean,
worse than C with per-motor mismatch (3 seeds: 6.1 / 11.4 cm, jitter 0.063).
Flown: seed 0, checkpoint 1.6M.

Real flights, 2026-11-06 (C, E) vs 2026-11-04 (I1), four undisturbed missions
each (0.6 m twice with a 0.3 m circle, 1.0 m and 1.5 m with 0.6 m), mean of the
per-flight values, distances from the drone's own Lighthouse estimate:

| Policy | Inputs | Hover | Circle | Landing spot | Jitter hover / circle | Rotation |
| --- | --- | --- | --- | --- | --- | --- |
| I1 (32-step history) | 149 | 4.0 cm | 7.4 cm | 2.8 cm | 0.057 / 0.061 | 33 deg/s |
| C (speeds + 4 steps) | 41 | 4.5 cm | 7.1 cm | 5.4 cm | 0.071 / 0.078 | 22 deg/s |
| E (speeds, no history) | 25 | 5.2 cm | 7.0 cm | 4.3 cm | 0.056 / 0.050 | 28 deg/s |

With measured motor speeds the long action history is not needed: circle
tracking is equal with 41 or even 25 inputs, I1 still hovers and lands slightly
better. E beat its simulation result, so the training's per-motor mismatch
(+-5%, +-500 rpm) is probably larger than the real one. Small samples: 1-2 cm
differences are within flight-to-flight spread.

Notes from these flights:

- Motor 3 intermittently fails the firmware's idle arming check (~230 rpm at
  idle instead of ~2,500; 1000-3500 required) and flies normally once armed.
  `rl_trigger.py` now waits for the supervisor's "can fly" bit and names the
  failing motor instead of taking off disarmed.
- `rl_trigger.py` logs the drone console (`[drone] ...`). "SUP: free falling!"
  right after landing comes from the motor cut 6 cm above the floor.
- A sharp styrofoam poke at 0.9 m (839 deg/s within 20 ms) flipped C; light pokes
  were absorbed. Training has no impulse disturbances.

## Simulator and reward options (plan Step 1)

From `pytorch_training_plan.html`. Every option is off by default: with none set,
training is bit-identical to before (checked: same evaluation returns over 60k
steps, same seed, old and new build).

| Option (`train_mission.sh`) | Kind | Effect |
| --- | --- | --- |
| `SMOOTHING=zigzag` (`ZIGZAG_WEIGHT=0.25`) | build | replaces the action-rate cost Σ(a_t − a_t−1)² by Σ(a_t − 2a_t−1 + a_t−2)²: zero for a steady ramp, large for chatter. At 50 Hz chatter, 0.25 · 16 = 1 · 4, so chatter costs what it did; a 2 Hz manoeuvre about 60× less. `SMOOTHING=none` drops both |
| `CRASH_PENALTY=P` | build | reward −P on a crash |
| `SURVIVAL_BONUS=c` | build | the per-step bonus (paper 2); 0 tests the crash penalty alone. Losing the bonus by crashing costs about c / (1 − γ) = 200 |
| `GYRO_CLIP=34.9` | runtime | the actor's gyro reading is clipped at the sensor's range (the firmware's ±2000°/s) |
| `OMEGA_CRASH=34.9` | runtime | rotating faster than this (rad/s, any axis) is a crash |
| `LANDING_CUT=1` | runtime | land like `rl_trigger.py`: in landing episodes and the mission test, the motors are cut when the target is within `cut_window` of the floor and the *estimated* height is below the cut height, or `floor_timeout` after the target reached the floor; the episode ends as a landing, paid the bonus it gives up (c / (1 − γ)). Values in `config/landing.yaml` |
| `POS_NOISE=1` | runtime | position noise drawn per episode and per axis, from `noise_measured.json` (`noise_survey.py`) or x, y 3–10 mm until then |
| `EVAL_EPISODES=100` | build | episodes of rl_tools' test every 10k steps (default 1000, upstream). It only writes the `Step: … mean return` lines and uses its own environment and random stream, so the weights are the same at any value (checked bit for bit at 100k steps; only the checkpoint's embedded test example differs). On the sensor path the 1000 episodes take most of a run's time |

`evaluate_robustness <actor.h5> N landing` flies the training environment's own
landing episodes and reports how they end (landed, failed landing, crashed, still
flying), with the cut height and touchdown speed. The mission test reports the motor
cut too when `L2F_LANDING_CUT=1`. With `MOO=1` the crash penalty or landing payout is
part 0 and the zig-zag cost part 5, so the parts still add up to the reward.

Run folders end in a tag (build, seed, runtime options, process id), so trainings
started in the same second no longer share a folder. `sim/compare.sh` runs a file of
option sets over several seeds and summarises them; `sim/step1_configs.txt` is the
comparison below.

Results (2026-11-21, macOS/Accelerate, variant E, 3 seeds, medians over checkpoints
1.2M–2.0M of the `train_mission.sh` mission test; "uneven" = with per-motor mismatch;
crashed = share of all test episodes):

| Option set | Circle | Uneven | Crashed | Jitter | Checkpoints without a crash |
| --- | --- | --- | --- | --- | --- |
| baseline (bonus 2, no penalty, action rate) | 6.3 cm | 12.6 cm | 3.1% | 0.067 | 3/27 |
| bonus 2 + crash penalty 200 | 7.6 cm | 13.1 cm | 2.1% | 0.065 | 10/27 |
| **no bonus, crash penalty 200** | **5.3 cm** | **10.0 cm** | **0.8%** | 0.070 | 17/27 |
| no bonus, crash penalty 400 | 6.3 cm | 9.6 cm | 0.7% | 0.069 | 19/27 |
| zig-zag 0.25 instead of action rate | 6.2 cm | 13.1 cm | 4.0% | 0.065 | 8/27 |
| no smoothing term | 6.8 cm | 13.0 cm | 2.9% | 0.073 | 10/27 |
| gyro clip + rotation crash + per-axis noise (x, y 3–10 mm) | 6.5 cm | 10.7 cm | 1.5% | 0.066 | 16/27 |
| landing cut (6 cm, `config/landing.yaml`) | 6.9 cm | 14.3 cm | 4.4% | 0.077 | 10/27 |

The baseline matches the WSL runs above (E: 6.1 / 11.4 cm, 2.8%). Per seed, the
baseline crashed 3.0–3.5% and the no-bonus crash penalty 0.2–1.7% (P 200) and
0.2–1.0% (P 400), so that difference is larger than the seed spread. In theory a bonus
of 2 and a crash penalty of 200 give the same incentive (2 / (1 − 0.99) = 200); in
training, the penalty without the bonus works clearly better. Zig-zag keeps the jitter
of the action-rate penalty (no smoothing term raises it to 0.073) but crashes slightly
more; this test has no sharp manoeuvres, where zig-zag should differ. With the cut at
6 cm, `evaluate_robustness … landing` lands 95–100% of the training landing episodes
for both the landing-cut and the baseline policies (cut at 6.0–7.0 cm, touchdown
1.10–1.16 m/s: the free fall from the cut height), so the cut changes little until the
cut height is lowered.

Target steps (plan Q15, `python3 l2f/py/step_response.py`, 2026-11-22): point-to-point episodes
(the target jumps 0.2–0.45 m), deterministic actions, no randomization, checkpoints 1.6M, 1.8M and
2.0M of the 3 seeds, 40 episodes each (360 per row). Half-way: time until the error is half the
step; settle: until it stays below max(10% of the step, 5 cm); no rise: the error never halved
within 2 s.

| Policy | Half-way | Overshoot | Settle | No rise | Crashed | Chatter \|Δa\| |
| --- | --- | --- | --- | --- | --- | --- |
| baseline (action rate) | 0.68 s | 27% | 1.53 s | 12% | 3 | 0.104 |
| zig-zag 0.25 | 0.67 s | 30% | 1.27 s | 21% | 6 | 0.107 |
| no smoothing term | 0.66 s | 25% | 1.43 s | 17% | 7 | 0.111 |
| no bonus, crash penalty 200 | 0.75 s | 23% | 1.50 s | 5% | 0 | 0.091 |

Zig-zag does not make steps faster (it frees fast purposeful command changes, but the policy
does not use them) and fails to respond more often; it settles faster. The no-bonus crash
penalty is the clearest gain here too: fewest failed responses, no crash, least overshoot and
chatter.

## PyTorch training (plan Step 3)

The training loop in PyTorch, on the unchanged C++ simulator. Build the simulator module with
the flags of the recipe (`train_mission.sh` prints them; the module is `l2f_sim`):

```
cmake -S l2f/learning_to_fly -B l2f/learning_to_fly/build_py_E -DCMAKE_BUILD_TYPE=Release <platform flags> \
  -DCMAKE_CXX_FLAGS="<the recipe's -D flags>" -DL2F_BUILD_PYTHON=ON -DPython_EXECUTABLE=$(which python3)
cmake --build l2f/learning_to_fly/build_py_E --target l2f_sim evaluate_robustness h5_to_header
python3 l2f/py/train.py --build l2f/learning_to_fly/build_py_E/src --variant E --seed 0 --evaluate
python3 l2f/py/train.py ... --moo parts --agg upgrad --total 1     # multi-objective, TorchJD aggregator
```

| File | What |
| --- | --- |
| `py/sim_binding.cpp` | pybind11 module: batched `Sim` (reset, observe, step with the 11 unweighted reward parts, curriculum weights), the C++ runner's rollout as a reference, rl_tools' actor and TD3 updates as references |
| `py/networks.py` | Actor and critic as in `src/config/actor_and_critic.h`: FAST_TANH with rl_tools' backward pass (1 − y²), PyTorch's default init (= rl_tools' `init_kaiming`), Adam lr 1e-3, ε 1e-8, weight decay p·wd/2 per step on weights |
| `py/export.py` | Actor checkpoints in rl_tools' `.h5` layout; the firmware header through `src/h5_to_header.cpp` |
| `py/td3.py`, `py/train.py` | TD3 with the C++ recipe's schedule; the buffer keeps the unweighted parts, so the reward is r = Σᵢ wᵢ(t) uᵢ with the current curriculum weights (rl_tools recalculates the buffer instead) |
| `py/moo.py` | Multi-objective TD3 with TorchJD (`autojac.backward` + `jac_to_grad`): framing A (8 reward parts, or `--parts 3`: tracking, smoothness, safety), framing B (tasks), `MOO_TOTAL` and `MOO_NORM` as in `src/moo/td3_moo.h`; per-objective gradient log in `logs/train_moo_*.csv` |
| `py/moo_report.py` | Per training window: the most opposed objective pairs (median cosine, share negative), aggregator weights, gradient norms |

Checks (all pass, 2026-11-22):

| Script | Checks |
| --- | --- |
| `py/test_sim.py` | the module reproduces the C++ runner exactly (observations, rewards, terminations), and the parts add up to the reward |
| `py/test_networks.py` | a C++ checkpoint in PyTorch gives rl_tools' actions (1e-6); export and firmware header are identical |
| `py/test_td3.py` | rl_tools' and PyTorch's TD3 updates from the same weights, batches and noise stay equal (E: 2.8e-5 of the largest weight after 2000 critic and 500 actor updates); `--control` adds a PyTorch copy on row-shuffled batches as the rounding floor (see A5 below for I1) |
| `py/test_moo.py` | the 8 part groups add up to the reward; TorchJD's `sum` step equals the plain TD3 gradient (3e-7); `--parts 1` makes exactly TD3's updates |

`test_td3.py` found that the port first used Adam ε = 1e-7 (rl_tools' `DefaultParametersTF`)
instead of the 1e-8 of `DefaultParameters` that the recipe uses; ε only matters for weights
with small gradients, but it changed the first update by a third of a step. Beyond ~1000
rounds the two runs drift apart by themselves: once the actor leaves saturation, Adam turns
rounding differences into full steps, and a PyTorch copy whose weights differ by 1e-7 drifts
from the original just as much. So only the statistics of whole runs can be compared (A5).

A5, PyTorch against C++ (`moo_summary.py checkpoints`, late checkpoints 1.2M–2.0M, 2026-11-22):

| Recipe | Trainer | Seeds | Circle | Uneven motors | Crashed | Jitter | Per seed: circle / uneven / crashed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E | C++ | 6 | 6.5 cm | 13.6 cm | 4.0% | 0.069 | 4.7–8.2 / 11.7–16.0 / 3.0–5.9% |
| E | PyTorch | 6 | 6.0 cm | 13.1 cm | 3.5% | 0.069 | 4.6–8.9 / 11.1–19.2 / 1.2–5.9% |
| I1 | C++ | 6 | 7.9 cm | 11.8 cm | 2.8% | 0.055 | 6.2–10.7 / 10.3–12.5 / 0.7–5.7% |
| I1 | PyTorch | 6 | 9.0 cm | 12.2 cm | 6.3% | 0.058 | 7.1–13.8 / 9.9–18.2 / 0–15.6% |

E matches within the seed spread (per-seed circle error and crash count: Mann-Whitney p = 0.75
for both). I1 does not differ significantly either (p = 0.26 circle error, 0.47 crashes), but
PyTorch I1 has 4 collapsed late checkpoints (8 or more of 15 episodes crashed), in 3 of 6 seeds
(s0 1.6M, s3 1.7M, s4 1.2M and 1.8M); C++ I1 has none, and E has none in either trainer. With
6 seeds that can still be chance (3/6 against 0/6 seeds: Fisher p ≈ 0.18).

The update rule is not the cause: `test_td3.py --control` on an I1 build (149 inputs) shows the
same arithmetic difference to rl_tools as on E. Its control is a second PyTorch copy fed every
batch with the rows shuffled (same loss, other float32 summation order at every update).
rl_tools sits a flat 2–24x above that floor at every check from 10 to 1000 rounds on both
builds (a bug would make the multiple grow). On I1 the actor exceeds the 1e-4 tolerance by
1000 rounds (9.7e-4) only because it is far more chaotic: the shuffled copy alone grows ~160x
between rounds 500 and 1000 on I1, ~1.2x on E. The training loop is shared with E (exploration
actions clipped to ±1 before the step, so the 32-action history holds the same values as in
rl_tools). So PyTorch is cleared for E, the recipe of the sensor path; I1 trained in PyTorch
should be picked by checkpoint evaluation (as every flown policy is), or get more seeds if the
tail matters.

## Sensor path (plan Step 2)

`SENSOR_SIM=1` (build flag `L2F_SENSOR_SIM`) makes the actor observe what it observes on the
drone: the output of the flown Kalman estimator, fed by simulated sensors. The critic keeps the
true state. Every sensor value is a prior in `config/sensors.yaml` (datasheets and earlier
measurements, `[lo, hi]` drawn per episode) until the survey measures it (plan Step 5).

| File | What |
| --- | --- |
| `estimator/` | `cf_estimator.c` + the firmware's own `kalman_core.c`, `mm_tof.c`, `mm_flow.c`, `mm_sweep_angles.c`, `mm_yaw_error.c`, `kalman_supervisor.c`, `outlierFilterLighthouse.c`, Lighthouse preprocessing, compiled unchanged from `~/crazyflie-firmware-2026.08` (`FW=`); `cf_estimator.c` re-implements only the FreeRTOS-bound loop (`kalmanTask` at 1 kHz, 100 Hz prediction), the measurement queue, the flying flag, the LH2 throttle and the driver rules, per instance |
| `include/.../simulator/sensor_sim.h` | Sensors per 10 ms step: IMU average for the filter, the gyro sample for the actor, range every 25 ms (15° cone), flow every ~10 ms (inverse of `mm_flow.c` with the true camera height), one Lighthouse V2 rotation per base station (~50 Hz) from the firmware's measurement model with the true geometry (stored + error), drop-outs; 10 estimator ticks |
| `config/sensors.yaml` | The sensor priors, the Lighthouse system file (`config/current/config.yaml`), the E4 switches (off) |

| Option | Effect |
| --- | --- |
| `SENSOR_SIM=1` | build: the sensor path (Gaussian observation noise off; not with `POS_NOISE=1`); `SENSORS=<file>` another sensor file |
| `FORCE_SCALE=<x>` | runtime: scale of the per-episode random external force (σ m·g/20 per axis; 1 = recipe) |

`estimator/build/replay_survey <noise_survey_*.csv> config/current/config.yaml` replays a survey log
through the library against the logged estimate (plan E1 check 2; `--selftest`: a synthetic 100 Hz
log with only sensor 0's angles, as the firmware logs them, reproduces the estimate within 0.6 mm).

Checks: `estimator/build/check_estimator` (ideal sensors through the library: converges, tracks a
0.5 m circle at 0.8 m/s with 4.3 mm RMS, several filters side by side are independent, the flow
alone at 5 cm reads 2.1× the speed) and `py/test_sensor_sim.py` (a trained actor flies the
mission mix; estimate against truth). Findings (2026-11-22):

- With ideal sensors and no random force the estimate is within 1.6 mm (median position),
  0.10° (tilt) and 0.05° (yaw) of the truth in closed loop: the frames, units and timing are right.
- The recipe's random external force (5% of the weight per axis) is read by the flown filter as
  tilt, because its flight model has thrust along body z only: 2-3° median tilt error even with
  ideal sensors (0.2° without the force). A real drone has much smaller steady sideways forces,
  so on this path the force makes the estimate worse than on the drone (open question in the plan).
- The drone keeps at most 50 LH2 samples per second (`lighthouse.lh2maxRate`): with two base
  stations at ~50 Hz it drops about half of them at random. The simulator does the same.
- Starting the filter exactly at a level truth hits a firmware rounding edge (`kalmanCoreFinalize`
  can give R₂₂ = 1 + 1 ulp, `acosf` of it in `mm_tof.c` is NaN). Episodes therefore start with
  the converged filter's errors (σ 2 mm, 1 cm/s, 0.18°). A firmware assert or a non-finite state
  ends the episode as a crash, as the assert halts the drone; none in 4500 early-training episodes.
- An old-path policy (E, seed 1, 1.6M) on this path with the priors: 50% of the hard training-mix
  episodes end early, against 32.5% on the old path (10% / 5% without the random force); the
  policy has never seen estimator output. Speed: 14.6 µs per step against 2.6 µs (Python loop).
  C++ training of 2M steps, 3 runs at a time on the 8-core Mac: 18–21.5 min against 5 min on the
  old path (2026-11-22; a first short run had suggested only 25% slower).

E4 (plan Q13, off by default): `estimator.flow_off_below` / `rl_trigger.py --flow-off-below 0.1`
switch the Flow deck off below 10 cm, and `L2F_RANGE_MIN_PATCH=1` in `patch_firmware.py` with
`estimator.range_min_distance: 0.04` raises the range σ below 4 cm, in the simulator and the
firmware alike.

### Training on the sensor path, and the random-force strength (plan Q16, 2026-11-22)

`sim/force_configs.txt`: the new baseline (no survival bonus, crash penalty 200, action rate,
rotation crash at 34.9 rad/s) trained on the sensor path at `FORCE_SCALE` 1, 0.5 and 0.25, and
on the old path at 1 for reference; variant E, 3 seeds each, sensor priors (not yet measured).
`train_mission.sh` tests each run at its own force, so `sim/cross_eval.sh` re-tests every run at
force 1, 0.5, 0.25 and 0 with the same mission test. Medians over checkpoints 1.2M–2.0M of the
3 seeds; crashes over the clean and uneven-motor tests (the sensor path has no 1% jumps, so that
test is left out); a collapsed checkpoint crashes ≥ 8 of 15 episodes.

| Trained at | Test force | Circle | Uneven | Crashed | Per seed | Jitter | Collapsed checkpoints |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sensor path, force 1 | 1 | 5.8 cm | 8.1 cm | **0.0%** | 0.0, 0.0, 0.0% | 0.113 | 0/27 |
| sensor path, force 1 | 0 | 5.7 cm | 8.1 cm | 0.2% | 0.4, 0.0, 0.4% | 0.114 | 0/27 |
| sensor path, force 0.5 | 1 | 5.4 cm | 8.6 cm | 5.3% | 1.5, 0.7, 13.7% | 0.107 | 1/27 |
| sensor path, force 0.5 | 0 | 5.3 cm | 8.4 cm | 4.8% | 1.5, 0.4, 12.6% | 0.107 | 1/27 |
| sensor path, force 0.25 | 1 | 6.5 cm | 10.2 cm | 3.1% | 0.0, 0.4, 8.9% | 0.124 | 1/27 |
| sensor path, force 0.25 | 0 | 6.5 cm | 10.1 cm | 3.1% | 0.0, 0.0, 9.3% | 0.124 | 1/27 |
| old path, force 1 | 1 | 5.3 cm | 9.0 cm | 0.9% | 1.1, 0.7, 0.7% | 0.075 | 0/27 |
| old path, force 1 | 0 | 4.7 cm | 8.6 cm | 0.1% | 0.4, 0.0, 0.0% | 0.075 | 0/27 |

(Test forces 0.5 and 0.25 lie between these rows; `cross_eval.sh` prints all of them.)

- The sensor path trains as well as the old path: with full force, 5.8 cm / 8.1 cm and no
  crash in 810 test episodes, against 5.3 cm / 9.0 cm / 0.9% on the old path.
- A weaker training force does not help. Tracking is about the same at 0.5 and worse at 0.25,
  and each weaker setting has one seed with a collapsed checkpoint (0.5: seed 2 at 1.9M, 30 of
  30 crashed; 0.25: seed 2 at 1.8M, 20 of 30); the other checkpoints of those seeds are fine.
  With 3 seeds this is no proof that weaker is worse, but nothing points to weaker being better.
- The test force hardly matters to the sensor-path policies (within 0.2 cm and 0.7 points
  between test forces 1 and 0; the old-path policies gain 0.6 cm and 0.8 points at force 0),
  although the estimate's tilt error scales with it (`py/test_sensor_sim.py`, one force-1
  policy, priors; median / 90%): force 1: 3.3° / 7.1°, 0.5: 1.4° / 3.3°, 0.25: 0.9° / 2.0°,
  0: 0.3° / 1.4°; position error 12 mm at every force.
- Jitter is about 50% higher on the sensor path (0.11–0.12 against 0.075). The priors' gyro
  noise (σ 0.05–0.09 rad/s on the actor's sample) and the estimator's noise reach the actor;
  the survey's `hover` mode measures the real values.

Decision (2026-11-22, plan Q16): keep the full random force; compare again after calibration.

### MOO on the sensor path (2026-11-22, lab computer, 8 seeds)

Same recipe at full force, PyTorch (`py/train.py`, build `build_py_E_nb200_ss`), seeds 0-7, the
old IH222A placement; medians of the per-seed late-checkpoint medians, crashes without the
1%-jumps test:

| Training | Circle | Uneven | Crashed | Collapsed seeds |
| --- | --- | --- | --- | --- |
| plain TD3 | 5.2 cm | 8.3 cm | 0.4% | 1 of 8 (seed 7: ends episodes early, 100% crashed) |
| sum, 3 groups | 5.2 cm | 8.5 cm | 0.0% | 0 of 8 |
| DualProj, 3 groups | 5.6 cm | 10.1 cm | 0.6% | 0 of 8 |

No method is better on circle error or crashes (Mann-Whitney p 0.11-0.95); DualProj is behind
plain TD3 with uneven motors (p = 0.05). A 3-seed round on the Mac had the sum behind (two seeds
with 12-15% crashes): seed spread, which the 8 seeds resolve. About one run in ten collapses in
any method, so flight checkpoints are always picked by a longer test (`policies/README.md`).

### First flight and first calibration (2026-11-22)

The TD3 candidate (`policies/2026-11-22_ss_td3_s11_1800k`, trained on the sensor priors) flew
`takeoff_and_switch` from the Mac: it held its position (spread 1.4-1.8 cm) but shook. Its motor
commands jumped by 20-50% of full range every 20 ms (motor 2 at 0 or full 20% of the time),
73% of the oscillation power at 15-25 Hz (the 50 Hz log's limit). The rotor-speed inputs were
right (`rltr.r*` equal the measured speeds). Two gaps to the simulator explain it:

- The drone's gyro noise is correlated from step to step (hover survey: lag-1 correlation 0.54-0.66
  on x, 0.24-0.35 on y, 0.23-0.39 on z); the simulator's was white.
- The policies chatter with more estimate delay than training covered (0-12 ms): in the mission
  test the TD3 candidate's jitter is 0.10 at 6 ms, 0.12 at 12 ms and 0.20-0.22 at 20 ms (the
  flight's level); the DualProj candidate (0.058 at 6 ms) crashed 46-49 of 50 episodes at 20 ms.

Calibration 1 (plan Step 5, `config/sensors.yaml`, commit 04ea435): the survey's `still` and hover
values, ranges widened ~20% (gyro per axis with AR(1) correlation, `sensor_sim.h`
`imu.gyro_noise_*_x/y/z`; filter gyro and accelerometer noise in hover; range noise; Lighthouse
angle noise 4-12x below the prior). The filter's gyro noise (0.015-0.06 rad/s) assumes the hover
noise is vibration, not body motion; the survey cannot tell them apart. Heights, slide and logs
were not run, so the range below 4 cm, the flow and the drop-outs keep their priors. Training
then uses `L2F_DELAY_MAX_MS=20`, and flight candidates are picked by 100-episode tests at 6, 12
and 20 ms of estimate delay.

Calibration 2: room-independent. Training uses only sensor-level survey values (none of the
room-dependent ones: estimate noise, Lighthouse jumps), and `lighthouse.placement: random` in
`config/sensors.yaml` draws a new base-station layout every episode (distance 1.5-3.5 m from the
room centre, height 1.4-2.6 m, 60-180 deg apart, aimed within 0.5 m of the centre at 0-1 m height;
calibrations from the room file), so a policy is not tied to one room's Lighthouse setup. The
drone stores the drawn layout; the true one adds the geometry error, as with a room file.
`L2F_LH_PLACEMENT=room` tests in the room file's layout instead (without the key: room, so older
recipes reproduce; checked identical). The IH222A-trained TD3 candidate ends 45% of
`test_sensor_sim` episodes early in random layouts against 20% in IH222A (yaw error 90th
percentile 9.5 against 4.3 deg): the room dependence this removes.

Found on the way: the flown filter's range update asserts when the drone rests on the floor with
an almost level estimate: `kalmanCoreFinalize` leaves R₂₂ = 1.000000119 and `acosf` of it in
`mm_tof.c` is NaN (`!isnan(HPHR)`, kalman_core.c:249). The simulator ends such an episode as a
crash, about 1 in 10,000 episodes; the same code runs on the drone, where it would halt the
estimator on the floor.

Bug fixed on the way (`evaluate_robustness … mission`, sensor path only): the mission test put
the drone on the floor after the episode start had started the estimator at the sampled state,
and it moved the target without moving the room frame (`world_offset`), so the sensors saw the
drone move against the target. Sensor-path policies crashed 82–92% in that test while training
normally. The test now restarts the estimator at the resting state, shifts the room frame and
the estimate as training does, and ends an episode as a crash when the estimator fails. The old
path is unchanged (bit-identical rows, checked). No earlier result used this test on the sensor
path.

## Multi-objective critic (`MOO=1`)

Groundwork for multi-objective training (gradient aggregation such as UPGrad).
`L2F_MOO` (`src/moo/td3_moo.h`) replaces the TD3 step: the reward
`2 - 0.5 * sum(w_k * cost_k)` is split into 8 parts that add up to it exactly
(survival constant, position, orientation, velocity, action, action rate,
angular acceleration, landing speed), the critics get one output per part,
and the actor gets one gradient per part. The twin-critic minimum is taken on
the summed value so the summed TD target equals the scalar one.
Without the flag training is bit-identical to before (checked on a 100k-step run).

The per-part gradients are combined with weights computed from their 8 x 8 Gram
matrix (`src/moo/aggregators.h`, double precision), chosen with `AGG=`:

| `AGG=` | Combination |
| --- | --- |
| `sum` (default) | plain sum: ordinary TD3 on the summed value |
| `upgrad` | UPGrad (Quinton & Rey, 2024): each gradient projected onto the dual cone of all of them, then added |
| `aligned_mtl` | Aligned-MTL (Senushkin et al., 2023): gradients rescaled to equal, smallest-eigenvalue strength |
| `svd_pga`, `osnmf_pga` | SVD-PGA / OS-NMF-PGA (`src/moo/aggregators_pga.h`), ported from R. Khosrowshahi's TorchJD fork ([rkhosrowshahi/torchjd](https://github.com/rkhosrowshahi/torchjd), branch `feat/pga-aggregators`) |

They follow TorchJD's defaults with an all-ones preference vector (TorchJD's
default is all 1/8; the weights are exactly 8 times those). Adam makes the actor
step almost blind to that overall scale. `python l2f/sim/check_aggregators.py`
builds `src/moo/check_aggregators.cpp` and compares its weights with TorchJD
on random, conflicting, rank-deficient and badly scaled Gram matrices, and with
`--log $BUILD/moo_gram_seed0.csv --agg upgrad` on the matrices and weights a
training run logged. Results (2026-11-18, relative error of the combined
direction): UPGrad 3e-12 or better, Aligned-MTL 2e-11 or better, except 1e-7
when the gradient norms span 6 decades (condition number 1e12, the limit of
any eigensolver). `AGG=sum` gives the same updates as before the change
(identical gradient logs over a 100k-step run). The PGA references need the
fork installed (otherwise they are skipped); SVD-PGA matches to 1e-12 or better
(5e-9 on the weights logged in training). OS-NMF-PGA matches except when a row
of its factor H is zero (~7% of real updates): the method then completes the
basis with an arbitrary null-space vector, so any two implementations differ
there (up to 60% in direction).

Aggregators on the 8 reward parts (variant E, `MOO_TOTAL=1`, seed 0, medians
over checkpoints 1.2M-2.0M): none beats the plain sum.

| `AGG=` | Circle | Uneven motors | Crashed | Jitter |
| --- | --- | --- | --- | --- |
| `sum` | 6.1 cm | 8.3 cm | 1.7% | 0.056 |
| `aligned_mtl` | 8.3 cm | 17.0 cm | 10.1% | 0.109 |
| `osnmf_pga` | 8.3 cm | 18.2 cm | 18.5% | 0.125 |
| `svd_pga` | 14.9 cm | 56.5 cm | 27% | 0.106 |
| `upgrad` | 348 cm | 199 cm | 66% | 0.214 |

UPGrad is unstable here (it raises the action-rate weight to 7-18x).

`train_mission.sh` options: `MOO=1`, `MOO_TOTAL=1` (critic loss below),
`AGG=<name>`, `MOO_PARTS=1` (all parts in one output, a code-path check),
`MOO_NORM=1`, `CRITIC_HIDDEN=<n>`. With `MOO=1`, every 50 actor updates, the
per-part gradient norms, pairwise cosines, aggregator weights and two
self-checks (parts add up to the stored reward; the combined per-part gradients
equal one backward pass on the weighted value) go to `$BUILD/moo_seed$SEED.csv`,
and the full Gram matrices with the weights to `$BUILD/moo_gram_seed$SEED.csv`.

Sanity check, variant E, 3 seeds, medians over checkpoints 1.2M-2.0M (mission
test as above):

| Critic | Circle | Uneven motors | Crashed | Jitter |
| --- | --- | --- | --- | --- |
| E, one output (rerun) | 6.1 cm | 11.4 cm | 2.8% | 0.063 |
| 8 outputs, one loss term per part | 8.2 cm | 14.5 cm | 12.9% | 0.148 |
| 8 outputs merged into 1 (`MOO_PARTS=1`) | 6.4 cm | 11.7 cm | 2.1% | 0.069 |
| 8 normalised outputs (`MOO_NORM=1`) | 110 cm | 112 cm | 100% | 0.185 |
| 8 outputs, 256 hidden (1 seed) | 9.7 cm | 19.0 cm | 5.7% | 0.083 |
| **8 outputs + total loss (`MOO_TOTAL=1`)** | **4.3 cm** | **8.8 cm** | **0.7%** | **0.059** |

Fitting each part on its own learned far slower (still crashing at 1M steps,
where E flies), because the shared 64-unit critic must get every part right
instead of only the total. Adding the total's TD error to the loss,
`(sum_k e_k)^2 + sum_k e_k^2`, keeps the total as accurate as in E and fixes it.
Use `MOO=1 MOO_TOTAL=1` for multi-objective runs.

Per-part gradients after 1M steps (3 seeds): survival conflicts with action
rate, action and position (median cosine -0.35 / -0.28 / -0.22, negative in 63%
of updates); orientation, velocity, landing speed and angular acceleration point
the same way (cosine ~+0.9).

### PyTorch MOO on the no-bonus reward (2026-11-22)

`l2f/py/train.py --moo parts --total 1` (TorchJD aggregators), variant E, no survival bonus,
crash penalty 200 (the best Step 1 reward), 2 seeds, late-checkpoint medians. Round 1: `sum`,
Aligned-MTL, UPGrad; round 2: DualProj, PCGrad, and plain PyTorch TD3 as the same-trainer
reference.

| Objectives | Aggregator | Circle | Uneven motors | Crashed | Jitter |
| --- | --- | --- | --- | --- | --- |
| – | plain TD3, C++ (3 seeds) | 5.3 cm | 10.0 cm | 0.8% | 0.070 |
| – | plain TD3, PyTorch | 4.6 cm | 8.8 cm | 0.9% | 0.073 |
| 8 parts | `sum` | 5.2 cm | 10.2 cm | 0.5% | 0.070 |
| 8 parts | `aligned_mtl` | 5.6 cm | 11.6 cm | 7.5% | 0.078 |
| 8 parts | `upgrad` | 33.7 cm | 88.0 cm | 56.5% | 0.195 |
| 8 parts | `dualproj` | 5.7 cm | 13.7 cm | 3.8% | 0.079 |
| 8 parts | `pcgrad` | 109.1 cm | 113.9 cm | 81.5% | 0.183 |
| 3 groups | `aligned_mtl` | 8.6 cm | 12.0 cm | 9.4% | 0.067 |
| 3 groups | `upgrad` | 6.2 cm | 11.1 cm | 0.9% | 0.071 |
| 3 groups | `dualproj` | 4.6 cm | 9.2 cm | 0.7% | 0.066 |
| 3 groups | `osnmf_pga` | 5.5 cm | 11.1 cm | 1.7% | 0.073 |
| 3 groups | `svd_pga` | 52.0 cm | 61.5 cm | 51.4% | 0.076 |

None beats the sum. DualProj on 3 groups ties it (between plain TD3 and `sum` on every column;
0.2 points of crashes is about one crash in 400 episodes). `python3 l2f/py/moo_report.py logs/train_moo_*.csv` shows why: with the bonus
gone, the crash-penalty value ("survival") conflicts strongly with smoothing and action (cosine
−0.8 to −0.9, negative in ~100% of updates, until ~1.2M steps), and late in training position
conflicts with survival, orientation, angular acceleration and landing speed (about −0.8). So the
conflict is real here, unlike with the bonus, but resolving it moves the trade-off away from the
tuned weights: UPGrad raises the smoothing weight to 8–12× (the policy gets too careful and
crashes), Aligned-MTL scales the step to the smallest singular value (weights 0.01–0.3, some
negative). On 3 groups UPGrad's weights return to ~1 late in training, hence its result near the
sum.

Round 2 shows the mechanism. Per-objective projections amplify a small objective that opposes a
large one: PCGrad replaces g_i by g_i − (g_i·g_j / |g_j|²) g_j when g_i·g_j < 0, so projecting
the survival gradient (norm 1.0) off smoothing (norm 0.10, cosine −0.95) adds about
0.95 · 1.0 / 0.10 ≈ 9.5 smoothing gradients. The logged PCGrad weights show it: smoothing 2.8x
early, 6.6–8.3x from 0.8M, the rest 1–3x (aggregation check 4e-7, so this is PCGrad itself, not
the wiring). UPGrad on 8 parts does the same; on 3 groups smoothing is merged with action, angular
acceleration and the rate term, so no group is that small. DualProj projects only the sum onto
the dual cone (the smallest change to the sum that hurts no objective): on 8 parts (seed 0) its
weights stay at 1 in all but one window (smoothing 2.2x at 0.4–0.8M), which is why it lands near
the sum.

PGA on 3 groups (round 3; the C++ rounds had tried it on 8 parts only, with the bonus). PGA
averages the oriented principal directions of the gradients with equal weight,
w = (1/t) Σ_j v_j / √λ_j for the Gramian's eigenpairs (λ_j, v_j): a whitening, so the step norm
is always 1/√3 and the largest gradient gets the smallest weight. Here the largest is safety,
the group that holds the crash penalty. SVD-PGA seed 0 never learns to fly (100% crashed;
training return falls from −283 to −682 over the run; safety weight 0.33–0.83 while its
gradient is the largest). Seed 1 flies (5.0 cm / 13.7 cm / 2.7%), but late in training its
weights turn negative (safety −0.98, tracking −0.26 at 1.6–2.0M), which climbs those costs.
Aggregation check ≤ 1.7e-6, no non-finite values, so this is SVD-PGA itself. OS-NMF-PGA keeps
the factor H nonnegative and stays usable (seeds 4.5 cm / 0.5% and 6.0 cm / 3.0%), but behind
the sum and DualProj.

Decision (2026-11-22, plan Q17): MOO stays, with DualProj on 3 groups as the default aggregator
(`train.py --moo parts --parts 3 --agg dualproj --total 1`, as in the table); the sum and plain
TD3 stay as references.

Untuned weights (`build_py_E_uniform`: `-DL2F_REWARD_UNIFORM`, every cost weight 1, no weight
curriculum, same no-bonus / crash-penalty-200 reward; `train.py` skips its weight curriculum
when the module reports `weight_curriculum = False`). Plain TD3, UPGrad and DualProj on 3
groups and Aligned-MTL on 8 parts, 2 seeds each: every run crashes in 100% of the evaluation
episodes (circle 97–120 cm), so the test cannot rank the aggregators. Two reasons, both from
the reward, not the trainer:

* Crashing is the optimum. With a good flying policy (the PyTorch TD3 reference above) the
  angular-acceleration part costs −1819 per step at weight 1, against −0.36 at the tuned
  2e-4; one step costs −1819 in total instead of −1.08. Over the ~100-step horizon of γ = 0.99
  that is about −182,000 against a one-time −200 for a crash. Without a survival bonus every
  cost part ends with the episode, so each of them prefers a crash; only the crash penalty does
  not, and here it is 1000x too small.
* The actor stops learning. The critics must fit values of the order of −10⁵; from ~0.4M
  steps the logged actor gradient is exactly 0 for every objective (FAST_TANH units clipped
  beyond ±3 pass no gradient), against 0.2–4 in the tuned runs. With all-zero gradients every
  aggregator returns the sum, so UPGrad and DualProj trained identical networks.

So half of what the hand-picked weights do is unit conversion (2e-4 turns rad²/s⁴ into the
scale of the other terms) and setting the total cost below the crash penalty, which no gradient
aggregator can do. A fair untuned test would first scale every part to the same typical size
and set the total cost per step to the tuned one (two chosen numbers instead of seven); it is
not run.

### Task objectives and conflict diagnostics

More `train_mission.sh` options (all off by default; the default build trains
bit-identically, checked on 100k-step runs):

| Option | Effect |
| --- | --- |
| `TASKS=1` | framing B: the actor objectives are the 5 mission episode types (free flight, take-off, landing, circle, line), each from its own minibatch; scalar critic, the policy does not see the task |
| `TASKS=9` | as `TASKS=1` plus figure-eight, fast circle, point-to-point (target step) and gust hover; every mission type 10% of the episodes |
| `TASKS=regime` | the 9 objectives are hidden dynamics regimes: delay 0-20 ms x mass +-20%, in thirds |
| `HARD=1` | the harder episode mix of `TASKS=9` without task objectives |
| `AGG=task_share` | (tasks only) each task weighted by its replay-buffer share = plain TD3 split per task |
| `REWARD=uniform` | every cost weight 1, no weight curriculum (untuned-reward test) |

With `MOO=1`, `$BUILD/moo_noise_seed$SEED.csv` repeats each objective's gradient
on a second, independent minibatch (own random generator, training unchanged):
`same_k` is the cosine of objective k across the two batches (is its direction
reproducible?), `cross_a__b` the a-b cosine without shared-sample noise.
`python l2f/sim/moo_summary.py conflicts $BUILD` summarises these logs per
training window (for reward parts also the tracking / smoothness / safety
groups); `moo_summary.py checkpoints <outputs>` gives the late-checkpoint medians
used in the tables here.

Findings (2026-11-18/19, seed 0 unless noted): per-objective gradients are not
noise (same-objective cosine 0.92-0.99). Splitting by task gives no conflict:
mission tasks, the harder tasks and the dynamics regimes all have pairwise
cosines of +0.87 to +0.99 after 800k steps, never negative, so a task split
looks like a random split; Aligned-MTL on tasks (3 seeds: 6.3 cm, 3.0% crashed)
matches `task_share` (5.6 cm, 3.4%) and plain E. Equal task weights (`AGG=sum`
with `TASKS=1`) collapse (100% crashed). Real conflict appears only between
reward parts, mainly survival against the costs during the curriculum (cosine
about -0.8 until ~1.2M steps, near 0 later). The untuned reward crashes every
episode for every aggregator (the actor saturates within ~50 updates).
