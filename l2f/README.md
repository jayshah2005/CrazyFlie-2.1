# Learning to Fly on the Crazyflie 2.1 Brushless

Port of [Learning to Fly in Seconds](https://arxiv.org/abs/2311.13081)
([arplaboratory/learning-to-fly](https://github.com/arplaboratory/learning-to-fly))
to our **Crazyflie 2.1 Brushless** (`cf21bl`) with Lighthouse + Flow deck.
First stable learned hover: 2026-11-01, 35 s, ~4 cm position error.
First full mission with no PID (take-off from the floor, 1.2 m wide circle,
landing): 2026-11-04, at 0.6 / 0.75 / 1.0 / 1.5 m, landing within 2-4 cm of the
start (see [Mission](#mission)).

Everything here runs in **WSL Ubuntu 24.04**. Flying runs on Windows from the
repo root with `rl_trigger.py`.

| Path | What |
| --- | --- |
| `sim/learning_to_fly-brushless.patch` | Brushless dynamics, reward terms, mission extension (below) and `evaluate_robustness` for learning-to-fly `d07592d` |
| `sim/train_brushless.sh` | Apply patch, train the hover policy (default 1M steps), run the robustness check |
| `sim/train_mission.sh` | Train the mission policy (recipe of the flown "I1") and compare all checkpoints |
| `firmware/patch_firmware.py` | Hooks into crazyflie-firmware 2026.08 (learned-controller packet, motor path, PID boot) |
| `firmware/build_firmware.sh` | Out-of-tree controller build for `cf21bl` (fixes below), output `~/l2f_2026/build/cf21bl.bin`; target velocity and position-error integral for mission policies |

## Brushless model

From two stock-PID flights (`sysid_hover.py` + `sysid_analyze.py`) and IDSIA
(Busetto et al., arXiv 2512.14450):

- mass 41.3 g; inertia 2.20e-5 / 2.97e-5 kg m² (IDSIA, mass-scaled); arm 35.35 mm
- thrust 3.86e-8 N/(rad/s)²; torque/thrust 0.0021 m; motor lag 0.06 s
- ESC map `rpm = 5400 * (ratio * vbat) + 3750` (inverted onboard, so commands compensate battery voltage)

## Steps

1. Toolchain (once): `sudo apt install cmake build-essential libhdf5-dev libopenblas-dev libboost-all-dev protobuf-compiler libprotobuf-dev gcc-arm-none-eabi`
2. Clone learning-to-fly at `d07592d` to `~/learning_to_fly`; init submodules
   `external/rl_tools`, `controller` (recursive) and, inside `external/rl_tools`,
   `external/cli11 external/highfive external/json external/tensorboard`.
3. `bash l2f/sim/train_brushless.sh` (~4 min for 1M steps).
4. `ACTOR=<checkpoint>.h bash l2f/firmware/build_firmware.sh`
5. Put `cf21bl.bin` into the official `firmware-cf21bl-2026.08.zip` in place of
   `cf21bl-2026.08.bin`, flash with cfclient **Bootloader → Cold boot → From file**,
   then power-cycle. Never flash this drone with `cfloader -w` (warm boot).
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

Simulator extension (`L2F_MISSION`, all in the patch):

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

Simulator (in the patch): `L2F_RPM_OBS` gives the actor the 4 speeds after the
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
