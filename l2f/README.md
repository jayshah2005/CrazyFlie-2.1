# Learning to Fly on the Crazyflie 2.1 Brushless

A learned controller for our **Crazyflie 2.1 Brushless** (`cf21bl`, Lighthouse + Flow deck),
after [Learning to Fly in Seconds](https://arxiv.org/abs/2311.13081). A small neural network
(the *policy*) flies the whole mission with no PID: take-off from the floor, hover, a circle,
landing. It is trained in simulation and runs on the drone at 100 Hz.

- First learned hover 2026-11-01; first full mission 2026-11-04 (policy I1, 4-8 cm tracking).
- Now (2026-11-22): training on the **sensor path**, where the policy sees the output of the
  drone's own Kalman filter fed by simulated sensors calibrated with the sensor survey, in
  random base-station layouts, so a policy is not tied to one room.
- Results and history: [`docs/results.md`](docs/results.md). Plan and open questions:
  [`pytorch_training_plan.html`](pytorch_training_plan.html) (open in a browser).

## What is where

| Path | What |
| --- | --- |
| `learning_to_fly/` | The simulator and the C++ trainer: upstream learning-to-fly with our changes ([its README](learning_to_fly/README.md)) |
| `py/` | PyTorch training on the same simulator (TD3, multi-objective TD3 with TorchJD), the simulator's Python module, checks |
| `sim/` | Training and comparison scripts: `train_mission.sh` (C++), `compare.sh`, `cross_eval.sh`, `summarize_checkpoints.py`, option sets |
| `estimator/` | The drone's Kalman filter, compiled from the firmware as a library for the sensor path |
| `firmware/` | Builds firmware with a trained policy (`build_firmware.sh`, `patch_firmware.py`) |
| `policies/` | Flight candidates: checkpoint, firmware header, test results ([README](policies/README.md)) |
| `docs/results.md` | What was measured, trained, compared and flown |
| `../config/sensors.yaml` | Sensor models of the sensor path (measured values from the survey) |
| `../config/landing.yaml` | Landing values shared by training and `rl_trigger.py` |

## How it works

- **Policy:** 25 inputs (variant E: position error to the target, orientation, velocity,
  angular velocity, the 4 measured rotor speeds, the running position-error sum) to 4 rotor-speed
  commands, 100 times a second.
- **Training:** TD3. The policy tries actions in the simulator and a second network (the critic,
  which sees the true state) learns how good they are. 2M steps of 10 ms; a checkpoint every
  100k steps.
- **Episodes** (5 s each): take-off from the floor, landing, circles, lines and free flight from
  random states; a curriculum ramps them in over 800k steps. Every episode varies mass, thrust,
  motor response, uneven motors, a push, ground effect and an estimate delay of 0-20 ms.
- **Reward:** costs for position error, tilt, speed, effort, command changes, angular
  acceleration and landing speed; a crash costs 200; spinning faster than 34.9 rad/s is a crash.
- **Sensor path** (`SENSOR_SIM=1`): the policy sees the Kalman filter's estimate, fed with
  simulated IMU, range, flow and Lighthouse sweeps (`../config/sensors.yaml`); the critic sees the
  truth.
- **Missions:** the flight script moves the target (up, out, around the circle, back, down) and
  cuts the motors at the end; the policy only follows the target.
- **Choosing a policy:** TD3 is noisy late in training, so every late checkpoint is tested and
  the best is flown, never just the last.

| Term | Meaning |
| --- | --- |
| checkpoint | The policy saved every 100k steps (`actor_<step>.h5`, firmware header `.h`) |
| variant I1, A-E | Which inputs the policy gets (`VARIANT=`): I1 the last 32 commands, E the measured rotor speeds and no commands |
| mission test | `evaluate_robustness <actor.h5> N mission`: the flight script's mission in simulation, clean and with uneven motors; circle error, crashes, jitter |
| jitter | Spread of the motor commands in hover (the paper's smoothness metric); lower is smoother |
| MOO | Multi-objective training: the critic has 3 outputs (tracking, smoothness, safety), the policy's update combines their gradients (`--moo parts --parts 3 --agg sum` or `dualproj`) |

## Setup

1. Toolchain. Ubuntu: `sudo apt install cmake build-essential libhdf5-dev libopenblas-dev
   libboost-all-dev protobuf-compiler libprotobuf-dev` (`gcc-arm-none-eabi` for firmware). macOS:
   Xcode tools, Homebrew `cmake hdf5 boost protobuf`, and Arm's GNU toolchain for firmware
   (`arm-none-eabi-gcc`); `sim/platform.sh` picks Apple's Accelerate instead of OpenBLAS.
2. Submodules, from the repository root:
   `git submodule update --init l2f/learning_to_fly/external/rl_tools l2f/learning_to_fly/controller`,
   `git -C l2f/learning_to_fly/external/rl_tools submodule update --init external/cli11 external/highfive external/json external/tensorboard`,
   and for firmware `git -C l2f/learning_to_fly/controller submodule update --init external/rl_tools`.
3. The firmware tree for the sensor path and firmware builds: `~/crazyflie-firmware-2026.08`
   (`firmware/build_firmware.sh` clones it on first use, or `git clone --branch 2026.08`).
4. Python 3.11+ with `torch numpy pybind11 h5py scipy quadprog` and the TorchJD fork with the PGA
   aggregators: `git clone -b feat/pga-aggregators https://github.com/rkhosrowshahi/torchjd ../torchjd`
   and `pip install -e "../torchjd[quadprog_projector]"`.

Training runs on CPU, one core per run; run many seeds in parallel. Builds on different
machines round differently (OpenBLAS and Accelerate), so compare runs from one machine only.

## Train

**PyTorch (current).** Build the simulator module with the recipe's flags, then train:

```
R="-DL2F_BRUSHLESS -DL2F_REAL_NOISE -DL2F_ANG_ACC_WEIGHT=2e-4 -DL2F_LANDING_SPEED_WEIGHT=5 -DL2F_POS_INTEGRAL
   -DL2F_MISSION -DL2F_STEP_LIMIT=2000001 -DL2F_REPLAY_CAP=700000 -DL2F_RPM_OBS -DL2F_HISTORY=0
   -DL2F_CRASH_PENALTY=200 -DL2F_SURVIVAL_BONUS=0 -DL2F_ACTION_RATE_WEIGHT=1 -DL2F_SENSOR_SIM"
source l2f/sim/platform.sh
cmake -S l2f/learning_to_fly -B l2f/learning_to_fly/build_py_E_nb200_ss -DCMAKE_BUILD_TYPE=Release \
  "${CMAKE_PLATFORM[@]}" -DL2F_BUILD_PYTHON=ON -DPython_EXECUTABLE=$(which python3) -DCMAKE_CXX_FLAGS="$R"
cmake --build l2f/learning_to_fly/build_py_E_nb200_ss -j --target l2f_sim evaluate_robustness h5_to_header
B=l2f/learning_to_fly/build_py_E_nb200_ss/src
python3 l2f/py/train.py --build $B --variant E --seed 0 --evaluate \
  --env L2F_OMEGA_CRASH=34.9 --env L2F_DELAY_MAX_MS=20                      # plain TD3
python3 l2f/py/train.py ... --moo parts --parts 3 --agg dualproj --total 1   # MOO (or --agg sum)
```

`--env KEY=VALUE` sets a runtime option for training and for the test after it. Runs go to
`learning_to_fly/checkpoints/multirotor_td3/<time>_py_E_..._<pid>/`; `--evaluate` prints the
mission-test table of the late checkpoints.

**C++.** `sim/train_mission.sh` builds and trains with options as environment variables, then
tests the checkpoints, e.g.
`VARIANT=E SURVIVAL_BONUS=0 CRASH_PENALTY=200 OMEGA_CRASH=34.9 SENSOR_SIM=1 EVAL_EPISODES=100 SEED=0 bash l2f/sim/train_mission.sh`.
Its header lists every option; the recipe of the flown I1 is the default.

### Training options

Build options change the compiled simulator (C++: `train_mission.sh` variables; PyTorch: the
`-D` flags); runtime options are environment variables (`L2F_*`, `--env` for `train.py`). With
none set, training reproduces the earlier recipes bit for bit.

| Option | Kind | Effect |
| --- | --- | --- |
| `VARIANT=I1\|A-E` | build | The policy's inputs (see the `train_mission.sh` header) |
| `CRASH_PENALTY=P`, `SURVIVAL_BONUS=c` | build | Reward −P on a crash; the per-step bonus (paper 2). The baseline is P 200, bonus 0 |
| `SMOOTHING=rate\|zigzag\|none` | build | Command-change cost: action rate (default), zig-zag Σ(a_t − 2a_t−1 + a_t−2)², or none |
| `SENSOR_SIM=1` | build | The sensor path; `SENSORS=<file>` another sensor file |
| `EVAL_EPISODES=100` | build | rl_tools' in-training test (only the `Step:` log lines; the weights are the same at any value); 100 makes C++ sensor-path runs several times faster |
| `HARD=1` | build | Harder episode mix: figure-eight, fast circle, target steps and gusts too |
| `REWARD=uniform` | build | Every cost weight 1, no weight curriculum (untuned-reward test) |
| `OMEGA_CRASH=34.9` | runtime | Rotating faster than this (rad/s) is a crash |
| `GYRO_CLIP=34.9` | runtime | Old path: the gyro reading clipped at the sensor's range (the sensor path clips its own) |
| `L2F_DELAY_MAX_MS=20` | runtime | Estimate delay drawn from 0 to this per episode (default 12; 20 is the model's cap) |
| `FORCE_SCALE=x` | runtime | Strength of the random push (1 = recipe; keep 1) |
| `LANDING_CUT=1` | runtime | Land like `rl_trigger.py`: motor cut near the floor ends the episode (`../config/landing.yaml`) |
| `POS_NOISE=1` | runtime | Old path: per-episode position noise from `sim/noise_measured.json` |
| `L2F_LH_PLACEMENT=room\|random` | runtime | Sensor path: base stations as in the room file, or a new layout per episode (`sensors.yaml` default: random) |

## Test and compare

- **One run:** `python3 l2f/py/train.py --build $B --evaluate-run <run folder> --episodes 100
  --env L2F_EVAL_DELAY_MS=12` tests every late checkpoint (100 episodes per condition; also
  `L2F_LH_PLACEMENT=room` to test in the real room's layout). Pick flight candidates across 6, 12
  and 20 ms of delay, not 6 ms only.
- **Several seeds:** `python3 l2f/sim/summarize_checkpoints.py [--no-jumps] <outputs>`: medians over
  checkpoints 1.2M-2.0M per file and pooled (`--no-jumps` on the sensor path).
- **Option sets:** `CONFIGS=l2f/sim/step1_configs.txt bash l2f/sim/compare.sh` runs each set of
  `train_mission.sh` options over seeds; `sim/cross_eval.sh` re-tests runs at other force levels.
- **Checks** (`py/`): `test_sim.py` (module = C++ simulator), `test_networks.py` (PyTorch actor =
  rl_tools, export), `test_td3.py` (PyTorch TD3 = rl_tools' updates), `test_moo.py` (MOO sum =
  TD3), `test_sensor_sim.py` (estimate against truth in closed loop),
  `estimator/build/check_estimator`; `step_response.py` (target steps), `moo_report.py` (gradient
  conflicts of MOO runs).

## Calibrate the sensors

`../noise_survey.py` measures the drone's sensors (`still`, `heights`, `hover`, `slide`, `logs`;
see its header). Only sensor-level values go into training: gyro and accelerometer noise and
correlation, range noise, Lighthouse angle noise, written into `../config/sensors.yaml` with their
measured ranges. Estimate-level values (position noise, Lighthouse jumps) depend on the room and
are not used.

## Firmware and flying

```
ACTOR=$PWD/l2f/policies/<folder>/actor.h HISTORY=0 RPM_OBS=1 \
  OFFICIAL_ZIP=<firmware-cf21bl-2026.08.zip> bash l2f/firmware/build_firmware.sh
python3 flash_cold.py ~/l2f_2026/build/firmware-cf21bl-2026.08-l2f.zip   # cold boot only, then power-cycle
python3 rl_trigger.py --mode mission                                      # from the repository root
```

- `HISTORY` / `RPM_OBS` must match the variant (E: 0 / 1). The official zip is the 2026.08 release
  asset; never flash this drone with a warm boot (`cfloader -w`).
- `rl_trigger.py` modes: `mission` (the policy flies everything from the floor), `takeoff_and_switch`
  (PID take-off, then the policy), `land`, `hover_learned`, `figure_eight`, `hover_original`.
- **Enter** starts a flight and **Enter** again stops it (`--hold`: fly only while held). A stop in
  the air is an automatic landing in `mission` mode and a motor cut otherwise; **Ctrl+C** cuts the
  motors at once. On macOS give the terminal the Input Monitoring permission;
  `rl_trigger.py --key-test 20` checks the keys without flying.
- Each flight writes `logs/rl_trigger_flight_*.csv`; `rltr.r1-r4` (`rin1-4`) are the rotor speeds
  the policy saw.

Firmware changes on top of the paper's controller: built for `cf21bl`; boots on PID (the learned
controller only runs when selected); policy at 100 Hz; motor commands every 1 kHz tick (DShot);
the action history starts from the measured rotor speeds; zero-thrust setpoints keep the
commander watchdog quiet; target offset and velocity in the learned-controller packet. Details in
`firmware/build_firmware.sh` and `docs/results.md`.

## What we changed in `learning_to_fly/`

Every change sits behind an `L2F_*` switch (`grep -rn L2F_ learning_to_fly/src learning_to_fly/include`
finds each, with a comment where it is read).

| File (under `learning_to_fly/`) | Our changes |
| --- | --- |
| `include/.../parameters/dynamics/crazyflie_brushless.h` | Measured model of this drone |
| `include/.../parameters/reward_functions/brushless.h` | Reward terms and the crash penalty |
| `include/.../simulator/multirotor.h` | Mission state, randomization, delay, glitch and sensor-path settings |
| `include/.../simulator/operations_generic.h` | Mission episodes, floor contact, ground effect, delay, rotor-speed inputs, the sensor path's steps |
| `include/.../simulator/sensor_sim.h` | Simulated sensors through the flown Kalman filter (`../estimator`) |
| `src/config/` | Which model, reward, noise and inputs a build uses; step limit, replay cap, test episodes |
| `src/steps/curriculum.h` | The mission ramp and the reward-weight curriculum |
| `src/evaluate_robustness.cpp` | The mission test and robustness evaluations |
| `src/h5_to_header.cpp` | Checkpoint to firmware header |
