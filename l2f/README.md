# Learning to Fly on the Crazyflie 2.1 Brushless

Port of [Learning to Fly in Seconds](https://arxiv.org/abs/2311.13081)
([arplaboratory/learning-to-fly](https://github.com/arplaboratory/learning-to-fly))
to our **Crazyflie 2.1 Brushless** (`cf21bl`) with Lighthouse + Flow deck.
First stable learned hover: 2026-11-01, 35 s, ~4 cm position error.

Everything here runs in **WSL Ubuntu 24.04**. Flying runs on Windows from the
repo root with `rl_trigger.py`.

| Path | What |
| --- | --- |
| `sim/learning_to_fly-brushless.patch` | Brushless dynamics, reward baseline, step-limit flag and `evaluate_robustness` for learning-to-fly `d07592d` |
| `sim/train_brushless.sh` | Apply patch, train (default 1M steps), run the robustness check |
| `firmware/patch_firmware.py` | Hooks into crazyflie-firmware 2026.08 (learned-controller packet, motor path, PID boot) |
| `firmware/build_firmware.sh` | Out-of-tree controller build for `cf21bl` (fixes below), output `~/l2f_2026/build/cf21bl.bin` |

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

## Known issue

Motor commands chatter between ~0% and ~100% in flight (gyro noise ~75 °/s vs
~3 °/s under PID). The simulator's sensor noise is probably far below the
real drone's. Next: retrain with measured noise and a stronger action penalty.
