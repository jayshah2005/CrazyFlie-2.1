# Flight candidates

Policies picked for flights, one folder each: `actor.h5` (checkpoint), `actor.h` (firmware
header, `src/h5_to_header`), `run.txt` (the training run's recipe line) and `test100.txt` (the
100-episode mission test of the run's late checkpoints).

## 2026-11-22: sensor path, new baseline (plan Steps 3 and 6)

Common recipe: variant E (measured rotor speeds, no action history), PyTorch training
(`l2f/py/train.py`, build `build_py_E_nb200_ss`: `-DL2F_SENSOR_SIM`, no survival bonus, crash
penalty 200, action-rate penalty), rotation crash at 34.9 rad/s, the recipe's full random force,
2M steps, trained on the lab computer at commit 9384031. The actor observed the flown Kalman
estimator fed by simulated sensors with the **priors** of `config/sensors.yaml` (not yet
calibrated with the survey) and the **new IH222A base-station placement** (config fingerprint
7f90c03d0232).

Picked from 12 runs (TD3, sum and DualProj on 3 groups, seeds 10-13) by a mission test of every
checkpoint from 1.0M to 2.0M with 100 episodes per condition (`test100.txt`):

| Folder | Training | Clean: crashed / circle | Uneven motors: crashed / circle | Jitter |
| --- | --- | --- | --- | --- |
| `2026-11-22_ss_td3_s11_1800k` | plain TD3, seed 11, 1.8M steps | 0/100, 3.4 cm | 0/100, 7.1 cm | 0.101 |
| `2026-11-22_ss_dualproj3_s10_1800k` | MOO, DualProj on 3 groups (`--moo parts --parts 3 --agg dualproj --total 1`), seed 10, 1.8M | 0/100, 3.2 cm | 0/100, 8.7 cm | 0.058 |

The DualProj policy commands the motors about half as unevenly (jitter); the TD3 policy tracks
slightly better with uneven motors. Fly both next to the current E policy.

**Flown 2026-11-22:** `2026-11-22_ss_td3_s11_1800k` held its position but shook (motor commands
between 0 and full at 15-25 Hz or faster): see `../docs/results.md` (First flight). Do not fly the
DualProj one: it crashes 84-92% at 20 ms of estimate delay in simulation.

## 2026-11-22 evening: calibration 2 (room-independent)

Recipe as above plus the survey's measured sensor values (correlated per-axis gyro noise), a
new random base-station layout every episode, and an estimate delay of 0-20 ms in training;
commit 58c5dfd. Picked from 12 runs (TD3, sum and DualProj, seeds 30-33) by 100-episode tests in
random layouts at 6, 12 and 20 ms of delay and in the IH222A layout at 12 ms (`test100.txt`).

| Folder | Training | Crashed (800 test episodes) | Circle / uneven motors | Jitter at 12 / 20 ms |
| --- | --- | --- | --- | --- |
| `2026-11-22_cal2_td3_s30_2000k` | plain TD3, seed 30, 2.0M | 2 (both at 20 ms, uneven motors) | 4.0 / 8.8 cm | 0.13 / 0.21 |

It survives delay and unseen layouts far better than the first candidates (on the same tests the
first TD3 candidate crashed 7% at 20 ms, the DualProj one 84-92%), but its jitter is about the
same as the first TD3 candidate's, so it may still shake: a smoothness comparison (action-rate
weight 2, 4, 8) is training.

## Build and flash

```
ACTOR=$PWD/l2f/policies/<folder>/actor.h HISTORY=0 RPM_OBS=1 \
  OFFICIAL_ZIP=<firmware-cf21bl-2026.08.zip> bash l2f/firmware/build_firmware.sh
python3 flash_cold.py ~/l2f_2026/build/firmware-cf21bl-2026.08-l2f.zip
```

`HISTORY=0 RPM_OBS=1` must match variant E. The official zip is the 2026.08 release asset
(github.com/bitcraze/crazyflie-release). Flash with a cold boot only, then power-cycle. Fly a
`takeoff_and_switch` hover first (`rl_trigger.py`, Windows only for now), then missions.
