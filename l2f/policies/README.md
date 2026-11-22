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

## Build and flash

```
ACTOR=$PWD/l2f/policies/<folder>/actor.h HISTORY=0 RPM_OBS=1 \
  OFFICIAL_ZIP=<firmware-cf21bl-2026.08.zip> bash l2f/firmware/build_firmware.sh
python3 flash_cold.py ~/l2f_2026/build/firmware-cf21bl-2026.08-l2f.zip
```

`HISTORY=0 RPM_OBS=1` must match variant E. The official zip is the 2026.08 release asset
(github.com/bitcraze/crazyflie-release). Flash with a cold boot only, then power-cycle. Fly a
`takeoff_and_switch` hover first (`rl_trigger.py`, Windows only for now), then missions.
