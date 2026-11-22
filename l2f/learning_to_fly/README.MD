# learning_to_fly (vendored, modified)

A copy of [arplaboratory/learning_to_fly](https://github.com/arplaboratory/learning_to_fly), the
simulator and TD3 trainer of *Learning to Fly in Seconds* (Eschmann, Albani, Loianno, RA-L 2024),
changed for this project. Start with [`../README.md`](../README.md); this folder is the
simulator and the C++ trainer that the scripts in `../sim` and `../py` build.

| Path | What |
| --- | --- |
| `include/learning_to_fly/simulator/` | The multirotor simulator: dynamics (`parameters/dynamics/crazyflie_brushless.h`, the identified Crazyflie 2.1 Brushless), rewards (`parameters/reward_functions/brushless.h`), mission episodes and curriculum, domain randomization, estimate delay (`operations_generic.h`), and the sensor path (`sensor_sim.h`: simulated sensors through the flown Kalman filter, `../estimator`) |
| `src/training.cpp`, `training.h`, `config/`, `steps/` | C++ TD3 training (`training_headless`), options set by `../sim/train_mission.sh` |
| `src/evaluate_robustness.cpp` | The mission test and robustness evaluations of checkpoints |
| `src/h5_to_header.cpp` | Checkpoint `.h5` to the firmware's actor header (also for PyTorch checkpoints) |
| `controller/`, `external/rl_tools` | Submodules: the firmware controller and RLtools |

Removed from upstream (2026-11-22) because this project does not use them: the web UI, the
CUDA benchmark, upstream's tests, the Docker files, the ablation study and the original
`scripts/trigger.py` (replaced by `rl_trigger.py` in the repository root). They are in
upstream and in this repository's git history.

Upstream's simulator has since moved into RLtools
([rl_tools/rl/environments/l2f](https://github.com/rl-tools/rl-tools/tree/master/include/rl_tools/rl/environments/l2f));
this copy is kept for the changes listed in `../README.md`.

```
@ARTICLE{eschmann2024learning,
  author={Eschmann, Jonas and Albani, Dario and Loianno, Giuseppe},
  journal={IEEE Robotics and Automation Letters},
  title={Learning to Fly in Seconds},
  year={2024},
  doi={10.1109/LRA.2024.3396025}}
```

License: [`LICENSE`](LICENSE) (upstream's).
