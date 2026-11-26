<!-- group: Start | reviewed: 2026-11-26 -->
# Crazyflie learned controller: project documentation

This folder explains the whole project: a small neural network, trained with reinforcement
learning in simulation, that flies a **Crazyflie 2.1 Brushless** drone by setting its four motor
speeds 100 times a second, with no PID controller. It covers how the simulator, the reward and
the training work, the multi-objective (MOO) research on top of it, every experiment so far, and
the open problems. It is written for readers who know the basics of reinforcement learning and
actor-critic methods; every other idea is explained, with a link to review the math.

> [!TIP]
> **Open `documentation/index.html` in a browser** for the interactive version: search (press
> `/`), diagrams, formulas, sortable tables and widgets you can drag and tune. The same pages
> also read fine on GitHub, which renders their formulas and diagrams too; only the widgets need
> the viewer.

## How to open it

| Way | How | Notes |
| --- | --- | --- |
| Viewer from disk | Double-click `documentation/index.html` | Uses the bundled copy of the pages (`assets/bundle.js`). The Markdown, math and diagram libraries load from the internet (the browser caches them); offline, pages show as plain text. |
| Viewer with live pages | `python3 -m http.server 8000` in the repository root, then open `http://localhost:8000/documentation/` | Reads each `.md` file fresh, so edits show at once. Links to code files open the files. |
| GitHub or an editor | Open the `.md` files | Formulas (`$…$`) and Mermaid diagrams render on GitHub; widgets show a short note instead. |

## Where to start

| If you want to… | Read |
| --- | --- |
| Understand the project in ten minutes | [Overview](01-overview.md) |
| Refresh the RL ideas the project uses | [RL foundations](02-rl-foundations.md) |
| See how the pieces fit, from simulator to flight | [System architecture](03-system.md), then [Simulator](04-simulator.md) |
| Understand what the policy sees and does | [Policy inputs and outputs](05-policy-io.md) |
| Understand the reward and why it is so delicate | [Reward](06-reward.md) |
| Know exactly how training works | [TD3 as implemented](07-td3.md) |
| Work on sim-to-real | [Sensor path](08-sensor-path.md), [Firmware and flights](11-flight.md) |
| Work on the multi-objective research | [Multi-objective TD3](09-moo.md), [Waypoints](10-waypoints.md) |
| See every experiment and its result | [Experiment log](12-experiments.md) |
| Know what is broken or unknown | [Issues and open questions](13-issues.md) |
| Run something | [How-to](14-howto.md) |
| Look up a term or a symbol | [Glossary](15-glossary.md) |
| Add or change a page | [Conventions](CONVENTIONS.md) |

## The pages

| # | Page | What you learn |
| --- | --- | --- |
| 01 | [Overview](01-overview.md) | The goal, the drone, what worked, the timeline and the current status |
| 02 | [RL foundations](02-rl-foundations.md) | MDPs, returns, Q-functions, TD learning, actor-critic, DDPG and TD3, shaping, multi-objective RL |
| 03 | [System architecture](03-system.md) | Hardware, software components, the pipeline from training to flight, the repository map |
| 04 | [Simulator](04-simulator.md) | Rigid-body and motor model, randomization, episodes, curriculum, termination, tests, options |
| 05 | [Policy inputs and outputs](05-policy-io.md) | The 25 inputs of variant E, the critic's inputs, the network, why measured rotor speeds replaced action history |
| 06 | [Reward](06-reward.md) | Every cost term, the weight curriculum, crash penalty against survival bonus, the crash incentive and the reward check |
| 07 | [TD3 as implemented](07-td3.md) | The exact training loop, losses, schedules, activation, optimizer, PyTorch against C++, checkpoint choice |
| 08 | [Sensor path](08-sensor-path.md) | Simulated sensors through the flown Kalman filter, calibration, findings |
| 09 | [Multi-objective TD3](09-moo.md) | Reward groups, critic designs, the aggregators, conflict, why they failed and then worked |
| 10 | [Waypoints and time to target](10-waypoints.md) | The waypoint task, why it collapsed, what fixed it, settling |
| 11 | [Firmware and flights](11-flight.md) | Export, firmware changes, flying with `rl_trigger.py`, safety, every flight day |
| 12 | [Experiment log](12-experiments.md) | Every study: question, setup, result, conclusion, with links to the records |
| 13 | [Issues and open questions](13-issues.md) | Known problems, open questions, process lessons, next steps |
| 14 | [How-to](14-howto.md) | Commands to build, train, test, run studies, export, flash, fly and analyse |
| 15 | [Glossary](15-glossary.md) | Terms, symbols and review links in one place |
| – | [Conventions](CONVENTIONS.md) | How these pages are written, checked and extended |

## Status at a glance (2026-11-26)

- **The learned controller flies the whole mission** (take-off, hover, circle, landing) on the
  real drone with no PID. The best policies track the circle within about 6-8 cm on the real
  drone (I1, C and E in early November; OS-NMF-PGA and UPGrad on 2026-11-25).
- **Current training recipe:** variant E (25 inputs, measured rotor speeds), PyTorch TD3, old
  (truth plus noise) observation path, crash penalty 400, simulator options for drag, motor lag
  and gain wander, mission plus short waypoint episodes.
- **Main research result:** with one critic network per reward group (`--split-critic`), the
  gradient aggregators beat plain summing, SVD-PGA matches hand-tuned TD3 in simulation, and on
  the 2026-11-25 flights every aggregator except Aligned-MTL beat plain TD3 on the circle.
- **Main open question:** can the aggregators replace hand-tuned reward weights? The fair test
  (untuned reward on split critics) has not been run. See [Issues](13-issues.md).

## Where the records are

These pages explain and connect. The detailed lab notebooks, with every table, stay where they
were written, and the pages here link to them:

| Record | Content |
| --- | --- |
| [`l2f/docs/results.md`](../l2f/docs/results.md) | Everything up to 2026-11-22: model, firmware, mission, measured RPM, sensor path, early MOO |
| [`l2f/docs/time_to_target_study.md`](../l2f/docs/time_to_target_study.md) | The first waypoint study (2026-11-23, lab computer) |
| [`l2f/docs/reward_studies.md`](../l2f/docs/reward_studies.md) | Studies 1-10 (2026-11-23 to 25): reward, waypoints, aggregators, critic designs |
| [`l2f/docs/moo_findings.md`](../l2f/docs/moo_findings.md) | The MOO findings, explained |
| [`l2f/docs/flight_day_2026-11-25.md`](../l2f/docs/flight_day_2026-11-25.md) | 28 real flights of seven methods |
| [`l2f/docs/results/README.md`](../l2f/docs/results/README.md) | Exported CSV results of the split-critic studies |
| [`l2f/docs/next_steps.md`](../l2f/docs/next_steps.md) | The plan after 2026-11-25 |
| [`l2f/pytorch_training_plan.html`](../l2f/pytorch_training_plan.html) | The plan of 2026-11-21/22 with its decisions |

## For maintainers

- Edit the `.md` files here, then run `python3 documentation/tools/build_docs.py`: it rebuilds
  `assets/bundle.js` and checks links, anchors, widgets and formulas.
- If you change a widget's math, run `python3 documentation/tools/check_widgets.py` (needs Node;
  checks the aggregators against TorchJD and the leg costs against the recorded table).
- Rules for writing and extending the pages: [Conventions](CONVENTIONS.md).
