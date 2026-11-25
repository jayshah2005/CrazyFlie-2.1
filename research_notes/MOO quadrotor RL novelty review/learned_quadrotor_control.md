# Learned low-level quadrotor control with RL: inventory and multi-objective novelty check

Scope: end-to-end RL policies that output motor commands (RPM / thrust), or CTBR (collective thrust + body rates). Searched as of 2026-11-25. Tool budget limited full-text reads to some papers. Items marked [UNVERIFIED] come from background knowledge and were not confirmed against the source in this session.

## Key papers: who, when, where, what

### Takeaway
The core lineage is: Hwangbo 2017 (first RL motor-command policy on a real quadrotor), Koch/GymFC 2019 (attitude), Molchanov 2019 (sim-to-multi-real, Crazyflie), Gronauer 2022, Eschmann 2024 (RPM policy trained in 18 s, Crazyflie), SimpleFlight 2025 (CTBR, Crazyflie), and RAPTOR 2025/26 (foundation policy for many quadrotors). Racing work (Song 2021/2023, Kaufmann Nature 2023, Ferede ICRA 2024) pushes agility. Every paper read optimizes one hand-weighted scalar reward.

### Cited Findings
- **Hwangbo, Sa, Siegwart, Hutter, "Control of a Quadrotor with Reinforcement Learning," 2017 (IEEE RA-L, DOI 10.1109/LRA.2017.2720851).** A neural net maps state directly to actuator commands with no predefined control structure. Uses a new "conservative but stable" learning algorithm. Recovers the vehicle after it is thrown upside-down at 5 m/s. Policy evaluation takes about 7 µs per step. — [arXiv 1707.05110](https://arxiv.org/abs/1707.05110)
- **Molchanov, Chen, Hönig, Preiss, Ayanian, Sukhatme, "Sim-to-(Multi)-Real: Transfer of Low-Level Robust Control Policies to Multiple Quadrotors," 2019 (IROS 2019 per the authors' publication file name).** State is mapped directly to motor outputs. The paper studies how cost-function changes, noise modeling and domain randomization affect transfer to several physical quadrotors. Reports robustness to disturbances and to thrown starts. — [arXiv 1903.04628](https://arxiv.org/abs/1903.04628); [author PDF](https://whoenig.github.io/publications/2019_IROS_Molchanov_Chen.pdf)
  - [UNVERIFIED] Algorithm is PPO, and the cost is a weighted sum of position error, velocity, orientation, angular velocity and action-magnitude terms. Platforms include Crazyflie 2.0. I could not extract the PDF text.
- **Koch, Mancuso, West, Bestavros, "Reinforcement Learning for UAV Attitude Control" (GymFC), ACM Trans. Cyber-Physical Systems 2019, arXiv 1804.04154.** [UNVERIFIED, not fetched] RL attitude (inner-loop) control in Gazebo, compared against PID. Later work became Neuroflight.
- **Mysore, Mabsout, Mancuso, Saenko, "Regularizing Action Policies for Smooth Control with Reinforcement Learning" (CAPS), ICRA 2021.** Adds a smoothness regularizer to the policy objective (temporal and spatial). Reports smoother actions on Gym tasks with comparable performance, and about 80% lower power use on a real quadrotor. — [arXiv 2012.06644](https://arxiv.org/abs/2012.06644v2); [project](https://ai.bu.edu/caps/index.html)
- **Gronauer et al., "Using simulation optimization to improve zero-shot policy transfer of quadrotors," IROS 2022 (arXiv 2201.01369).** [UNVERIFIED bibliographic details] Used as an RL baseline (Crazyflie) in Eschmann 2024, trained for 16M steps. — [Eschmann 2024 HTML](https://arxiv.org/html/2311.13081)
- **Lambert et al., "Low-Level Control of a Quadrotor With Deep Model-Based RL," RA-L 2019.** Model-based RL on a Crazyflie. Reaches stable hover after about 10,000 data points. — [PDF](https://drl.ece.utah.edu/wp-content/uploads/sites/155/2021/07/lambert-ral2019.pdf)
- **Eschmann, Albani, Loianno, "Learning to Fly in Seconds," IEEE RA-L 2024, arXiv 2311.13081.** Asymmetric actor-critic (off-policy) with curriculum learning and a fast simulator (RLtools). Direct RPM control transfers sim-to-real after 18 s of training on a consumer laptop (2020 MacBook Pro), roughly 300k steps. Runs on the Crazyflie microcontroller. Code and simulator are open source. — [arXiv abs](https://arxiv.org/abs/2311.13081); [HTML](https://arxiv.org/html/2311.13081)
  - Actor input is 18 + N_H·4 dimensions (includes action history). The critic gets 28 dimensions of privileged information (rotor speeds, force and torque disturbances). — [HTML](https://arxiv.org/html/2311.13081)
- **Song, Steinweg, Kaufmann, Scaramuzza, "Autonomous Drone Racing with Deep RL," IROS 2021**, and **Song, Romero, Müller, Koltun, Scaramuzza, "Reaching the limit in autonomous racing: Optimal control versus reinforcement learning," Science Robotics 2023.** [UNVERIFIED, not fetched] CTBR racing policies using a progress-based reward.
- **Kaufmann, Bauersfeld, Loquercio, Müller, Koltun, Scaramuzza, "Champion-level drone racing using deep reinforcement learning" (Swift), Nature 620, 2023.** — [Nature](https://www.nature.com/articles/s41586-023-06419-4) (paywall, text not fetched)
  - [UNVERIFIED] CTBR outputs. The reward sums progress, perception (camera pointed at the gate), command-smoothness and crash terms with fixed coefficients. Swift won several races against human world champions and set the fastest lap.
- **Ferede, De Wagter, Izzo, de Croon, "End-to-end Reinforcement Learning for Time-Optimal Quadcopter Flight," ICRA 2024, pp. 6172–6177.** Direct motor commands compared with a CTBR + INDI baseline. The end-to-end policy is 1.39 s faster in simulation and 0.17 s faster in the real world. Closes the sim-to-real gap with a learned residual model and an adaptive component. — [arXiv 2311.16948](https://arxiv.org/abs/2311.16948v1); [DOI](https://api.crossref.org/works/10.1109%2FICRA57147.2024.10611665)
- **Yu et al., "Multi-Agent RL for the Low-Level Control of a Quadrotor UAV," 2023 (arXiv 2311.06144).** Splits control into a translational agent and a yaw agent, each with its own reward. Adds regularization terms against steady-state error and aggressive inputs. — [arXiv 2311.06144](https://arxiv.org/pdf/2311.06144)
- **"Equivariant RL Frameworks for Quadrotor Low-Level Control," 2025 (arXiv 2502.20500).** Equivariant RL for low-level control, covering the full dynamics including yaw. — [arXiv](https://arxiv.org/pdf/2502.20500)
- **Xing, Geles, Song, Aljalbout, Scaramuzza, "Multi-Task Reinforcement Learning for Quadrotors," IEEE RA-L (accepted Nov 2024), arXiv 2412.12442.** One shared actor and **one critic per task**, with a shared task-agnostic encoder. Tasks are stabilization, velocity tracking and racing. Output is CTBR, sent to BetaFlight. — [HTML](https://arxiv.org/html/2412.12442); [PDF](https://arxiv.org/pdf/2412.12442)
- **Chen, Yu, et al. (Tsinghua), "What Matters in Learning a Zero-Shot Sim-to-Real RL Policy for Quadrotor Control? A Comprehensive Study" (SimpleFlight), IEEE RA-L 2025, arXiv 2412.11764.** PPO in OmniDrones. Output is CTBR. Tested on a Crazyflie 2.1. Identifies five factors: relative-position actor inputs plus a rotation matrix, a time vector as privileged critic input, a smoothness reward ‖u_t−u_{t−1}‖, system identification with selective domain randomization (thrust coefficient only), and large batch sizes. — [HTML v4](https://arxiv.org/html/2412.11764v4); [abs](https://arxiv.org/abs/2412.11764)
- **Multitask RL for quadcopter attitude stabilization, 2025 (arXiv 2503.08259).** Uses SAC with a graph policy. The reward is a linear combination of features, and task weights select the objective. — [arXiv](https://arxiv.org/pdf/2503.08259)
- **Eschmann, Albani, Loianno, "RAPTOR: A Foundation Policy for Quadrotor Control," arXiv 2509.11481 (Sept 2025, revised April 2026).** Trains 1,000 RL teacher policies on sampled quadrotors and distills them into one recurrent student with 2,084 parameters. The student adapts zero-shot within milliseconds on 10 real quadrotors from 32 g to 2.4 kg. — [arXiv](https://arxiv.org/abs/2509.11481); project page raptor.rl.tools
- **Crazyflie brushless and sim-to-real.** A Cal Poly thesis transferred an Isaac Lab hover policy zero-shot to a Crazyflie 2.1 Brushless, with mean position error 0.081 m over 28 flights. — [Cal Poly thesis](https://digitalcommons.calpoly.edu/theses/3375/). Other tools: the Crazyflow JAX simulator with a system-identification pipeline — [arXiv 2610.08225](https://arxiv.org/pdf/2610.08225); a Bitcraze roundup of RL research on the Crazyflie — [Bitcraze blog 2026-05](https://www.bitcraze.io/2026/05/five-unexpected-research-applications-of-the-crazyflie-platform/); QuadSwarm, a simulator with direct thrust control — [arXiv 2306.09537](https://arxiv.org/pdf/2306.09537).
- **Survey:** "A Survey of Offline and Online Learning-Based Algorithms for Multirotor UAVs," 2024. — [arXiv 2402.04418](https://arxiv.org/pdf/2402.04418)

### Inferences
- The Crazyflie (2.0/2.1) is the de facto benchmark platform for RPM-level RL (Molchanov, Lambert, Gronauer, Eschmann, SimpleFlight). Brushless Crazyflie RL work is very sparse: I found only one thesis.
- The field has split by action level. RPM / motor command: Hwangbo, Molchanov, Eschmann, Ferede, RAPTOR. CTBR: Swift, Song, Xing, SimpleFlight.

### Gaps
- I did not read the full text of Koch/GymFC, Gronauer, Song 2021/2023, Kaufmann 2023 (Nature paywall; the PMC page was blocked by bot verification) or Molchanov (PDF text not extractable). Their reward details are marked [UNVERIFIED].
- I did not run a "cited by" sweep of Eschmann 2024 on Google Scholar, so there may be more 2025–2026 Crazyflie RL papers.
- Not covered: Zhang et al. near-hover controller for very different quadcopters, DATT (Huang et al., CoRL 2023, used as a SimpleFlight baseline), Neural-Fly (adaptive, not RL).

## How rewards are defined and how trade-offs are handled

### Takeaway
Every paper I could read combines precision, effort, smoothness, speed and safety into one scalar reward with hand-tuned fixed weights. They tune it with curricula on the weights (Eschmann), a weight swept on real hardware (SimpleFlight λ), regularizers (CAPS), or survival bonuses and crash penalties.

### Cited Findings
- **Eschmann 2024 reward:** r = −C_rp‖p‖² − C_rq(1−q_w²) − C_rv‖v‖² − C_rω‖ω‖² − C_ra‖a−C_rab‖² + C_rs. The survival bonus C_rs counters "learning to terminate." The curriculum moves penalty weights from C_init toward stricter C_target by multiplying them by a factor every 100k steps, so position errors and actions are punished harder as training goes on. Exploration noise decays on the same schedule. — [HTML](https://arxiv.org/html/2311.13081)
- **SimpleFlight reward:** r = r_task + λ·r_smooth, with both terms normalized to [0,1]. λ = 0.4 was picked from real-world sweeps. The authors state that a larger λ "can degrade tracking performance due to restricted agility, while smaller λ may lead to unstable flight." They also found that action clipping limits agility and low-pass filtering does not generalize. — [HTML v4](https://arxiv.org/html/2412.11764v4)
- **Xing et al. multi-task rewards:** each task has its own scalar reward with fixed coefficients (their Table I). Racing: progress, perception, action-change penalty, body-rate penalty, gate bonus, crash penalty. Stabilization: height, attitude, velocity, body-rate and action-change penalties, plus a hover success reward. Velocity tracking: velocity error, body-rate penalty, action-change penalty. — [HTML](https://arxiv.org/html/2412.12442)
- **CAPS:** smoothness is added as a regularization term in the policy loss, not as a separate objective. — [arXiv 2012.06644](https://arxiv.org/abs/2012.06644v2)
- **Yu et al. 2023:** regularization terms for steady-state error and aggressive inputs, inside per-agent rewards. — [arXiv 2311.06144](https://arxiv.org/pdf/2311.06144)
- **Molchanov 2019:** studies how changing the cost function affects flight performance, which is an ablation over scalar cost designs. — [arXiv 1903.04628](https://arxiv.org/abs/1903.04628)
- **2503.08259:** the reward is a linear combination of features, with task weights choosing the objective. — [arXiv](https://arxiv.org/pdf/2503.08259)

### Inferences
- The precision versus smoothness/effort trade-off is openly acknowledged (SimpleFlight's λ sweep, CAPS's power savings, Eschmann's tightening curriculum). It is always settled by one manually chosen weight vector and never reported as a front. This is the gap a multi-objective formulation would fill.
- Eschmann's curriculum is effectively a hand-scheduled change of scalarization weights over training. That is a useful point of contrast for MOO or gradient-aggregation methods, which balance the terms automatically.

### Gaps
- I do not have the exact coefficient values for Eschmann (supplementary material) or for the racing papers.

## Does any paper use multi-objective RL, reward decomposition, per-objective critics, gradient aggregation, or constrained RL?

### Takeaway
None of the learned low-level quadrotor control papers found uses true multi-objective RL (Pareto fronts, preference vectors, vector-valued critics per reward term) or gradient conflict resolution (PCGrad, CAGrad, MGDA and similar). The nearest neighbours are Xing et al. 2024/25 (one critic per **task**, not per objective, with no gradient surgery), Yu et al. 2023 (per-agent rewards across control axes), and Lagrangian or safe RL used at the navigation level, not motor level. Multi-objective RL with vector critics and Pareto methods does exist for UAV **formation** control and for quadrupeds, but not for low-level quadrotor control.

### Cited Findings
- **Xing et al. (RA-L):** "A separate critic function is used for each task." The objective is the average return over uniformly sampled tasks, trained with PPO. The paper cites conflict-averse gradient descent (CAGrad) as related work but describes no gradient surgery or conflict resolution of its own. — [HTML](https://arxiv.org/html/2412.12442)
- **Yu et al. 2023:** decomposes control across two agents (translation and yaw) with separate rewards. This is multi-agent decomposition, not multi-objective RL. — [arXiv 2311.06144](https://arxiv.org/pdf/2311.06144)
- **SimpleFlight:** describes the task/smoothness trade-off but does not frame it as multi-objective optimization. — [HTML v4](https://arxiv.org/html/2412.11764v4)
- **Constrained and safe RL near this area:** Lagrangian constrained policy learning for safe quadrotor *navigation* in PX4/Gazebo — [arXiv 2610.16119](https://arxiv.org/html/2610.16119v1). Reachability-constrained RL on a 2D quadrotor in safe-control-gym, with SAC- and PPO-Lagrangian baselines, ICML 2022 — [PMLR](https://proceedings.mlr.press/v162/yu22d/yu22d.pdf). Agile flight with safety guarantees via a safety filter — [ResearchGate](https://www.researchgate.net/publication/377507508_Learning_Agile_Quadrotor_Flight_in_Restricted_Environments_with_Safety_Guarantees). I found no Lagrangian method applied to RPM or CTBR-level quadrotor control.
- **MORL in UAVs outside low-level control:** multi-UAV formation control with a vector reward (flight, formation, collision) that is then scalarized — [arXiv 2410.18495](https://arxiv.org/pdf/2410.18495). Formation control with a 4-D reward (progress, energy, coherence, safety), vector critics and a Pareto archive, Drones 2025 — [DOI 10.3390/drones9120845](https://doi.org/10.3390/drones9120845).
- **Preference-conditioned MORL in robotics, but not drones:** PROMO for quadrupeds, where 67 of 100 sampled preferences were non-dominated — [arXiv 2610.01260](https://arxiv.org/abs/2610.01260). Heavy-duty vehicle driving — [arXiv 2601.18783](https://arxiv.org/html/2601.18783).
- Targeted searches for "multi-objective" combined with quadrotor low-level, attitude control, Pareto front, preference-conditioned quadrotor, or Lagrangian low-level quadrotor returned no matching paper. — searches run 2026-11-25. Representative results: [2412.12442](https://arxiv.org/abs/2412.12442v1), [2503.08259](https://arxiv.org/pdf/2503.08259).

### Inferences
- A paper that treats tracking precision, effort/smoothness, and safety/crash avoidance as separate objectives would appear to be novel for RPM-level (or CTBR-level) quadrotor RL, especially one validated on a Crazyflie or Crazyflie Brushless. Such a paper would use per-objective critics, aggregate per-objective policy gradients (for example with conflict resolution), and possibly report a Pareto front.
- Xing et al. is the prior work reviewers will most likely raise. The distinction is that their critics are per **task** with separate environments, while per-objective critics share one task and decompose its reward. Mysore et al. ("Multi-Critic Actor Learning," ICLR 2022, [UNVERIFIED details]) is another multi-critic-per-task precedent from the CAPS/Neuroflight group and should be checked.
- The Eschmann weight curriculum and the SimpleFlight λ sweep give direct motivation: the community already tunes these trade-offs by hand.

### Gaps
- I could not confirm whether Mysore et al.'s "Multi-Critic Actor Learning" (ICLR 2022) includes a quadrotor experiment. The search did not surface it.
- Google Scholar "cited by" was not swept, so a recent preprint on RPM-level quadrotor MORL cannot be fully ruled out. The confidence that none exists is moderate to high.

## Evaluation protocols

### Takeaway
The standard protocol is real-world figure-eight (Lissajous) tracking at slow, normal and fast speeds on a Crazyflie under motion capture, reported as position RMSE, plus position-control success counts. Baselines are PID, geometric (Mellinger–Kumar), nonlinear (Brescianini), INDI, and earlier RL policies (Molchanov, Gronauer, DATT, Fly). Racing papers report lap times and success rates.

### Cited Findings
- **Eschmann 2024:** 10 seeds per configuration (the first 10 of 50, to avoid cherry-picking). Position control: 20 s episodes, mean/median/min xy error, and successful flights out of 10. Figure-eight p(t)=[cos(2πt/T), sin(4πt/T)/2, const] at T = 15 / 5.5 / 3.5 s, 4 cycles each. The fast case reaches up to 3 m/s and 0.9 g. Flown at NYU ARPL (10×6×4 m) with Vicon. — [HTML](https://arxiv.org/html/2311.13081)
- **Eschmann 2024 RMSE ē (m), slow / normal / fast:**

  | Controller | Slow | Normal | Fast |
  |---|---|---|---|
  | PID | 0.23 | 0.72 | 0.88 |
  | Geometric | 0.06 | 0.16 | 0.36 |
  | Nonlinear | 0.29 | 0.38 | failed |
  | INDI | 0.21 | 1.13 | 1.04 |
  | Molchanov | 0.15 | 0.25 | 0.52 |
  | Gronauer | 0.06 | 0.23 | failed |
  | Ours | 0.08 | 0.17 | 0.24 (best) |

  Molchanov and Gronauer were trained for 84M and 16M steps. Neither produced a flying policy within Eschmann's 300k or 3M step budgets. — [HTML](https://arxiv.org/html/2311.13081)
  - Note: the paper's text appears to contain typos in its simulator-throughput claims ("55 months" of flight per second, and a 6420× speedup that does not match 12.84M vs 200k steps/s, which is about 64×). Treat these with care.
- **SimpleFlight on Crazyflie (mean Euclidean error, m), figure-eight slow / normal / fast:** Fly 0.093 / 0.181 / 0.282; DATT 0.050 / fail / fail; SimpleFlight 0.016 / 0.028 / 0.051. On the custom "Air" platform, PAMPC beats SimpleFlight on some trajectories (slow figure-eight 0.011 vs 0.014). The authors caution that comparisons across simulators are not fully fair. — [HTML v4](https://arxiv.org/html/2412.11764v4)
- **Xing et al.:** 10 seeds in Flightmare. Metrics: racing success rate, gate-passing error, lap time, hover time, velocity error. Real-world runs on an Agilicious quadrotor with Vicon, 5 runs per task, 100% success. Racing error was 0.152 m in simulation and 0.189 m in the real world. Single-task RL keeps a slightly lower velocity-tracking error (1.18 vs 1.21). — [HTML](https://arxiv.org/html/2412.12442)
- **Ferede 2024:** lap-time advantage versus a CTBR+INDI baseline, 1.39 s in simulation and 0.17 s in the real world. — [arXiv 2311.16948](https://arxiv.org/abs/2311.16948v1)
- **Hwangbo 2017:** step responses and recovery from an upside-down throw. — [arXiv 1707.05110](https://arxiv.org/abs/1707.05110)
- **RAPTOR:** zero-shot transfer across 10 real platforms (32 g to 2.4 kg) and activation mid-flight from a fast starting velocity. — [arXiv 2509.11481](https://arxiv.org/abs/2509.11481)
- **Brushless Crazyflie thesis:** 0.081 m mean hover position error over 28 flights. — [Cal Poly](https://digitalcommons.calpoly.edu/theses/3375/)

### Inferences
- A multi-objective paper on a Crazyflie would be directly comparable to Eschmann and SimpleFlight if it reuses the figure-eight at T = 15 / 5.5 / 3.5 s with RMSE, adds effort and smoothness metrics (‖Δu‖, power, motor saturation) and crash/success rates, and reports per-objective trade-off curves.
- None of the reviewed papers reports smoothness or effort metrics with the same rigor as tracking error, except CAPS (power). That is an open evaluation niche.

### Gaps
- I did not obtain Swift's (Nature 2023) exact race statistics or reward coefficients in this session.
