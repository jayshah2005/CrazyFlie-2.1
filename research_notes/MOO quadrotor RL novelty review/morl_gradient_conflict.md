# Multi-objective / multi-task RL methods that resolve or aggregate gradient conflict (actor-critic, robotics, drones)

Scope note: 15-call research budget. Primary sources were read for GCR-PPO, MOAC, CAGrad, Nash-MTL, PCGrad, CoMOGA, and Xing et al. (quadrotor MTRL). Envelope, PGMORL, and CAPQL are cited through the MORL-Baselines index only; their internal details are in Gaps. Where an item is from memory and not fetched, it is marked UNVERIFIED.

## Q1. GCR-PPO (arXiv 2509.14816): critic, conflict measure, tasks, quadcopter result

### Takeaway
GCR-PPO uses PPO with a vector-valued (multi-head) critic, one head per reward component, and PCGrad-style projection ordered by priority: task gradients beat regulariser gradients. It reports +4.6% on IsaacLab and +9.5% overall over massively parallel PPO. The IsaacLab Quadcopter task has the lowest measured conflict of all tasks (0.15) and only a +1.09% gain, which the authors put down to that low conflict alone.

### Cited Findings
- Authors: Humphrey Munn, Brendan Tidd, Peter Böhm, Marcus Gallagher, David Howard. arXiv, 18 Sep 2025, cs.RO/cs.LG. The arXiv page names no peer-reviewed venue. — [arXiv abs](https://arxiv.org/abs/2509.14816)
- Abstract claims: better scalability than parallel PPO (p = 0.04), small compute overhead, and an average +9.5% over large-scale PPO, with larger gains on high-conflict tasks. — [arXiv abs](https://arxiv.org/abs/2509.14816)
- Critic: a vector-valued V_φ: S → R^K with one head per reward component, trained jointly by least squares on bootstrapped per-component targets. The paper text does not say whether the heads share a trunk. — [arXiv HTML](https://arxiv.org/html/2509.14816)
- Advantages: component-wise GAE on per-component TD residuals, stacked into a vector. Normalisation uses one global scale chosen so the summed advantage has unit variance, which keeps the ratios between components (Eq. 1). Per-component PPO clipping uses the baseline threshold (about 0.2). — [arXiv HTML](https://arxiv.org/html/2509.14816)
- Conflict definition: a negative inner product (negative cosine) between component policy gradients. The reported metric, "Avg. conflict" in Table I, is the mean number of component pairs at angles above 90°, averaged over training steps and seeds. Fig. 5 shows cosine similarities at the start and end of training. — [arXiv HTML](https://arxiv.org/html/2509.14816)
- Resolution rule: PCGrad-style projection. When a task gradient and a regulariser gradient conflict, the regulariser gradient is projected onto the normal plane of the task gradient. Task–task and regulariser–regulariser conflicts are handled symmetrically, as in PCGrad. The update is the sum of the projected gradients. — [arXiv HTML](https://arxiv.org/html/2509.14816)
- Priority is set in the code through a `reward_component_task_rew` list of the task-reward names. The repository targets IsaacLab 2.1.0 with RSL-RL. — [GitHub humphreymunn/GCR-PPO](https://github.com/humphreymunn/GCR-PPO)
- Tasks (13 IsaacLab tasks): Quadcopter, Reach-UR10, Reach-Franka, Repose-Cube-Shadow, Velocity-Rough-Unitree-Go2, Lift-Cube-Franka, Velocity-Rough-H1, Open-Drawer-Franka, Ant, Humanoid, Velocity-Rough-G1, Repose-Cube-Allegro, Tracking-LocoManip-Digit. There are also 2 custom multi-objective tasks: G1 full-body throwing (+6 objectives) and humanoid running (+4 objectives). — [arXiv HTML](https://arxiv.org/html/2509.14816)
- Quadcopter result (Table I): baseline 64.205 ± 1.610, GCR-PPO 64.911 ± 1.583, SPC +1.09%. Avg. conflict 0.15 is the lowest in the table. Fig. 1 says the two methods "perform similarly on the Quadcopter, where gradient conflict is low." The paper does not list the quadcopter reward terms. — [arXiv HTML](https://arxiv.org/html/2509.14816)
- The only explanation the paper gives for the small gain is low conflict. Across tasks, conflict and improvement correlate with Spearman ρ = 0.736 (p = 0.0041). — [arXiv HTML](https://arxiv.org/html/2509.14816)
- Baselines and ablations: RSL-RL PPO with default IsaacLab configs, multi-head PPO without surgery, and GCR-PPO-NoPriority. PCGrad, CAGrad, and CoMOGA are not run as baselines; CAGrad and CoMOGA appear only in related work. — [arXiv HTML](https://arxiv.org/html/2509.14816)
- No real-robot deployment; all experiments are in IsaacLab simulation. — [arXiv HTML](https://arxiv.org/html/2509.14816)
- Stated or implied limitations:
  - Per-iteration overhead up to 54% in the worst case.
  - The priority scheme encodes designer intent and gives no Pareto guarantee.
  - Higher seed variance.
  - Entropy coefficient re-tuned per method and task.
  - About 2% of runs had numerical instabilities and were rerun.
  - A significant drop on Velocity-Rough-Unitree-Go2 (p = 0.043).
  - No within-task conflict–SPC correlation on the custom tasks.

  — [arXiv HTML](https://arxiv.org/html/2509.14816)
- Internal inconsistencies:
  - The IsaacLab average is given as 4.6% in the introduction and 4.55% in §IV-A.
  - The custom-task gain is 41.2% in Table IV, while the text gives 79.7% (humanoid running) and 2.6% (throwing).

  — [arXiv HTML](https://arxiv.org/html/2509.14816)

### Inferences
- The IsaacLab Quadcopter is a simple hover/goal-reaching task with few reward terms, so low conflict is plausible. A low-level quadrotor controller with many regulariser terms (action smoothness, energy, body rates) could conflict more. GCR-PPO did not test that case, so it is open ground.
- GCR-PPO is on-policy only (PPO). It leaves open the off-policy (TD3/SAC) case, where critics are Q-functions and the actor gradient is ∇_a Q_k · ∇_θ π.

### Gaps
- Whether the critic heads share a trunk: neither the paper text nor the README says so; the code was not inspected.
- The quadcopter reward terms and their count: the paper does not give them. The IsaacLab Quadcopter env source would need checking.

## Q2. MOAC (arXiv 2405.03082)

### Takeaway
MOAC is a theory paper (ICML 2024) about actor-critic MORL. Each objective has its own linear critic, all on a shared feature map. The actor uses an MGDA min-norm common direction, with momentum on the weights. Experiments are small: MO-Gymnasium toy tasks and an offline recommender dataset. There is no robotics experiment.

### Cited Findings
- Paper: "Finite-Time Convergence and Sample Complexity of Actor-Critic Multi-Objective Reinforcement Learning", by Tianchen Zhou, Hairi, Haibo Yang, Jia Liu, Tian Tong, Fan Yang, Michinari Momma, and Yan Gao. ICML 2024. — [arXiv abs](https://arxiv.org/abs/2405.03082)
- Critic: one linear value estimate per objective, V^i(s) ≈ φ(s)ᵀw^i, with "the same feature mapping … shared among all objectives". It uses mini-batch TD learning. — [arXiv HTML](https://arxiv.org/html/2405.03082)
- Actor: solves min‖Σλ^i g^i‖² on the simplex (MGDA), then smooths the weights as λ_t = (1−η_t)λ_{t−1} + η_t λ̂*_t. With η_1 = 1, the first samples set the initial weights. The paper claims convergence and sample-complexity bounds that do not depend on the number of objectives. — [arXiv HTML](https://arxiv.org/html/2405.03082)
- Experiments:
  - MO-Gymnasium resource-gathering-v0, with FishWood in the appendix.
  - KuaiRand video-recommendation logs (7 rewards), run off-policy from a behavior policy.
  - Baselines: Behavior-Clone, PDPG, TSCAC, and a fixed-weights ablation.
  - Results over BC: about +6.6% Like and +10% Follow. TSCAC is better on Click and WatchTime.

  — [arXiv HTML](https://arxiv.org/html/2405.03082)

### Inferences
- MOAC is the closest theoretical precedent for "per-objective critics + MGDA actor". It is linear, small-scale, and not about continuous-control robots.

### Gaps
- None material. The remaining ~54k characters of the paper (appendix) were not read.

## Q3. Gradient-surgery and aggregation methods in multi-task RL (PCGrad, CAGrad, Nash-MTL, MGDA, CoMOGA, others)

### Takeaway
In RL, PCGrad, CAGrad, and Nash-MTL have mostly been tested on Meta-World MT10/MT50 with multi-task SAC. Per-task gradients there are taken over shared actor and critic parameters, with a task one-hot input, and not from separate per-objective critics. The reported MT10 success ladder is: MT-SAC 0.49, PCGrad 0.72, CAGrad 0.83, Nash-MTL 0.91, and one SAC per task 0.90. CoMOGA is the main off-policy, per-objective-critic aggregation method for MORL with robot (legged) tasks.

### Cited Findings
- **PCGrad** (Yu et al., "Gradient Surgery for Multi-Task Learning"):
  - The method projects each conflicting task gradient onto the normal plane of the other.
  - In actor-critic RL it replaces "task gradients for both the actor and the critic" on SAC.
  - On MT10 it solves all 10 tasks; on MT50 about 70%. It is far more sample-efficient than independent SAC agents.
  - Ablations show that both the direction change and the magnitude change matter.
  - A "tragic triad" analysis (conflict, dominance, high curvature) on reach / press-button-top finds that conflict is frequent until the second task is learned.
  - The arXiv v4 is dated Dec 2020. The NeurIPS 2020 venue is from memory, UNVERIFIED on the fetched page.

  — [arXiv HTML 2001.06782](https://arxiv.org/html/2001.06782)
- **CAGrad** (Liu et al., NeurIPS 2021):
  - Update: maximise the worst-case local improvement inside a ball of radius c‖g₀‖ around the average gradient g₀. c = 0 gives GD; large c approaches MGDA.
  - RL setup: SAC in the MTRL/CARE codebase, 2M steps, 10 seeds, c = 0.9 (MT10) and 0.5 (MT50).
  - Success: MT10 0.83 ± 0.045, MT50 0.52 ± 0.023. PCGrad scores 0.72 / 0.50, multi-head SAC 0.61 / 0.45, CARE 0.84 / 0.54, and one SAC per task 0.90 / 0.74.
  - MGDA is absent from the RL table.
  - Speed-up claims are inconsistent: 22×/55× in the main text, about 2×/5× in appendix Table 7.

  — [arXiv HTML 2110.14048](https://arxiv.org/html/2110.14048)
- **Nash-MTL** (Navon et al., ICML 2022):
  - Solves (GᵀG)α = 1/α, a Nash bargaining rule that is scale-invariant.
  - MT10 with shared MT-SAC: 0.91 ± 0.031, compared with CAGrad 0.83, PCGrad 0.72, and STL SAC 0.90.
  - Runtime per episode: about 40.7 s (Nash-MTL-100 variant: 0.87 at 7.9 s), CAGrad about 20.9 s, PCGrad about 9.7 s.

  — [arXiv HTML 2202.01017](https://arxiv.org/html/2202.01017)
- **CoMOGA** (Kim, Hong, Park, Oh; "Conflict-Averse Gradient Aggregation for Constrained Multi-Objective RL"; v1 was titled "Scale-Invariant Gradient Aggregation"):
  - Critics: separate preference-conditioned reward critics per objective and cost critics per constraint.
  - Off-policy with a replay buffer and SAC-style reparameterised policy gradients.
  - Aggregation: a QP that turns objectives into constraints so no objective decreases, inside a trust region (Fisher matrix). The resulting policies are distilled into one preference-conditioned policy.
  - Tasks: legged quadruped/biped (velocity tracking and energy, plus 3 constraints), Safety-Gymnasium, and MO-MuJoCo.
  - Baselines: LP3, plus PD-MORL (TD3) and CAPQL (SAC) with Lagrangians, and a CAGrad+Lagrangian ablation that did worse.
  - No drones and no real robot.

  — [arXiv HTML v2](https://arxiv.org/html/2403.00282v2); ICLR 2025 proceedings PDF listed at [proceedings.iclr.cc](https://proceedings.iclr.cc/paper_files/paper/2025/file/59ddfff7979b43f54690fa986c0e5138-Paper-Conference.pdf) (venue year inferred from URL)
- **MGDA in policy optimisation**: MOAC (Q2) is the clearest actor-critic MGDA instance found. — [arXiv HTML](https://arxiv.org/html/2405.03082)
- **Multi-critic actor learning (MultiCriticAL)**, ICLR 2022: separate critics per task with one multi-task actor, to avoid negative interference from a shared critic in multi-style learning. It reports up to 56% gain over single-critic baselines. The aggregation rule was not verified. — [ICLR 2022 poster](https://iclr.cc/virtual/2022/poster/6568)
- 2025–26 robotics papers with multi-critic or multi-head PPO and reward-group decomposition (snippet-level, full text NOT read):
  - APEX (Go2 motion tracking): two critic heads, style vs task+regularisation.
  - Humanoid soccer: task / style / regularisation critics.
  - Discrete-time hybrid automata skateboarding: multi-critic PPO.
  - TactileStep: dense/sparse dual critic.
  - None of these, as far as the snippets show, does gradient surgery; they decompose value estimation only.

  — [APEX](https://arxiv.org/html/2505.10022), [Soccer](https://arxiv.org/pdf/2511.03996), [Skateboarding](https://arxiv.org/pdf/2503.01842), [TactileStep](https://arxiv.org/pdf/2609.28959)

### Inferences
- In multi-task RL, "gradient conflict" usually means conflict between tasks on shared parameters, with one critic conditioned on the task. In multi-objective (reward-term) RL it means conflict between reward components of a single task. Only GCR-PPO, MOAC, and CoMOGA work on the second kind with per-objective critics.
- Nash-MTL ≈ STL > CAGrad > PCGrad on MT10. Note, though, that those numbers come from a shared codebase and may be copied between papers (the baseline rows are identical in the CAGrad and Nash-MTL tables).

### Gaps
- Not covered within budget: FAMO, Aligned-MTL, IMTL, GradDrop, and RotoGrad RL results; Google Scholar citations of PCGrad/CAGrad in RL after 2025.
- PCGrad's goal-conditioned pushing results are in an appendix that was not read.

## Q4. MORL frameworks and their vector critics

### Takeaway
The main MORL frameworks (Envelope Q, PGMORL, CAPQL, PD-MORL, GPI-PD, PCN, MORL/D) are collected in MORL-Baselines (NeurIPS 2023). They mostly learn a single network with a vector output, or a preference-conditioned vector Q, and scalarise in the actor or policy. They do not apply per-objective gradient surgery.

### Cited Findings
- MORL-Baselines (Felten et al., NeurIPS 2023 per the repo) implements GPI-LS/GPI-PD, MORL/D, Envelope Q-Learning (arXiv 1908.08342), CAPQL (OpenReview TjEzIsyEsQ6), PGMORL, PCN (AAMAS 2022), LCN, Pareto Q-learning, MO Q-learning, OLS, EUPG, IPRO, and NLMOPPO. — [GitHub morl-baselines](https://github.com/LucasAlegre/morl-baselines)
- CoMOGA uses PD-MORL (TD3-based) and CAPQL (SAC-based) as off-policy MORL baselines. — [arXiv HTML](https://arxiv.org/html/2403.00282v2)

### Inferences
- From memory, UNVERIFIED in this session:
  - Envelope Q-learning (Yang et al., NeurIPS 2019) uses one preference-conditioned network with a vector Q output.
  - PGMORL (Xu et al., ICML 2020) is PPO with a vector value head plus evolutionary, prediction-guided weight selection on MO-MuJoCo.
  - CAPQL (Lu, Herman, Yu, ICLR 2023) is SAC with a preference-conditioned vector Q and an augmented concave reward.
  - Hayes et al., "A practical guide to multi-objective RL and planning" (AAMAS/JAAMAS 2022, arXiv 2103.09568) is the standard survey.
  - None of these computes and projects per-objective actor gradients.

### Gaps
- No primary paper pages were fetched for Envelope, PGMORL, CAPQL, PD-MORL, or Hayes et al. The OpenReview fetch for CAPQL was blocked by a browser check. There is no verified source for MO-TD3, MO-DDPG, or MOMA-TD3.

## Q5. Real robots and control level; drones and quadrotors

### Takeaway
Among the gradient-conflict methods found, none reports a real-robot deployment. GCR-PPO and CoMOGA are simulation only. The closest drone work, Xing et al. (RA-L), uses one shared actor with one critic per task, PPO, and CTBR actions, deployed on a real Agilicious quadrotor. It names conflicting objectives but applies no gradient surgery.

### Cited Findings
- Xing, Geles, Song, Aljalbout, Scaramuzza, "Multi-Task Reinforcement Learning for Quadrotors", arXiv 2412.12442, RA-L. — [arXiv abs](https://arxiv.org/abs/2412.12442)
- Design:
  - A shared actor fed by a shared dynamics encoder and task-specific encoders (32-d each).
  - A separate critic per task that takes the full observation and does not share the actor's encoder.
  - PPO in Flightmare.
  - Actions are collective thrust + body rates (CTBR), tracked by Betaflight.

  — [arXiv HTML](https://arxiv.org/html/2412.12442)
- Interference: the paper acknowledges "conflicting objectives" and cites CAGrad, but does not use or compare PCGrad or CAGrad. Its MTRL-Separate ablation (no feature fusion) gets 0% racing success, which the authors attribute to the policy favouring the easier task. — [arXiv HTML](https://arxiv.org/html/2412.12442)
- Results at 40M samples: racing 100% success, lap 5.80 s vs 5.83 s single-task, and tracking error 1.21 vs 1.18 single-task. Real-world tests: 5 runs per task, 100% success, gate error 0.189 m vs 0.152 m in simulation. — [arXiv HTML](https://arxiv.org/html/2412.12442)
- GCR-PPO quadcopter: IsaacLab simulation only, lowest conflict (0.15), +1.09%. — [arXiv HTML](https://arxiv.org/html/2509.14816)

### Inferences
- No source found applies per-objective critics with gradient conflict resolution at the motor-command level on a real quadrotor (for example a Crazyflie), either on-policy or off-policy. This looks like a gap in the literature, within the limits of this search.

### Gaps
- No exhaustive search of RA-L/ICRA/IROS 2025–26 for drone MORL. The search did not surface multi-critic papers on low-level (motor thrust or RPM) quadrotor control.

## Q6. Off-policy (TD3/SAC/DDPG) methods with per-objective critics and gradient aggregation in the actor

### Takeaway
CoMOGA is the clear verified instance: SAC-style off-policy learning, separate per-objective reward and cost critics, and a conflict-averse QP aggregation in the actor. MOAC is an MGDA variant with per-objective linear critics, run off-policy on logged data. PD-MORL and CAPQL are off-policy, but they scalarise a preference-conditioned vector Q and do not do gradient surgery (UNVERIFIED details). The Meta-World results for PCGrad, CAGrad, and Nash-MTL use SAC, but they aggregate per-task gradients of shared networks, not per-reward critics.

### Cited Findings
- CoMOGA: separate preference-conditioned reward and cost critics, replay buffer, SAC reparameterisation, QP aggregation. — [arXiv HTML](https://arxiv.org/html/2403.00282v2)
- MOAC: per-objective linear critics and an MGDA actor, adapted off-policy with a behavior policy for KuaiRand. — [arXiv HTML](https://arxiv.org/html/2405.03082)
- PCGrad applies surgery to both actor and critic gradients in SAC. — [arXiv HTML](https://arxiv.org/html/2001.06782)
- CAGrad and Nash-MTL use multi-task SAC with shared parameters. — [CAGrad](https://arxiv.org/html/2110.14048), [Nash-MTL](https://arxiv.org/html/2202.01017)

### Inferences
- TD3 with one separate critic network per reward group, plus PCGrad, CAGrad, MGDA, Nash-MTL, or priority projection of the deterministic actor gradients ∇_a Q_k, does not appear in the verified literature. The most similar works are CoMOGA (SAC-like, constrained MORL, simulation) and GCR-PPO (on-policy). Novelty claims should still be checked with a targeted Scholar search on "TD3 PCGrad multi-critic" and on the citers of GCR-PPO.

### Gaps
- No verified MO-TD3, MO-DDPG, or MOMA-TD3 paper was found within budget. The Google Scholar citations of GCR-PPO, PCGrad, and CAGrad were not searched.
