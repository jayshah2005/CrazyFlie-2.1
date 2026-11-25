# Multi-objective / multi-task / constrained RL for UAVs and quadrotors (inventory as of 2026-11-25)

Scope note: about 17 search and fetch calls. Sources are mostly arXiv abstract and HTML pages. Citation counts were not checked. "Most-cited" claims are marked as unverified.

## Q1. MORL for UAV navigation: energy vs time, coverage, communication (2018-2026)

### Takeaway
Most UAV "multi-objective" RL sits at the high level (trajectory, path, or resource allocation) and comes from the wireless and communications community, almost always evaluated in simulation. Usually it is a fixed weighted-sum reward. True Pareto or preference MORL (evolutionary MO-PPO, Pareto Q-sets, attention-based MORL) appears mainly from 2023 onward. None of it touches attitude or motor control.

### Cited Findings
- **DRL-EC3 (Liu, Chen, Tang, Xu, Piao, IEEE JSAC 36(9):2059-2070, 2018).** A DRL controller for a UAV group. Its single "energy-efficiency" objective jointly weighs coverage, fairness, energy consumption and connectivity. This is scalarized, not Pareto. It is high-level movement control, in simulation. It is likely among the most-cited early works, but the count is unverified. — [DBLP](https://dblp1.uni-trier.de/rec/journals/jsac/LiuCTXP18.html); [record](https://searchworks.jioinstitute.edu.in/articles/edb__133211107)
- Do not confuse it with the near-identical title by Nemer et al., Sensors 2022 (SBG-AC). — [PubMed](https://www.pubmed.gov/35271067/)
- Related follow-ups found only in reference lists (not verified): Liu et al., "Distributed Energy-Efficient Multi-UAV Navigation for Long-Term Communication Coverage by DRL," IEEE TMC 2020; Liu et al., "Trajectory Design and Power Control for Multi-UAV Assisted Wireless Networks: A ML Approach," IEEE TVT 2019. — cited inside [Rasool et al. arXiv 2303.05465](https://arxiv.org/abs/2303.05465)
- **Evolutionary multi-objective PPO for a UAV edge server (arXiv 2501.06410).** Task delay vs UAV energy, formulated as an MOMDP. Multi-strategy PPO plus target-distribution learning approximates a Pareto set. High-level, in simulation. — [arXiv 2501.06410](https://arxiv.org/pdf/2501.06410)
- **Evolutionary MORL for a UAV swarm (arXiv 2502.05824).** Transmission rate vs flight energy, with a Pareto-set approximation. Simulation. — [arXiv 2502.05824](https://arxiv.org/html/2502.05824v1)
- **Attention-based MORL for UAV IoT data collection (Toure et al., arXiv 2601.14092, 2026).** Data collected vs energy. Claims better generalization to unseen scenarios than prior RL. Simulation. — [arXiv 2601.14092](https://arxiv.org/html/2601.14092v1)
- A non-RL analytic baseline derives energy-vs-age-of-information Pareto trajectories for UAV data collection. — [arXiv 2106.03822](https://arxiv.org/pdf/2106.03822)
- MORL with a Bayesian belief network for manned/unmanned UAV data networks. It keeps a non-dominated (Pareto) set of Q-vectors over three objectives. The venue is a Canadian Science Publishing journal, so it is lower visibility. — [cdnsciencepub](https://cdnsciencepub.com/doi/10.1139/dsa-2022-0043)
- A DRL approach for multi-objective UAV sensor data collection (data vs flight time) uses decomposition into subproblems with actor-critic plus a pointer network. Springer conference chapter, 2023. — [Springer](https://link.springer.com/chapter/10.1007/978-981-99-1354-1_15)
- Pareto optimization layer on multi-agent actor-critic for UAV formation control (efficiency, energy, safety). MDPI Drones 2025, so treat with care. — [Drones 9(12):845](https://doi.org/10.3390/drones9120845)
- Surveys: Bai et al., IEEE COMST 2023, a multi-UAV wireless RL survey that discusses trade-offs among energy efficiency, coverage and user satisfaction — [PDF](https://acris.aalto.fi/ws/portalfiles/portal/130752954/Toward_Autonomous_Multi-UAV_Wireless_Network_A_Survey_of_Reinforcement_Learning-Based_Approaches.pdf). A DRL UAV comms survey, 2025 — [arXiv 2502.12875](https://arxiv.org/abs/2502.12875v1). A review of multi-objective UAV path planning in hostile environments (mostly evolutionary or metaheuristic), MDPI Drones 2024 — [Drones 8(12):769](https://www.mdpi.com/2504-446X/8/12/769)

### Inferences
- The high-level MORL literature on UAVs is large but orthogonal to low-level flight control. Its "objectives" are network or mission metrics (coverage, AoI, throughput, energy, delay), and the low-level flight dynamics are abstracted away.
- The dominant pattern is either a hand-weighted scalar reward or an evolutionary Pareto-set search over policies. Gradient-aggregation methods (MGDA, PCGrad, etc.) did not appear in this body of work in my searches.

### Gaps
- Citation counts and a reliable "most-cited" ranking were not obtained. Google Scholar or IEEE Xplore sorted by citations would be needed.
- I did not verify the author lists or venues of 2501.06410 and 2502.05824 beyond the arXiv pages. They may be journal papers in IEEE TMC or IoT-J.

## Q2. Model-assisted RL of a Quadrotor (arXiv 2311.06914)

### Takeaway
This is a single-author arXiv preprint (no peer-reviewed venue). It uses PPO at a high level (waypoint displacement plus speed, tracked by a PID) on a Crazyflie model. MORL appears only as a formulation for run-time conservativeness tuning. No MORL experiments are reported, and real hardware flights are not clearly shown.

### Cited Findings
- Arshad Javeed, "Model-assisted Reinforcement Learning of a Quadrotor," arXiv:2311.06914 [cs.RO], submitted 12 Nov 2023. No venue listed. — [arXiv](https://arxiv.org/abs/2311.06914)
- Control level: high-level and hierarchical. The action is a relative displacement (Δx, Δy, Δz), each capped at 0.05 m, plus a scalar target speed, executed by a closed-loop low-level PID. The state is the 12-D kinematics plus 3 action-error states. — [arXiv HTML](https://arxiv.org/html/2311.06914)
- Two objectives: a navigation reward (negative squared distance to goal) vs an action-error reward (negative discounted cumulative squared action-tracking error, which stops the agent from exploiting the navigation reward). — [arXiv HTML](https://arxiv.org/html/2311.06914)
- MORL: a scalarized expected return (SER) formulation. The proposal is to learn a set of policies, one per weight vector, and switch among them at run time to set conservativeness. The MO section has no experiments. The reported results are single-objective (ESR, fixed weights [1, 0.5]). PPO (stable-baselines3), [64, 64] network, 1M steps, with periodic step disturbances in training. — [arXiv HTML](https://arxiv.org/html/2311.06914)
- Results: the error-feedback ("dist-err") model is smoother and converges more often than the baselines. It has a larger backward reachability tube (HJ PDE, with a bang-bang approximation of the policy), even though the baseline earns a higher RL return. The paper concludes that a higher return does not imply more robustness. — [arXiv HTML](https://arxiv.org/html/2311.06914)
- Hardware: the abstract says "navigating a CrazyFlie quadrotor," but the experiments appear to be in simulation. Hardware trials are not clearly confirmed. — [arXiv HTML](https://arxiv.org/html/2311.06914)

### Inferences
- This is the closest "MORL plus Crazyflie plus run-time trade-off" hit, but it is weak evidence: no hardware, no MO experiments, a high-level action space, and not peer reviewed. It does not pre-empt low-level multi-objective control on a Crazyflie.

### Gaps
- Whether any real Crazyflie flight was done could not be confirmed from the HTML.

## Q3. Multi-Task Reinforcement Learning for Quadrotors (arXiv 2412.12442, RA-L)

### Takeaway
Xing et al. (UZH RPG, RA-L accepted Nov 2024) train one shared CTBR actor with one critic per task and shared plus task-specific encoders. It is validated on a real Agilicious quadrotor. Gradient conflict is mentioned (CAGrad is cited) but not studied or resolved. The actor simply optimizes the average return over uniformly sampled tasks.

### Cited Findings
- Jiaxu Xing, Ismail Geles, Yunlong Song, Elie Aljalbout, Davide Scaramuzza, IEEE RA-L (accepted Nov 2024; published 2025). Tasks: high-speed stabilization, velocity tracking, autonomous racing. — [arXiv abs](https://arxiv.org/abs/2412.12442); [RPG PDF](https://rpg.ifi.uzh.ch/docs/RAL25_Xing.pdf)
- Control level: collective thrust plus body rates (CTBR), a 4-D tanh output. On hardware, BetaFlight tracks the CTBR commands. — [arXiv HTML](https://arxiv.org/html/2412.12442)
- Hardware: an Agilicious quadrotor (0.6 kg, thrust-to-weight 5.78) with VICON. Flightmare simulation. Real-world tests: 5 runs per task, 100% success. Racing gate error is 0.152 m in simulation and 0.189 m real. — [arXiv HTML](https://arxiv.org/html/2412.12442)
- Architecture: "A separate critic function is used for each task," not used at deployment. The critics take the full observation and do not share the policy encoder. A shared dynamics encoder (position, orientation, velocities, previous action) produces a 32-D embedding, a per-task encoder produces another 32-D embedding, and the 64-D concatenation feeds the shared actor. — [arXiv HTML](https://arxiv.org/html/2412.12442)
- Gradient combination: the actor maximizes the average expected return across uniformly sampled tasks. No PCGrad, CAGrad or other conflict-resolution method is used. The introduction mentions "conflicting objectives" and cites CAGrad. — [arXiv HTML](https://arxiv.org/html/2412.12442)
- Results (10 seeds): at 20M samples, 100% racing success and 0.187 m gate error, vs 95% and 0.224 m for single-task RL. At 40M samples it is roughly on par with single-task (tracking 1.21 vs 1.18 m/s worse). Ablations: MTRL-Actor (only the actor shared) and MTRL-Separate (task encoders do not see the shared observation; 0% racing). — [arXiv HTML](https://arxiv.org/html/2412.12442)

### Inferences
- "Per-task critics plus a shared actor plus averaged actor loss" is the closest published structural analogue to per-objective critics at the CTBR level, with real hardware. The objectives here are tasks (different MDPs), not reward terms within one task. Conflict handling is an open, unaddressed question in this work.

### Gaps
- Whether the authors measured gradient cosine similarity in supplementary material was not checked. The HTML shows no such analysis.

## Q4. Multitask RL for quadcopter attitude stabilization and tracking with a graph policy (arXiv 2503.08259)

### Takeaway
Liu, Vale, Ahmad, Ventura, Basiri (2025, arXiv; ISR Lisboa / MPI) use a multitask SAC with a GCN policy that takes the task weight vector as input. It outputs attitude commands on an F450 with Pixhawk-6X at about 400 Hz, flown in real life. Gradient conflict is the stated motivation and is mitigated by graph structure (zero or tunable edges), but it is not measured.

### Cited Findings
- arXiv 2503.08259, submitted 11 Mar 2025. Code: robot-perception-group/GraphMTSAC_UAV. Venue not stated on arXiv. — [arXiv](https://arxiv.org/abs/2503.08259v1); [ISR page](https://isr.tecnico.ulisboa.pt/publications/multitask-reinforcement-learning-for-quadcopter-attitude-stabilization-and-tracking-using-graph-policy)
- Tasks and rewards: a linear reward r = φᵀw over 7 features (roll/pitch error, yaw error, roll/pitch rate, yaw rate, action smoothness). Tracking uses w = [1,1,1,1,0,0,0], smooth tracking adds 0.3 smoothness weights, stabilization uses [1,0,…]. There are up to 57 task weight vectors, normalized and sampled per environment. The policy is conditioned on w (plus an RMA latent). The replay buffer stores (s, a, s′), and rewards are recomputed per sampled w. This is effectively a successor-feature or preference-conditioned MORL setup. — [arXiv HTML](https://arxiv.org/html/2503.08259v1)
- Control level: roll, pitch and yaw attitude-level commands mapped to rotor thrusts. Altitude and target attitude come from ArduPilot cascade PIDs. — [arXiv HTML](https://arxiv.org/html/2503.08259v1)
- Gradient conflict: graph adjacency encodes decoupling (for example, pitch angle to roll control set to zero). This blocks gradient flow on irrelevant edges, and the adjacency is tunable during training. The paper does not measure conflict. — [arXiv HTML](https://arxiv.org/html/2503.08259v1)
- Network: 2 layers × 24 neurons, 697 parameters. Baselines: single-task SAC and Graph SAC, plus multitask feed-forward, parallel and compositional networks. Other architectures reached higher final accuracy than the graph policy. 3 seeds, figures only. — [arXiv HTML](https://arxiv.org/html/2503.08259v1)
- Real flight: attitude tracking within about 10°. Stabilization was tested by cutting the rotors for 1-3 s, with recovery in 13 of 13 trials. No quantitative real-world tracking error is reported. — [arXiv HTML](https://arxiv.org/html/2503.08259v1)

### Inferences
- This is the closest prior work to "reward-weight-conditioned (preference) policies at the attitude level on real hardware." It uses a single critic, not per-objective critics, and no gradient aggregation.

### Gaps
- Peer-reviewed venue unknown (possibly IROS or RA-L 2025; unverified).

## Q5. Safe and constrained RL for quadrotors at the low level

### Takeaway
Safe RL on quadrotors is dominated by safety filters (CBF, CLBF) wrapped around RL policies, often on Crazyflie hardware, but usually at the position or velocity level, with the Crazyflie firmware doing the inner loops. Lagrangian or CPO constrained RL directly on attitude or motor commands is sparse. The closest hits are a 2026 PPO-Lagrangian reduced-order paper and a thesis.

### Cited Findings
- Pizarro Bejarano, Brunke, Schoellig, "Safety Filtering While Training: Improving the Performance and Sample Efficiency of RL Agents," IEEE RA-L 10(1):788-795, Jan 2025, DOI 10.1109/LRA.2024.3512374. Real Crazyflie 2.0 experiments. It studies the effect of filtering during training on performance, sample efficiency and chattering. — [PDF](https://www.dynsyslab.org/wp-content/papercite-data/pdf/pizarro-ral25.pdf)
- Du et al., arXiv 2305.09793 (2023). Model-free actor-critic that learns a control Lyapunov barrier function (reachability plus safety). Deployed on a Crazyflie 2.0 for a 2-D navigation task. — [arXiv](https://arxiv.org/pdf/2305.09793)
- safe-control-gym (Yuan et al., arXiv 2109.06325). Benchmark with symbolic dynamics, constraints and disturbances for cart-pole and 1-D/2-D quadrotor stabilization and tracking. Explicitly focused on safe low-level control. Compares control, learning-based control and safe RL. — [arXiv](https://arxiv.org/abs/2109.06325)
- SafeSwarm (arXiv 2501.07566, 2025). PPO/TD3 with a control barrier net for swarm landing. Crazyflie 2.1 swarm with Vicon. High level. — [arXiv](https://arxiv.org/html/2501.07566v1)
- AttentionSwarm (arXiv 2503.07376, 2025). RL with an attention-based CBF on Crazyflie. The safe policy's outputs are relayed to the onboard Crazyflie controller (high level). — [arXiv](https://arxiv.org/pdf/2503.07376)
- Disturbance-observer CBF with RL. 2-D quadrotor simulation. The guarantee can be lost if the CBF is invalid under uncertainty or input limits. — [arXiv 2211.17250](https://arxiv.org/pdf/2211.17250)
- Safe observer-controller synthesis with CBFs on a Crazyflie 2.0. The safety-filter output is mapped to motors by the firmware. Not RL. — [arXiv 2211.14364](https://arxiv.org/pdf/2211.14364)
- PPO-Lagrangian on a reduced-order CMDP where attitude becomes a commanded reference tracked by an inner loop (zero-shot safety for cascade systems, 2026). — [arXiv 2604.10429](https://arxiv.org/pdf/2604.10429)
- CALOS (Univ. Bologna thesis, 2026). A QP-based hard tilt-angle constraint layer on a learned quadrotor policy. Fewer attitude violations. Thesis, full text unavailable. — [amslaurea](https://amslaurea.unibo.it/id/eprint/38567/)
- PPO outer loop that tunes PID gains with safety bounds and a Lagrangian penalty. MDPI Aerospace 2026, an engineering safeguard, not a proof. — [MDPI](https://www.mdpi.com/2226-4310/13/5/446/htm)

### Inferences
- Constraint-based (Lagrangian) treatment of one objective, such as tilt or crash as a cost, is the main "separate objective" mechanism in quadrotor safe RL. At the attitude or motor level on real hardware, it appears essentially unexplored outside filters.

### Gaps
- I could not extract Pizarro et al.'s RL algorithm, filter type or action level (the PDF body did not decode).
- I did not search specifically for CPO, FOCOPS or PID-Lagrangian on quadrotor rate control.

## Q6. Gradient aggregation (MGDA/PCGrad/CAGrad/Nash-MTL/UPGrad) or per-objective critics at the low level

### Takeaway
Several targeted searches found no drone or quadrotor paper that applies MGDA, PCGrad, CAGrad, Nash-MTL or UPGrad to RL control at any level, sim or real. Per-objective or per-task critics at the low level exist only as multi-task critics (Xing et al., CTBR) or as multi-agent axis decompositions (per-axis critics). None combines critics' actor gradients with a conflict-aware aggregator.

### Cited Findings
- General MTL references: CAGrad (NeurIPS 2021) subsumes GD and MGDA. On Meta-World MT10/MT50 it reaches 0.83/0.52 success vs PCGrad 0.72/0.50. — [arXiv 2110.14048](https://arxiv.org/abs/2110.14048); [NeurIPS PDF](https://papers.nips.cc/paper/2021/file/9d27fdf2477ffbff837d73ef7ae23db9-Paper.pdf). Sparse-training benchmark of PCGrad, CAGrad, GradDrop, MGDA, IMTL-G and NashMTL — [arXiv 2411.18615](https://arxiv.org/abs/2411.18615)
- Xing et al. cite CAGrad but do not use it (see Q3). — [arXiv HTML](https://arxiv.org/html/2412.12442)
- Yu and Lee, "Multi-Agent RL for the Low-Level Control of a Quadrotor UAV," arXiv 2311.06144 (2023/2024). Translational/roll-pitch agent plus yaw agent, each with its own observations and rewards. A mixer produces the 4 motor thrusts, and critics are centralized. Sim-to-sim only, per the abstract. — [arXiv](https://arxiv.org/abs/2311.06144)
- Multi-agent TD3 for quadrotor attitude with separate roll, pitch and yaw critics, each using its own reward plus a global cooperative reward. IJCDS (Univ. of Bahrain journal), a low-visibility venue; hardware unknown. — [IJCDS PDF](https://iiict.uob.edu.bh/IJCDS/papers/1571033189.pdf)
- Graph-policy MT-SAC mitigates conflict architecturally, not by gradient aggregation (see Q4). — [arXiv HTML](https://arxiv.org/html/2503.08259v1)
- cc-DRL (arXiv 2408.13054). Convex combination of DRL controllers for a morphing quadrotor. A policy-blending approach, not an MOO gradient method. Details not read. — [arXiv](https://arxiv.org/html/2408.13054v1)

### Inferences
- An apparent novelty gap: conflict-aware gradient aggregation over per-objective (reward-term) critics for attitude or motor-level quadrotor RL, especially on Crazyflie hardware, has no prior work in these searches. This is absence-of-evidence from about 6 searches, not proof.

### Gaps
- Google Scholar full-text search for "PCGrad" plus "quadrotor", or "Nash-MTL" plus "UAV", was not possible. Chinese-language venues and IEEE TAES / IoT-J were not exhaustively covered.

## Q7. Real-hardware (Crazyflie or similar) multi-objective low-level controllers

### Takeaway
No real-hardware Crazyflie RL controller with explicit multi-objective treatment at the attitude or motor level was found. The nearest real-hardware works are Xing et al. (CTBR, Agilicious, multi-task critics), Liu et al. (attitude, F450/Pixhawk, weight-conditioned rewards) and the Crazyflie safety-filter papers (Pizarro et al., Du et al.; constraint plus performance, not MORL). A quadruped analogue (PROMO) shows run-time preference-conditioned trade-offs on real hardware.

### Cited Findings
- Xing et al.: real Agilicious, CTBR, per-task critics. — [arXiv HTML](https://arxiv.org/html/2412.12442)
- Liu et al.: real F450/Pixhawk-6X, attitude commands, policy conditioned on reward weights w. — [arXiv HTML](https://arxiv.org/html/2503.08259v1)
- Pizarro Bejarano et al.: real Crazyflie 2.0, RL with a safety filter during training. — [PDF](https://www.dynsyslab.org/wp-content/papercite-data/pdf/pizarro-ral25.pdf)
- PROMO (arXiv 2610.01260, 2026). Preference-conditioned MORL for a quadruped (Unitree Go2). Tracking, stability and efficiency are a run-time input. On the real robot, preference changes alone cut specific energy by up to 30.4%, position error by 38.7% and peak attitude deviation by 59.0% vs the balanced preference. Policy at about 50 Hz over a 200 Hz low-level loop. Not a drone. — [arXiv](https://arxiv.org/abs/2610.01260)
- Single-objective low-level baselines for context: Molchanov et al., Sim-to-(Multi)-Real, state to motor outputs, transferred across several real quadrotors including Crazyflie (single objective) — [arXiv 1903.04628](https://arxiv.org/pdf/1903.04628). Hwangbo et al., "Control of a Quadrotor with RL" (RA-L 2017), direct rotor thrust — [arXiv 1707.05110](https://arxiv.org/pdf/1707.05110)

### Inferences
- For the IJCNN framing: the closest competitors to cite and differentiate from are Xing et al. 2412.12442 (multi-critic, but tasks, CTBR, and averaged gradient), Liu et al. 2503.08259 (weight-conditioned rewards at the attitude level, single critic, structural conflict mitigation), Javeed 2311.06914 (MORL formulation only, high level, no experiments) and PROMO (preference MORL on a quadruped). A per-reward-term critic with gradient aggregation on Crazyflie at the rate or motor level appears unclaimed.

### Gaps
- I did not check for 2026 workshop papers or theses on Crazyflie MORL, or Bitcraze community projects. I did not verify whether Molchanov et al. explicitly used a Crazyflie; to my recollection they did (Crazyflie 2.0 and others), but this is not confirmed in this session.
