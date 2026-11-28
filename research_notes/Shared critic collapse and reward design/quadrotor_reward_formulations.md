# Reward formulations of published learned quadrotor controllers (input to our 4-group multi-objective reward design)

Read-status tags used below:
- **[R]**: I read the reward, termination and training passages in the primary text (arXiv PDF text, arXiv HTML, or Europe PMC full text). I did not read the whole paper.
- **[C]**: I read the reward function in the released code (default branch, cloned 2026-11-27). Code can differ from the published configuration.
- **[A]**: abstract only.
- **[S]**: search-result snippet only (not verified).

Equations are transcribed from the extracted text with symbols normalised (subscripts and Unicode). "Ours" means the setup in the task brief (`l2f/docs/shared_critic_research_prompt.md`, sections 1-2).

---

## Q1. Learning to Fly in Seconds and its follow-ups: exact reward, survival bonus, termination, curriculum

### Takeaway
L2F (Eschmann et al., RA-L 2024) uses a **negative squared cost plus a constant survival reward** (+2 per step after a 0.5 scale). The terminal step gets **reward 0**, with no explicit crash penalty: a crash costs the lost stream of +2 rewards. A multiplicative curriculum every 100k steps raises the position (x1.2, cap 40), velocity (x1.4, cap 1) and action (x1.4, cap 1) weights, and the replay buffer's rewards are **recalculated** at each change. Removing the curriculum collapsed real-flight success. The same group's 2025 follow-up RAPTOR changed the design: **unsquared norms**, an **action-rate** term, **+1.5 survival with an explicit -100 termination penalty**, and SAC instead of TD3. It **dropped the curriculum**.

### Cited Findings
**L2F paper [R]**
- Reward equation (p. 4, training-setup section): "We use a negative squared cost with an additive constant incentivizing survival to mitigate the 'learning to terminate' problem [34]: r(s, a, s′) = −C_rp‖p‖²₂ − C_rq(1 − q_w²) − C_rv‖v‖²₂ − C_rω‖ω‖²₂ − C_ra‖a − C_rab‖²₂ + C_rs." The constants are "supplied in the supplementary material". — [Eschmann, Albani, Loianno, arXiv 2311.13081 (RA-L, accepted April 2024)](https://arxiv.org/abs/2311.13081)
- Reference [34] in that paper is J. Eschmann, "Reward function design in reinforcement learning," in *Reinforcement Learning Algorithms: Analysis and Applications*, pp. 25-33, 2021 (I did not read it). — [L2F reference list](https://arxiv.org/abs/2311.13081)
- Curriculum rationale (p. 4): "a simple curriculum that is transitioning from an initial set of constants C_init,∗ to a more restrictive C_target,∗ (punishing position errors and particularly control actions more harshly) benefits the sample complexity and Sim2Real transfer". "Every 100 000 steps the weights are adjusted by multiplying them by the Cp∗ factors … until they reach the C_target,∗ values, where they remain constant. We also decay the exploration noise using the same exponential scheme". — [L2F](https://arxiv.org/abs/2311.13081)
- Task framing: "we train a position controller, not merely a stabilizing controller … The goal of our position controller is to return to the origin point with zero linear velocity from any initial conditions". — [L2F](https://arxiv.org/abs/2311.13081)
- Evaluation metric: "it is hard to judge the level of flying capabilities of a particular policy because it is highly dependent on the reward function formulation. We believe the episode length is a more understandable measure than the return because if the agent crashes or flies away from a tight box around the origin, the episode is terminated." — [L2F, p. 5](https://arxiv.org/abs/2311.13081)
- Ablation (Table II). Each row removes one component. The # values are successful real flights out of 10 (no crash) in each of the three real-world tests, which include position control and figure-eight tracking. I could not recover the column order from the extracted text.
  - All components: 10/10, 10/10, 10/10.
  - No curriculum: 1/10, 1/10, 9/10.
  - No reward recalculation: 9/10, 9/10, 7/10.
  - No asymmetric actor-critic: 6/10, 4/10, 9/10.
  - No action history: 5/10, 5/10, 0/10.
  - No rotor delay: 0/10, 0/10, 0/10.
  - No asymmetric actor-critic and no curriculum: 0/10, 0/10, 2/10.
  - [L2F Table II](https://arxiv.org/abs/2311.13081)

**L2F code (upstream `arplaboratory/learning-to-fly`, commit d07592d, May 2025) [C]**
- Reward logic: if the next state is terminal, `reward = termination_penalty`. Otherwise `reward = −scale·weighted_cost + constant`, clipped at 0 only if `non_negative` is set. The orientation cost is `1 − q_w²`, and the action cost is `‖a − action_baseline‖²` with baseline 0.334. Linear- and angular-acceleration cost terms also exist in the code. — [squared.h L62-L67](https://github.com/arplaboratory/learning-to-fly/blob/master/include/learning_to_fly/simulator/parameters/reward_functions/squared.h#L62-L67)
- Constants actually used:
  - Initial reward `reward_squared_position_only_torque`: non_negative false, scale 0.5, constant 2, termination_penalty 0, position 5, orientation 5, linear velocity 0.01, angular velocity 0, linear/angular acceleration 0, action 0.01.
  - Target reward `…_curriculum_target`: the same except position 40, linear velocity 1.0, action 1.0.
  - The config selects these two via `initial_reward_function` and `target_reward_function`.
  - [default.h L410-L438](https://github.com/arplaboratory/learning-to-fly/blob/master/include/learning_to_fly/simulator/parameters/reward_functions/default.h#L410-L438); [config/parameters.h L32-L34](https://github.com/arplaboratory/learning-to-fly/blob/master/src/config/parameters.h#L32-L34)
- Termination: the config uses `termination::fast_learning` = {enabled, position 0.6, linear velocity 1000, angular velocity 1000}, so in practice only the position box terminates. A "classic" variant uses 10 and 10. The position check is **per axis** (`abs(state.position[i]) > position_threshold`). — [termination/default.h](https://github.com/arplaboratory/learning-to-fly/blob/master/include/learning_to_fly/simulator/parameters/termination/default.h#L15-L19); [parameters.h L63](https://github.com/arplaboratory/learning-to-fly/blob/master/src/config/parameters.h#L63); [operations_generic.h L697](https://github.com/arplaboratory/learning-to-fly/blob/master/include/learning_to_fly/simulator/operations_generic.h#L697)
- Curriculum code:
  - Every 100,000 steps: action weight x1.4 (cap 1.0), position x1.2 (cap 40), linear velocity x1.4 (cap 1).
  - When `RECALCULATE_REWARDS` is set (the default), `recalculate_rewards(...)` rewrites the rewards stored in the replay buffer.
  - From step 500k, exploration noise and target-policy noise (std and clip) are multiplied by 0.9 every 100k steps.
  - [curriculum.h](https://github.com/arplaboratory/learning-to-fly/blob/master/src/steps/curriculum.h#L8-L55); [ablation.h](https://github.com/arplaboratory/learning-to-fly/blob/master/src/config/ablation.h)
- Other reward presets in the same file show the authors' exploration of alternatives (not used in the paper): `reward_squared_4` (termination penalty −100, constant 0), `reward_squared_5` (termination penalty −10000), `reward_squared_1` (constant 300, scale 0.01), and several "AbsExp" (absolute-value/exponential) presets. — [default.h](https://github.com/arplaboratory/learning-to-fly/blob/master/include/learning_to_fly/simulator/parameters/reward_functions/default.h)

**RAPTOR (Eschmann, Albani, Loianno; arXiv 2509.11481 v2, Science Robotics) [R]**
- Teacher reward (Materials and Methods, Eq. 1; verified against the LaTeX source in the arXiv HTML): r(s_t, a_t, s_{t+1}) = −‖p‖₂ − 0.2·arccos(1 − |q_z|) − ‖a_t − a_{t−1}‖₂ + 1.5 − 100·1[terminal(s_{t+1})]. — [arXiv HTML v2](https://arxiv.org/html/2509.11481v2)
- Termination: position beyond 20·l_arm, linear velocity beyond 2 m/s, or angular velocity beyond 35 rad/s. Initial states are sampled up to 10·l_arm, 90°, 1 m/s and 1 rad/s, and with 10% probability the initial state is the target state. γ = 0.99. — [arXiv HTML v2](https://arxiv.org/html/2509.11481v2). These details and the quotes below came through a summarising fetch tool; only Eq. 1 was checked character by character.
- Stated changes to the L2F pipeline: "Switching from TD3 to SAC because we observed slightly more robust training dynamics in SAC"; "adding a penalty for termination and for the action derivative"; "Removing the curriculum because we found that the changes to the reward function stabilize the training without the need for a curriculum"; "We tuned the reward function for reliable training, particularly the survival bonus and the termination penalty." — [arXiv HTML v2](https://arxiv.org/html/2509.11481v2)
- Returns are a poor cross-platform metric ("More agile quadrotors … achieve much higher returns"), so episode length is used instead. — [arXiv HTML v2](https://arxiv.org/html/2509.11481v2)
- Publication: the repository README links the Science Robotics DOI 10.1126/scirobotics.aec1481. — [rl-tools/raptor README](https://github.com/rl-tools/raptor); [DOI](https://doi.org/10.1126/scirobotics.aec1481)

### Inferences
- **L2F's crash price is implicit.** With constant +2 and terminal reward 0, termination forfeits about 2/(1−γ) = 200 of reward at γ = 0.99. Ours (survival 0, terminal −400) is about **twice** that price. RAPTOR's price is 1.5/(1−γ) + 100 ≈ 250. (My arithmetic, using γ = 0.99, which matches RAPTOR's stated value and our TD3 setting.)
- **L2F's survival term only protects while the per-step reward is positive**, that is while 0.5·cost < 2. With the initial weights (position 5) the break-even position error is √(4/5) ≈ 0.89 m, outside the 0.6 m box, so early in training surviving always beats terminating. With the target weights (position 40) the break-even is √(4/40) ≈ 0.32 m. Late in the curriculum, a state with a large error has negative reward and termination (reward 0) becomes attractive unless recovery is quick. The curriculum delays the harsh weights until the policy no longer visits such states. (My inference.)
- RAPTOR's changes point to a broader lesson: **an explicit termination penalty plus an action-rate term replaced the need for a weight curriculum**. The authors, not a controlled ablation, credit this for training stability.
- L2F's reward recalculation matters for any curriculum: without it, the replay buffer mixes rewards from different weightings. Removing it cost real-flight successes in Table II.

### Gaps
- The L2F supplementary `parameters.pdf` was not read. Constants come from the current upstream code, which matches our description but may not match the paper version exactly.
- I did not extract the L2F paper's own statement of γ and episode length.
- I did not check the other follow-ups (RLtools, JMLR, [arXiv 2306.03530](https://arxiv.org/abs/2306.03530); motor-delay system identification, [arXiv 2404.07837](https://arxiv.org/abs/2404.07837)) for reward changes.
- RAPTOR's teacher episode length is not stated (per the fetch tool).
- A 2026 follow-up comparing against RAPTOR, "Embodiment-conditioned Generalist Control for Multirotor Aerial Robots" ([arXiv 2606.10857](https://arxiv.org/abs/2606.10857)) [S], was not read.

---

## Q2. Comparative table of reward formulations (low-level, hover/tracking, agile/racing)

### Takeaway
Real-hardware low-level controllers fall into three families:
1. **Unbounded costs** (squared or absolute norms). Hwangbo, Molchanov, Gronauer, L2F, RAPTOR, DATT and Zhang use these. Most add a survival constant or a moderate terminal penalty (−10 to −100). Kaufmann 2022 is the exception at −500.
2. **Bounded, positive, often gated rewards**. OmniDrones and SimpleFlight use exp(−x) or 1/(1+x²); Aerial Gym and gym-pybullet-drones also bound their rewards. Termination then needs no explicit penalty.
3. **Progress-based rewards** for time-optimal and agile flight (Song 2021/2023, Penicka, Kaufmann 2023, Ferede, Xing). These use small, often bounded crash penalties (−5 to −20).

An **action-rate (Δa) term** is the most consistently used and best-supported smoothness term.

### Cited Findings

#### Table A: hover, position, attitude and tracking controllers

| Work | Level / algorithm | Per-step reward (exact form, weights) | Form | Survival | Termination → terminal reward | Curriculum | Source |
|---|---|---|---|---|---|---|---|
| Hwangbo et al., RA-L 2017 [R] | rotor thrusts, plus a low-gain PD attitude loop and gravity thrust bias during training; own on-policy method; γ = 0.99 | cost r_t = 4×10⁻³‖p_t‖ + 2×10⁻⁴‖a_t‖ + 3×10⁻⁴‖ω_t‖ + 5×10⁻⁴‖v_t‖ (Eq. 9) | absolute norms | none | finite trajectories; tail value from V(s) | none reported | [arXiv 1707.05110, p. 5](https://arxiv.org/abs/1707.05110) |
| Molchanov et al., IROS 2019 [R][C] | rotor thrusts, Crazyflie 2.0 and larger frames | c_t = (‖e_p‖₂ + α_v‖e_v‖₂ + α_ω‖e_ω‖₂ + α_a‖a‖₂ + α_R·cos⁻¹((Tr(R)−1)/2))·dt (Eq. 14); chosen α_ω = 0.1, α_a = 0.05, α_R = α_v = 0 | absolute norms | none | paper: none in the text I read. Code: crash (z ≤ arm length) adds crash·1[crashed] (default 1.0); the episode is not ended (`#or self.crashed` commented out) | none | [arXiv 1903.04628 Eq. 14, Table III](https://arxiv.org/abs/1903.04628); [quadrotor.py](https://github.com/amolchanov86/quad_sim2multireal/blob/master/quad_sim/quadrotor.py#L603-L660) |
| Koch et al., GymFC v1 (ACM TCPS 2019; venue not verified) [R] | motor outputs, attitude-rate tracking | r_t = −clip(Σ|Ω*_t − Ω_t| / 3Ω_max) (Eq. 7), normalised to [−1, 0] | absolute, normalised and clipped | none | – | – | [arXiv 1804.04154](https://arxiv.org/abs/1804.04154) |
| Koch et al., Neuroflight / GymFC v1.5 (arXiv 2019) [R][C] | motor outputs, attitude-rate | r = r_e + r_y + r_Δ (Eq. 3); r_e = −(e_φ² + e_θ² + e_ψ²) (Eq. 4); r_Δ = β Σᵢ max{0, Δy_max − (Δy_i)²} (Eq. 5), active only inside an error band; r_y rewards low motor output. Code: −1000·max\|y − y_prev\|; min-output reward ≤ 1000·(1 − mean y)·inband; e_penalty = shaping_t − shaping_{t−1} with shaping = −Σe²; −10⁹·oversaturation; a "doing nothing" penalty | squared error, bounded hinge, difference-of-potential term | none | – | – | [arXiv 1901.06553](https://arxiv.org/abs/1901.06553); [gymfc_nf rewards.py](https://github.com/wil3/gymfc/blob/master/examples/gymfc_nf/envs/rewards.py) |
| Gronauer et al., 2022 (arXiv; venue not verified) [R][C] | PWM / attitude-rate / attitude, Crazyflie 2.1 | PDF text: "r(s,a,t) = ‖e‖₂ + 0.0001‖a_t‖₂ + 0.001‖a_{t−1} − a_t‖₂ + 0.001‖ω‖₂ + r_f" (Eq. 10); the code computes reward = −dist − penalties, so the norms are costs | absolute norms | none | ‖e‖₂ > 0.25 m → r_f = −100, terminate; 500-step (5 s) episodes | none | [arXiv 2201.01369 Eq. 10](https://arxiv.org/abs/2201.01369); [hover.py](https://github.com/SvenGronauer/phoenix-drone-simulation/blob/master/phoenix_drone_simulation/envs/hover.py#L169-L187) |
| Eschmann et al., L2F, RA-L 2024 [R][C] | RPM setpoints (motor level), TD3, asymmetric critic, action history | r = −C_rp‖p‖² − C_rq(1−q_w²) − C_rv‖v‖² − C_rω‖ω‖² − C_ra‖a − C_rab‖² + C_rs. Code: scale 0.5, +2, position 5→40, orientation 5, velocity 0.01→1, ω 0, action 0.01→1, baseline 0.334 | squared | +2 per step after scale | \|p_i\| > 0.6 m → reward = 0 (termination_penalty 0) | x1.2 / x1.4 every 100k steps, replay rewards recalculated | see Q1 |
| Eschmann et al., RAPTOR (Sci. Robot.; arXiv 2509.11481) [R] | motor-level teachers (SAC), 1000 teachers distilled into one student | r = −‖p‖₂ − 0.2·arccos(1−\|q_z\|) − ‖a_t − a_{t−1}‖₂ + 1.5 − 100·1[terminal] | absolute norms | +1.5 per step | \|p\| > 20 l_arm, \|v\| > 2 m/s, \|ω\| > 35 rad/s → −100 | removed | see Q1 |
| Zhang et al., "Extreme adaptation", arXiv 2409.12949 (2024) [R] | CTBR → motor speeds; IL + RL | sum of: −‖a_t − a_{t−1}‖; "Survival Reward: δt"; −‖c_Σ − c_Σ,des‖; −‖τ_t − τ_des‖, with τ_des = J·K(ω_des − ω) + ω × Jω and K = diag(20, 20, 4) s⁻¹ | absolute norms | δt per step | not found in the text I read | imitation weight α = e^(−0.001·t_epoch) | [arXiv 2409.12949, Reward Design](https://arxiv.org/abs/2409.12949) |
| Huang et al., DATT, CoRL 2023 [R][C] | CTBR, PPO, Crazyflie 2.1 | "r = ‖p_t − p_t^d‖ + 0.5‖ψ_t‖ + 0.1‖v_t‖" (Eq. 3b), used as a cost (code returns −cost; velocity term 0.1·min(‖v‖, 1)). Hover task clips position at min(‖p‖, 1.0) | absolute, partly clipped | none | – | – | [arXiv 2310.09053 Eq. 3b](https://arxiv.org/abs/2310.09053); [trajectory.py](https://github.com/KevinHuang8/DATT/blob/main/learning/tasks/trajectory.py#L52-L62); [hover.py](https://github.com/KevinHuang8/DATT/blob/main/learning/tasks/hover.py#L11-L20) |
| Kaufmann, Bauersfeld, Scaramuzza, benchmark, ICRA 2022 [R] | single-rotor thrust vs CTBR vs linear velocity; PPO, γ = 0.98 | r_t = −(x_t − x_ref)ᵀQ(x_t − x_ref) − (u_t − u_ref)ᵀR(u_t − u_ref) − r_crash (Eq. 5); Q = diag{0.1·1₃, 0.02·1₉, 0.002·1₃, 0.01·1₃}, R = diag{0.001·1₄} | squared | none | p_z ≤ 0 → r_crash = −500, terminate | – | [arXiv 2202.10796 Sec. IV-A](https://arxiv.org/abs/2202.10796) |
| Chen et al., SimpleFlight ("What matters…"), RA-L 2025 [R][C] | CTBR, PPO, Crazyflie 2.1 | r = r_task + λ·r_smooth (Eq. 4), "We normalize both r_task and r_smooth to the range [0, 1]"; r_smooth based on ‖u_t − u_{t−1}‖². Code: pos = s·exp(−d); up = w·0.5/(1 + tilt²); spin = w·0.5/(1 + spin²); action norm, Δaction, acc, jerk, snap = w·exp(−x); total = pos + pos·(up + spin) + others | exp and 1/(1+x²), bounded | implicit (all terms ≥ 0) | not extracted | penalty weights ramp linearly to a maximum during training | [arXiv 2412.11764 Eq. 4, Table I](https://arxiv.org/abs/2412.11764); [track.py](https://github.com/thu-uav/SimpleFlight/blob/main/omni_drones/envs/single/track.py#L473-L516) |
| Xu et al., OmniDrones Hover task (RA-L 2024; venue not verified) [C] | rotor commands (simulation platform) | pose = 1/(1 + (k·d)²), d over position and heading error; up = ((up_z + 1)/2)²; spin = 1/(1 + spin²); effort = w·exp(−effort); smooth = w·exp(−Δthrottle); r = pose + pose·(up + spin) + effort + smooth | 1/(1+x²) and exp, bounded | implicit (≥ 0) | z < 0.2 m or d > 4 → terminate, no penalty; time-limit truncation | – | [hover.py L308-L340](https://github.com/btx0424/OmniDrones/blob/main/omni_drones/envs/single/hover.py#L308-L340) |
| Panerati et al., gym-pybullet-drones HoverAviary (IROS 2021; venue not verified) [C] | RPM and others | r = max(0, 2 − ‖p* − p‖⁴) | bounded, clipped ≥ 0 | implicit | \|x\|,\|y\| > 1.5, z > 2, \|roll\|,\|pitch\| > 0.4 → **truncation**, not termination | – | [HoverAviary.py L68-L112](https://github.com/utiasDSL/gym-pybullet-drones/blob/main/gym_pybullet_drones/envs/HoverAviary.py#L68-L112) |
| Kulkarni et al., Aerial Gym, end-to-end position task (RA-L 2025) [C] | motor thrusts | exp(x; m, w) = m·exp(−x²/w); pos = Σ exp(e; 10, 10) + Σ exp(e; 2, 2) (z error x11); upright, alignment, angular velocity, velocity as exp; action and Δaction as m·(exp(−x²/w) − 1); progress 10·Δd (15·Δd when moving away); r = progress + [pos·(align + vel + angvel + Δa) + (angvel + vel + upright + pos + action)]/100 | exp, bounded, plus progress | implicit | distance > crash_dist → crash | – | [compute_reward L267-L309](https://github.com/ntnu-arl/aerial_gym_simulator/blob/main/aerial_gym/task/position_setpoint_task_sim2real_end_to_end/position_setpoint_task_sim2real_end_to_end.py#L267-L309); [math.py L77-L89](https://github.com/ntnu-arl/aerial_gym_simulator/blob/main/aerial_gym/utils/math.py#L77-L89) |
| safe-control-gym quadrotor (Yuan et al., RA-L 2022; paper not read) [C] | thrusts | r = −Σ w_x·e_x² − Σ w_u·e_u²; optional r ← exp(r), commented "Convert rew to be positive and bounded [0,1]" | squared, or exp(−squared) | – | – | – | [quadrotor.py L819-L845](https://github.com/utiasDSL/safe-control-gym/blob/main/safe_control_gym/envs/gym_pybullet_drones/quadrotor.py#L819-L845) |
| Flightmare quadrotor env (Song et al., CoRL 2020; paper not read) [C] | thrust / rates | weighted error terms with coefficients position −0.002, orientation −0.002, linear velocity −0.0002, angular velocity −0.0002, action −0.0002; "survival reward" +0.1 | (norm form cut off in my excerpt) | +0.1 per step | z ≤ 0.02 → −0.02 | – | [quadrotor_env.cpp L108-L147](https://github.com/uzh-rpg/flightmare/blob/master/flightlib/src/envs/quadrotor_env/quadrotor_env.cpp#L108-L147); [quadrotor_env.yaml](https://github.com/uzh-rpg/flightmare/blob/master/flightlib/configs/quadrotor_env.yaml#L18-L22) |

#### Table B: agile, time-optimal, racing and multi-task controllers

| Work | Level / algorithm | Per-step reward (exact form) | Form | Survival | Termination → terminal reward | Curriculum | Source |
|---|---|---|---|---|---|---|---|
| Song et al., IROS 2021 [R] | CTBR, PPO | r(t) = r_p(t) + a·r_s(t) − b‖ω_t‖² + {r_T if gate collision, else 0} (Eq. 4); r_p(t) = s(p_c(t)) − s(p_c(t−1)), projected progress along the gate-to-gate segment (Eq. 2); safety r_s(t) = −f²·(1 − exp(−0.5·d_n²/v)) (Eq. 3) with f = max[1 − d_p/d_max, 0] and v = max[(1 − f)(w_g/6), 0.05]; r_T = −min[(d_g/w_g)², 20.0] (Eq. 5) | progress, bounded safety term, capped crash | none | gate collision → r_T (bounded at −20, scaled by miss distance) | "a quadratic penalty on the body rates … is added at an initial training stage" | [arXiv 2103.08624 Sec. III-B.1](https://arxiv.org/abs/2103.08624) |
| Penicka et al., RA-L 2022 [R] | CTBR, PPO | r(t) = k_p·r_p(t) + k_s·s(p(t)) + k_wp·r_wp + r_T − k_ω‖ω‖ (Eq. 10); r_p(t) = s(p(t)) − s(p(t−1)), progress along a topological guiding path (Eqs. 7-9) | progress | – | collision → r_T (value not extracted) | – | [arXiv 2203.15052 Sec. III-B](https://arxiv.org/abs/2203.15052) |
| Kaufmann et al., Swift, Nature 2023 [R] | CTBR, PPO, 1,500-step episodes, 100 agents | r_t = r_t^prog + r_t^perc + r_t^cmd − r_t^crash (Eq. 7); r^prog = λ1·[d_{t−1}^Gate − d_t^Gate]; r^perc = λ2·exp[λ3·δ_cam⁴] (Eq. 8); r^cmd = λ4·a_t^ω + λ5·‖a_t − a_{t−1}‖² (Eq. 9; norm marks on the body-rate term lost in text extraction); λ values in Extended Data Table 1a (not extracted) | progress, exp, squared | none | p_z < 0, gate collision, or leaving a bounding box → r_crash = 5.0, episode ends | – | [Nature 2023, Methods](https://www.nature.com/articles/s41586-023-06419-4); [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC10468397/) |
| Song et al., Science Robotics 2023 [R] | CTBR, PPO | "gate progress reward (Eq. 5)" (exact form not extracted); "penalized with a negative reward if it experiences a collision … rewarded with a positive reward upon successfully completing the maximum episode length" | progress | completion bonus | collision → negative reward | – | [arXiv 2310.10943, Materials and Methods](https://arxiv.org/abs/2310.10943) |
| Ferede et al., ICRA 2024 [R] | **direct motor commands** (end-to-end) with a learned residual model; PPO, γ = 0.999, 12 s episodes | r(k) = +10 − 10‖p_k − p_{g_k}‖ if a gate is passed; −10 if collision; ‖p_{k−1} − p_{g_k}‖ − ‖p_k − p_{g_k}‖ otherwise | progress | none | ground contact, or crossing the gate plane outside the gate → −10 | – | [arXiv 2311.16948 Sec. II](https://arxiv.org/abs/2311.16948) |
| Xing et al., multi-task RL, RA-L 2025 [R] | CTBR, PPO, **one critic per task**, shared actor | racing: r = r_prog + r_perc + r_act + r_br + r_pass + r_crash (Eq. 3), with r_prog = α1(d_Gate(t−1) − d_Gate(t)), r_perc = α2·exp(α3·δ_cam⁴), r_act = α4‖u_t − u_{t−1}‖, r_br = α5‖ω_B,t‖, r_pass = α6, r_crash = α7 (Eq. 4); stabilisation: r = r_height + r_attitude + r_velocity + r_br + r_act + r_success (Eq. 5); velocity tracking: r = r_velocity + r_br + r_act (Eq. 7); "All the rewards are formulated here using the L2 norm multiplied by a corresponding constant coefficient" | progress and norms | success bonus (stabilisation) | crash → α7, ends the episode | "…by 10% for every 100,000 data samples … stop once the predefined upper limits … are reached" (stabilisation task; the sentence's subject was cut in extraction) | [arXiv 2412.12442 Sec. III-C](https://arxiv.org/abs/2412.12442) |

#### Why each design was chosen, and what failed
- **Hwangbo 2017**:
  - Weighting rule: "Only the position has a high cost coefficient since that is what we care about the most. We roughly set the rest of the coefficients such that the other cost terms have about one tenth of the magnitude of the position error." — [arXiv 1707.05110](https://arxiv.org/abs/1707.05110)
  - What failed: "DDPG was not able to converge to adequate performance in reasonable amount of time". The simulation "sometimes become unstable (get NaN …) when the angular velocity becomes very high", and "the gradient observed when the quadrotor is upside-down is very low and discontinuous", which motivated the auxiliary PD controller. — [arXiv 1707.05110](https://arxiv.org/abs/1707.05110)
- **Molchanov 2019**:
  - The goal was "a simpler cost function with fewer hyper-parameters".
  - Table III: α_ω = 0 (with α_a = 0.05) → "Training failed"; α_ω = 0.05 → "No Takeoff" in RotorS. "we can train a successful policy with a cost that only penalizes position, angular velocity, and actions, as long as α_ω is larger than 0.05 but smaller than 1"; "higher α_ω can result in significantly higher errors in position" on the real quadrotor.
  - Adding α_R > 0 "improves position and angular errors in simulation, but results in slightly larger position errors on the physical quadrotor", while it "also stabilizes yaw".
  - Adding a linear-velocity cost (Hwangbo-like) "is harder to tune because of the larger number of hyper-parameters" and gave slightly worse errors.
  - [arXiv 1903.04628 Sec. VI-A](https://arxiv.org/abs/1903.04628)
  - The released code's position loss is pos·(log_weight·log(dist + 0.1) + linear_weight·dist), commented "log to create a sharp peak at the goal". Its defaults are pos 1, offset 0.1, log weight 1, linear weight 0.1, effort 0.01, crash 1, orient 1. — [quadrotor.py L603-L607, L892-L903](https://github.com/amolchanov86/quad_sim2multireal/blob/master/quad_sim/quadrotor.py#L892-L903)
- **Koch GymFC v1**: "Rewards are normalized to provide standardization and stabilization during training"; "We found sparse binary rewards to give poor performance." — [arXiv 1804.04154](https://arxiv.org/abs/1804.04154)
- **Neuroflight**: "The reward system used by GYMFC-V1 is strictly a function of the angular velocity error. This is inadequate in developing neuro-flight controllers that can be used in the real world. Essentially this produces controllers that closely resemble the behavior of an over-tuned PID controller … leading to output oscillation." The two control-output terms were "critical for transferability to the real world and achieving stable flight". — [arXiv 1901.06553 Sec. IV-B](https://arxiv.org/abs/1901.06553)
- **Gronauer 2022**: in real flights, "rollouts also failed when the distance between the drone and the set-point exceeded the episode termination criterion of ‖e‖₂ > 0.25m". The policy had not learned to act outside the training box. — [arXiv 2201.01369 Sec. 5](https://arxiv.org/abs/2201.01369)
- **Zhang 2024**: "The output oscillation penalty encourages smooth inputs, discouraging high-frequency control signals. The survival reward encourages the quadcopter to learn to fly longer until the end of the training episode." — [arXiv 2409.12949](https://arxiv.org/abs/2409.12949)
- **DATT code** comments that the action cost was used "for old versions of hover by accident!". The action cost is computed but excluded from the returned cost. — [hover.py](https://github.com/KevinHuang8/DATT/blob/main/learning/tasks/hover.py#L11-L20)
- **SimpleFlight (Table I, real Crazyflie tracking error in m, slow / normal / fast; ∞ = failed)**:
  - action clipping: 0.035 / 0.077 / 0.310
  - low-pass filter: ∞ / ∞ / ∞
  - ‖acc‖²: 0.183 / 0.329 / ∞
  - ‖jerk‖²: 0.024 / 0.047 / ∞
  - ‖snap‖²: 0.026 / ∞ / ∞
  - ‖u_t‖²: 0.044 / 0.066 / 0.110
  - **‖u_t − u_{t−1}‖²: 0.016 / 0.028 / 0.051 (best)**
  - Other findings: "For λ = 0.2, not all fast trajectory experiments are successful". They also note the trade-off: r_smooth "can also restrict the quadrotor's ability to exploit agile maneuvers".
  - [arXiv 2412.11764 Sec. III-C, Table I](https://arxiv.org/abs/2412.11764)
- **Song 2021** on progress rewards: lap time is sparse, so "reward shaping techniques based on the projected progress along the center-line … have shown to work well". The safety reward is "designed to reduce the risk of crashing in training settings that feature large track changes", and "a ∈ R is a hyperparameter that trades off between progress maximization and risk minimization". — [arXiv 2103.08624](https://arxiv.org/abs/2103.08624)
- **Penicka 2022**: the reached-distance term s(p(t)) is added "to counteract the fact that the progress along the line segments can have many singularities … which can block learning a policy that flies through all waypoints when negative progress occurs in the singularities." — [arXiv 2203.15052](https://arxiv.org/abs/2203.15052)
- **Song 2023**: RL "can optimize a highly nonlinear and nonconvex gate-progress reward directly, removing the need for a reference time trajectory", which "leads to more robust control performance in the presence of unmodeled dynamics". — [arXiv 2310.10943](https://arxiv.org/abs/2310.10943)
- **Ferede 2024**: direct motor-command RL worked in real flight with a learned residual model, but the abstract reports "the performance drop observed from simulation to reality". — [arXiv 2311.16948](https://arxiv.org/abs/2311.16948)
- **Kaufmann 2022**: with this reward, single-rotor-thrust policies crashed on all race trajectories, while CTBR policies succeeded (Table: "RaceA crash 5.6±1.7 …"). — [arXiv 2202.10796](https://arxiv.org/abs/2202.10796)

### Inferences
- **The Δa (action-rate) term has the strongest support.** It is the best real-world smoothness term in SimpleFlight's ablation, and Neuroflight, RAPTOR, Kaufmann 2023, Xing, Gronauer and Zhang all use it. Our efficiency group already contains it (weight 1, squared).
- **An angular-velocity penalty mattered for motor-level stabilisation** in Molchanov (training failed without one). L2F sets the angular-velocity weight to 0 and relies on orientation and action terms. We use angular acceleration (2e-4) and Δa instead. This works for us, but it means no group directly prices the body rate below the 34.9 rad/s termination.
- **Absolute (unsquared) norms dominate the most recent motor-level designs** (RAPTOR, Zhang, Gronauer, DATT, Molchanov). They keep a constant pull near the target, which helps cm-level precision, and grow only linearly far away. Squared costs (L2F, ours) have a vanishing gradient near the target and a large cost far from it. (Functional reasoning, not a reported ablation.)
- **Progress rewards are the default for "reach the target quickly"** (racing, waypoints). Their return telescopes to λ·(d_0 − d_T), so the value range is bounded by the initial distance. Our "1 per step until reached" time cost is the sparse alternative that these papers moved away from.

### Gaps
- Kaufmann 2023's λ1-λ5 (Extended Data Table 1a) were not in the full-text XML I parsed.
- Song 2023's Eq. 5 form, its collision and completion reward values, and Penicka's r_T value were not extracted.
- Xing's α coefficients, Gronauer's γ and code penalty constants, and Molchanov's algorithm and γ were not extracted.
- Venues for Koch (ACM TCPS), Neuroflight, Gronauer (likely IROS 2022), OmniDrones and gym-pybullet-drones were not verified from the documents.
- The Flightmare and safe-control-gym papers were not read; only their code was.

---

## Q3. Bounded or normalised rewards: who uses them, and what is reported about stability and sim-to-real

### Takeaway
Bounded per-term rewards are common in the GPU-simulator lineage (OmniDrones, SimpleFlight, Aerial Gym, gym-pybullet-drones) and in some benchmark and attitude-control work (Koch, safe-control-gym, Mysore). The stated reasons are **commensurate terms** (so a weight means a relative importance), **training stability**, and an implicit survival incentive. **No quadrotor paper I found compares bounded and unbounded rewards head to head**, so the claimed stability benefits are author statements, not controlled evidence.

### Cited Findings
- **SimpleFlight** normalises the task and smoothness rewards to [0, 1] "allowing λ to represent the relative weight of the smoothness reward compared to the task-specific reward". Its code builds every term as w·exp(−x) or w·0.5/(1 + x²). — [arXiv 2412.11764](https://arxiv.org/abs/2412.11764); [track.py](https://github.com/thu-uav/SimpleFlight/blob/main/omni_drones/envs/single/track.py#L473-L516)
- **OmniDrones Hover** uses 1/(1 + (k·d)²) for pose and spin, exp(−·) for effort and smoothness, and **multiplicative gating**: pose·(up + spin). An alternative `exp(-distance * scale)` is left commented out. — [hover.py](https://github.com/btx0424/OmniDrones/blob/main/omni_drones/envs/single/hover.py#L308-L340)
- **Aerial Gym** builds terms from m·exp(−x²/w) rewards and m·(exp(−x²/w) − 1) penalties. Two position kernels of different widths (w = 10 and w = 2) are summed, a progress term 10·Δd (15·Δd when moving away) is added, and the bounded part is divided by 100. — [Aerial Gym compute_reward](https://github.com/ntnu-arl/aerial_gym_simulator/blob/main/aerial_gym/task/position_setpoint_task_sim2real_end_to_end/position_setpoint_task_sim2real_end_to_end.py#L267-L309)
- **gym-pybullet-drones** HoverAviary: r = max(0, 2 − ‖p* − p‖⁴), which is non-negative and bounded. — [HoverAviary.py](https://github.com/utiasDSL/gym-pybullet-drones/blob/main/gym_pybullet_drones/envs/HoverAviary.py#L68-L79)
- **safe-control-gym** offers `rew_exponential`: r = exp(−quadratic cost), "Convert rew to be positive and bounded [0,1]". — [quadrotor.py](https://github.com/utiasDSL/safe-control-gym/blob/main/safe_control_gym/envs/gym_pybullet_drones/quadrotor.py#L841-L845)
- **Koch GymFC v1**: the reward is clipped to [−1, 0]; "Rewards are normalized to provide standardization and stabilization during training". — [arXiv 1804.04154](https://arxiv.org/abs/1804.04154)
- **Mysore et al., "How to Train your Quadrotor" (arXiv 2012.06656; venue not verified) [R]** propose **multiplicative composition** of normalised components:
  - Eq. 8: r_t = g(r) = (Π_k min(1, r_k + ε))^(1/K).
  - Rationale: "With multiplicative composition and normalized reward components, the importance of each component reward signal is preserved. If any one reward component falls too low, the overall reward signal is lowered … the agent is not allowed to offset its poor performance on one objective by being better on another. This property is essential if the objectives defined are competitive."
  - Reported results: "multiplicative composition allows more simplified reward composition with little to no tuning", with "significantly reduced variance while achieving comparable or better rewards" on OpenAI Gym benchmarks.
  - [arXiv 2012.06656 Sec. 4](https://arxiv.org/abs/2012.06656)
- Bounded **sub-terms inside unbounded designs**:
  - Kaufmann 2023 perception term λ2·exp(λ3·δ_cam⁴). — [Nature 2023](https://www.nature.com/articles/s41586-023-06419-4)
  - Song 2021 safety term −f²(1 − exp(−0.5·d_n²/v)) and capped crash term −min[(d_g/w_g)², 20]. — [arXiv 2103.08624](https://arxiv.org/abs/2103.08624)
  - DATT hover clipping min(‖p‖, 1). — [hover.py](https://github.com/KevinHuang8/DATT/blob/main/learning/tasks/hover.py#L11-L20)
  - Our own 0.2 m position cap (`L2F_POS_CAP`) is of this type.
- Cross-domain: legged_gym uses exp(−error²/σ) tracking rewards ("tracking reward = exp(-error^2/sigma)", σ = 0.25) and clips negative totals at zero (see Q4). — [legged_robot_config.py L148-L149](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot_config.py#L148-L149)
- **Unbounded designs** that flew on real hardware: Hwangbo, Molchanov, Gronauer, L2F, RAPTOR, DATT, Zhang, Kaufmann 2022, Neuroflight (see the Q2 tables). Boundedness is therefore not required for sim-to-real.

### Inferences
- With bounded per-step terms r_k ∈ [0, 1] and γ = 0.99, every group's value lies in [0, 100] by construction. That is the commensurability our 4 groups lack: TD-target std is 3.2, 7.8, 17 and 152 now. (My arithmetic.)
- Bounded kernels trade gradient shape for scale:
  - exp(−d²/w) saturates far from the target (vanishing pull).
  - 1/(1 + d²) has a heavier tail.
  - Aerial Gym's two widths plus a progress term look designed to keep both near-field and far-field gradients.
  - Our 0.2 m cap on a squared cost is a crude version of the same saturation.
  - (Design reasoning, not reported by the authors.)
- Multiplicative gating (OmniDrones and SimpleFlight's pos·(up + spin), Aerial Gym's pos·(...), Mysore's product) couples objectives inside one scalar. For our group split, a product across groups would break the "groups sum to the scalar reward" property. Keep any gating inside a group, or treat products as a different scalarisation for the baseline.

### Gaps
- I found no controlled quadrotor study of bounded vs unbounded reward forms with sim-to-real or critic-stability results.
- I did not extract Mysore's quadrotor-specific component definitions (only the composition operator and the stated rationale).

---

## Q4. Crash handling: penalty size, survival bonus, early termination, "learning to crash" and over-caution

### Takeaway
Explicit crash penalties in the literature range from **−0.02 (Flightmare) and −5 (Kaufmann 2023) to −100 (Gronauer, RAPTOR) and −500 (Kaufmann 2022)**. Ours is −400. Most motor-level controllers that train reliably combine a **survival bonus** with zero or a moderate terminal penalty (L2F +2/0; RAPTOR +1.5/−100; Zhang δt; Flightmare +0.1/−0.02), or use **non-negative bounded rewards** so that termination is always a loss. The "learning to terminate" failure is named explicitly by L2F and by legged_gym's `only_positive_rewards`. I found **no report of over-cautious behaviour** caused by a large crash penalty in quadrotor RL.

### Cited Findings
| Work | Crash / termination condition | Explicit terminal reward | Survival bonus | γ | Source |
|---|---|---|---|---|---|
| L2F 2024 | \|p_i\| > 0.6 m (per axis) | 0 (replaces the step reward) | +2 per step (after 0.5 scale) | (ours 0.99) | [squared.h](https://github.com/arplaboratory/learning-to-fly/blob/master/include/learning_to_fly/simulator/parameters/reward_functions/squared.h#L62-L67); [termination](https://github.com/arplaboratory/learning-to-fly/blob/master/include/learning_to_fly/simulator/parameters/termination/default.h) |
| RAPTOR 2025 | \|p\| > 20 l_arm, \|v\| > 2 m/s, \|ω\| > 35 rad/s | −100 | +1.5 per step | 0.99 | [arXiv HTML](https://arxiv.org/html/2509.11481v2) |
| Gronauer 2022 | ‖e‖ > 0.25 m | −100 | none | n/e | [arXiv 2201.01369](https://arxiv.org/abs/2201.01369) |
| Kaufmann 2022 | p_z ≤ 0 | −500 | none | 0.98 | [arXiv 2202.10796](https://arxiv.org/abs/2202.10796) |
| Kaufmann 2023 | p_z < 0, gate collision, leaving a bounding box | −5.0 | none (positive progress reward) | n/e | [Nature](https://www.nature.com/articles/s41586-023-06419-4) |
| Song 2021 | gate collision | −min[(d_g/w_g)², 20] | none (positive progress) | n/e | [arXiv 2103.08624](https://arxiv.org/abs/2103.08624) |
| Song 2023 | collision | negative (value n/e) | positive reward on completing the maximum episode length | n/e | [arXiv 2310.10943](https://arxiv.org/abs/2310.10943) |
| Ferede 2024 | ground contact, or gate plane outside the gate | −10 | none (positive progress) | 0.999 | [arXiv 2311.16948](https://arxiv.org/abs/2311.16948) |
| Xing 2025 | collision with gates or ground | α7 (n/e) | success bonus (stabilisation) | n/e | [arXiv 2412.12442](https://arxiv.org/abs/2412.12442) |
| Zhang 2024 | n/e | none listed | "Survival Reward: δt" | n/e | [arXiv 2409.12949](https://arxiv.org/abs/2409.12949) |
| Flightmare (code) | z ≤ 0.02 | −0.02 | +0.1 per step | – | [quadrotor_env.cpp](https://github.com/uzh-rpg/flightmare/blob/master/flightlib/src/envs/quadrotor_env/quadrotor_env.cpp#L133-L147) |
| OmniDrones Hover (code) | z < 0.2 or distance > 4 | none (rewards ≥ 0) | implicit | – | [hover.py](https://github.com/btx0424/OmniDrones/blob/main/omni_drones/envs/single/hover.py#L336-L340) |
| gym-pybullet-drones (code) | box and tilt limits → **truncated** | none (rewards ≥ 0) | implicit | – | [HoverAviary.py](https://github.com/utiasDSL/gym-pybullet-drones/blob/main/gym_pybullet_drones/envs/HoverAviary.py#L100-L112) |
| Molchanov (code) | z ≤ arm length (floor) | crash·1[crashed] per step, no termination | none | – | [quadrotor.py L1386-L1398](https://github.com/amolchanov86/quad_sim2multireal/blob/master/quad_sim/quadrotor.py#L1386-L1398) |
| Ours | > 0.6 m on any axis, tilted floor contact > 60°, body rate > 34.9 rad/s, failed landing | −400 (replaces other terms) | 0 | 0.99 | task brief |

(n/e = not extracted.)

- L2F introduces the survival constant explicitly "to mitigate the 'learning to terminate' problem". — [arXiv 2311.13081](https://arxiv.org/abs/2311.13081)
- legged_gym (Rudin et al.) config: `only_positive_rewards = True  # if true negative total rewards are clipped at zero (avoids early termination problems)`. — [legged_robot_config.py L148](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot_config.py#L148)
- RAPTOR: "We tuned the reward function for reliable training, particularly the survival bonus and the termination penalty." — [arXiv HTML](https://arxiv.org/html/2509.11481v2)
- Zhang 2024: "The survival reward encourages the quadcopter to learn to fly longer until the end of the training episode." — [arXiv 2409.12949](https://arxiv.org/abs/2409.12949)
- Risk attitude with a small crash penalty (Kaufmann 2023, crash penalty 5.0): "The autonomous drone is unaware of its opponent and pushes for fastest expected completion time no matter what, potentially risking too much when in the lead and too little when trailing behind"; "Swift was not trained to recover after a crash." — [Nature 2023](https://www.nature.com/articles/s41586-023-06419-4)
- The closest statement to over-caution concerns MPC tuning, not RL rewards: "Poorly tuned parameters lead to an overly aggressive policy that crashes into gates or an overly conservative one". — [Song et al. 2023](https://arxiv.org/abs/2310.10943)
- Termination boxes shape real-world failure: Gronauer's real flights failed when the drone exceeded the 0.25 m training termination box. — [arXiv 2201.01369](https://arxiv.org/abs/2201.01369)

### Inferences
- **A survival bonus and a crash penalty are interchangeable up to a constant** (my derivation). In a discounted task with bootstrapped truncation, suppose episode returns differ only by a bonus b on every non-terminal transition versus a terminal penalty P = γ·b/(1−γ) on the terminal transition. Then every trajectory's returns differ by exactly b/(1−γ), so the optimal policies are identical. The equivalent per-step "crash price" is b_eq = b + P(1−γ)/γ:
  - L2F: about 2.
  - RAPTOR: 1.5 + 1.01 ≈ 2.5.
  - Ours: about 4.04.
  - Kaufmann 2022: 500·0.02/0.98 ≈ 10.2.
  - Ferede: about 0.01 (plus forfeited progress).
- **Consequence for our MOO problem:** moving our −400 into a +4/step survival term would **not** shrink the safety group's value spread. Q-values only shift by about 404, and the gap between doomed and safe states stays about 400. The spread (TD-target std 152) is set by the **crash price**, not by where the price is booked. Only re-pricing (for example L2F's implicit 200) or re-representing the crash (Q6) changes it. (My inference.)
- **"Learning to crash" exposure in our build.** Survival is 0 and every step costs something, so crashing beats continuing whenever the expected discounted cost of recovering exceeds 400. In aggressive states the per-step cost can be large:
  - These are costs before the 0.5 scale:
  - orientation up to 5 (5·(1 − q_w²)),
  - Δa up to 16 (‖Δa‖² with a ∈ [−1, 1]⁴),
  - landing term 5·v_down²,
  - angular-acceleration term 2e-4·α², which reaches about 18 at α = 300 rad/s².
  - After the 0.5 scale, a reward of −10 to −20 per step is plausible, and 20-40 such steps cost as much as a crash.

  This is a plausible mechanism for the seed that "learned to crash (100 %)" in the towers experiment. L2F's own survival term protects only while 0.5·cost < 2 (Q1). legged_gym's positive clipping and the bounded-positive designs (OmniDrones, SimpleFlight) remove the incentive by construction. (My inference.)
- **Crash prices in successful motor-level controllers cluster at b_eq ≈ 2-2.5** (L2F, RAPTOR). Racing controllers use tiny explicit penalties because forfeited progress already makes a crash expensive.

### Gaps
- No quadrotor paper I found reports over-cautious behaviour (for example refusing to move toward a waypoint) attributed to the crash-penalty size.
- No paper reports a controlled sweep of the crash penalty or survival bonus. RAPTOR says these were tuned but gives no sweep.

---

## Q5. Safety as a constraint, and explicitly multi-objective quadrotor RL

### Takeaway
Treating crashes as a **constraint** (CMDP cost, safety filter, barrier function) is an active area, but in quadrotor work it is mostly tied to navigation or attitude-limit constraints, and I verified it only at abstract or snippet level. **I found no quadrotor paper with per-reward-component critics combined by a multi-objective gradient aggregator.** The closest are Xing et al. (one critic per **task**, shared actor) and Mysore et al. (multiplicative scalarisation of normalised objectives).

### Cited Findings
- **Xing et al. 2025 (RA-L) [R]**: "We train the shared actor policy to maximize the expected return across all tasks, while the individual critic networks are trained to evaluate the value function for each task." They also chose "not sharing the policy feature encoder with the critic networks, as it has been shown to improve the performance of the policy". — [arXiv 2412.12442 Sec. III](https://arxiv.org/abs/2412.12442)
- **Mysore et al. [R]**: geometric-mean (product t-norm) composition of normalised objectives, so one objective cannot compensate for another (quotes in Q3). — [arXiv 2012.06656](https://arxiv.org/abs/2012.06656)
- **Pizarro Bejarano, Brunke, Schoellig, "Safety Filtering While Training", RA-L 2025 [A]**: integrates a safety filter into RL training rather than only at deployment, with real experiments on a Crazyflie 2.0. The abstract states that the modifications improve performance and sample efficiency; I did not see the details of penalties for filter intervention. — [arXiv 2410.11671](https://arxiv.org/abs/2410.11671)
- **safe-control-gym** ships a safe-exploration PPO controller (`controllers/safe_explorer/safe_ppo.py`) next to its quadrotor environment, and keeps the RL reward separate from the constraint machinery (code tree; paper not read). — [safe-control-gym repo](https://github.com/utiasDSL/safe-control-gym)
- Search snippets only [S], not verified:
  - control Lyapunov barrier functions for safe RL on a Crazyflie 2.0 ([arXiv 2305.09793](https://arxiv.org/abs/2305.09793));
  - primal-dual constrained differentiable policy learning for quadrotor navigation with zero-shot real transfer ([arXiv 2610.16119](https://arxiv.org/abs/2610.16119));
  - CALOS, a runtime attitude-constraint safety layer for quadrotor deep RL ([arXiv 2609.17758](https://arxiv.org/abs/2609.17758));
  - PPO-Lagrangian on reduced-order quadrotor models ([arXiv 2604.10429](https://arxiv.org/abs/2604.10429));
  - multi-agent RL splitting quadrotor low-level control into translational and yaw agents with a reward that balances conflicting objectives ([arXiv 2311.06144](https://arxiv.org/abs/2311.06144)).

### Inferences
- **Novelty:** within quadrotor control, decomposing the reward into groups with one critic per group and combining actor gradients with a multi-objective aggregator appears unreported. Xing's multi-critic design is per task (separate MDPs), not per reward group of one MDP. Reviewers will likely cite it as the nearest prior multi-critic quadrotor work.
- **A constraint-style safety signal is the most natural fix for the scale gap.** With a cost of 1 on a crash and 0 otherwise, Q_safe is the discounted crash probability, bounded in [0, 1]. Its scale then no longer depends on an arbitrary 400 price, and the trade-off moves to a Lagrange multiplier or to the aggregator's weighting. This is how CMDP-style safe RL represents failure. I did not find a quadrotor paper doing exactly this with TD3. (My proposal.)

### Gaps
- I did not read in full any constrained-RL quadrotor paper with a crash or termination constraint. The ones above are at abstract or snippet level.
- Bullet-Safety-Gym (Gronauer) was cloned but not examined for drone cost definitions.

---

## Q6. Lessons for our reward design (4 groups: time, precision, efficiency, safety)

### Takeaway
The literature suggests our safety-group scale problem is **mostly a crash-price and representation choice, not a TD3 or critic artefact**. Moving the −400 into a survival bonus will not fix it, because the two are equivalent up to a constant. The options with literature support are:
1. a lower crash price, in line with L2F's and RAPTOR's ≈ 2-2.5 per step;
2. bounded, positive per-group rewards with termination zeroing all streams (OmniDrones, SimpleFlight, legged_gym), which makes every group's value range [0, 1/(1−γ)];
3. a constraint-style crash indicator whose weight is set outside the reward.

Independently: keep Δa, consider unsquared norms for precision, a progress reward for the time group, and either recalculate replay rewards with the curriculum or drop the curriculum as RAPTOR did.

### Cited Findings
(The facts these proposals rest on are cited in Q1-Q5.)
- L2F: survival +2, terminal 0, squared costs, curriculum with replay recalculation (Q1). — [arXiv 2311.13081](https://arxiv.org/abs/2311.13081); [curriculum.h](https://github.com/arplaboratory/learning-to-fly/blob/master/src/steps/curriculum.h)
- RAPTOR: unsquared norms, Δa term, +1.5 / −100, no curriculum (Q1). — [arXiv HTML](https://arxiv.org/html/2509.11481v2)
- Δa is the best real-world smoothness term (Q2). — [arXiv 2412.11764](https://arxiv.org/abs/2412.11764)
- Bounded, normalised terms are used so that weights are relative (Q3). — [arXiv 2412.11764](https://arxiv.org/abs/2412.11764); [arXiv 2012.06656](https://arxiv.org/abs/2012.06656)
- Progress rewards are standard for time-optimal flight (Q2). — [arXiv 2103.08624](https://arxiv.org/abs/2103.08624); [arXiv 2311.16948](https://arxiv.org/abs/2311.16948)

### Inferences
All items below are my proposals, built on the cited facts.

1. **Do not expect survival-vs-penalty to fix the spread.** By the derivation in Q4, −400 terminal with 0 survival equals about +4/step survival with 0 terminal. Only the crash price (b_eq) or its representation changes the safety group's value range. Our b_eq ≈ 4 is about 1.6-2x L2F's and RAPTOR's.
2. **Option A, re-price (cheapest).** Use a −200 terminal penalty (L2F's implicit price), or RAPTOR's split (+1.5/step in the safety group plus −100). This halves the safety spread at most, so it alone cannot close a ~50x gap. Since the scalar objective changes, re-check variant E with it before MOO comparisons. Minimal screen: 2 seeds of variant E plus the split-critic best, comparing crash %, circle error and waypoint %.
3. **Option B, bounded positive groups (L2F/OmniDrones/SimpleFlight style, re-split by group).**
   - Per step: r_time = 1[target reached] (already 0/1), r_prec = exp(−‖p‖²/σ_p) or 1/(1 + (‖p‖/σ_p)²) combined with a velocity kernel, r_eff = exp(−c_eff/σ_e), r_safe = an uprightness × landing-speed kernel in [0, 1].
   - Termination ends all four streams. Every group's value is then in [0, 100] at γ = 0.99, and the crash price is distributed: each group forfeits its own future reward, at most 4 × 100 = 400 in total, which coincides with our current −400.
   - Benefit: the groups become commensurate without normalisation. Crash avoidance appears in every group's value, which aligns the groups on safety instead of isolating it in one huge head, and may stop scale-invariant aggregators from starving safety.
   - Risks: it changes the objective (variant E must be retrained on the same reward to stay the fair baseline); exp kernels lose far-field gradient (add Aerial-Gym-style wide kernels or a progress term); and keep products within a group.
4. **Option C, constraint-style crash indicator.** The safety cost is 1 on the crash step and 0 otherwise, so Q_safe is a discounted crash probability in [0, 1]. Move the "400" into the aggregator weighting or a Lagrange multiplier (target crash rate). Keep the dense orientation and landing terms in a bounded r_safe. This is the most principled framing for an IJCNN multi-objective paper, but it is the least tested in quadrotor RL that I found (Q5).
5. **Curriculum and normalisation interact.**
   - L2F's curriculum multiplies w_pos up to 8x (5 → 40) and w_vel and w_act up to 100x (0.01 → 1) between start and cap.
   - A per-group scale measured once (our `--norm` at 15k steps) is therefore stale for precision and efficiency by up to these factors, and is inflated for safety by early crashes.
   - If the curriculum stays, recalculate replay rewards as L2F does (its ablation lost real-flight successes without it), and use running statistics.
   - Alternatively, follow RAPTOR: add the termination and Δa penalties and remove the curriculum.
   - Check whether our build keeps `RECALCULATE_REWARDS`.
6. **Precision terms.** Consider ‖p‖ (unsquared, as RAPTOR, Hwangbo, Molchanov and Gronauer use) or a Huber/log form (Molchanov's code uses log(d + 0.1) "to create a sharp peak at the goal") instead of the capped ‖p‖². This gives a stronger pull at the 1-5 cm errors that dominate circle tracking.
7. **Time group as progress.** For waypoint jumps, replace "0.5 per step until reached" with λ·(d_{t−1} − d_t), as in Song 2021, Kaufmann 2023, Ferede, Penicka and the Aerial Gym `towards_goal` term. It is dense, its return telescopes (bounded by λ·d_0), and the racing papers show it drives time-optimal behaviour. Watch for Penicka's caveat: negative-progress singularities arise when the reference path has corners.
8. **Keep Δa and consider an explicit body-rate term.** Δa is best supported (SimpleFlight Table I). Molchanov found an angular-velocity penalty necessary for motor-level training; we rely on angular acceleration and Δa, with a hard 34.9 rad/s termination (RAPTOR uses 35 rad/s).

### Gaps
- None of these proposals has been tested in our setup. The literature gives no controlled comparison of the three crash representations (price, bounded-positive, constraint) for quadrotors.
- Whether bounded per-group rewards would keep a shared critic's heads distinct (the collapse problem in the brief) is untested and not addressed by any paper found here. Other researchers' notes on critic architectures should be consulted.
