# Reward and objective design for multi-objective TD3 with a rare, large crash penalty

Scope: how to formulate and decompose the Crazyflie reward (4 groups: time, precision, efficiency,
safety; a crash ends the episode with -400 and no other terms; no survival bonus) so that the
groups are commensurate without losing crash avoidance.

Reading-status tags: **[sec]** = the relevant sections were read in an HTML or ar5iv rendering
(not the whole paper); **[abs]** = abstract or landing page only; **[secondary]** = only search
summaries, third-party pages or a bibliographic listing; **[code]** = source code read. No paper
was read cover to cover. Everything under "Inferences" is my own reasoning, labelled as such.
Facts about our setup come from the local files cited by path.

## Q1. Termination penalties vs survival bonuses: value learning, bias, "learning to terminate early", bootstrapping

### Takeaway
Our TD3 bootstraps at the 500-step limit (partial-episode bootstrapping, PEB). Under PEB, a
survival bonus b on every non-terminal step is the same objective as a crash penalty
P = b/(1-gamma): the two returns differ by the constant b/(1-gamma) on every trajectory. So
L2F's +2 bonus equals a 200-point crash penalty, and our -400 equals a bonus of 4. The choice
between them changes value offsets and learning dynamics, not the optimal policy. It also does not
change the safety group's value spread. The finding that matters more for the decomposition is
this: every per-step cost is negative, and the whole crash reward sits in the safety group. So the
value of every other group goes up when the episode ends (the "termination bias" reported for
strictly negative rewards). Safety is the only group that opposes crashing. Any rebalancing that
weakens it should raise crashes, and this is what every normalisation run showed.

### Cited Findings
- **Time limits.** Pardo et al. separate two cases. If the time limit is part of the task, the
  agent should get the remaining time as an input. If the limit only helps learning, the agent
  should bootstrap from the value of the last state of each partial episode. Ignoring this
  distinction causes state aliasing, invalidates experience replay, and leads to suboptimal
  policies and unstable training. The abstract says nothing about reward sign or incentives to end
  episodes early — [Pardo, Tavakoli, Levdik, Kormushev, ICML 2018, PMLR 80:4042-4051](https://arxiv.org/abs/1712.00378) [abs]
- **Our code already does PEB:** "no bootstrap where terminated; truncated episodes bootstrap".
  A new episode starts after a termination or 500 steps — [l2f/py/td3.py docstring](/Users/khushpatel/code/CrazyFlie-2.1/l2f/py/td3.py) [code]
- **Where the crash reward goes.** On a terminated step the whole reward (crash penalty or landing
  payout) goes to the 'survival' part, which is in the safety group — [l2f/py/moo.py docstring](/Users/khushpatel/code/CrazyFlie-2.1/l2f/py/moo.py) [code]
- **Our crash-pays criterion.** The reward check says: "With no survival bonus every r <= 0, and a
  crash ends the episode at -P ... From any state where the costs still ahead add up to more than
  P, crashing is the better choice". In the time-to-target study (P = 200), a waypoint leg cost
  250-500 and "every method learned to crash" — [l2f/py/reward_check.py docstring](/Users/khushpatel/code/CrazyFlie-2.1/l2f/py/reward_check.py) [code]
- **Our own conflict measurement.** The regrouping comment records that an ungrouped run found
  "survival fights the dense costs, time agrees with position" — [l2f/py/moo.py, REGROUPS comment](/Users/khushpatel/code/CrazyFlie-2.1/l2f/py/moo.py) [code]. With the split critic, precision and safety gradients are opposed in 76 % of updates — [research prompt §2](/Users/khushpatel/code/CrazyFlie-2.1/l2f/docs/shared_critic_research_prompt.md)
- **Survival bias and termination bias (Kostrikov et al.):**
  - "a strictly positive reward function prevents the agent from solving tasks in a minimal number
    of steps", and it "implicitly provides a survival bonus".
  - The strictly negative variant "penalizes every step and leads to collapsing in environments
    with a survival bonus".
  - Many methods "assign zero reward value, often implicitly, to absorbing states". The fix is to
    "explicitly learn rewards for absorbing states" (an indicator dimension marks absorbing states).
  - Computing the terminal return analytically instead of recursively "was much less stable".

  Sources: [Kostrikov, Agrawal, Dwibedi, Levine, Tompson, ar5iv rendering](https://ar5iv.labs.arxiv.org/html/1809.02925) [sec]. Venue ICLR 2019: [ICLR 2019 poster page](https://iclr.cc/virtual/2019/poster/836)
- **legged_gym practice.** The default PPO config has `only_positive_rewards = True  # if true
  negative total rewards are clipped at zero (avoids early termination problems)`. The default
  termination reward scale is `termination = -0.0` — [legged_gym legged_robot_config.py](https://raw.githubusercontent.com/leggedrobotics/legged_gym/master/legged_gym/envs/base/legged_robot_config.py) [code]
- **Constant shifts in episodic tasks.** In a continuing task, adding a constant c to every reward
  adds c/(1-gamma) to every value and leaves the policy unchanged (Sutton & Barto Ex. 3.15). In an
  episodic task, the shift grows with episode length and can change the optimal policy
  (Ex. 3.16) — [Bonde's chapter-3 solutions](https://yashbonde.github.io/blogs/bartosutton/chap3_ex.html); [gpldecha solutions](https://gpldecha.github.io/ml/docs/RL_Solutions_Chap3.pdf) [secondary: third-party solutions; textbook not re-read]
- **Reward centering (Naik et al.):**
  - "Reward centering is not directly applicable to episodic problems ... subtracting a constant
    from all the rewards may change an episodic problem".
  - The discounted value splits into r_bar/(1-gamma) plus a differential value.
  - "Approximation errors in estimating the offset can easily mask the relative differences in
    actions".
  - In continuing problems, a constant reward shift makes standard methods "perform much worse",
    while centred methods are "unaffected".

  Sources: [Naik, Wan, Tomar, Sutton, RLC 2024](https://arxiv.org/abs/2405.09999); [ar5iv](https://ar5iv.labs.arxiv.org/html/2405.09999) [abs + sec]
- **Reward shifting (Sun et al.).** A linear reward shift is equivalent to changing the
  Q-function's initialisation. A positive shift gives conservative exploitation (applied to offline
  RL). A negative shift gives curiosity-driven, optimistic exploration (applied to online
  continuous control with several value functions with different shift constants, and to discrete
  control) — [Sun, Han, Yang, Ma, Guo, Zhou, NeurIPS 2022, "Exploit Reward Shifting in Value-Based Deep-RL"](https://proceedings.neurips.cc/paper_files/paper/2022/hash/f600d1a3f6a63f782680031f3ce241a7-Abstract.html); [arXiv 2209.07288, now titled "Optimistic Curiosity Exploration and Conservative Exploitation with Linear Reward Shaping"](https://arxiv.org/abs/2209.07288); [code](https://www.github.com/holarissun/RewardShifting) [abs]
- **Shaping equals initialisation.** A learner whose Q-values start at the shaping potential makes
  the same updates as a learner given potential-based shaping rewards — [Wiewiora, JAIR 19:205-208, 2003](https://arxiv.org/abs/1106.5267) [abs]
- **CaT assumes non-negative rewards:** "we assume positive rewards r ⩾ 0 for simplicity (without
  loss of generality w.r.t. any other lower bounded definition)". Its naive early-termination
  baseline gets a reward floor by adding a constant — [Chane-Sane et al., IROS 2024, arXiv HTML](https://arxiv.org/html/2403.18765v1) [sec]
- **L2F's original reward.** It used a +2 per-step survival bonus and no crash penalty. Our earlier
  reward check found that it left more states where crashing pays than our -400 penalty does —
  task brief (not re-checked against Eschmann et al. 2024, [arXiv 2311.13081](https://arxiv.org/abs/2311.13081))

### Inferences
#### A. Bonus and penalty are the same objective under PEB (derivation)
- **Setup.** Let tau be the crash step. The crash step has reward r_c, then an absorbing state with
  reward 0 follows; there is no bootstrap, as in td3.py.
  - Bonus reward: -c_t + b on non-terminal steps, r_c on the crash step. This is how
    reward_check's `bonus=` candidate is built: no bonus on the terminal step.
  - Penalty reward: -c_t on non-terminal steps, r_c - P on the crash step.
- **Difference in return.**
  - Trajectory that crashes at tau: G_bonus - G_penalty = b(1 - gamma^tau)/(1 - gamma) + gamma^tau P.
  - Trajectory that never terminates (PEB treats the 500-step cut as non-terminal): the difference
    is b/(1 - gamma).
  - Both equal the same constant b/(1-gamma) for every tau if and only if P = b/(1-gamma).
- **Result.** Q_bonus(s,a) = Q_penalty(s,a) + b/(1-gamma) for all (s,a). So the greedy policy,
  dQ/da and the set of crash-pays states are all the same. With gamma = 0.99: b = 2 matches
  P = 200, and P = 400 matches b = 4. This explains without simulation why the reward check found
  that L2F's +2 leaves more crash-pays states than -400: it is a 200-point penalty.
- **Conditions.**
  - The landing termination must also get +b/(1-gamma) in the bonus version. Otherwise the bonus
    also penalises landing by 400 (for b = 4). This is the survival bias Kostrikov et al. describe.
  - Without PEB, the implicit penalty of a bonus is b(1 - gamma^H)/(1 - gamma) with H steps left.
    It fades to zero near the time limit, which is the state-aliasing problem Pardo et al. warn
    about. Our code avoids this.

#### B. Where bonus and penalty differ: learning dynamics, not the optimum
- **Offset.** The bonus version adds +b/(1-gamma) (400 for b = 4) to every Q-value. Following Sun
  et al. and Wiewiora, this works like a different initialisation:
  - Networks initialised near 0 under a positive shift start pessimistic, which gives conservative
    exploitation.
  - Our all-negative costs start the critic optimistic, which favours exploration.

  Naik et al. warn that errors in a large offset can mask differences between actions.
- **Spread is unchanged.** In both encodings, the gap between a crash-bound and a safe continuation
  is P. So the safety group's TD-target spread (152 vs time 3.2) and its 100-3000x gradient
  dominance in a shared trunk stay the same. **Reward sign is not a lever for commensurability.**
- **Bootstrapping.** In both encodings, crash information enters only through the rare terminal
  transitions with fixed targets (-P or 0). The bonus version also requires bootstrapping a
  +400 offset into every state (effective horizon 1/(1-gamma) = 100 steps), which is slow from a
  near-zero initialisation.

#### C. Per-group termination bias in our 4-group decomposition
- **The bias.** Take any cost group k: time, precision, efficiency, and also the orientation and
  landing-speed terms inside safety. The crash step gives group k nothing and ends all its future
  costs. So Q_k(s, crash-bound a) >= Q_k(s, safe a). Wherever actions can change the crash
  probability, each such group's actor gradient points toward crashing. Only the -400 points away.
  This is Kostrikov et al.'s termination bias applied per objective. The legged_gym comment
  ("avoids early termination problems") and CaT's r >= 0 assumption address the same issue. Our
  own data fits this picture ("survival fights the dense costs"; precision vs safety opposed in
  76 % of updates), although other conflicts also contribute.
- **Link to H1.** In variant E's scalar reward, the termination bias of the costs cancels inside the
  sum. Once the groups are learned and weighted separately, the cancellation depends on one thing:
  the safety head's slope must be accurate and large enough to outweigh three heads that all
  correctly "want" the episode to end. Normalisation (B.1-B.3) and scale-invariant aggregation
  (PGA) shrink exactly this counterweight. This explains, from the reward structure, why "critics
  with distinct, accurate group values (R2 ~ 0.99) still produce policies that crash". Accurate
  values are not enough when three of them correctly say that crashing helps them.

#### D. Reformulations that follow (my proposals)
- **R1 — split the terminal penalty across groups.**
  - **What it is.** Give each group k a crash-step reward -P_k with sum_k P_k = 400. Split the
    landing payout the same way.
  - **What changes.** The summed reward does not change, so the utility is identical to variant
    E, and the "sum" aggregator has the same optimum.
  - **How to choose P_k.** Make P_k proportional to group k's typical discounted future cost C_k.
    Then group k prefers flying iff C_k < 400 C_k / sum_j C_j, which holds iff sum_j C_j < 400.
    So every group prefers flying exactly when the scalar reward does, which is what the reward
    check already verifies.
  - **Effect.** No group gains from crashing. Each head carries crash information in its own units,
    so per-head normalisation or scale-invariant aggregation can no longer switch crash avoidance
    off.
  - **Risks.**
    1. The shared crash component raises the correlation between group values. Measure
       "collapse" on the dense parts only.
    2. In the unnormalised loss sum_k e_k^2, the crash error's share falls by a factor
       sum_k w_k^2, for example 0.30 for w = 0.1/0.2/0.4/0.3. This is a mild H1 risk; the `total`
       loss term keeps it whole.
    3. It changes the conflict statistics reported for the split critic.
  - **Cost.** A few lines where the terminal 'survival' part is assigned to groups (moo.py).
- **R1b — make crashes neutral for the cost groups.**
  - **What it is.** Bootstrap the non-safety groups through the crash, as if it were a time-out.
    This is analogous to DAC's learned absorbing-state rewards.
  - **What changes.** The objective changes: a crash then costs 400 plus the continuation costs,
    so the policy becomes more crash-averse than variant E.
  - **Risk.** It bootstraps from crashed states that lie outside the training data. DAC found
    analytic terminal returns less stable than recursive bootstrapping.
- **R6 — survival bonus b = 4 instead of P = 400 (with +400 added to the landing payout).**
  - **What changes.** The utility is the same, and every value gets a +400 offset.
  - **Value.** No gain in commensurability. Useful only as a prerequisite for CaT-style
    terminations, or for a CMDP whose reward must itself be termination-averse (Q2).

### Gaps
- Pardo et al. was read at abstract level only. I found no statement there on reward sign or
  early-termination incentives. The DAC bias analysis is about imitation-learning rewards. Applying
  it to hand-designed costs and to individual objectives is my inference.
- I found no paper that analyses termination bias per objective in a decomposed or multi-head
  critic, or that redistributes a terminal penalty across reward-decomposition groups. I did not
  search this exhaustively, so it is a possible novelty point but untested.
- I found no source on keeping rare terminal (crash) transitions represented in a 700k replay
  buffer as the crash rate falls (anchor samples for the -400 become scarce late in training).
- The "+2 bonus, no crash penalty" description of L2F comes from the task brief. I did not verify
  it in Eschmann et al. 2024.

## Q2. Constraints instead of rewards: CMDPs, Lagrangian methods, constraints-as-terminations, cost critics, constrained MORL

### Takeaway
Safe RL's standard design is to treat the crash as a CMDP constraint with its own critic, which
learns a bounded crash probability, plus an adaptive multiplier. Doing this takes the 400-point
term out of the reward critic, so the remaining groups differ by about 5x instead of 50x.

- PID-Lagrangian adds damping and a gradient-norm scaling that makes the reward/cost balance
  independent of their scales, and it explicitly uses a separate cost critic.
- CoMOGA (ICLR 2025) is a constrained multi-objective aggregator. It treats the objectives
  scale-invariantly but keeps the safety constraints as hard constraints in absolute units.

The catch for us: with all-negative costs, the reward objective itself favours crashing (Q1), so a
pure constraint would be active all the time. Keep a base penalty, or make the reward groups
termination-averse, and use the constraint as an adaptive top-up.

### Cited Findings
- **CMDP reference.** Altman's book is the standard reference for CMDPs (finite state space,
  unbounded costs) — [Altman 1999, Chapman & Hall/CRC](https://www.routledge.com/Constrained-Markov-Decision-Processes/Altman/p/book/9781315140223) [secondary: bibliographic listing only]
- **CPO** is trust-region policy search with constraints, with "guarantees for near-constraint
  satisfaction at each iteration". It was tested on simulated robot locomotion with safety
  constraints — [Achiam, Held, Tamar, Abbeel, ICML 2017](https://arxiv.org/abs/1705.10528) [abs]
- **RCPO** is "a novel multi-timescale approach ... which uses an alternative penalty signal to
  guide the policy towards a constraint satisfying one", with a convergence proof. It is motivated
  by agents that "exploit loopholes and misspecifications in the reward signal" — [Tessler, Mankowitz, Mannor, ICLR 2019](https://arxiv.org/abs/1805.11074) [abs]
- **PID Lagrangian (Stooke, Achiam, Abbeel):**
  - Standard Lagrangian dynamics "exhibit oscillations and overshoot which ... leads to
    constraint-violating behavior during agent training". The classic multiplier update acts as
    integral control; PID adds proportional and derivative terms. It set a new state of the art on
    Safety Gym, and it adds "invariance to the relative numerical scales of reward and cost" — [ICML 2020](https://arxiv.org/abs/2007.03964) [abs]
  - Update: lambda <- (K_P Delta + K_I I + K_D d)_+, with Delta = J_C - d_limit,
    I <- (I + Delta)_+ and d = (J_C - J_C,prev)_+.
  - Scale-invariant objective: grad L = (1 - u) grad J - u beta grad J_C, with u = lambda/(1+lambda)
    and beta = ||grad J|| / ||grad J_C||.
  - "cost overshoot and oscillations are in fact inherent to the learning dynamics" (integral
    control gives a 90° phase lag).
  - It uses "separate value- and cost-value approximators" instead of one critic for r + lambda c,
    because lambda may change quickly.

  Sources for the last four points: [ar5iv rendering](https://ar5iv.labs.arxiv.org/html/2007.03964) [sec]. The extraction also said the scaled form keeps learning dynamics nearly unchanged over about two orders of magnitude of reward scale. Check this against the paper's figure before quoting it.
- **Safety Gym** argues for constrained RL as the main formalism for safe exploration, with safety
  specifications kept separate from task performance. OpenAI's post reports that the Lagrangian
  approach did surprisingly better than CPO — [Ray, Achiam, Amodei 2019](https://cdn.openai.com/safexp-short.pdf); [OpenAI Safety Gym post](https://openai.com/index/safety-gym) [secondary]
- **Rewards plus constraints for legged robots.** Kim et al. report "significantly less reward
  engineering" (only a single reward coefficient tuned) and describe constraints as "interpretable
  and generalizable" — [Kim et al., IEEE T-RO 2024](https://arxiv.org/abs/2308.12517) [abs]
- **Value-constrained control.** Bohez et al.: "It can, however, be hard to find the correct
  trade-off between cost and return that results in the desired behavior". Their constraint-based
  RL uses Lagrangian relaxation to learn the multipliers, satisfies constraints in expectation or
  per step, and trains one policy that trades off dynamically. Tasks: energy-optimised quadruped
  locomotion and a real robot arm — [Bohez et al. 2019, arXiv](https://arxiv.org/abs/1902.04623) [abs]
- **Real-world legged learning.** Ha et al. use a multi-task procedure and a "safety-constrained RL
  framework" to deal with falls — [Ha, Xu, Tan, Levine, Tan 2020](https://arxiv.org/abs/2002.08550) [abs; mechanism not verified]
- **CaT (Constraints as Terminations) mechanism:**
  - Termination probability: delta = max_i p_i^max · clip(c_i^+ / c_i^max, 0, 1). c_i^max is an
    EMA of the batch-maximum violation; using the raw maximum was "slightly less stable".
  - In PPO it is applied by setting rewards to r(1 - delta) and "dones" to delta ("the sum of all
    future rewards at t and after time are re-scaled by (1-delta_t)").
  - Hard constraints (p_max = 1): base/knee contact and foot contact force. Soft constraints
    (p_max ramped 0.05 -> 0.25): torque, joint velocity and acceleration, action rate, style.
  - Results:
    - Flat-terrain simulation (4 seeds): N-P3O reached 593.2 reward with 8 % torque-violation
      time; CaT reached 682.9 with 0.5 %.
    - Hard constraints only: learning "fails completely".
    - The ET-MDP baseline "entirely fails".
    - Real Solo-12 parkour success: 90.6 % with tracking as constraints, 79.4 % with tracking as
      rewards.

  Sources: [Chane-Sane, Leziart, Flayols, Stasse, Souères, Mansard, arXiv HTML](https://arxiv.org/html/2403.18765v1) [sec]. Venue IROS 2024, pp. 13303-13310: [researchr](https://researchr.org/publication/Chane-SaneLFSSM24/bibtex)
- **ET-MDP** recasts safe RL as an Early Terminated MDP, "an unconstrained MDP with the same optimal
  value function as its corresponding CMDP". It claims "better asymptotic performance and improved
  learning efficiency" than methods that solve the CMDP directly — [Sun et al. 2021, arXiv](https://arxiv.org/abs/2107.04200) [abs]. This is contradicted, in high-dimensional legged locomotion, by CaT's report that ET-MDP "entirely fails" — [CaT](https://arxiv.org/html/2403.18765v1)
- **Conservative Safety Critics** learn a conservative safety estimate that can "provably upper
  bound the likelihood of catastrophic failures at every training iteration". Task performance is
  competitive, with significantly lower failure rates during training (navigation, manipulation,
  locomotion) — [Bharadhwaj, Kumar, Rhinehart, Levine, Shkurti, Garg, ICLR 2021](https://arxiv.org/abs/2010.14497) [abs]
- **SQRL safety critic:**
  - Q_safe is the probability of future failure.
  - Bellman target I(s) + (1 - I(s)) gamma_safe Qbar_safe(s', a'); squared TD loss with a delayed
    target network.
  - gamma_safe = 0.7 by default.
  - Actions with Q_safe > epsilon_safe are masked.
  - It is a separate network from the task critics.

  Source: [Srinivasan, Eysenbach, Ha, Tan, Finn 2020, ar5iv](https://ar5iv.labs.arxiv.org/html/2010.14603) [sec]
- **WCSAC** adds a separate safety critic to SAC that models the distribution of accumulated cost,
  with a CVaR constraint. It reports better risk control than expectation-based methods. The
  journal extension compares Gaussian and IQN-quantile safety critics and finds quantile better —
  [Yang, Simão, Tindemans, Spaan, AAAI 2021](https://ojs.aaai.org/index.php/AAAI/article/view/17272); [Machine Learning 2023 extension](https://link.springer.com/article/10.1007/s10994-022-06187-8) [secondary]
- **CoMOGA:**
  - It casts multi-objective maximisation as a constrained optimisation problem. Each objective
    must improve by a preference-scaled amount, omega_i e_i <= J_Ri(theta + dtheta) - J_Ri(theta),
    with e_i = epsilon ||g_i||_{H^-1}. So the aggregated step never conflicts with any objective.
  - Safety costs enter the same QP as linear constraints, b_k^T dtheta + J_Ck <= d_k.
  - When the policy violates a constraint, "we take a recovery step by solving the following QP
    problem, which only consists of the safety constraints".
  - It uses separate reward critics and cost critics, with SAC-style reparameterised gradients.
  - Tasks: Safety Gymnasium, legged locomotion (2 objectives, 3 constraints), MO-Gymnasium MuJoCo.
    It had the highest hypervolume (HV) and satisfied all constraints in the legged tasks. A
    CAGrad + Lagrangian ablation violated a constraint, which the authors attribute to "the
    concurrent update of policy and multipliers".

  Sources: [Kim, Hong, Park, Oh, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/59ddfff7979b43f54690fa986c0e5138-Paper-Conference.pdf); [arXiv v2 HTML](https://arxiv.org/html/2403.00282v2) [sec, first ~80 % of the text]. Earlier title "Scale-Invariant Gradient Aggregation for Constrained Multi-Objective RL": [paperswithcode listing](https://astro.paperswithcode.com/paper/scale-invariant-gradient-aggregation-for) [secondary]
- **Lexicographic MORL** maximises the first reward, then the second subject to the first, and so
  on. It provides action-value and policy-gradient algorithm families with convergence proofs, and
  is applied to impose safety constraints and compared against constrained RL methods — [Skalse, Hammond, Griffin, Abate, IJCAI 2022](https://arxiv.org/abs/2212.13769) [abs]

### Inferences
#### Variant E is a Lagrangian with a frozen multiplier
With the cost c_t = 1{crash at t}, the Lagrangian reward r - lambda c is exactly our reward with
crash penalty lambda. So variant E is a CMDP Lagrangian with lambda fixed at 400. RCPO or PID
would let lambda adapt to a target crash rate. At convergence, lambda* is the shadow price of the
constraint: the crash penalty that makes it just tight. This framing makes the constraint version
easy to defend to reviewers: it is variant E with the 400 learned instead of hand-set.

#### What changes in our decomposition if the crash becomes a constraint
- **Objective.** Maximise the dense groups (time, precision, efficiency, and "limits" =
  orientation + landing speed) subject to J_C = E[sum gamma^t 1{crash}] <= d.
- **Critics.**
  - Reward critic (a shared trunk is fine): its groups' TD-target spreads are 3.2 / 7.8 / 17 plus
    limits, within about 5x instead of 50x.
  - Cost critic Q_C: a separate network, as in PID-Lagrangian, SQRL, CoMOGA and WCSAC. Its targets
    c + (1 - c) gamma_C Q'_C lie in [0, 1] whatever the penalty.
- **Actor.** Aggregate the dense groups' gradients (any aggregator), then add the crash gradient
  through lambda, either with Stooke's beta-scaled form or as a hard linear constraint
  (CoMOGA-style). The crash term never goes through a scale-invariant aggregation, so the observed
  PGA failure mode (unit-norm averaging that gives safety almost no weight) cannot happen.
- **Closed loop.** The multiplier is driven by the measured crash rate, an outcome, not by value
  accuracy. This directly addresses the observation that value accuracy on replay data does not
  predict control quality. If the crash slope is wrong, lambda rises until crashes fall.

#### Effect on the gradient-scale problem
- In the critic, no 400-point targets reach the shared trunk, and the cost critic's loss is O(1).
- In the actor, the crash gradient's size is lambda times the reward-gradient norm (beta
  normalisation), and it adapts. There is no fixed 100-3000x ratio.

#### The catch: termination bias and threshold choice
- **Termination bias.** With all-negative costs, J_R improves when the drone crashes (Q1-C). So the
  constraint is always active, and the solution sits at crash rate = d instead of near zero.
  Mitigations:
  - (a) Keep a base penalty P0 (for example 400) in the reward, so that variant E is the starting
    point, and let the constraint add lambda on top only when the crash rate exceeds d.
  - (b) Make the dense groups termination-neutral (R1/R1b) or non-negative (R6/CaT). Safety Gym
    and CaT do not face this problem: their rewards are positive and costs do not end the episode
    (Safety Gym), or rewards are assumed >= 0 (CaT).
- **Threshold.** Variant E reached 0.6 % crashed test episodes. A training-time constraint is
  measured on episodes with exploration noise, so d must be set on that basis or estimated from
  periodic evaluation episodes.
- **Multiplier dynamics.** Expect oscillation (Stooke); prefer PID, or CoMOGA's QP with a recovery
  step.
- **Rare-event cost critic.** Crash transitions are rare and become rarer. SQRL's gamma_safe = 0.7
  is about 3 steps (30 ms) at our 100 Hz, far too short for a quadrotor. Choose gamma_C by time
  horizon: 0.95 is about 0.2 s, 0.99 about 1 s.
- **Reviewer concern about network count.** A reward critic plus a cost critic is the standard
  safe-RL structure. The worry about "4 separate networks" then becomes "1 shared reward critic +
  1 cost critic".

#### Options from this family (what changes / main risk)
- **R2 — Lagrangian crash constraint on top of a base penalty.**
  - **Setup.** Separate crash-probability critic (sigmoid output). PID-lambda on the running crash
    rate. Stooke's beta-scaled actor objective. Dense groups only in the shared critic.
  - **Main risk.** If P0 is dropped, the termination bias keeps the constraint active all the time.
    Keep P0, or apply R1 first.
- **R3 — CoMOGA-style QP.**
  - **Setup.** Dense groups are improvement constraints with thresholds proportional to their own
    gradient norms. This is scale-invariant (my derivation: rescaling J_Ri by c scales g_i and e_i
    by c, so the feasible set does not change). The crash is an absolute-unit constraint with a
    recovery step.
  - **Main risk.** It was designed around SAC with preference sampling, and recovery steps need a
    good estimate of J_C. The QP itself is small (4-5 constraints).
- **R4 — Lexicographic, safety first with slack.**
  - **Main risk.** The slack needs tuning, and there is little TD3 evidence.
- **R5 — CaT for the soft safety terms only** (tilt, angular rate, landing speed, motor
  saturation). The crash is already a hard termination.
  - **What changes.** Those reward terms disappear, along with the safety group. Termination then
    enters every head's target as a factor (1 - delta), in each head's own units, which is
    commensurate by construction. TD3 needs one stored delta per transition; the target becomes
    (1 - delta)(r + gamma Q').
  - **Requirements.** Rewards must be >= 0 (for example the R6 bonus), and the landing payout must
    equal the continuation value.
  - **Main risks.** The objective is no longer variant E's, which makes the baseline comparison
    harder. CaT's evidence is on-policy PPO for legged robots. delta values computed when a
    transition was collected go stale in a 700k replay buffer. Hard-only constraints failed in CaT.

### Gaps
- I did not access Altman's book. The CPO, RCPO, CSC, Kim et al., Bohez et al., Ha et al. and
  ET-MDP results were read at abstract level, so their mechanisms and numbers are not verified
  here.
- The evidence on ET-MDP conflicts: its abstract claims, CaT's report of failure. I could not
  resolve this without the ET-MDP paper.
- I found no CMDP or Lagrangian study for end-to-end quadrotor control with crash as the
  constraint, and no study of a constrained formulation on top of an all-negative cost reward
  (the termination-bias catch above is my inference).
- I did not examine OmniSafe or safe-control-gym in this session.
- The extracted claim of "~two orders of magnitude" scale robustness for the PID beta scaling is
  unverified.

## Q3. Reward scaling and normalisation in off-policy actor-critic, and per-objective normalisation in MORL

### Takeaway
Reward scale clearly matters in off-policy actor-critic. Henderson et al. found large,
inconsistent effects for DDPG, and SAC's reward scale is its inverse temperature. The published
fixes target either a single return (PopArt; DreamerV3's symlog/twohot critic and percentile return
normalisation) or separate tasks (multi-task PopArt, which deliberately gives each task equal
influence). MORL methods that claim scale invariance (MO-MPO, CoMOGA, PID-Lagrangian's beta) put
the invariance in the policy-improvement step and train per-objective critics in raw units. I
found no paper that normalises additive reward components inside one shared critic's loss and
reports a benefit.

Std-based scales are dominated by the rare crash (our s_safe = 6389), so they suppress exactly the
crash signal. Percentile scales are robust to it. But no normaliser removes the basic tension: a
scale that shrinks the crash errors weakens crash learning, and one that doesn't leaves safety
dominating the trunk.

### Cited Findings
- **Henderson et al. on reward scale:**
  - "reward rescaling can have a large effect, but results were inconsistent across environments
    and scaling values" (DDPG on HalfCheetah and Hopper; very small scales, below about 0.01 in the
    main-text example, failed).
  - "Layer normalization changes how the rescaling factor affects results".
  - They call for principled adaptive rescaling and point to van Hasselt et al. (2016) as a first
    step.

  Source: [Henderson, Islam, Bachman, Pineau, Precup, Meger, AAAI 2018, ar5iv](https://ar5iv.labs.arxiv.org/html/1709.06560) [sec]
- **SAC on reward scale:** SAC "is particularly sensitive to the scaling of the reward signal,
  because it serves the role of the temperature". If the scale is too small, the policy becomes
  nearly uniform; if too large, it becomes nearly deterministic, with poor local minima. "we found
  reward scale to be the only hyperparameter that requires tuning" — [Haarnoja, Zhou, Abbeel, Levine 2018, ar5iv](https://ar5iv.labs.arxiv.org/html/1801.01290) [sec; ICML 2018 venue not re-checked]
- **PopArt:** "Most learning algorithms are not invariant to the scale of the function that is
  being approximated". PopArt adaptively normalises the targets, which lets Atari agents drop
  reward clipping, since "a clipped reward function can result in qualitatively different
  behavior" — [van Hasselt, Guez, Hessel, Mnih, Silver, NIPS 2016](https://arxiv.org/abs/1602.07714) [abs]
- **Multi-task PopArt:** tasks with denser or larger rewards look more "salient" and capture a
  shared learner's resources. The fix adapts each task's contribution "so that all tasks have a
  similar impact on the learning dynamics" (57 Atari games, 30 DMLab tasks) — [Hessel, Soyer, Espeholt, Czarnecki, Schmitt, van Hasselt, AAAI 2019](https://ojs.aaai.org/index.php/AAAI/article/view/4266); [arXiv](https://arxiv.org/abs/1809.04474) [abs]
- **DreamerV3:**
  - symlog(x) = sign(x) ln(|x| + 1).
  - The critic is a categorical distribution over exponentially spaced bins, trained with twohot
    targets and cross-entropy. "the loss only depends on the probabilities assigned to the bins",
    which "decouples the size of the gradients from the size of the targets".
  - "Predicting large targets using a squared loss can lead to divergence whereas absolute and
    Huber losses stagnate learning".
  - Actor returns are scaled by S = EMA(Per(R, 95) - Per(R, 5), 0.99), and advantages are divided
    by max(1, S): "to avoid amplifying noise from function approximation under sparse rewards, we
    only scale down large return magnitudes".
  - Percentiles are used because "The return distribution can be multi-modal and include outliers"
    and normalising by the extremes "would then scale returns down too much".

  Sources: [Hafner, Pasukonis, Ba, Lillicrap, arXiv HTML](https://arxiv.org/html/2301.04104) [sec]; journal version Nature 640:647-653 (2025), DOI 10.1038/s41586-025-08744-2: [RePEc record](https://ideas.repec.org/a/nat/nature/v640y2025i8059d10.1038_s41586-025-08744-2.html)
- **Classification losses for value functions.** Training value functions with categorical
  cross-entropy instead of MSE "significantly improves performance and scalability". The authors
  attribute the gains mainly to mitigating noisy targets and non-stationarity (Atari, robotic
  manipulation, chess, Wordle) — [Farebrother et al. 2024, arXiv](https://arxiv.org/abs/2403.03950) [abs; venue not confirmed on the page]
- **MO-Gymnasium tooling.** It offers `MONormalizeReward(env, idx, gamma, epsilon)`, which
  normalises one reward component by index (one wrapper per objective) — [MO-Gymnasium wrappers docs](https://mo-gymnasium.farama.org/wrappers/wrappers) [secondary]
- **MO-MPO:**
  - Preferences are encoded as per-objective KL budgets epsilon_k ("We use these ϵk to encode
    preferences over objectives"). The temperatures computed from them end up "neutralizing the
    effect of the scaling of rewards".
  - Linear scalarisation is shown to be scale-sensitive. In a bandit, multiplying objective 1's
    rewards by 20 shifted scalarised MPO's choices. In humanoid run, a 10x action-norm penalty
    stopped scalarised MPO from reaching high task reward, while MO-MPO did not change.
  - "We train a separate Q-function per objective, following the Q-decomposition approach".

  Sources: [Abdolmaleki et al., ICML 2020, PMLR 119:11-22](https://proceedings.mlr.press/v119/abdolmaleki20a.html); [ar5iv](https://ar5iv.labs.arxiv.org/html/2005.07513) [sec, first ~75 %]
- **Our normalisation results.** Measured once at 15k steps, the scales were safety 6389 vs
  precision 26, "inflated by early crashes, 2.5 % of steps". Crash rates: norm + sum 55 %; running
  TD-error RMS + sum 53 %; normalising by TD-target std fixed collapse (R2 0.97-0.99) but crashed
  28 % with sum (plain shared critic 4.5 %, variant E 0.6 %) — [research prompt §2, §4](/Users/khushpatel/code/CrazyFlie-2.1/l2f/docs/shared_critic_research_prompt.md)

### Inferences
- **Global vs relative scale.** Our critic uses Adam with per-parameter normalisation, so the global
  reward scale has little effect on step sizes (unlike in Henderson's DDPG setting). The relative
  scale of loss terms that share parameters does matter. In the shared trunk, the largest group's
  gradient dominates each parameter's second-moment estimate, so the other groups barely steer the
  update. Adam cannot fix what normalisation tries to fix, and normalisation cannot fix it without
  changing what the trunk learns.
- **Where the literature puts invariance.**
  - MO-MPO, CoMOGA and PID-Lagrangian make preferences or balance scale-invariant in the policy
    update, while each objective's critic regresses in its own raw units, usually as a separate
    network.
  - Multi-task PopArt normalises inside a shared network, but to give separate tasks equal
    influence. That is a deliberate utility choice. It is not a neutral preconditioner for additive
    parts of one return.
  - For us, the utility is the real-unit sum (variant E). Any normalisation that reaches the actor's
    weighting is a different utility, and normalisation inside the critic loss reallocates trunk
    capacity (H1). The B.1-B.3 failures are what this view predicts.
- **Std vs percentile scales.** With crashes at 0.6-2.5 % of episodes, a std- or RMS-based scale for
  a group that contains the crash is set by the crash outliers (our 6389), and dividing by it
  removes the crash signal. A DreamerV3-style P95-P5 range would mostly exclude crash returns and
  measure the group's ordinary variation. Dividing by it would not shrink the crash. This also
  means it would not rebalance safety's gradient share. Hence the tension in the takeaway. My
  conclusion: resolve it structurally, not with a better normaliser. Move the crash out of the
  shared trunk (R2), spread it across groups (R1), or remove its decision-irrelevant variance
  (R-PBRS, Q4).
- **If normalisation is kept:**
  - Normalise only the dense groups.
  - Use PopArt-style output preservation and feed real-unit values to the actor.
  - Prefer percentile ranges with a floor, and scale down only.
  - Never clip or squash the crash term in the actor's objective; PopArt's motivation is that
    clipping changes behaviour qualitatively.
- **Recommended practice in MORL.** Per-objective normalisation is available in tooling
  (MO-Gymnasium) and recommended when preferences are meant to be scale-free (MO-MPO). It is not
  neutral when a fixed real-unit utility is given. I found no MORL guidance that recommends
  per-objective return normalisation for a decomposition whose sum is the true objective.

### Gaps
- I found no paper that normalises additive reward components inside a shared decomposed or
  multi-head critic and reports the effect on control. This is consistent with our results being
  new, but also with the configuration being unusual.
- PopArt's output-preservation details were not read (abstract only), nor was multi-task PopArt's
  mechanism.
- I found no TD3-specific study of reward scale. Henderson's study is DDPG, and SAC's scale
  sensitivity comes through entropy, which TD3 lacks.
- DreamerV3's percentile normalisation is for a single return in a model-based agent. Applying it
  per objective is my extrapolation.

## Q4. Potential-based reward shaping for progress and time-to-target terms

### Takeaway
Potential-based shaping F = gamma Phi(s') - Phi(s) leaves the optimal policy unchanged (Ng et al.
1999). In episodic tasks this holds only if terminal states have zero potential (Grześ 2017).

- Because the shaping return telescopes, a shaping term's value is -Phi(s), which does not depend
  on the action. It changes TD-target scales but not dQ/da. As its own group it would give the
  aggregator a pure-noise gradient.
- A minimum-time cost cannot be written as a potential difference, and replacing the position cost
  with a progress term would remove the tracking objective.
- Our progress candidate's "no refund" crash step makes it shaping plus a 2K·d crash penalty, so it
  is not policy-invariant.
- The one useful role I see for shaping in commensurability: a crash-risk potential for the safety
  head. It removes state-driven variance from that head's targets without changing the utility or
  the action-gradients.

### Cited Findings
- **Ng, Harada, Russell (1999)** characterise the reward changes that preserve the optimal policy.
  The potential-difference form F = gamma Phi(s') - Phi(s) is the shaping result — [Ng, Harada, Russell, ICML 1999, pp. 278-287](https://mlanthology.org/icml/1999/ng1999icml-policy/) [secondary: bibliographic record; paper not re-read in this session]
- **Shaping equals initialisation.** Potential-based shaping is equivalent to initialising Q at the
  potential: the same updates throughout learning, and behaviour indistinguishable for a broad
  class of policies — [Wiewiora, JAIR 2003](https://arxiv.org/abs/1106.5267) [abs]
- **Episodic tasks need zero terminal potential.** In episodic tasks, terminal states must have
  zero potential (Phi(s_terminal) = 0) to guarantee policy invariance; otherwise the shaped
  learner can converge to a different solution — [Grześ, AAMAS 2017, pp. 565-573](https://kar.kent.ac.uk/60614) [secondary: via search snippets and later papers' descriptions]
- **Our progress candidate.** The reward check scores "progress=K" as position cost
  min(d, C)^2 plus K(d_t - gamma d_{t+1}). On a crash step it uses `extra[-1] = -k * d[-1]  # no
  refund of the distance still to fly`. Its analysis treats a crash as worth -(P + k d_t) — [l2f/py/reward_check.py lines ~123-147](/Users/khushpatel/code/CrazyFlie-2.1/l2f/py/reward_check.py) [code]
- **Our time term.** The time group is a constant 0.5 per step (time-to-target cost) with a
  TD-target spread of 3.2 — [research prompt §1](/Users/khushpatel/code/CrazyFlie-2.1/l2f/docs/shared_critic_research_prompt.md)

### Inferences
- **A shaping group has an action-independent value.** For any policy, with Phi(terminal) = 0 and
  discounting (PEB), the shaping return telescopes to -Phi(s_0). So the shaping term's own
  Q^pi_F(s, a) = -Phi(s) for every action. As a separate group, its true action-gradient is zero.
  The learned head would feed only noise to the aggregator, and scale-invariant aggregators
  (PGA/UPGrad) would scale that noise up. **Never give a shaping term its own head.** Merge it into
  the group it is meant to guide (precision).
- **Inside a group, shaping changes target spread, not decision scale.** Q_shaped = Q - Phi(s). So
  shaping changes the group's TD-target spread, and therefore its share of the trunk loss and any
  spread-based normaliser. It does not change the action-dependent part the actor uses. TD-target
  std is therefore a poor measure of a group's decision importance. Advantage or dQ/da size is the
  relevant scale.
- **What shaping cannot do.** Replacing the squared-distance cost with a progress term removes the
  tracking objective, because shaping is policy-invariant and neutral by construction. The
  minimum-time cost depends on the path (the arrival time), so it cannot telescope and is a genuine
  objective. The time and position terms therefore cannot be made "commensurate" by recasting them
  as potentials.
- **The progress candidate is not pure shaping.** Exact shaping with Phi = -K d and
  Phi(terminal) = 0 needs +K d_t on the crash step (the "refund"). The implemented -K d_t equals
  shaping plus an extra crash penalty of 2K·d_t. This is not policy-invariant (Grześ's condition)
  and is more crash-averse far from the target, which may be intended. If a paper calls this term
  "potential-based", it should either add the refund or describe the extra term. For landings
  (d ≈ 0) the difference is negligible.
- **R-PBRS (my proposal): a crash-risk baseline for the safety head.**
  - **What it is.** Add shaping with Phi_safe(s) = -P · p_hat(s), where p_hat is a fixed crash-risk
    model on the privileged state (for example a logistic fit of crash-within-1-s on tilt, height,
    angular rate and velocity, fitted once on replay), and Phi(terminal) = 0.
  - **Effect.** The safety head then learns Q_safe - Phi(s), the action-dependent residual. Its
    state-driven spread (doomed vs safe states, which inflates the 152 std and the trunk gradient)
    shrinks. dQ_safe/da and the optimal policy stay exactly the same. This is the only tool I found
    in the shaping literature that changes a group's target scale without changing the utility or
    the actor's gradients.
  - **Risks.**
    - A poor p_hat adds variance (it stays policy-invariant regardless).
    - F must use the critic's gamma exactly, with Phi = 0 on crash and landing steps.
    - The benefit depends on how much of safety's spread is state-driven. Measure this first:
      compare Var[Q_safe] with Var[Q_safe - V_safe(s)] on replay data from an existing split-critic
      run.

### Gaps
- I did not re-read Ng et al. or Grześ in full. The Phi(terminal) = 0 condition is taken from
  secondary descriptions.
- I found no quadrotor or robot-control study comparing a minimum-time cost with progress shaping,
  or using a learned or hand-designed crash-risk potential as a baseline for one head of a
  decomposed critic. R-PBRS is untested.

## Q5. Choosing commensurate objectives in MORL: utility-based guidance, normalisation practice, which terms belong together

### Takeaway
The utility-based MORL guide (Hayes et al. 2022) says to fix the utility first. Hand-tuned linear
weights are a guess-and-check substitute, and safety-critical decisions are closer to
constraint-based or risk-sensitive formulations than to weighted terms. In that view, our utility
is variant E's real-unit sum, so normalising groups changes the utility unless it is compensated.

The decomposition should separate the rare catastrophic event (the crash) from dense costs,
because the two differ in scale, in timing, and in how each relates to termination. Scale-invariant
preference encodings exist (MO-MPO's KL budgets; CoMOGA's norm-proportional thresholds), but they
pair scale-free objectives with absolute-unit safety constraints.

My ranking of options:

1. Crash as a constraint with its own bounded critic, on top of a base penalty.
2. Split the crash penalty across groups (same utility as variant E, cheapest test).
3. A categorical / risk-sensitive crash critic.
4. A shaping baseline (R-PBRS) for the safety head.
5. CaT for the soft safety terms.
6. A survival bonus, which is only a re-encoding.

### Cited Findings
- **Hayes et al., guide to MORL (JAAMAS 36:26, 2022):**
  - **Hand-tuning fails.** The reward is "re-engineered if the behaviour is not satisfactory".
    Since only a few scalarisations are tried, an acceptable solution "can be arbitrarily far away
    from optimal utility".
  - **Linear rewards are limited.** "Not all human preferences can be handled by scalar additive
    reward functions", and "the relationship between the reward weights and the actual objective
    outcomes may well be non-linear".
  - **Safety.** "Strong parallels exist between multi-objective decision making and risk-sensitive
    or safety-aware decision making". Lexicographic MDPs use slack, which "allows some degree of
    loss in the primary objective in order to obtain gains in secondary objectives".
  - **SER vs ESR.** SER (utility of expected returns) applies when utility comes from averages over
    many runs. ESR (expected utility of each run) applies when "every policy execution must
    ensure" an outcome. With a linear utility they coincide.
  - **Refining objectives.** Objectives can be refined after review, for example pollution "should
    be refined, i.e., split into two objectives".

  Sources: [Hayes, Rădulescu, et al., ar5iv](https://ar5iv.labs.arxiv.org/html/2103.09568) [sec, first ~45 % of the text; no passage on normalising objectives found in the part read]. Journal record (DOI 10.1007/s10458-022-09552-y): [Springer PDF](https://rd.springer.com/content/pdf/10.1007/s10458-022-09552-y.pdf) [secondary]
- **MO-MPO** argues that objectives "are often in different units and/or scales" and lets
  preferences be set "in a scale-invariant way", with separate Q-functions per objective — [Abdolmaleki et al., ICML 2020](https://proceedings.mlr.press/v119/abdolmaleki20a.html); [arXiv abs](https://arxiv.org/abs/2005.07513) [sec]
- **CoMOGA** keeps the objective constraints relative to each objective's gradient norm, and the
  safety constraints in absolute cost units with thresholds d_k — [CoMOGA arXiv v2 HTML](https://arxiv.org/html/2403.00282v2) [sec]
- **Safety Gym** keeps "safety specifications ... separate from task performance" — [Ray, Achiam, Amodei 2019](https://openai.com/index/benchmarking-safe-exploration-in-deep-reinforcement-learning) [secondary]
- **Our measured structure:**
  - TD-target spreads: time 3.2, precision 7.8, efficiency 17, safety 152.
  - Per-step reward correlations between groups: -0.08 to +0.39.
  - The safety group mixes the crash penalty with dense orientation and landing-speed costs (and,
    per moo.py, the landing payout).

  Sources: [research prompt §1](/Users/khushpatel/code/CrazyFlie-2.1/l2f/docs/shared_critic_research_prompt.md); [moo.py](/Users/khushpatel/code/CrazyFlie-2.1/l2f/py/moo.py)
- **Our alternative grouping.** The code already has a crash / progress / effort / limits regrouping
  (`REGROUPS`): the crash term alone, position + velocity + time in "progress", and the dense
  safety terms in "limits" — [moo.py](/Users/khushpatel/code/CrazyFlie-2.1/l2f/py/moo.py) [code]

### Inferences
#### Principles for which terms belong together (my synthesis)
1. **Separate the rare catastrophic event from dense costs.** The crash differs from every dense
   term in four ways:
   - it is rare;
   - its value is a terminal event, so it is bounded as a probability;
   - it is the only term that opposes termination (Q1-C);
   - it is what normalisers mis-measure (Q3).

   Orientation and landing speed are dense costs. They belong with precision or in a "limits"
   group, as `REGROUPS` already does, not with the crash. The landing payout is a terminal event
   like the crash and should follow the same design as the crash term.
2. **Group terms that agree, and split terms that conflict.** That is what the conflict-measured
   regrouping did ("time agrees with position"). Hayes's advice to refine objectives after review
   supports splitting "safety" into a catastrophic part and a dense-limits part.
3. **Never make a group whose true action-gradient is zero** (a pure shaping term, Q4), or a group
   whose sign structure makes it want the episode to end unless something compensates (Q1-C).
4. **Judge scale by decision relevance** (advantages, dQ/da), not by TD-target std. State-driven
   variance inflates target std without affecting the actor.
5. **Keep the utility fixed.** If the claim is "same objective as variant E, better optimiser", any
   per-group rescaling must be compensated in the actor (f_k = w_k s_k, as `--norm` did). A
   reviewer will otherwise argue the objective changed. If the claim is "a better objective", state
   it as a constrained or lexicographic utility (crash rate <= d), which the MORL guide treats as a
   legitimate utility choice.

#### Ranked reformulation options (my synthesis; best first)
| # | Option | What changes in our setup | Evidence | Main risk | Minimal 2-seed screen |
|---|---|---|---|---|---|
| 1 | **R2: crash as constraint + separate crash-probability critic + PID-lambda (beta-scaled), base penalty P0 kept** | Safety splits into the crash (cost critic, targets in [0,1]) and limits (dense, stay in reward critic). The shared reward critic sees only 3-4 dense groups (about 5x apart). The actor step is Agg(dense) plus a lambda-weighted crash gradient. lambda tracks the measured crash rate. | Stooke 2020 (separate cost critic, scale invariance, damping); Tessler 2019; Ray 2019; Kim 2024; Bohez 2019; SQRL/CSC critic forms; CoMOGA's CAGrad + Lagrangian ablation (unstable) | Termination bias keeps the constraint active if P0 is dropped. Multiplier oscillation. Threshold set on noisy training episodes. Rare-event critic. | Keep -400. Add PID-lambda for crash episodes <= 0.5 % with a small separate crash critic (gamma_C 0.99). Rerun the shared reward critic with B.2/B.3 normalisation. Prediction: crash rate no longer rises with normalisation. |
| 2 | **R1: split the crash penalty (and landing payout) across groups, P_k proportional to C_k, sum 400** | Per-group terminal rewards. Scalar reward and utility are identical to variant E. No group gains from crashing. Every head carries crash information in its own units. | First principles: invariance of the decomposition's sum plus termination bias (Kostrikov 2019; S&B Ex. 3.16; legged_gym). No direct literature. | Higher inter-group correlation from the shared crash component. Crash's share of sum e_k^2 falls by sum w_k^2. Changes the split-critic conflict statistics. | Shared critic + R1 with (a) sum, (b) `--critic-norm-by target`, (c) OS-NMF-PGA. Measure collapse on dense parts only. |
| 3 | **R7: categorical (Bernoulli/twohot) crash critic, optionally CVaR**, used as the cost critic of #1 or as the safety critic of the split design | The crash head predicts discounted crash probability with a cross-entropy loss. The actor uses -P·Q_C (variant E) or -lambda·Q_C. | DreamerV3 (gradient size decoupled from target size); Farebrother 2024; WCSAC (CVaR safety critic); IQN | In a shared trunk, decoupling the gradient is exactly the H1 risk, so keep it separate. Choice of CVaR level. | Split critic with the safety net replaced by a Bernoulli crash net plus a dense-limits head. |
| 4 | **R-PBRS: crash-risk potential baseline on the safety head** | The safety targets become an action-dependent residual. The utility and dQ/da are unchanged. Safety's state-driven spread and gradient share shrink. | Ng 1999; Wiewiora 2003; Grześ 2017 | A poor p_hat adds variance. Exact gamma and zero terminal potential are required. Little gain if the spread is action-driven. | First measure Var[Q_safe] vs Var[Q_safe - V_safe(s)] offline. Then the shared critic with Phi_safe, sum and B.3. |
| 5 | **R5: CaT for soft safety limits, with non-negative rewards** | The tilt, rate and landing-speed terms become stochastic terminations. The safety group disappears. Termination enters every head multiplicatively. | CaT (IROS 2024; real Solo-12); legged_gym; DAC survival bias | The objective is no longer variant E's. Positive-reward bias against landing (payout must equal continuation value). Only on-policy PPO evidence. Stale delta in replay. | After #1/#2. Needs a reward redesign first. |
| 6 | **R6: survival bonus b = 4 (equivalent to P = 400 under PEB) plus +400 landing payout** | Encoding only, with a +400 value offset. | Q1 derivation; Sun 2022 (shift equals initialisation); Naik 2024 (offset masks action differences) | No change in commensurability. Offset error. | Only as part of #5. |
| - | **Normalisation hygiene** (if any normalisation remains) | Dense groups only. Percentile range, scale down only. Output-preserving. Real units to the actor. Never clip the crash term. | DreamerV3; PopArt; our B.1 (s_safe 6389) | Does not resolve the tension by itself. | - |

### Gaps
- The visible ~45 % of the Hayes guide has no explicit guidance on normalising objectives or on
  grouping reward terms. The rest (including the worked example) was not read.
- I found no MORL paper that studies how to group the components of a hand-designed robot reward
  into objectives, or that compares groupings.
- I found no paper that combines a decomposed or multi-head reward critic with a separate cost
  critic for a terminal catastrophe in continuous control. CoMOGA uses separate reward and cost
  critics, but a shared multi-head reward critic is not reported there in the part read.

## Q6. Distributional and risk-sensitive critics for rare catastrophic events

### Takeaway
A distributional critic can represent the bimodal crash/no-crash return explicitly, and with a
classification loss it decouples gradient size from target size (DreamerV3; Farebrother et al.).
Risk-sensitive CVaR safety critics (WCSAC) report better risk control in safe RL. Two cautions:

- In tabular and linear settings, distributional RL matches expected RL (Lyle et al.), so any gain
  for a rare penalty must come from representation learning or from deliberately risk-averse
  objectives.
- In a shared trunk, the gradient decoupling repeats the H1 risk (crash errors no longer dominate
  the features).

The natural use for us is a separate crash critic that predicts crash probability (a [0, 1]
quantity), with an optional CVaR, not a distributional safety head on the shared trunk.

### Cited Findings
- **IQN** uses quantile regression to approximate the full quantile function of returns; this
  "gives rise to a large class of risk-sensitive policies" (Atari) — [Dabney, Ostrovski, Silver, Munos, ICML 2018](https://arxiv.org/abs/1806.06923) [abs]
- **Continuous-control distributional critics.** D4PG is the distributional (C51-style categorical)
  DDPG with N-step returns and prioritised replay. TQC (SAC-based) combines quantile critics,
  truncation of the top atoms, and ensembling to control overestimation, and reports gains on the
  continuous-control suite — [D4PG, Barth-Maron et al.](https://arxiv.org/abs/1804.08617); [TQC, Kuznetsov et al.](https://arxiv.org/abs/2005.04269) [secondary: search summaries; venues not checked]
- **Lyle et al.** show that in many tabular and linear settings "distributional RL behaves exactly
  the same as expected RL". Where the two differ, distributional RL can hurt. Their nonlinear
  experiments look for where the gains come from — [Lyle, Castro, Bellemare, AAAI 2019](https://arxiv.org/abs/1901.11084) [abs]
- **DreamerV3** uses a twohot categorical critic that "decouples the size of the gradients from the
  size of the targets", and notes that squared loss on large targets can diverge while
  absolute/Huber losses "stagnate learning" — [DreamerV3 arXiv HTML](https://arxiv.org/html/2301.04104) [sec]
- **Classification value losses** help mainly by mitigating noisy targets and non-stationarity — [Farebrother et al. 2024](https://arxiv.org/abs/2403.03950) [abs]
- **WCSAC** uses a separate distributional safety critic with a CVaR constraint (SAC-Lagrangian
  style) and reports better risk control than expectation-based methods. The journal version finds
  IQN-quantile safety critics better than the Gaussian approximation — [WCSAC, AAAI 2021](https://ojs.aaai.org/index.php/AAAI/article/view/17272); [Machine Learning 2023](https://link.springer.com/article/10.1007/s10994-022-06187-8) [secondary]
- **SQRL and CSC** both learn a separate failure-probability critic. SQRL uses a squared TD loss on
  a probability target; CSC makes the estimate conservative to bound failure probability during
  training — [SQRL ar5iv](https://ar5iv.labs.arxiv.org/html/2010.14603) [sec]; [CSC](https://arxiv.org/abs/2010.14497) [abs]

### Inferences
- **The safety return is bimodal.** For a given (s, a), the safety group's return is near 0 (no
  crash) or near -400 gamma^tau (crash). Its expectation is about -400 times the discounted crash
  probability. An MSE critic sees rare -400 targets as high-variance TD errors. These are the
  100-3000x gradient spikes. Modelling the crash as a probability (Bernoulli/BCE on
  c + (1 - c) gamma Q'_C) is the minimal "distributional" fix: the targets are in [0, 1] and the
  gradients are bounded, and the actor multiplies by P (or lambda).
- **The expected-value fixed point is unchanged.** Per Lyle et al., a distributional crash head does
  not change what the expected value converges to. Benefits would come from better features (an
  auxiliary-task effect on a shared trunk) or from choosing a risk-averse objective (CVaR), not from
  "representing the penalty better" in expectation.
- **Shared trunk vs separate network.** The same bounded-gradient property that helps a separate
  crash critic would, on a shared trunk, cut the crash's share of the features. That is the H1
  failure we saw with normalisation. Put the distributional or CVaR crash critic in its own network
  (#1/#3 in the Q5 table).
- **Risk sensitivity and sim-to-real.** A CVaR or quantile crash critic gives a principled knob for
  extra caution under model mismatch (domain randomisation of motor delay and gains): it optimises
  the worst alpha-fraction of crash outcomes. Our tracking already degrades under mismatched motors
  (circle error 11.9 cm for the split critic vs 8.9 cm for variant E, research prompt §2), and that
  is the kind of mismatch where such a knob would matter. The evidence is from safe-RL benchmarks,
  not quadrotors.
- **Sample scarcity.** Crash transitions are rare and become rarer, and no distributional head
  changes that. Keeping them represented in replay is a separate, unsolved issue for us (see Q1
  gaps).

### Gaps
- I found no study of distributional or risk-sensitive critics for rare terminal penalties in
  quadrotor or other end-to-end low-level control.
- The D4PG, TQC and WCSAC details come from search summaries only, and I found no TD3-based
  distributional variant with published crash or safety metrics.
- I found no evidence on whether a distributional safety head on a shared trunk helps or hurts the
  other heads. This would need our own experiment.
