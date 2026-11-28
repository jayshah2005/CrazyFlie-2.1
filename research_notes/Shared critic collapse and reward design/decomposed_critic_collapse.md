# Collapse and interference between heads of multi-head, decomposed and vector-valued critics

Scope and method (2026-11-27). This search covers the literature on shared-torso critics with several value outputs, and on whether the heads collapse into copies or interfere. It goes deeper than two earlier notes: `research_notes/MOO quadrotor RL novelty review/decomposed_critics.md` and `morl_gradient_conflict.md`. In particular it reads SAC-D's appendices, settles GCR-PPO's trunk sharing from its code, and verifies "Shared Actors Need Not Share Critics" (which the earlier review could not confirm). It also adds ensemble-diversity, MARL parameter-sharing, explainability-critic, preference-conditioned and TD-representation-dynamics evidence.

Reading status is marked on each source:
- **[full]**: full text read, or extracted locally and searched.
- **[partial]**: some sections read.
- **[abstract]**: abstract or listing only.
- **[snippet]**: search-engine summary only.
- **[code]**: source code inspected.

The fetch tool limits verbatim quotes to about 125 characters, so many quotes below are exact sentence fragments, with their section given where known. Quotes from locally extracted PDFs (Juozapaitis et al., HDPG, PD-MORL) are verbatim and longer.

## Q1. Does any paper report per-head collapse, rank loss across heads, or a shared critic hiding objective conflict? (novelty flags)

### Takeaway
No paper found reports what we observe. The specific observation is that the per-objective heads of a reward-decomposed, shared-trunk TD critic become near-copies (high correlation), even though each head has its own reward and TD target, and that this hides inter-objective gradient conflict from the actor. Searches with five different phrasings and every paper fetched came up empty on this point. Several precedents do narrow the claim and must be cited:
- **SAC-D** reports that rare terminations cause very large component losses and "catastrophic gradient conflicts" in a shared multi-head critic. It also reports that CAGrad gives lower weight to the higher-loss components, and that the hidden-layer gradients were divided by the number of components.
- **Bootstrapped DQN** reports that a shared torso costs head diversity, and that summed head gradients raise the torso's learning rate (fixed with 1/K).
- **MED-RL** already uses the term "value function collapse", for ensemble members (separate networks, same target).

### Cited Findings
**Flag 1: SAC-D. Shared multi-head critic, rare terminations, conflict-averse fix down-weights high-loss heads.** SAC-D is MacGlashan, Archer, Devlic, Seno, Sherstan, Wurman and Stone, "Value Function Decomposition for Iterative Design of Reinforcement Learning Agents", NeurIPS 2022. It is a decomposed SAC, not "distributional SAC". [full appendices via ar5iv; main text read in the earlier review]
- Architecture: "Alter Q-function networks to have m outputs instead of 1, where m is the number of reward components." Hidden layers are [256, 256], with "SAC-D's hyperparameters [tied] to SAC's." — [ar5iv 2206.13901](https://ar5iv.labs.arxiv.org/html/2206.13901)
- Appendix C, exact: "In SAC-D and SAC-D-Naive training, the gradients of the hidden layers are divided by the size of reward dimension." No reason is given. — [ar5iv 2206.13901](https://ar5iv.labs.arxiv.org/html/2206.13901)
- Appendix E (Ant), exact fragments:
  - "Unlike other environments, in Ant, it is uncommon for even an untrained policy to terminate in failure."
  - "these rare instances in the replay buffer were associated with very large critic losses"
  - "Because terminating transitions constitute a large component value difference compared to non-terminating transitions"

  — [ar5iv 2206.13901](https://ar5iv.labs.arxiv.org/html/2206.13901)
- Main text, as quoted in the earlier review from the NeurIPS PDF:
  - "simultaneous optimization of all Q_i components may introduce training problems common in multi-objective optimization: conflicting gradients, high curvature and large differences in gradient magnitudes"
  - "In Ant, SAC-D (without CAGrad) underperforms all other methods… infrequent environment termination causes large Q-function errors and leads to catastrophic gradient conflicts"
  - "SAC-D meets or exceeds SAC's performance in all environments we tested"

  — [NeurIPS 2022 PDF](https://papers.neurips.cc/paper_files/paper/2022/file/4eb2c0adafbe71269f3a772c130f9e53-Paper-Conference.pdf)
- How the fix works:
  - CAGrad (c = 0.5) is applied over the per-component TD losses, with entropy as the (m+1)th component.
  - It acts on the union of both twin networks' parameters. The policy takes an ordinary gradient step.
  - Inside the CAGrad inner solve, gradients are normalised by an averaged norm.

  — [ar5iv 2206.13901](https://ar5iv.labs.arxiv.org/html/2206.13901)
- Appendix F, exact: "the higher the critic loss, the lower the weight CAGrad tended to assign the component." — [ar5iv 2206.13901](https://ar5iv.labs.arxiv.org/html/2206.13901)
- Twin choice: "a better approach is to use all the predictions from the network with the minimum composite Q-function". This is the same as our per-sample choice by the smaller summed value. — [ar5iv 2206.13901](https://ar5iv.labs.arxiv.org/html/2206.13901)
- SAC-D reports no per-component accuracy or head-similarity metric, and no shared-vs-separate ablation. Its ablations vary only the twin-min rule and CAGrad. — [ar5iv 2206.13901](https://ar5iv.labs.arxiv.org/html/2206.13901)

**Flag 2: Bootstrapped DQN. A shared torso lowers head diversity; summed head gradients raise the torso's step; fixed with 1/K.** Osband, Blundell, Pritzel and Van Roy, NeurIPS 2016. [partial]
- "The network consists of a shared architecture with K bootstrapped "heads" branching off independently." The design gives "significant computational advantages at the cost of lower diversity between heads". Each head "is trained against its own target network". — [ar5iv 1602.04621](https://ar5iv.labs.arxiv.org/html/1602.04621)
- Section 6.1, exact: "We found the best final scores by normalizing the gradients by 1/K, but this also leads to slower early learning." The stated reason is a fragment: "effectively increases the learning rate for this portion of the network". — [ar5iv 1602.04621](https://ar5iv.labs.arxiv.org/html/1602.04621)
- Diversity comes from initialisation, not data: "Bootstrapped DQN relies upon random initialization of the network weights as a prior to induce diversity." The mask p made little difference ("the algorithm performed similarly irrespective of p and all outperformed DQN"). With p = 1, "all network heads share all the data, so they are not actually a traditional bootstrap at all." — [ar5iv 1602.04621](https://ar5iv.labs.arxiv.org/html/1602.04621)

**Flag 3: The term "value function collapse" is already taken (ensembles).** Sheikh and Bölöni, "Preventing Value Function Collapse in Ensemble Q-Learning by Maximizing Representation Diversity", arXiv 2006.13823 (NeurIPS 2020 Deep RL workshop). This became MED-RL, Sheikh, Phielipp and Bölöni, ICLR 2022. [partial, arXiv version]
- Abstract: "Unfortunately, these learners can converge to the same point in the parametric or representation space". Body: "these networks collapse to closely related representations". — [ar5iv 2006.13823](https://ar5iv.labs.arxiv.org/html/2006.13823)
- Measurement:
  - Linear-kernel CKA between ensemble members.
  - Output layers show "more than 96% similarity" for baselines against "around 90%" when regularised.
  - In a regression toy, outputs go from "0% similar" to "98% similar" after training.

  — [ar5iv 2006.13823](https://ar5iv.labs.arxiv.org/html/2006.13823)
- The members are separate networks ("parameterized by {ψ1,…,ψN}") trained on different batches. Collapse still happens: "neural networks can learn similar representation while trained on different batches." — [ar5iv 2006.13823](https://ar5iv.labs.arxiv.org/html/2006.13823)
- The five regularisers (Atkinson, Gini, Theil, variance of logs, MeanVector) act on the L2 norms of each network's parameters, not on Q-values or features. — [ar5iv 2006.13823](https://ar5iv.labs.arxiv.org/html/2006.13823)
- The ICLR 2022 version reports gains "in some cases by more than 300%" over unregularised ensembles across 6 MuJoCo and 6 Atari tasks (from search summary, not verified). — [ICLR 2022 poster](https://iclr.cc/virtual/2022/poster/6498)

**Flag 4: Scale-driven dominance of a shared torso is established in multi-task RL.**
- Multi-task PopArt (Hessel, Soyer, Espeholt, Czarnecki, Schmitt, van Hasselt; AAAI 2019) [partial]:
  - "the saliency of a task for the agent increases with the scale of the returns observed in that task"
  - "This causes the algorithm to focus on those salient tasks at the expense of generality."
  - "causing some tasks to have a larger impact on the learning dynamics than other equally important ones."

  — [ar5iv 1809.04474](https://ar5iv.labs.arxiv.org/html/1809.04474)
- Distral (Teh et al., NeurIPS 2017) [partial]:
  - The abstract says gradients from different tasks "can interfere negatively, making learning unstable and sometimes even less data efficient".
  - Introduction: "gradients from other tasks behave as noise, interfering with learning" and "one of the tasks might dominate the others".

  — [ar5iv 1707.04175](https://ar5iv.labs.arxiv.org/html/1707.04175)

**Flag 5: Closest statements that a shared critic "hides" differences.** Neither concerns objective gradients.
- Liu, Meng, Zhao, Liu and Chen, "Shared Actors Need Not Share Critics: Effects of Value Mismatch in Parallel RL", arXiv 2608.26481, Aug 2026 [partial: 200k of 232k characters]:
  - "This average can also hide a rarely sampled environment with a large pointwise offset"
  - "The practical tradeoff is therefore between statistical sharing and separation of conflicting value targets."

  This is about a scalar critic shared across environments, not reward heads. — [arXiv 2608.26481](https://arxiv.org/html/2608.26481v1)
- COP-Q (Li, Zanger, Spaan, Kooij; arXiv 2606.04749, Jun 2026) [partial]:
  - Each critic "uses a shared trunk with two output heads, corresponding to the reward and safety Q-values".
  - The authors note "these objectives are typically correlated, so their estimation uncertainties are entangled".
  - They also note numerical trouble when "the ensemble Q-vectors are nearly collinear in the bi-objective space".

  They treat the correlation as something to model (Cholesky-ordered projection), not as a pathology. — [arXiv 2606.04749](https://arxiv.org/html/2606.04749)
- SAC-D on surfacing conflict, quoted in the earlier review: "Even though the scalar values of the composite Q-function are identical to those used in SAC," per-component optimisation brings in conflicts. — [NeurIPS 2022 PDF](https://papers.neurips.cc/paper_files/paper/2022/file/4eb2c0adafbe71269f3a772c130f9e53-Paper-Conference.pdf)

**Flag 6: Terminology collisions.**
- "Value function interference" is an existing MORL term with a different meaning, from Vamplew, Watkins, Foale and Dazeley, arXiv 2402.06266 (2024, v2 2026) [partial]. It is defined at the level of expected returns: the vector "represent[s] the expected return averaged over multiple different future returns", and under a non-linear utility "the utility of this expected return vector may differ from that of the returns from which it is derived". It arises "even in the context of otherwise stable learning algorithms based on tabular representation of Q-values". — [arXiv 2402.06266v2](https://arxiv.org/html/2402.06266v2)
- "Representational collapse" in D3PO and "collapse" in MOPPO mean policy or entropy collapse, not critic heads (verified in the earlier review). — [D3PO arXiv 2602.07764](https://arxiv.org/html/2602.07764v1); [MOPPO arXiv 2407.16807](https://arxiv.org/pdf/2407.16807)

### Inferences
- **Novelty that survives.** The following set of claims appears unreported:
  1. Heads that are separately supervised, and therefore identified, collapse to near-copies (head-value correlation 0.93-1.00) in a reward-decomposed, shared-trunk TD critic.
  2. The collapse is visible in the TD targets too, so it looks self-reinforcing through bootstrapping.
  3. It makes the per-objective actor gradients near-parallel, so every actor-side aggregator reduces to the sum. The shared critic hides the conflict.
  4. It is not a capacity effect: a parameter-matched wider shared critic still collapses, while separate critics do not.

  None of SAC-D, HRA, HDPG, GCR-PPO, HARC, PGMT, COP-Q, Kim et al., MOMA-AC, PD-MORL, Envelope or Abels et al. measured head-to-head similarity.
- **Claims to qualify or attribute.**
  1. The observation that an unscaled K-head critic moves the summed value by more than a scalar critic does, and the 1/K fix, have precedent: Osband et al. 2016 give exactly this reason, and SAC-D uses the same scaling. This corrects the earlier review's "looks novel" statement.
  2. An aggregator inside the critic that down-weights large-loss components is documented for CAGrad in SAC-D, Appendix F. Our scale-invariant PGA mechanism is a sharper, quantified version of it.
  3. A rare termination penalty causing very large losses in a multi-head critic is documented in SAC-D, Appendix E.
- **Naming.** Avoid "value function collapse" (used by MED-RL for ensembles) and "value function interference" (used by Vamplew et al. in a different sense). Name the phenomenon distinctly, for example "head collapse in decomposed critics", and define it operationally.
- **Reviewer-proofing (first principles, my reasoning).** Per-step reward correlation is the wrong null hypothesis for value correlation. Under a fixed policy, every group value is Q_k = (I − γP^π)^{-1} r_k. All heads share the same resolvent, and its slow modes (for example time-to-termination or distance-to-target decay) are amplified by 1/(1 − γλ). Two weakly correlated reward streams can therefore have strongly correlated true values. The right reference is the correlation between per-group discounted Monte Carlo returns on the same states, or the split critic's values (+0.30 to +0.75). Showing that the shared critic sits well above that reference makes the collapse claim robust.

### Gaps
- MultiCriticAL's full text (ICLR 2022) is behind an OpenReview bot check, so I read the project page only. It may contain a multi-head vs multi-critic comparison.
- SAC-D's Appendix E and F figures (per-component loss curves, CAGrad weights) were read as text only. The size of the component losses is not recorded here.
- Papers citing SAC-D or HRA after 2024 were not searched through Google Scholar, so a recent report of head collapse in a citing paper cannot be ruled out.
- "Reward-Wise Value Estimation for Multi-Reward Optimization in LLMs" (ICML 2026 workshop) is listing-only. I found no full text, so I could not check it for a head-similarity analysis.

## Q2. Hybrid Reward Architecture (van Seijen et al. 2017): shared vs separate heads, design rationale, reported interference

### Takeaway
HRA uses heads on a shared layer and argues that each head is easier to learn because it depends on few state variables. Its biggest gains come when each head sees only its own relevant inputs ("+1"), which is less sharing, not more. It reports no head-to-head interference. Its only scale problem was at the aggregator (Pac-Man), fixed by normalising heads. Later work showed that HRA's per-head max bootstrapping is unsound.

### Cited Findings
- Rationale: the decomposed value "can be approximated more easily by a low-dimensional representation", when "each reward function is mainly affected by only a small number of state variables". — [ar5iv 1706.04208](https://ar5iv.labs.arxiv.org/html/1706.04208) [partial]
- On sharing: "In general, different agents can share multiple lower-level layers of a deep Q-network." The paper does not discuss interference or other trade-offs of sharing. — [ar5iv 1706.04208](https://ar5iv.labs.arxiv.org/html/1706.04208)
- Fruit collection:
  - Network: "a fully connected hidden layer of length 250", and "This layer is connected to 10 heads consisting of 4 linear nodes each". A fixed layer sums the heads.
  - Result: "HRA shows a clear performance boost over DQN, even though the network is identical."
  - "+1" removes irrelevant features per head, so each head sees only the agent position and its own fruit. This gives further large gains.

  — [ar5iv 1706.04208](https://ar5iv.labs.arxiv.org/html/1706.04208)
- All 10 fruit heads had equal reward scale (one fruit each). In Pac-Man, the scale mismatch showed up at the aggregator: "the aggregator gets overwhelmed by the ghost avoider values". The authors fixed it by "modifying the additive aggregator with a normalisation over the score heads", with "the ghost multiplier ... modified to -10". — [ar5iv 1706.04208](https://ar5iv.labs.arxiv.org/html/1706.04208)
- Each head bootstraps from its own max, y_k = R_k + γ max_a′ Q_k(s′, a′), while "Action selection for HRA is based on the sum of the agent's Q-value functions." — [ar5iv 1706.04208](https://ar5iv.labs.arxiv.org/html/1706.04208)
- Juozapaitis, Koul, Fern, Erwig and Doshi-Velez, "Explainable RL via Reward Decomposition" (IJCAI/ECAI 2019 XAI workshop) [full, PDF extracted]. On HRA's target: "HRA adjusts the value of each Qc toward rc + γmaxa′Qc(s′,a′), which completely ignores the influence of other components to the overall Q-function and greedy action." The heuristic "generally overestimates the true Q-values, and the corresponding greedy policy can perform arbitrarily poorly, even with infinite training." — [NSF PAR PDF](https://par.nsf.gov/servlets/purl/10159391)

### Inferences
- HRA's success with dense heads on a shared layer came with equal-scale rewards and a tiny state space. It is not evidence that dense sharing tolerates a 100-3000x gradient-scale gap or a rare 400-point terminal penalty.
- HRA's "+1" gain from per-head input masking is the earliest evidence that restricting what each head shares helps. It supports partial-sharing or per-head feature-selection designs (see Q7).
- Our TD3 heads bootstrap with the shared actor's action a′ = π′(s′). That is the sound "global action" rule (drQ and SARSA-style), not HRA's per-head max, so HRA's unsoundness critique does not apply to us.

### Gaps
- HRA's full text was not reread for a shared-vs-separate network ablation. None was visible in the sections read, and the earlier review found none either.

## Q3. Decomposed and vector-valued critics: architecture, head-distinctness measurements, shared-vs-separate ablations

### Takeaway
Every decomposed or vector critic found uses a shared trunk with a vector output, or separate networks. None measures whether heads stay distinct. One structural split matters for transfer:
- **Routed heads**: multi-task PopArt, Liu et al. 2026, MultiCriticAL and the quadrotor MTRL paper. Each sample trains only its task's or environment's head.
- **Dense heads**: HRA, SAC-D, HDPG, GCR-PPO, HARC, PGMT, COP-Q, Kim et al., MOMA-AC, and our critic. Every sample trains every head, so the largest-error head drives the trunk at every sample.

Most dense multi-head critics that work either have comparable component scales or normalise each component.

### Cited Findings
| Work | Critic architecture | Dense or routed | Head distinctness measured? | Shared-vs-separate ablation | Source |
|---|---|---|---|---|---|
| SAC-D (NeurIPS 2022) | One net, m outputs, hidden [256,256]; hidden-layer gradients ÷ m; CAGrad c=0.5 | Dense | No | No | [ar5iv](https://ar5iv.labs.arxiv.org/html/2206.13901) |
| GCR-PPO (arXiv 2509.14816) | **Code-verified**: `nn.Sequential` MLP [256,256,256] ending in `nn.Linear(..., num_reward_components)`, so a fully shared trunk. Value loss `(value_batch - returns_batch).pow(2)` **summed over components**, with no per-component weighting in the loss (return normalisation in rollout storage not checked). Actor is a separate MLP; one AdamW optimiser | Dense | No | No (multi-head PPO vs GCR-PPO only) | [GitHub actor_critic.py](https://github.com/humphreymunn/GCR-PPO/blob/HEAD/rsl_rl/rsl_rl/modules/actor_critic.py) [code]; [ppo.py](https://github.com/humphreymunn/GCR-PPO/blob/HEAD/rsl_rl/rsl_rl/algorithms/ppo.py) [code] |
| MOMA-AC (Callaghan, Mason, Mannion; arXiv 2511.18181; *Neurocomputing* 664, 2026) | "A centralised critic estimating vector-valued Q-functions using joint observations and actions", conditioned on ω. Loss ℓj = ‖Qθj(o,a,ω) − y‖² (unweighted vector MSE). Twin chosen by ω⊤Q in the equations, while the prose says "pointwise minimum" | Dense | No | No (TD3 vs DDPG only) | [arXiv HTML](https://arxiv.org/html/2511.18181v1) [partial: 100k of 104k chars] |
| MOAC (ICML 2024) | Linear critics V^i(s) ≈ φ(s)ᵀw^i on "the same feature mapping … shared among all objectives" | Dense, fixed features | No | No | [arXiv 2405.03082](https://arxiv.org/html/2405.03082) (earlier review) |
| CoMOGA (ICLR 2025) | Separate preference-conditioned reward critics and cost critics | n/a | No | No | [arXiv 2403.00282](https://arxiv.org/html/2403.00282v2) (earlier review) |
| Xing et al., quadrotor MTRL (arXiv 2412.12442) | "a shared actor policy with multiple individual critic networks, each corresponding to a specific task"; no rationale given, no shared-critic ablation, critic sizes not stated | Routed (task per env) | No | No | [arXiv HTML](https://arxiv.org/html/2412.12442) [partial] |
| Multi-task PopArt (AAAI 2019) | Shared torso, "a multi-task value function 𝐯(S) with N outputs, one for each task"; "only the ith head in the value net is updated" | Routed | No (performance only) | "MultiHead-IMPALA" (heads, no PopArt) "actually performs slightly worse than the original IMPALA both with and without clipping" | [ar5iv](https://ar5iv.labs.arxiv.org/html/1809.04474) |
| Distral (NeurIPS 2017) | Per-task policies distilled into a shared π₀; per-task values V̂ᵢ with a small L2 term to "encourage a small amount of transfer of knowledge in value estimates too" | n/a | No | Against multitask A3C: unstable; some runs "did not learn well, while others may learn initially then suffer degradation later" | [ar5iv](https://ar5iv.labs.arxiv.org/html/1707.04175) |
| HDPG, bipedal (Huang, Wang, Zhou, Zhang, Lin; arXiv 2107.01908v2, 2021) | DDPG; state 69→128 and action 10→128, concatenated → 256 → 256 → 6 Q-values. "Only the last layer is independent of each other." "All component rewards are normalized to the same value range (e.g., (0,1))" by max-min | Dense | No | Multi-head vs scalar only: "The performance of MHDDPG is much better than vanilla DDPG" | [arXiv PDF](https://arxiv.org/pdf/2107.01908) [full, extracted] |
| HARC (arXiv 2608.09762, 2026) | Splits the centralised critic "into one value head for each reward component while sharing a common feature extraction trunk"; each head has "its own reward component, TD target, and gradient path"; 2 heads (sparse task, grasp shaping); "Share encoder: True" | Dense | No | No | [arXiv HTML](https://arxiv.org/html/2608.09762) [partial] |
| PGMT, humanoid motion tracking (arXiv 2609.08511, 2026) | "The three heads share a common critic backbone"; motivation: "A single scalar critic may entangle reward components with substantially different scales and learning dynamics." | Dense | No | Split-return vs single-head aggregate critic (Fig. 4): after perception injection the split critic reaches about 380-step episodes and about 80% timeout rate, while the aggregate critic stays near zero. No separate-network baseline | [arXiv HTML](https://arxiv.org/html/2609.08511) [partial] |
| COP-Q (arXiv 2606.04749, 2026) | Ensemble of critics, each with "a shared trunk with two output heads, corresponding to the reward and safety Q-values" | Dense | Ensemble covariance only, not head similarity | No | [arXiv HTML](https://arxiv.org/html/2606.04749) [partial] |
| Kim et al., "Not Only Rewards But Also Constraints" (IEEE T-RO 2024; arXiv 2308.12517) | "a multi-head cost value function, in which a single neural network predicts all cost values rather than just one"; each cost advantage "undergoes additional standardization and zero-mean normalization" | Dense (10 constraint heads, PPO) | No | Yes (see Q6) | [arXiv HTML v4](https://arxiv.org/html/2308.12517v4) [partial] |
| Liu et al. 2026, "Shared Actors Need Not Share Critics" | Shared critic, FiLM, multihead (shared encoder, one head per environment), multihead + PopArt, separate value network | Routed (logged env index) | Qualitatively: "the learned multihead values separate by gravity, whereas the shared value lies between them" | Yes (see Q6) | [arXiv HTML](https://arxiv.org/html/2608.26481v1) [partial] |

**Reward-decomposition explainability critics**
- drQ and drDQN (Juozapaitis et al. 2019) [full]:
  - On architecture: "Qc may correspond to a neural network, possibly with the networks for different reward types sharing initial network layers".
  - drQ "first computes the overall greedy action" and updates "each component Qc(s,a) toward rc + γQc(s′,a+)", which gives tabular convergence to the correct components.
  - Lunar Lander used "1 hidden layer of 64 ReLU units and an output layer with one linear unit per action". "drDQN and drDSARSA were able to achieve a good policy of consistently landing safely, whereas HRA's policy continuously degraded."
  - The decomposition exposed dominance: "the policies drDQN and drDSARSA are almost entirely governed by the shaping reward components", and "HRA doesn't learn to land and prefers to crash itself."

  — [NSF PAR PDF](https://par.nsf.gov/servlets/purl/10159391)
- DRDRL (Lin, Zhao, Yang, Qin, Liu, Yang; NeurIPS 2019) [partial]:
  - The sub-distributions "all depend on the same state feature input", made channel-specific through a learned one-hot channel embedding.
  - A disentanglement term avoids "naive decomposition such as 0-1 or half-half". It maximises a KL/cross-entropy between the return distributions at each sub-return's greedy action, with λ = 0.0001.
  - Evidence is qualitative: "significant differences between curves with and without the KL term".
  - Note that DRDRL learns the decomposition without sub-reward labels, so its heads are not identified.

  — [ar5iv 1911.02166](https://ar5iv.labs.arxiv.org/html/1911.02166)
- RD² (Lin, Yang, Zhao, Qin, Yang, Liu; NeurIPS 2020) [abstract]: it constrains the "uniqueness and compactness" of the state features linked to each sub-reward, so that each sub-reward uses a minimal, distinct feature set. — [NeurIPS 2020](https://papers.nips.cc/paper/2020/hash/82039d16dce0aab3913b6a7ac73deff7-Abstract.html)

**MARL value factorisation and parameter sharing**
- VDN (Sunehag et al., AAMAS 2018) [partial]:
  - Per-agent values are not individually supervised: "Q~i is learned implicitly rather than from any reward specific to agent".
  - The "lazy agent" problem: "a second agent is discouraged from learning because its exploration would hinder the first agent".
  - Weights are shared, with a "1-hot encoding of their identity".
  - Scale mismatch appears in Checkers: "the magnitude of rewards (and hence the value function) from one agent is ten times higher than for the other agent."

  — [ar5iv 1706.05296](https://ar5iv.labs.arxiv.org/html/1706.05296)
- CDS (Li, Wang, Wu, Zhao, Yang, Zhang; NeurIPS 2021) [partial]:
  - With shared parameters, "agents tend to acquire homogeneous behaviors because they typically adopt similar actions under similar observations" (stated as motivation, not measured).
  - Fix: "we additionally equip each agent i with an individual local Q-function", Q_i = Q^S + Q^I_i, with an L1 penalty on Q^I that is "critical to achieving a balance between diversity and cooperation". A mutual-information term is added.
  - Ablation: "CDS-All-Shared performs even worse than QPLEX" on academy_counterattack_hard.

  — [ar5iv 2106.02195](https://ar5iv.labs.arxiv.org/html/2106.02195)
- QMIX was not fetched. Its monotonic mixer has the same unidentified-utility structure as VDN, which is from memory and not verified this session.

### Inferences
- **Routed vs dense is the key distinction.** Multi-task PopArt, Liu et al. and MultiCriticAL show that heads stay distinct, or that distinct critics help, when heads see different samples. Their heads differ because their data differ. In a reward-decomposed critic, every head gets a gradient at every sample, so the trunk is shaped by the sum of all heads' errors, weighted by size. This is the setting where dominance by the 100-3000x safety gradient (H1, H4) can turn every head into a readout of safety-shaped features. Evidence from routed critics therefore does not carry over to ours directly.
- **Identified vs unidentified decompositions.** VDN, QMIX and DRDRL need regularisers or structure because only the sum is supervised, so degenerate splits are expected. Our heads each have their own reward and target, so their fixed points are unique in the tabular case (drQ-style convergence). Collapse in our critic is therefore a function-approximation and representation-learning artefact, not an identifiability problem. That makes it a stronger and cleaner finding. MOAC's linear heads on fixed shared features cannot collapse at all, because there is no feature learning. The shared-feature learning is what fails.
- **The on-policy vs off-policy pattern.** Most shared dense multi-head critics reported to work are PPO with GAE: GCR-PPO, Kim et al., PGMT. GAE targets are closer to Monte Carlo than 1-step TD; λ was not verified per paper, and 0.95 is the RSL-RL default.
  - The off-policy exceptions are HDPG, HARC, COP-Q and SAC-D.
  - HDPG (DDPG) works, but with every component reward max-min normalised to (0,1), so scales are equal.
  - HARC and COP-Q have only 2 heads and report no head analysis.
  - SAC-D is the off-policy, 1-step-bootstrapped case that needed a fix (CAGrad).
  - This is consistent with H3 (see Lyle et al. in Q5) and with H4 (scale), but no paper separates the two.
- **GCR-PPO's low quadcopter conflict is ambiguous.** GCR-PPO measured the lowest gradient conflict on the IsaacLab quadcopter (0.15, earlier review) with a shared, unnormalised summed-MSE trunk. Under our findings, a partly collapsed shared critic would also report low conflict. GAE advantages depend mostly on realised per-component rewards, so this effect should be weaker in PPO. This is speculation and should not be cited as evidence.

### Gaps
- MultiCriticAL (multi-head vs multiple critics) was not read in full.
- The PGMT authors were not captured. The Fig. 4 numbers are approximate, read from the text.
- No primary text was fetched for UNREAL, Horde or GVF networks this session. I found no report of interference between auxiliary value heads in them.

## Q4. Preference-conditioned critics: does conditioning on objective weights keep per-objective heads distinct?

### Takeaway
No evidence either way. Envelope Q-learning, Abels et al., PD-MORL, MOMA-AC and D3PO all train one shared network that outputs a per-objective vector conditioned on ω. None measures whether the per-objective outputs stay accurate or distinct. PD-MORL deliberately couples the vector Q to ω through an angle term in the critic loss. Envelope adds a scalarised-utility loss similar to our 5th "total value" loss.

### Cited Findings
- Envelope Q-learning (Yang, Sun, Narasimhan; NeurIPS 2019) [partial]:
  - The input is (s, ω) and the output is |A| × m Q-values.
  - Loss: L = (1−λ)·L^A + λ·L^B. L^A is a vector MSE. L^B is an absolute error between ω⊤y and ω⊤Q, which gives "an auxiliary pull along the direction with better utility". λ is annealed from 0 to 1.
  - Reason for the combination: "the optimal frontier contains a large number of discrete solutions, which makes the landscape of loss function considerably non-smooth".
  - N_ω preferences are sampled per update, hindsight-style.
  - Per-objective accuracy is not analysed.

  — [ar5iv 1908.08342](https://ar5iv.labs.arxiv.org/html/1908.08342)
- Abels, Roijers, Lenaerts, Nowé and Steckelmacher, "Dynamic Weights in Multi-Objective Deep RL" (ICML 2019) [partial]:
  - The Conditioned Network is a UVFA "adapted to output Q-value-vectors conditioned on an input weight vector". The loss is on vector TD errors.
  - Training on shifting weights caused forgetting and bias: the agent "could overfit to the current 𝐰" and "could be biased towards experiences for recent 𝐰's".
  - Fix: Diverse Experience Replay plus training each transition on the current w and "a random previously encountered weight vector". The stated rationale is to make the network "identify that different weight vectors can have different Q-values for the same state".
  - The multi-network baseline "fails to converge when weights regularly change". Per-objective Q distinctness is not measured.

  — [ar5iv 1809.07803](https://ar5iv.labs.arxiv.org/html/1809.07803)
- PD-MORL (Basaklar, Gumussoy, Ogras; ICLR 2023) [full, PDF extracted]:
  - MO-TD3-HER critic: "The critic network takes state s, preference ω, and action a as input and outputs L Q-values for each objective". It has 1 hidden layer of 400 units, Table 5.
  - Preferences: "For each episode during training, we randomly sample a preference vector ... from a uniform distribution", plus 3 hindsight preferences per transition.
  - Target: "target Q-value is calculated using whichever of the two critics gives a smaller ωTQ."
  - Coupling: "we include a directional angle term g(ω,Q(s,a,ω; θ)) to both actor's and critic's loss function … [it] provides an alignment between preferences and the Q-values." Eq. 11 adds g(ω_p, Q) to the critic's MSE.

  — [arXiv 2208.07914v3](https://arxiv.org/pdf/2208.07914v3.pdf)
- MOMA-AC: ω is "sampled uniformly from the unit simplex and held fixed for the episode". The critic is a shared centralised vector-Q network trained with an unweighted vector MSE. The paper has no interference, normalisation or distinctness analysis. — [arXiv 2511.18181](https://arxiv.org/html/2511.18181v1)
- D3PO (earlier review): a PPO critic with a shared body over (s, ω) and d heads, each predicting "the unweighted value of a single objective". It reports policy "representational collapse", not head collapse. — [arXiv 2602.07764](https://arxiv.org/html/2602.07764v1)
- CAPQL (ICLR 2023) was not read this session (OpenReview blocked). It is listed in MORL-Baselines as SAC with a preference-conditioned vector Q. — [MORL-Baselines](https://github.com/LucasAlegre/morl-baselines)

### Inferences
- **Mechanism (my reasoning).** Varying ω could in principle break head collapse. When the policy changes with ω, the per-objective values move in opposite directions: a larger efficiency weight raises Q_eff and lowers Q_prec. The trunk must then carry ω-dependent features with opposite signs across heads, which a collapsed representation cannot fit. No paper tests this.
- **The direction-coupling caveat.** PD-MORL's angle term pulls the critic's vector Q toward the preference direction, and Envelope's L^B pulls it toward scalarised utility. Both actively shape the head vector rather than letting each head follow its reward. They are not evidence for distinct heads, and may work against it.
- **The deployment constraint.** Preference-conditioning the actor adds inputs to the deployed actor, which our constraint forbids. Conditioning only the critic on ω while the actor ignores ω leaves Q_k(s, a, ω) with no ω-dependence to learn, so it brings no benefit. I rank this direction low for us.

### Gaps
- None of the preference-conditioned papers reports per-objective R², per-objective error or head correlation, so the question has no empirical answer in the literature.
- CAPQL's critic details were not verified.

## Q5. Bootstrapping-specific fixes, ensemble-head diversity, and auxiliary value heads

### Takeaway
I found no paper that uses separate Polyak rates, separate learning rates, or a stop-gradient between reward heads of a decomposed critic. The literature offers related tools instead:
- 1/K trunk-gradient scaling (Bootstrapped DQN, SAC-D).
- Per-head target outputs from one shared target network (Bootstrapped DQN, HRA, HARC).
- Diversity regularisers for ensembles (MED-RL on parameter norms, EDAC on the cosine between action-gradients).
- Disentanglement regularisers for unsupervised decompositions (DRDRL, RD²).
- Gradient-similarity gating of auxiliary losses (Du et al.).
- Detached features between value learners at different horizons (Che et al. 2026).

On theory, Lyle et al. (AISTATS 2021) show that TD learning, unlike Monte Carlo, pulls the deviations of K value functions toward the same top-K eigenfunctions of the transition operator, whatever the reward. This is the closest formal account of how bootstrapping can imprint shared structure across heads (H3).

### Cited Findings
- **Lyle, Rowland, Ostrovski, Dabney (AISTATS 2021)** [partial: 100k of 146k chars]:
  - Proposition 3.5 (paraphrased): with K value functions, the span of the TD deviations approaches the span of the top-K eigen-basis functions of P^π. Per the fetch summary, this holds even with zero reward (V^π = 0).
  - "an agent performing TD learning still picks up information about the transition structure of the environment".
  - By contrast, the Monte Carlo trajectory "simply linearly interpolates between V₀ and V^π" and "does not pick up any additional information about the environment in the value function as learning proceeds".
  - With random cumulants, "the distribution of the limiting representation does not collapse". Without such auxiliary signals, "the agent's representation is at risk of collapsing in such cases".

  — [ar5iv 2102.13089](https://ar5iv.labs.arxiv.org/html/2102.13089); [PMLR v130](https://proceedings.mlr.press/v130/lyle21a.html)
- **Per-head targets**:
  - Bootstrapped DQN: each head "is trained against its own target network" (a target head on a shared target torso). — [ar5iv 1602.04621](https://ar5iv.labs.arxiv.org/html/1602.04621)
  - HARC: each head V_k has its own target parameters θ̄_k (Eq. 8, paraphrased from the fetch summary). — [arXiv 2608.09762](https://arxiv.org/html/2608.09762)
  - None of these varies Polyak rates by head (none found).
- **1/K trunk-gradient scaling**: Bootstrapped DQN ("normalizing the gradients by 1/K", best final scores but "slower early learning") and SAC-D ("the gradients of the hidden layers are divided by the size of reward dimension"). — [ar5iv 1602.04621](https://ar5iv.labs.arxiv.org/html/1602.04621); [ar5iv 2206.13901](https://ar5iv.labs.arxiv.org/html/2206.13901)
- **Action-gradient diversity, EDAC** (An, Moon, Kim, Song; NeurIPS 2021) [partial]:
  - "the performance of the learned policy degrades significantly when the Q-functions share a similar local structure."
  - The loss adds, for each member, (η/(N−1)) Σ_{i≠j} of the inner products of normalised ∇_a Q_i and ∇_a Q_j at dataset (s, a) pairs.
  - η sweeps were {0, 1, 5} on Gym and up to 1000 on Adroit.
  - Members have their own parameters; whether layers are shared is not stated.

  — [ar5iv 2110.01548](https://ar5iv.labs.arxiv.org/html/2110.01548)
- **Representation diversity, MED-RL**: CKA-measured collapse of separate ensemble networks, with regularisers on parameter-norm inequality (Flag 3). — [ar5iv 2006.13823](https://ar5iv.labs.arxiv.org/html/2006.13823)
- **Disentanglement for unsupervised decomposition**: DRDRL's KL between the return distributions of the sub-return greedy actions; RD²'s "uniqueness and compactness" of per-sub-reward features (Q3). — [ar5iv 1911.02166](https://ar5iv.labs.arxiv.org/html/1911.02166); [NeurIPS 2020](https://papers.nips.cc/paper/2020/hash/82039d16dce0aab3913b6a7ac73deff7-Abstract.html)
- **Gradient-similarity gating of auxiliary heads**: Du, Czarnecki, Jayakumar, Farajtabar, Pascanu, Lakshminarayanan (arXiv 1812.02224). They minimise "the auxiliary loss as long as its gradient has non-negative cosine similarity with the" main-task gradient, claim convergence to the main task's critical points, and test on Atari among other domains. [snippet] — [ar5iv 1812.02224](https://ar5iv.labs.arxiv.org/html/1812.02224)
- **Detached features across value learners**: Che, Ivanovic, Pavone, "Short-to-Long: Detached Features for Long-Horizon Value Learning" (arXiv 2610.30461, Oct 2026) [abstract]:
  - "Detached connections prevent the behavior value loss from updating the sources, allowing each source to follow its own value objective while the behavior discount remains unchanged."
  - "Fixed-trajectory measurements further show reduced drift in source value targets with mixed discounts."
  - The sources are separate small-discount networks, not heads on one trunk.

  — [arXiv 2610.30461](https://arxiv.org/abs/2610.30461)
- **Stop-gradient on the value-to-torso path (policy–value, not value–value)**: Hao, Yue, Lai et al., arXiv 2610.01630 (Oct 2026) [partial]:
  - X0 lets the value loss update features shared with the actor, "so value-to-torso gradients are generally nonzero". X1 blocks them.
  - Under Adam at scale 4.0, MinEx had maintenance events in 60/60 runs (X0) against 4/60 (X1).
  - The authors caution that direct routing is "neither necessary nor sufficient" across their matrix.

  — [arXiv 2610.01630](https://arxiv.org/html/2610.01630)

### Inferences
- **Lyle et al. applied to our critic (my reasoning, idealised linear theory).** All four heads share the same P^π, because they have the same actor and dynamics. TD learning therefore pushes the shared trunk toward the same leading eigenfunctions of P^π for every head. A narrow trunk (64-64, FAST_TANH) whose fitting is also dominated by the safety error would end up with all four heads reading the same few dominant directions. That predicts high head correlation, which bootstrapped targets then carry forward (H3). Monte Carlo, or targets closer to Monte Carlo (n-step or λ-returns), should weaken the effect. This has not been tested for reward heads.
- **Diversity regularisers transfer poorly.** EDAC, MED-RL and DRDRL push heads apart regardless of what the rewards say. Our heads should differ only as much as their rewards and returns make them differ, and the actor learns only from ∇_a Q_k. Forcing apart the action-gradients (EDAC-style) risks wrong slopes (H2). They are useful as diagnostics (CKA between head-specific features, pairwise cos of ∇_a Q_k), not as fixes.
- **Gradient-similarity gating may make collapse worse.** Du et al.-style gating that drops a head's trunk gradient when it conflicts with the total-value gradient would remove exactly the conflicting signal that keeps heads distinct. It would likely produce collapse rather than prevent it.

### Gaps
- No paper found on per-head Polyak rates, per-head learning rates, or per-head separate target networks in decomposed critics.
- No paper found that applies Lyle et al.'s spectral analysis to reward-decomposed critics or measures head correlation under TD vs Monte Carlo targets.
- UNREAL and Horde were not checked for reported interference between auxiliary heads.
- Osband et al. 2018 (randomised prior functions) and later bootstrapped-ensemble diversity papers were not fetched.

## Q6. Explicit shared-vs-separate critic ablations in multi-objective or multi-task actor-critic

### Takeaway
Few controlled ablations exist, and their results are mixed. They are on-policy (PPO) with normalisation, and none measures head distinctness:
- **Kim et al. (T-RO 2024)**: a multi-head cost critic matches separate cost critics with about 10x fewer parameters.
- **Liu et al. 2026**: routed multihead beats both a shared critic and separate value networks, which learn "much more slowly".
- **Liang et al. 2026 (diffusion)**: separate heads beat a shared head.

No off-policy (TD3/SAC) shared-vs-separate decomposed-critic ablation was found. Our ablation would be the first.

### Cited Findings
- **Kim et al., T-RO 2024 (arXiv 2308.12517)**, 10 constraints, PPO [partial]:
  - Baselines: "Use independent neural networks with no shared parameters for each cost value function", and a reduced version where "The reduced neural network size is applied for each cost value function … to match the total number of trainable parameters with the multi-head method."
  - Results:
    - "The multi-head cost value function requires 0.1 million trainable parameters", against about 1 million for separate networks.
    - Reward: "the policy optimized with the multi-head version yielded a similar reward as the single-head version".
    - Prediction loss: single-head 0.435 ± 0.139, multi-head 0.468 ± 0.120, reduced single-head 0.534 ± 0.205.
    - Compute time: 0.23 s multi-head, 0.56 s single-head, 0.33 s reduced.
  - Conclusion: "This indicates that sharing the neural network backbone is permissible for each cost-value function".
  - Each cost advantage is standardised.

  — [arXiv HTML v4](https://arxiv.org/html/2308.12517v4)
- **Liu et al. 2026**, routed per-environment heads, PPO, Procgen, BipedalWalker and MuJoCo [partial]:
  - Procgen: normalised return rises 21.2% (FiLM) and 40.8% (multihead) over the shared critic.
  - BipedalWalker on unseen terrains: 90.7 (shared), 155.6 (FiLM), 190.4 (multihead).
  - The separate value network "learns much more slowly than FiLM and multihead".
  - "Routing every sample to the same head closely tracks the shared critic".
  - Adding PopArt to multihead "reduces the aggregate normalized training gain on the pinned levels from +24.2% to +10.0%".

  — [arXiv 2608.26481](https://arxiv.org/html/2608.26481v1)
- **Liang, Zhang, Yang, "Explicit Critic Guidance for Aligning Diffusion Models"** (arXiv 2605.27736, May 2026), PPO with 3 rewards [partial]:
  - "separate heads consistently outperform the shared head across all rewards (CLIP, HPSv2.1, and GenEval)", "suggesting that decoupling value prediction reduces interference between heterogeneous objectives".
  - "A naive shared-head design consistently yields slow PPO progress."
  - Table 5: separate vs shared head gives CLIP 0.2896 vs 0.2804, HPSv2.1 0.3441 vs 0.3138, GenEval 0.4986 vs 0.4238.
  - Both variants sit on "a shared latent backbone together with one lightweight value head per reward". The ablation is about sharing at head level, not the trunk.

  — [arXiv HTML 2605.27736](https://arxiv.org/html/2605.27736)
- **PopArt multi-task**: per-task heads without PopArt are slightly worse than single-head IMPALA. With PopArt, the median human-normalised score is 110.7% against 59.7% (clipped), 107.0% against 0.3% (unclipped), and 72.8% against 58.4% on DmLab-30. — [ar5iv 1809.04474](https://ar5iv.labs.arxiv.org/html/1809.04474)
- **Dual vs unified critics, humanoid** (arXiv 2606.11891, earlier review): separate critics against a scalar critic, with the update rule confounded and single unseeded runs. The author notes a controlled ablation isolating the critic change is still needed. — [arXiv 2606.11891](https://arxiv.org/html/2606.11891)
- **Multi-critic vs single-critic in legged robotics** [snippet]:
  - APEX: multi-critic variants kept tracking accuracy across reward scales, while single-critic variants degraded when velocity rewards dominated.
  - CWI: the multi-critic design had higher final reward and faster convergence.
  - Whole-body twist tracking: the single critic learned only locomotion or manipulation, not both.
  - These compare against a scalar critic, not a multi-head critic.

  — [APEX arXiv 2505.10022](https://arxiv.org/html/2505.10022); [CWI arXiv 2606.27676](https://arxiv.org/pdf/2606.27676); [arXiv 2507.08656](https://arxiv.org/pdf/2507.08656)
- **MultiCriticAL (ICLR 2022)** [project page]: separate per-style critics "mitigate[] interference between task-value estimates", with up to 56% gains over single-critic baselines (Papers with Code lists 45%). Whether a multi-head variant was tested could not be verified. — [project page](https://ai.bu.edu/MultiCriticAL); [ICLR 2022](https://iclr.cc/virtual/2022/poster/6568)

### Inferences
- **Kim et al. is the strongest counter-example to "shared fails".** A dense multi-head cost critic was fine in PPO with per-head advantage standardisation and GAE. What differs in our case:
  1. TD3's 1-step bootstrapping.
  2. A rare 400-point terminal penalty sitting in one head.
  3. Our 64-64 trunk against their larger network.
  4. The actor learning from ∇_a Q_k rather than from advantages.

  Any of these could explain why our shared critic fails. Point 1 is directly testable with n-step targets (Q7).
- **Liu et al.'s PopArt result echoes our `--critic-norm-by target` result.** Normalisation made our heads distinct but the policy worse. Normalising targets in a multi-head critic can cost control quality even when heads separate, which is consistent with H1.

### Gaps
- No ablation found for TD3, SAC or DDPG comparing a shared-trunk decomposed critic with separate networks per component (with or without matched parameters).
- None of the ablations reports per-head accuracy or correlation, apart from Kim et al.'s pooled prediction loss.

## Q7. Which literature fixes could transfer to our 4-head TD3 critic (actor size fixed)? Ranked

### Takeaway
The literature gives three fixes that are both principled and cheap to transfer, plus one diagnostic every reviewer will expect:
1. Reduce bootstrapping per head (n-step or Monte Carlo auxiliary targets), grounded in Lyle et al.'s TD-vs-MC result.
2. Asymmetric sharing that gives safety its own network or exclusive control of the trunk. Supported by HRA+1 and RD² (per-head features), safe-RL separate cost critics, Kim et al.'s multi-head cost critic, and Che et al.'s detached features.
3. SAC-D's exact recipe (1/K hidden-layer gradient scaling + CAGrad c = 0.5 + composite twin min), run as the literature baseline.

The diagnostic is to compare head correlation against the correlation of per-group Monte Carlo returns. Full per-head normalisation (PopArt), diversity regularisers and preference conditioning are poor fits, because of evidence already in hand and our constraints.

### Cited Findings
- TD dynamics imprint transition structure common to all heads, while Monte Carlo does not (Proposition 3.5). — [ar5iv 2102.13089](https://ar5iv.labs.arxiv.org/html/2102.13089)
- SAC-D's recipe for a shared multi-head critic: hidden-layer gradients ÷ m, CAGrad c = 0.5 on per-component losses, twin chosen by composite Q. CAGrad assigns lower weight to higher-loss components. — [ar5iv 2206.13901](https://ar5iv.labs.arxiv.org/html/2206.13901)
- 1/K trunk scaling gives the best final scores in Bootstrapped DQN, but slower early learning. — [ar5iv 1602.04621](https://ar5iv.labs.arxiv.org/html/1602.04621)
- Evidence for restricting what each head shares:
  - HRA+1 (per-head input masking): large gains. — [ar5iv 1706.04208](https://ar5iv.labs.arxiv.org/html/1706.04208)
  - RD² (minimal distinct features per sub-reward). — [NeurIPS 2020](https://papers.nips.cc/paper/2020/hash/82039d16dce0aab3913b6a7ac73deff7-Abstract.html)
  - CDS (shared Q plus an L1-regularised individual Q per agent). — [ar5iv 2106.02195](https://ar5iv.labs.arxiv.org/html/2106.02195)
- Safety or cost value as its own learner: CoMOGA uses separate cost critics. — [arXiv 2403.00282](https://arxiv.org/html/2403.00282v2)
- COP-Q puts reward and safety on one trunk with two heads. — [arXiv 2606.04749](https://arxiv.org/html/2606.04749)
- A shared multi-head cost critic works in PPO with standardised per-head advantages. — [arXiv 2308.12517v4](https://arxiv.org/html/2308.12517v4)
- Detached (stop-gradient) feature transfer between value learners. — [arXiv 2610.30461](https://arxiv.org/abs/2610.30461)
- PopArt-style normalisation equalises task influence. — [ar5iv 1809.04474](https://ar5iv.labs.arxiv.org/html/1809.04474)
- PopArt can reduce control gains in a multihead critic. — [arXiv 2608.26481](https://arxiv.org/html/2608.26481v1)
- Diversity regularisers (EDAC, MED-RL) push estimators apart regardless of targets. — [ar5iv 2110.01548](https://ar5iv.labs.arxiv.org/html/2110.01548); [ar5iv 2006.13823](https://ar5iv.labs.arxiv.org/html/2006.13823)

### Inferences
All items below are my proposals, grounded in the sources above, for the 4-head TD3 critic. The actor is unchanged.

0. **Diagnostic, needed before any claim** (about 30 lines, no training).
   - On held-out test rollouts, compute per-group discounted Monte Carlo returns G_k(s_t). Report corr(G_i, G_j) alongside the head-value correlations of the split and shared critics, plus per-head R² against G_k.
   - Optionally, compute linear CKA between the hidden features that each head's gradient uses, as MED-RL does.
   - This turns "values are 0.93-0.99 correlated" into "values exceed the true return correlation by X", and rules out the resolvent-amplification explanation (Q1 Inferences).
   - Addresses H3. Risk: none.
1. **Reduce bootstrapping per head** (n-step or λ-returns, or a Monte Carlo auxiliary loss).
   - Store per-group rewards and use n-step per-group targets (n of about 5-20), or add an auxiliary regression of each head toward its stored per-group discounted episode return. The total value and the actor stay as in TD3.
   - Basis: Lyle et al.'s TD-vs-MC contrast. The shared multi-head critics reported to work with unequal scales (GCR-PPO, Kim et al., PGMT) all use GAE-based PPO targets, which are closer to Monte Carlo. The off-policy HDPG works only with equal-scale, max-min-normalised rewards.
   - Addresses H3, and partly H4.
   - Cost: about 50-120 lines for replay changes. Compute is unchanged.
   - Risk: n-step off-policy bias. It is small here because TD3's exploration std is 0.1 and decays. Higher target variance from the rare crash could hurt the safety head, which can be mitigated by applying n-step targets only to time, precision and efficiency.
   - Expected effect: lower head correlation, pulled toward the Monte Carlo correlation. Whether control improves is uncertain, given the repeated observation that distinct, accurate values have still produced crashing policies.
2. **Asymmetric sharing that respects H1** (safety gets its own trunk). Two variants:
   - (a) Safety is a separate 64-64 network, and time, precision and efficiency share one 64-64 trunk with three heads. That is about 13k parameters against 26k for the split critic.
   - (b) One trunk is trained only by the safety loss (or by the total loss), and the other heads read stop-gradient trunk features through a small private layer each, in the style of Che et al.'s detached features and CDS's shared-plus-individual design.

   This keeps safety's ownership of the representation, which the H1 results say matters, while ending its domination of the other heads' features. Our earlier towers experiment shared the first layer among all four groups, including safety, so variant (a) is untested.
   - Addresses H1 and H4.
   - Cost: about 40-80 lines for (a); (b) mostly needs `detach()` calls.
   - Risk: the remaining three-head trunk could still collapse. The time, precision and efficiency scales differ by about 5x rather than 100-3000x, so dominance should be milder.
3. **SAC-D's recipe as the literature baseline**: hidden-layer (trunk) gradients ÷ 4 instead of our output scaling, CAGrad c = 0.5 inside the critic, and the composite twin min we already use.
   - We already tested CAGrad c = 0.5 (12.3% crashes, still collapsed). The untested piece is SAC-D's trunk-gradient scaling in place of our output scaling, which slowed head learning.
   - Reviewers will expect SAC-D as the shared-critic comparison, and our results suggest it does not prevent collapse at our scale gap. That is a publishable negative result, with the caveat that SAC-D never measured head distinctness.
   - Cost: about 5-10 lines.
   - Risk: low. Expected effect: TD3-like step size without slowing the heads, but dominance stays.
4. **Partial normalisation** (my proposal; no literature found). Divide each head's error by σ_k^α with α between 0 and 1, or apply PopArt to time, precision and efficiency only while keeping safety in real units.
   - This sits between "no normalisation" (collapse) and "PopArt" (distinct heads but crashes 18-56%; Liu et al. also saw PopArt reduce gains).
   - Addresses H1 and H4. Cost: about 10 lines.
   - Risk: α is a tuned hyperparameter, and reviewers may call it ad hoc unless α is fixed a priori, for example α = 0.5 as a geometric mean.
5. **Per-head feature masks** (HRA+1, RD²): learned or fixed masks over trunk units per head, so each head reads a sparse subset.
   - Cost: moderate (about 50 lines, plus a mask-sparsity loss).
   - Risk: masks can converge to the same units unless a uniqueness term like RD²'s is added, which brings back a regulariser.
6. **Not recommended**:
   - Diversity or decorrelation regularisers (EDAC action-gradient cosine, MED-RL, DRDRL KL). They force disagreement whatever the returns say, which risks wrong slopes (H2).
   - Gradient-similarity gating (Du et al.). It removes the conflicting signal.
   - Preference conditioning (Envelope, Abels, PD-MORL, MOMA-AC). It would change the deployed actor's inputs, there is no evidence it keeps heads distinct, and PD-MORL's angle term couples heads to ω.

### Gaps
- None of the ranked directions 1-5 has been tested in the literature on a reward-decomposed TD3 or SAC critic, so their expected effects are first-principles estimates.
- How Lyle et al.'s linear and continuous-time results carry over to a 64-64 FAST_TANH trunk is unknown.
- The claim from the search snippet that "with dense rewards, sharing the trunk is beneficial, while with sparse rewards the opposite is true" (attributed to Moalla et al., "No Representation, No Trust", NeurIPS 2024, [arXiv 2405.00662](https://arxiv.org/abs/2405.00662)) could not be verified from the abstract. Do not cite it until the full text is checked.
