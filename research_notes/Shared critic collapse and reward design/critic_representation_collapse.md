# Representation collapse and plasticity loss in RL critics, and whether they explain multi-head value collapse in a shared critic

Reading status. I did not read any paper cover to cover. Tags: **[P]** = I extracted the full text and read the passages I cite (definitions, method, main results, and the appendix tables where named). **[S]** = I read it only through a web summariser, so confidence is lower. **[NR]** = I cite it only through another paper; I did not read it. Results from DQN/Atari, from on-policy methods (PPO, IMPALA) and from supervised learning carry those labels. Everything else is off-policy continuous-control actor-critic (SAC, TD3).

Facts about our network come from the local code, [l2f/py/networks.py](/Users/khushpatel/code/CrazyFlie-2.1/l2f/py/networks.py):
- The critic is (privileged obs 31 + action 4) -> 64 -> 64 -> outputs, with FAST_TANH on the hidden layers and a linear output.
- FAST_TANH clips x to [-3, 3], then computes y = x(27 + x^2)/(27 + 9x^2). Its backward pass is dy/dx = 1 - y^2.
- At x = 3, y = 3·36/108 = 1 exactly. A unit whose pre-activation satisfies |z| >= 3 therefore outputs +-1 and passes exactly zero gradient.
- The actor also uses FAST_TANH on every layer. `SplitCritic` builds 4 independent `Critic` nets. `TowerCritic` shares only the first layer.

---

## 1. Implicit under-parameterisation (srank collapse, Kumar et al., ICLR 2021) and DR3 (Kumar et al., ICLR 2022): mechanism, metric, multi-head effects, fixes

### Takeaway
- **What it is.** Bootstrapped TD training with gradient descent lowers the effective rank (srank) of a Q-network's penultimate features. Supervised regression to fixed targets does not do this.
- **Where it was shown.** The paper shows it for SAC on Gym (online and offline) as well as for DQN.
- **What it predicts for us.** Its kernel-regression theory has a direct consequence for a shared trunk (my extrapolation, below): every head's value is the same, increasingly low-rank operator applied to that head's reward. That alone predicts that heads become near-collinear, while separate critics would not.
- **Evidence for the fixes.** I found no paper that tests srank or DR3 with multiple heads. The rank penalty (Kumar 2021) and DR3 have evidence only in offline RL (Atari DQN/CQL; D4RL CQL), not in online TD3/SAC.

### Cited Findings
**srank metric** — [Kumar et al. 2021, ICLR, arXiv 2010.14498](https://arxiv.org/abs/2010.14498) [P]
- Definition: srank_δ(Φ) = min{ k : Σ_{i=1..k} σ_i(Φ) / Σ_{i=1..d} σ_i(Φ) >= 1 - δ }, with δ = 0.01.
- Φ is the penultimate-layer feature matrix, computed on "a minibatch of state-action pairs sampled i.i.d. from the training data" (the replay buffer when online). The σ_i are taken in decreasing order.
- Interpretation: srank near d means the network maps states to near-orthogonal features. A small srank means state-action pairs are "aliased" into a small subspace.

**Empirical findings** — [Kumar et al. 2021](https://arxiv.org/abs/2010.14498) [P]
- Offline: after an initial learning period, srank falls in every domain tested (gridworld, Atari, Gym). In Atari, "only 20-100 singular components are active for 512-dimensional features". Policy performance generally falls as srank drops.
- Supervised regression to Q* "results in a much higher srank" and "does not cause rank collapse". The collapse needs bootstrapping.
- Online (DQN on Atari, SAC on Gym): with more gradient steps per environment step (n), srank drops more aggressively and earlier, keeps falling with more training, and performance is worse.
- Capacity: "even high-capacity value networks behave as low-capacity networks as more training is performed with a bootstrapped objective".

**Theory** — [Kumar et al. 2021](https://arxiv.org/abs/2010.14498) [P]
- Model: kernel-regression fitted Q-iteration. The update is Q_{k+1} = G(cI+G)^{-1} y_k with y_k = R + γP^π Q_k. Unrolling gives Q_k = A Σ_i (γP^π A)^{k-i} R = M_k R, where A = G(cI+G)^{-1}.
- Theorem 4.1: assume S = γP^π A is normal. Then along an infinite increasing sequence of fitting iterations, the ratio of any two singular values of M_k shrinks, so srank_δ(M_k) does not increase. If S is positive semi-definite, srank "continuously decreases in each fitting iteration".
- A second analysis (deep linear networks under gradient flow) yields an implicit regulariser "that encourages disproportionate singular values" of the feature weights.

**Rank-penalty fix** — [Kumar et al. 2021](https://arxiv.org/abs/2010.14498) [P]
- Penalty: L_p(Φ) = σ_max²(Φ) - σ_min²(Φ), computed by SVD on the minibatch features and added to the TD loss with weight α = 0.001.
- Effect on rank: in the gridworld, rank no longer collapses (it plateaus). On offline Atari, rank rises.
- Effect on performance (offline Atari, 5% data): DQN improves on 16/16 games (median +74.5%); CQL improves on 11/16 (median +14.1%).
- I did not find continuous-control results for the penalty in the passages I read.

**DR3 mechanism ("feature co-adaptation")** — [Kumar et al. 2022, ICLR, arXiv 2112.04716](https://arxiv.org/abs/2112.04716) [P]
- Definition: when value functions are trained by bootstrapping, "the feature representations of consecutive state-action pairs exhibit a large value of the dot product φ(s,a)^T φ(s',a')".
- The large dot product can come from high cosine similarity or from high feature norms. It appears "even when there is no explicit objective to increase feature similarity".
- The authors derive it from the implicit regulariser of TD learning under SGD noise. It co-occurs with oscillatory learning dynamics.

**DR3 regulariser and results** — [Kumar et al. 2022](https://arxiv.org/abs/2112.04716) [P]
- Practical form: R_exp(θ) = Σ_{i∈D} φ(s_i,a_i)^T φ(s'_i,a'_i), on last-layer features only (the simplification Σ*_M = I). The total loss is L = L_ALG + c0·R.
- Coefficient: c0 = 0.001 for REM and 0.03 for CQL on Atari, chosen from {0.001, 0.01, 0.03, 0.3} on 5 games.
- D4RL with CQL (offline): antmaze-medium-play-v2 goes from 24.0 to 73.0 and antmaze-large-diverse-v2 from 1.33 to 25.67 (2M steps).
- All DR3 experiments are offline (Atari, D4RL, robotic manipulation from images). I found no online TD3/SAC result.

**Rank and performance are only loosely linked** — [Gulcehre et al. 2022, arXiv 2207.02099](https://arxiv.org/abs/2207.02099) [P; offline, DQN-family, Atari]
- "The rank and the performance are correlated in restricted settings, but that correlation disappears when we increase the range of hyperparameters and the models."
- An auxiliary loss (CURL), SAM, or a change of activation function "can increase the effective rank but does not necessarily improve the performance".
- The correlation is strong in some settings, e.g. DQN with ReLU and many learning steps over a fixed dataset.
- The PDF I read is labelled "Preprint". I did not verify the venue.

**Multiple heads or auxiliary tasks**
- [Kumar et al. 2021](https://arxiv.org/abs/2010.14498) [P] has no multi-head analysis. Its conclusion suggests auxiliary losses "that learn useful features while passively preventing underparameterization".
- Section 4 gives the evidence that extra heads raise rank, and the shared-operator view of TD auxiliary heads ([Lyle et al. 2021](https://arxiv.org/abs/2102.13089)).

### Inferences
- **The shared-operator argument (my extrapolation of Theorem 4.1, not a published result).**
  - In Kumar's kernel model, a shared trunk means one shared kernel G for all heads, and so one shared M_k. Each group g then gets Q_k^(g) = M_k R^(g).
  - As M_k's spectrum sparsifies, M_k ≈ σ1 u1 v1^T. Then every head's value approaches σ1 (v1^T R^(g)) u1: a multiple of the same vector u1. So |corr(Q^(g), Q^(h))| tends to 1 whatever the reward correlation.
  - With 4 separate critics, each group has its own kernel G_g, hence its own A_g and M_k^(g) with its own dominant direction, so the values need not align.
  - This matches four of our observations: (i) collapse happens under bootstrapping; (ii) the TD targets are as correlated as the values (0.93-0.99); (iii) the 142-unit critic still collapses (Kumar: under-parameterisation is not a capacity limit); (iv) the split critic tracks the reward structure.
  - Caveats: the theorem assumes a fixed kernel and a normal S. In a real network the kernel changes during training, and the shared kernel is shaped mostly by the head with the largest error (safety), which would pull u1 toward safety-relevant directions.
- **True values may correlate too.** The true values also share one operator, (I - γP^π)^{-1}, so they can be more correlated than the rewards even without any collapse. The split critic's value correlations (+0.30..+0.75) against reward correlations of -0.08..+0.39 fit this. The shared critic's 0.93-1.00 goes beyond it.
- **Which rank fix matches the mechanism.** Kumar's L_p flattens the singular spectrum of the shared features, so it acts directly on the mechanism above. DR3 targets a different pathology (similarity of consecutive features) and has only offline evidence.
- **α must be re-tuned for our scale.** α = 0.001 was set for Atari losses. Our TD loss is in real units, with safety TD errors in the hundreds, so a plain α = 0.001 would probably have no effect.
- **Test before fixing.** If this mechanism is at work, the shared trunk's srank, or the head-weighted feature variance (Section 5), should fall as head correlation rises. The PopArt-like `--critic-norm-by target` critic, whose values stay distinct, should keep a higher srank.

### Gaps
- No paper I found measures srank (or any rank) of a multi-head or vector-valued TD critic, or tests whether rank collapse worsens with the number of heads or with differences in head scale.
- I did not find continuous-control results for L_p, or online-RL results for DR3, in the passages I read.
- The OpenReview discussions of either paper were not checked.

---

## 2. Capacity loss, plasticity loss, primacy bias and resets, dormant neurons and ReDo: mechanisms, metrics, fixes, and tanh vs ReLU

### Takeaway
- **Where the evidence sits.** Most of it comes from high update-to-data regimes or from explicit non-stationarity, and the RL results are mainly DQN/Atari.
- **Off-policy continuous-control evidence:**
  - Resets help SAC on DMC (IQM 501 to 656).
  - Dormant neurons grow in the SAC actor and critic, but ReDo gives no gain on MuJoCo.
  - Large network-regulariser studies (Section 3) find resets the most robust fix at high replay ratio.
- **Saturating activations.** The literature treats a tanh unit stuck in its constant region as a "dead" unit (Dohare; Lyle 2024). ReDo's dormancy score does not catch it: a saturated tanh has |h| = 1, so it scores as highly active.
- **Most relevant mechanism.** Lyle et al. 2024 find that a large regression-target magnitude alone causes plasticity loss in a dose-response way. This bears directly on a trunk shared with a safety head whose targets are about 50x larger in spread.

### Cited Findings
**Capacity loss and feature rank** — [Lyle, Rowland, Dabney 2022, ICLR, arXiv 2204.09560](https://arxiv.org/abs/2204.09560) [P; DQN/Atari]
- Capacity is measured as a network's ability to fit new random target functions: the MSE after a fixed number of steps.
- Feature rank: ρ̂_n(φ, X, ε) = |{σ ∈ SVD(φ(X)/√n) : σ > ε}|, evaluated with n = 5000. Unlike srank, it is "not normalize[d] by the maximal singular value", so it also catches collapse "where the network features, and thus also their singular values, converge to zero".
- Agents: DDQN, QR-DQN, and DQN with a random-cumulant auxiliary task (RC-DQN). Rainbow checkpoints fall into two clusters: "a sufficiently high feature rank thus appears to be a necessary condition" for learning progress, but it is "not sufficient".
- Feature rank is higher with non-trivial prediction tasks; e.g. RC-DQN has higher rank in Montezuma's Revenge.
- InFeR fix: L = E_{x~B}[ Σ_{i=1..k} (g_i(x;θ) - β·g_i(x;θ0))² ], i.e. k auxiliary heads regress onto β times their own outputs at initialisation. Settings: k = 10, β = 100, weight α = 0.1.
- InFeR results: a net improvement over Rainbow on 57 Atari games, with significant gains where agents are below human level. On DDQN it helps "slightly on average", significantly only on sparse-reward games. It raises feature rank (App. C.2).

**Plasticity and curvature** — [Lyle et al. 2023, ICML, arXiv 2303.01486](https://arxiv.org/abs/2303.01486) [P]
- Plasticity loss "is deeply connected to changes in the curvature of the loss landscape, but ... it often occurs in the absence of saturated units".
- Feature rank, the number of saturated units, the parameter norm and weight-matrix rank "do not identify robust causal relationships". The sign of their correlation with plasticity depends on the task.
- [DQN/Atari] Adding layer normalisation after each hidden layer of double DQN (57 games, 3 seeds each, RMSProp) "robustly improves performance across the benchmark, without any additional hyper parameter tuning". The largest gains are in games where the default network's gradient covariance is degenerate or its Hessian ill-conditioned.

**Independent mechanisms and target magnitude** — [Lyle et al. 2024, "Disentangling the causes of plasticity loss", arXiv 2402.18762](https://arxiv.org/abs/2402.18762) [P; preprint]
- Plasticity loss "can be decomposed into multiple independent mechanisms":
  - pre-activation distribution shift, which causes dead units, "linearization" ("zombie") units and degraded signal propagation;
  - parameter-norm growth;
  - regression-target magnitude.
- Target magnitude: "pretraining on large offsets significantly reduces the network's ability to learn on new targets", with a dose-response relation to the offset size. LayerNorm "can mitigate this trend, it does not completely eliminate it, particularly in the case of centered fine-tuning targets".
- They also find "target magnitude alone can explain a significant fraction of previous examples of plasticity loss in deep reinforcement learning".
- On tanh: a dead unit can be "a tanh [with] very large preactivation values that take it into its 'constant' region". A non-saturating activation such as Leaky ReLU fixes dead units, but pre-activation shift can still cause pathologies.
- Fix: LayerNorm plus weight decay "is highly effective at maintaining plasticity" in synthetic non-stationary tasks and in RL on the Arcade Learning Environment [DQN/Atari].

**Normalisation and the effective learning rate** — [Lyle et al. 2024, "Normalization and effective learning rates in RL" (NaP), arXiv 2407.01800](https://arxiv.org/abs/2407.01800) [P, abstract only; the version read is marked "Preprint. Under review"]
- Normalisation layers make parameter-norm growth equivalent to a decay in the effective learning rate, which can fall "to near zero too quickly".
- Normalize-and-Project adds weight projection to keep the effective learning rate constant. Evaluated on synthetic plasticity benchmarks and on single-task and sequential ALE [DQN/Atari].

**Loss of plasticity in continual learning** — [Dohare et al., arXiv 2306.13812, "Maintaining Plasticity in Deep Continual Learning"; published as "Loss of plasticity in deep continual learning", Nature 2024](https://arxiv.org/abs/2306.13812) [P on the arXiv version; the Nature page could not be fetched]
- Effective rank: erank(Φ) = exp(-Σ_k p_k log p_k), with p_k = σ_k/||σ||_1. It is continuous and ranges from 1 to rank(Φ).
- Dead units (ReLU): units outputting zero for all of 2,000 random examples, counted at the start of each task. For tanh or sigmoid, the analogue counts units within ε of either extreme value ("saturated"). The ε value is not given.
- [Supervised, Online Permuted MNIST] Dead units grow (up to 25% after 800 tasks at step size 0.01), the mean weight magnitude grows, and the effective rank falls.
- Adam and dropout "significantly exacerbate loss of plasticity"; L2 and Shrink & Perturb reduce it.
- [On-policy PPO] Slippery-Ant: PyBullet Ant whose friction is resampled log-uniformly from [1e-4, 1e4] every 10M steps; policy and value nets 2×256 with tanh, trained with Adam.
  - PPO "degraded rapidly".
  - Continual PPO (continual backprop), PPO + Shrink & Perturb and PPO + L2 all do "substantially better". Continual PPO is best, with "a minimal drop" over 100M steps.

**Primacy bias and resets** — [Nikishin et al. 2022, ICML, arXiv 2205.07802](https://arxiv.org/abs/2205.07802) [P]
- Primacy bias: early data dominates training. In the "heavy priming" test, SAC is heavily updated on its first 100 transitions and then "is unable to recover".
- SAC reset protocol: "reset agent's networks entirely every 2×10^5 steps", including both Q-networks, the target networks and the optimiser statistics. The replay buffer is kept.
- DMC results (10 seeds per task): SAC IQM 501 rises to 656 with resets; DrQ 569 to 762.
- Results are "the best results over different values of replay ratio and n". Resets enable training at higher replay ratios and longer n-step targets.
- [Atari/SPR] Reset only the final linear layer every 2×10^4 steps: IQM 0.478 on Atari 100k.

**Dormant neurons and ReDo** — [Sokar et al. 2023, ICML, arXiv 2302.12902](https://arxiv.org/abs/2302.12902) [P]
- Score: s_i^ℓ = E_{x∈D}|h_i^ℓ(x)| / ( (1/H^ℓ) Σ_k E_{x∈D}|h_k^ℓ(x)| ). A neuron is τ-dormant if s_i^ℓ <= τ.
- Thresholds: the analyses use τ = 0. ReDo itself uses τ = 0.1 ("better ... than using a threshold of 0 or 0.025").
- DQN settings: τ = 0.025 by default and 0.1 otherwise; recycling period 1,000; score minibatch 64.
- SAC settings: τ = 0, recycling period 200,000, actor/critic 2×256, lr 3e-4, batch 256, score minibatch 256.
- ReDo algorithm: for each dormant neuron, "reinitialize input weights" and "set outgoing weights ... to 0".
- [DQN] The fraction of dormant neurons correlates strongly with replay ratio.
- SAC on MuJoCo (HalfCheetah, Hopper, Walker2d, Ant; 10 seeds):
  - Dormant neurons increase over training in both the actor and the critic.
  - Pruning them every 200k steps "does not affect the performance".
  - ReDo "maintains the performance", while Reset and weight decay "cause a performance drop in most cases". The authors hypothesise no gain because "the state space is considerably low and the typically used network is sufficiently over-parameterized".
- Activations: "RL networks typically use ReLU activations, which can saturate at zero outputs, and hence zero gradients". With a different activation (App. C.1, not read) there is "a mild decrease in the number of dormant neurons, but the phenomenon is still present".

**Dormancy as a gradient property** — [He 2026, "Rethinking Plasticity in Deep Reinforcement Learning", arXiv 2603.21173](https://arxiv.org/abs/2603.21173) [S; preprint labelled "an incomplete version"]
- Dormancy metrics are descriptive; "the absence of gradient signals is the primary driver of dormancy".
- The paper proposes gradient-based plasticity metrics: a mean absolute gradient intensity and a zero-gradient index.
- Setting: PPO on HalfCheetah, ReLU networks only.

**Saturation in supervised nets** — [Glorot & Bengio 2010, AISTATS](https://proceedings.mlr.press/v9/glorot10a.html) [P; supervised]
- Sigmoid "can drive especially the top hidden layer into saturation". Saturated units "can move out of saturation by themselves, albeit slowly", which explains training plateaus.
- "A new non-linearity that saturates less can often be beneficial."

### Inferences
- **Our update ratio is low.** We run 0.1 critic updates per environment step (about 2×10^5 critic updates in 2M steps). That is 10x below standard SAC (1 update per step) and far below the high-ratio settings where resets, ReDo and rank collapse matter most (Kumar: more steps per env step means faster collapse; Sokar: dormancy tracks replay ratio; Nikishin and Nauman: resets matter most at high replay ratio). The split critic runs on the same schedule and works. So generic plasticity loss is an unlikely primary cause of the head collapse.
- **The collapse is too early for slow plasticity loss.** The head-scaled critic reaches correlation 1.00 by 200k env steps, only about 18k critic updates after warm-up. That timing points to optimisation geometry (scale dominance plus the shared operator of Section 1), not slow degradation.
- **Target magnitude does not depend on the update ratio.** The Lyle 2024 mechanism is a function of target scale. It predicts that a trunk fitted to the safety head's large targets (crash penalty 400; TD-target std 152 vs 3.2 for time) loses the ability to fit the small, roughly centred targets of the other heads. LN reduces but does not remove this.
  - Caveat: their setting is sequential (pretrain, then fine-tune); ours trains all heads at once.
- **FAST_TANH saturation is a dead-unit pathology that ReDo misses.**
  - A unit with |z| >= 3 on every sample is a constant ±1. It acts as an extra bias for the next layer, carries no variance, and lowers the centred feature rank.
  - A unit saturated on some samples behaves like a step function of the input.
  - ReDo's score uses mean |h|, so saturated units score high (≈1 × the layer mean or more) and are never flagged. Dohare's "within ε of ±1" count, or a zero-gradient index as in He 2026, is the right detector. For FAST_TANH, zero gradient is exactly |z| >= 3.
- **Why the large head may push the trunk into saturation.** tanh features are bounded (|h_j| <= 1), so |Q_k| <= ||w_k||_1 + |b_k|. A head whose values span hundreds needs output weights with L1 norm in the hundreds, or features pinned at ±1. Under Adam at lr 1e-3, weights grow by about 1e-3 per step, so in early training the cheapest route to large outputs may be driving the trunk into saturation. This is my hypothesis and is untested; Section 5 has the measurement.
- **The critic's activation can be changed freely.** Only the actor is deployed, so the deployment constraint does not bind the critic. The evidence says that switching to ReLU or Leaky ReLU removes the saturation mode but not the other plasticity mechanisms (Sokar; Lyle 2024; Dohare). LayerNorm (Section 3) is the better-supported generic fix.
- **Resets in our regime.** Evidence for resets at our low update ratio is weak. Head-scaled collapse re-forms within about 200k env steps, so periodic resets may only reset the clock. Nikishin notes performance drops right after a reset, which here would mean crash spikes.

### Gaps
- No paper I found measures tanh or sigmoid saturation in an off-policy continuous-control critic. All dormancy and reset studies of SAC use ReLU.
- Lyle 2022's ε value for feature rank was not in the passages I read. Later papers often use 0.01, but I did not verify this.
- Dohare does not give the ε for tanh saturation, and I did not verify the outcome of the activation sweep (sigmoid, tanh, ELU, Leaky ReLU, ReLU, Swish).
- How ReDo handles Adam moments, and which alternative activation Sokar tested (App. C.1), were not read.
- The Nature version of Dohare et al. (2024) could not be fetched. Details that appear only there, such as the Adam β settings, were not checked.

---

## 3. Normalisation in critics (LayerNorm, BatchRenorm/CrossQ, SimBa, BRO, spectral normalisation): measured effects on TD3/SAC stability and representation rank

### Takeaway
- **Strong evidence for LayerNorm and spectral norm.** In SAC (and in one TD3 variant), critic LayerNorm and spectral normalisation are among the most effective cheap stabilisers. They reduce value divergence and overestimation, and they enable critic scaling.
- **Resets** are the most robust fix only at high replay ratio.
- **Effect on rank is unmeasured.** I found no paper that reports how these normalisations change srank or dormancy in TD3/SAC critics. Nauman et al. 2024 compute such proxies but report correlations, not per-method rank values.
- **The one multi-head, scale-imbalanced shared critic** in this literature (TOPPO, on-policy PPO) found that LayerNorm in the critic reduces the co-linearity of per-task critic gradients.
- **All of this evidence uses ReLU.**

### Cited Findings
**Large-scale regulariser study** — [Nauman et al. 2024, ICML, "Overestimation, Overfitting, and Plasticity in Actor-Critic: the Bitter Lesson of RL", arXiv 2403.00514](https://arxiv.org/abs/2403.00514) [P]
- Setting: SAC, "14 diverse tasks from 2 simulat[ion benchmarks]" (DeepMind Control and Meta-World).
- Headline result: "general neural network regularizers significantly outperform most RL-specific algorithmic improvements". Layer normalisation "is more effective in reducing overestimation than techniques specifically designed for mitigating Q-value overestimation". "Replacing ... Clipped Double Q-learning with network regularization techniques leads to significant performance gains."
- Recommendations by method:
  - "Periodical network resetting is the most robust intervention across two benchmarks in a high replay ratio regime."
  - "Layer norm is essential for Dog environments."
  - Layer norm is "generally recommended for DMC, while spectral norm is more effective for MW"; "across a diverse range of tasks ... spectral norm [is] more robust than layer norm."
  - "Weight decay has generally low performance when used alone with SAC."
  - "When using network or plasticity regularization, critic regularization leads to reduced performance."
- Metrics: plasticity proxies are % dormant neurons (Sokar), representation rank (Kumar), gradient norm and parameter norm. Overestimation is measured as approximation error; overfitting as the ratio of validation to training TD error. Relations are analysed with Spearman correlations.
- Interactions: "interventions motivated by a specific problem ... can have a pronounced impact on the metrics associated with overestimation or plasticity."

**BRO** — [Nauman et al. 2024, "Bigger, Regularized, Optimistic", arXiv 2405.16158](https://arxiv.org/abs/2405.16158) [P]
- BroNet architecture: dense -> LayerNorm -> ReLU, then residual blocks with LayerNorm (N = 2, about 5M parameters). "Layer Norm ... is essential to unlocking scaling."
- It also uses weight decay and full-parameter resets.
- Evaluated on 40 tasks (DMC, Meta-World, MyoSuite, Dog and Humanoid). The largest ablation effects come from scaling critic capacity, from the replay ratio, and from removing Clipped Double Q.
- I did not verify the venue in the text I read.

**RLPD** — [Ball et al. 2023, ICML, "Efficient Online RL with Offline Data", arXiv 2302.02948](https://arxiv.org/abs/2302.02948) [P]
- Claim: LayerNorm in the critic bounds Q-values. With LN (without bias) on the representation ψ and ReLU features, ||Q(s,a)|| <= ||w||·||relu(ψ)|| <= ||w||, so "Q-values are bounded by the norm of the weight layer, even for actions outside the dataset".
- "Introducing LayerNorm into the critic greatly improves performance through mitigating critic divergence." This is SAC-style online RL with offline data.

**CrossQ** — [Bhatt et al. 2024, ICLR, arXiv 1902.05605](https://arxiv.org/abs/1902.05605) [P]
- Changes: removes target networks, uses Batch (Re)Normalization in the critic "applied in a particular manner", and uses wider critic layers, all at UTD = 1.
- Results: it matches or beats REDQ and DroQ (which use UTD 20) with "20× fewer gradient steps than REDQ" and up to 4x less wall-clock time.
- "CrossQ (TD3)" applies the same changes to TD3. On Humanoid and HumanoidStandup, "CrossQ and CrossQ (TD3) both substantially surpass all baselines."

**SimBa** — [Lee et al. 2025, ICLR, arXiv 2410.09754](https://arxiv.org/abs/2410.09754) [P]
- Components: (i) an observation-normalisation layer "that standardizes inputs with running statistics", (ii) pre-LN residual feedforward blocks, (iii) a final LayerNorm "to control feature magnitudes".
- SAC with SimBa "matches or surpasses state-of-the-art" on DMC, MyoSuite and HumanoidBench. Scaling a plain MLP's parameters degrades SAC; scaling SimBa improves it.

**Unit-ball feature normalisation** — [Hussing et al. 2024, RLC/RLJ, "Dissecting Deep RL with High Update Ratios: Combatting Value Divergence", arXiv 2403.05996](https://arxiv.org/abs/2403.05996) [P]
- Setting: SAC; the authors state the findings "extend to other methods such as TD3".
- Finding: at high update ratios, "overinflated Q-values are found not only on out-of-distribution but also in-distribution data and can be linked to overestimation on unseen action prediction propelled by optimizer momentum".
- Fix: projecting features onto the unit sphere "decouples learning the scale of the values from the first layers of the network and moves it to the last linear layer". It "fully mitigates diverging Q-values in the priming experiment" and performs strongly on dm_control, including the dog tasks.

**Spectral normalisation in deeper critics** — [Bjorck, Gomes, Weinberger 2021, NeurIPS, "Towards Deeper Deep RL with Spectral Normalization", arXiv 2106.01151](https://arxiv.org/abs/2106.01151) [P]
- Naively using large modern networks in actor-critic agents (SAC, DrQ) is unstable. Dataset size is "not the limiting factor": "instability from taking gradients through the critic is the culprit".
- Spectral normalisation, applied "to each linear layer except the first and last" in the setup read, enables stable training of large networks with significant gains on DMC.

**Spectral normalisation as optimisation** — [Gogianu et al. 2021, ICML, "Spectral Normalisation for Deep RL: an Optimisation Perspective", arXiv 2105.05246](https://arxiv.org/abs/2105.05246) [P, in part; DQN/C51 Atari]
- Normalising only the penultimate layer (SN[-2]) lifts C51 beyond Rainbow.
- SN "almost never degrades the performance of the baseline" and "extends the usable range of Adam's hyper-parameters" (a sweep of 14,000 DQN agents).
- The paper links the benefit to how SN changes the optimiser step. I read the optimiser-ablation section only in part.

**LayerNorm in a shared multi-task critic** — [Li, Lin, Qu, Miao 2026, "TOPPO: Rethinking PPO for Multi-Task RL with Critic Balancing", arXiv 2605.11473 (v3, 7 Nov 2026)](https://arxiv.org/abs/2605.11473) [S, parts read; on-policy PPO, V-critic, Meta-World+ MT50]
- LayerNorm in the shared critic ("LN-c") reduces "representation-level co-linear collapse in the per-task Gram matrix": the mean off-diagonal |cos| falls from 0.34 to 0.20 in early-stage diagnostics.
- Per the summariser, the Gram matrix is built from per-task critic gradients.

### Inferences
- **Most transferable cheap fix: LayerNorm on the critic's hidden pre-activations, before FAST_TANH.**
  - Evidence: SAC (Nauman 2024; BRO; RLPD; SimBa), DQN (Lyle 2023/2024), and the multi-task shared critic in TOPPO.
  - For tanh specifically (my derivation): when the LN gain is about 1 and its bias about 0, the 64 normalised pre-activations of each sample have mean 0 and mean square 1. By Markov's inequality on z², at most 1/9 of the units can sit at |z| >= 3 for any sample. LN therefore caps hard saturation per sample however large the safety head's targets are. The learnable gain can undo this, so monitor it or apply weight decay to it.
  - Caveat: none of the papers tested LN with tanh in an off-policy critic.
- **Unit-ball normalisation of the 64-d penultimate features (Hussing) is the fix that best matches H1/H4 by mechanism.** It moves the value scale into the head weights, so the safety head cannot use the trunk's feature norm to reach its large values.
  - Caveat: |Q_k| <= ||w_k|| + |b_k| when ||φ|| = 1. Safety (spread about 152) then needs ||w_safe|| in the hundreds, which Adam at lr 1e-3 reaches only slowly.
  - A fixed per-head output multiplier c_k fixes this without changing the trunk gradient at a given function (my reasoning). The multiplier reparameterises the head; the trunk gradient Σ_k e_k·w_k stays the same.
- **Spectral normalisation is the most relevant option for H2 (action-gradient quality).** It bounds the critic's Lipschitz constant, and so dQ/da. Bjorck locates the instability in "taking gradients through the critic", and Nauman finds SN the most robust across diverse tasks.
  - Risk: it may smooth away the sharp crash-value cliff that the actor needs to feel.
- **CrossQ is the only normalisation result shown on TD3.** Removing target networks changes the bootstrapping loop (H3).
  - Cost and risk: BatchRenorm and the joint (s,a)/(s',a') batch are more code, and BN statistics enter the actor's dQ/da.
- **Check the critic inputs first.** SimBa's running-statistics observation normalisation is a reminder: if any of the critic's 31 privileged inputs or 4 actions are not standardised, layer 1 can saturate regardless of the heads.

### Gaps
- No paper I found reports srank, feature rank or dormancy before and after LayerNorm or SN in a TD3/SAC critic. Nauman 2024 computes these proxies but reports their correlations, not values per method.
- No test of LN, SN or CrossQ in a multi-head or decomposed off-policy critic.
- TOPPO's exact Gram-matrix definition and its ablations (LN-c alone, PopArt alone) were not in the parts I read (about 200k of 246k characters, via a summariser).
- SimbaV2 (hyperspherical normalisation, arXiv 2502.15280) was not checked.

---

## 4. Evidence on a shared representation trained by TD with several value heads of very different scales (multi-task critics, auxiliary value heads)

### Takeaway
- **Large-scale heads dominate.** Multi-task RL consistently reports that the tasks with the largest reward or value magnitude dominate the updates of a shared network:
  - Hessel et al. 2019, on-policy IMPALA;
  - TOPPO 2026, on-policy PPO, per-task critic gradient norms about 497x apart, "co-linear collapse";
  - Lyle 2024, regression-target magnitude.
- **Per-task normalisation (PopArt) is the standard fix.** This matches our finding that target-std normalisation removes the head collapse.
- **Balanced auxiliary heads raise rank.** Auxiliary TD heads of comparable scale tend to increase feature rank (InFeR, random-cumulant DQN, Lyle 2021).
- **What the literature lacks.** All of this assumes tasks of equal importance. I found no work on a decomposed critic whose dominant head is also the one the policy's objective (the real-unit sum) depends on most. That is the trade-off behind our H1.

### Cited Findings
**PopArt in multi-task actor-critic** — [Hessel et al. 2019, "Multi-task Deep RL with PopArt", arXiv 1809.04474](https://arxiv.org/abs/1809.04474) [P; on-policy IMPALA actor-critic, Atari]
- Learners "can get distracted by certain tasks ... [that] appear more salient ... because of the density or magnitude of the in-task rewards", focusing "on those salient tasks at the expense of generality".
- "Even with reward clipping, in a multi-task setting the magnitude of updates can still differ significantly between tasks."
- The magnitude of updates depends on "1) the scale of rewards, 2) the sparsity of rewards, 3) the competence of the agent", which are non-stationary, so updates cannot be normalised a priori.
- PopArt makes the update invariant to these factors. One agent reached a 110% median human-normalised score on 57 games; the earlier multi-task IMPALA reached a 59.7% median (49.4% mean).

**Scale imbalance in a shared multi-task critic** — [Li et al. 2026, TOPPO, arXiv 2605.11473](https://arxiv.org/abs/2605.11473) [S; on-policy PPO, Meta-World+ MT50, shared 400×3 MLP with per-task PopArt heads; preprint, venue not stated]
- Imbalance: "a few easier tasks learn fast and produce large critic gradients", and harder tasks must "fit their value functions through shared features optimized primarily for other tasks". Per-task critic gradient norms span about 497x, versus about 4.1x for actor gradients after per-task advantage normalisation. "The critic fits unnormalized value targets unless explicitly corrected."
- Fix: TOPPO combines PopArt, LayerNorm in the critic, FairGrad (α = 1) critic-gradient aggregation and PCGrad on the actor. It reaches 90.9 ± 1.6% mean and 56.5 ± 7.8% worst-10 success on MT50.
- PopArt's marginal gain is +17.0 IQM points on V1 rewards but +0.3 on V2.
- FairGrad added to LN-c alone lowered V1 IQM from 64.3 to 62.0.
- Critic-only LN (96.8%) beat LN on both actor and critic (93.1%, about 3x wider seed spread) on MT10.

**Scaling multi-task critics** — [McLean et al. 2025, "Multi-Task RL Enables Parameter Scaling", arXiv 2503.05126](https://arxiv.org/abs/2503.05126) [S; preprint]
- Setting: multi-task multi-head SAC on Meta-World with one-hot task IDs.
- "Critics benefit more significantly from increased model capacity in multi-task settings."
- Dormant neurons increase with parameter count on MT10, but there is "almost no indication of plasticity loss" on MT50. "Large models trained on a large number of tasks exhibit much lower rates of dormant neurons." The threshold is not stated, and no normalisation layers are used.

**Theory of auxiliary heads under TD** — [Lyle et al. 2021, AISTATS, "On the Effect of Auxiliary Tasks on Representation Dynamics", arXiv 2102.13089](https://arxiv.org/abs/2102.13089) [P, abstract and introduction]
- For auxiliary tasks such as ensemble predictions and random cumulants trained under TD, the paper proves "convergence of the induced representations to subspaces defined by certain decompositions of the environment's transition operator".
- Random-cumulant auxiliary tasks perform strongly in sparse-reward Atari.

**Auxiliary heads raise rank** — [Lyle et al. 2022](https://arxiv.org/abs/2204.09560) [P; DQN/Atari]
- A random-cumulant auxiliary head (RC-DQN) gives higher feature rank.
- 10 auxiliary heads regressing onto scaled initial outputs (InFeR) preserve feature rank and capacity.

**Large target offsets hurt other targets** — [Lyle et al. 2024](https://arxiv.org/abs/2402.18762) [P]
- Pretraining on large target offsets impairs later fitting of other targets, including centred ones. LayerNorm reduces the effect but does not eliminate it.

### Inferences
- **The literature reproduces our collapse fix.** Per-task scale normalisation removes the collapse (Hessel; TOPPO), as our `--critic-norm-by target` run did (values -0.98..+0.93, R² 0.97-0.99).
- **The literature does not predict our crashes.** In multi-task RL every task is equally important by construction, so equalising their influence is the goal. In our decomposition the actor optimises the real-unit sum, so the safety head's dominance of the trunk is partly legitimate: it is also the head whose action-gradient matters most. This is consistent with H1. I found no paper that studies this trade-off, which supports a novelty claim around it.
- **Extra heads are not the problem; imbalanced heads are.** Heads with similar scales tend to raise feature rank (InFeR, RC-DQN). A 4-head trunk becomes effectively a single-task trunk when one head's error is 100-3000x larger, and Section 1's shared-operator argument then makes the other heads collinear readouts of that head's features.
- **Rebalancing the trunk gradient is not a promising next step.** TOPPO's critic-gradient balancing (FairGrad) is close to our `--critic-agg` experiments, which crashed 7.7-95% in our task. TOPPO also found FairGrad alone did not help without PopArt.
- **Width alone is not enough.** McLean's "critics benefit from capacity" fits the general picture, but our 142-unit shared critic still collapsed. That matches Kumar's point that under-parameterisation is implicit, not a capacity limit.

### Gaps
- I found nothing on TD-learned multi-head critics with heads of very different scales in off-policy continuous control. Multi-task SAC papers (McLean) do not discuss reward-scale imbalance.
- The Hybrid Reward Architecture and decomposed-critic papers belong to another researcher's scope and were not checked here.
- The original PopArt paper (van Hasselt et al. 2016, arXiv 1602.07714) [NR] was not read. Its output-preserving rescaling (the "Pop" part) matters if our `--critic-norm-by target` variant rescales only the losses.

---

## 5. Cheap diagnostics for our critic (feature srank, saturated or dormant units, per-head effective rank) and what values indicate collapse

### Takeaway
- **Cost.** Every metric below needs one forward pass over a few thousand replay samples, plus one SVD of a 5000×64 matrix. That takes milliseconds.
- **No absolute thresholds.** The literature uses these metrics comparatively: Kumar compares against supervised regression, Lyle 2022 found two clusters, and Gulcehre showed the link to performance depends on the setting. Our split critic, and the PopArt-like critic that stayed distinct, are the natural references.
- **Saturation needs its own detector.** For FAST_TANH, the right saturation metric is the fraction of samples with |z| >= 3. ReDo's τ-dormant score cannot see saturation.

### Cited Findings
**Definitions to reuse exactly:**
- srank_δ(Φ), δ = 0.01, on penultimate features of a minibatch of replay (s,a) pairs — [Kumar et al. 2021](https://arxiv.org/abs/2010.14498) [P]. DR3 also plots srank with δ = 0.01 — [Kumar et al. 2022](https://arxiv.org/abs/2112.04716) [P].
- Feature rank: the number of singular values of Φ/√n above ε, with n = 5000 and no normalisation by σ_max — [Lyle et al. 2022](https://arxiv.org/abs/2204.09560) [P].
- Effective rank: erank = exp(-Σ p_k log p_k), p_k = σ_k/||σ||_1 — [Dohare et al.](https://arxiv.org/abs/2306.13812) [P].
- Saturated units in tanh nets: units within ε of either extreme on the sample, which is the tanh analogue of ReLU dead units counted over 2,000 examples — [Dohare et al.](https://arxiv.org/abs/2306.13812) [P].
- τ-dormant: s_i = E|h_i| / mean_k E|h_k| <= τ, with τ = 0 for SAC, and 0.025 or 0.1 for DQN (0.1 for ReDo) — [Sokar et al. 2023](https://arxiv.org/abs/2302.12902) [P].
- Feature co-adaptation: φ(s,a)^T φ(s',a'), and its cosine version — [Kumar et al. 2022](https://arxiv.org/abs/2112.04716) [P].
- Gradient-based plasticity metrics (zero-gradient index) — [He 2026](https://arxiv.org/abs/2603.21173) [S].
- Proxy set used for SAC: % dormant, representation rank, gradient norm, parameter norm — [Nauman et al. 2024](https://arxiv.org/abs/2403.00514) [P].
- Gradient-cosine Gram matrix across tasks in a shared critic — [TOPPO](https://arxiv.org/abs/2605.11473) [S].

**How to read the values:**
- Atari: 20-100 active components out of 512 is the paper's example of "significant underutilization" — [Kumar et al. 2021](https://arxiv.org/abs/2010.14498).
- High feature rank is necessary but not sufficient for learning progress — [Lyle et al. 2022](https://arxiv.org/abs/2204.09560).
- The rank-performance link varies with hyperparameters and architecture — [Gulcehre et al. 2022](https://arxiv.org/abs/2207.02099).

### Inferences
**Recipe (my own proposal).** Use N = 4096-5000 replay (s,a) pairs, and separately (s, π(s)) pairs, since the actor sees the critic at the policy's actions. Apply it to the online critic and the target critic, at the checkpoints where head correlation is already logged (200k, 600k, 2M). Run it on: the shared critic, each split-critic net, the 142-unit shared critic, and the `--critic-norm-by target` critic.

1. **Saturation (FAST_TANH-exact).** For each hidden unit j in layers 1 and 2:
   - sat_j = mean_n 1[|z_nj| >= 3]. This is exactly the share of samples on which the unit passes zero gradient.
   - Report the share of fully dead units (sat_j = 1), the histogram of sat_j, and the mean gradient gain mean(1 - h²) per layer.
   - Also report the std of z_j per unit. A very low std marks Lyle 2024's "linearised" units. No threshold is published, so compare across critics.
2. **Feature rank of the layer-2 output Φ (N×64).** Report srank_0.01 on raw Φ (comparable to Kumar). Also report srank_0.01 and erank on column-centred Φ: constant (saturated) columns inflate σ1 of raw Φ, and centring isolates the variance that the heads can actually use. Add Lyle's feature rank with ε = 0.01 as a secondary measure.
3. **Head-level collapse (shared critic).**
   - (a) Compute the 4×4 correlation matrix of the head outputs and its effective rank, erank of its eigenvalues, which runs from 1 (full collapse) to 4.
     - Worked example with equal pairwise correlations: corr 0.95 gives erank ≈ 1.22, while corr 0.45 (a split-critic-like value) gives ≈ 3.10.
   - (b) Head read-out concentration. Take the eigen-decomposition (λ_j, u_j) of the centred feature covariance, and compute f_k = λ_1 (w_k·u_1)² / Σ_j λ_j (w_k·u_j)², the share of head k's output variance carried by the top feature direction.
     - f_k near 1 for all four heads means the heads read the same trunk direction. That is the signature of the shared-operator mechanism (Section 1) and of safety-dominated features (H1/H3).
     - If srank is high but f_k ≈ 1, the problem is how the heads read the features, not missing rank.
4. **Per-head trunk-gradient Gram matrix.** Compute the cosines and norms of the 4 per-head loss gradients on the trunk parameters. We already have the 100-3000x norm ratios; TOPPO tracked the mean off-diagonal |cos|.
5. **DR3 co-adaptation.** mean φ(s,a)·φ(s',a') and the mean cosine, with a' from the target policy plus smoothing noise.
6. **Norms.** Weight norms per layer, and the norm of each head's output weights. These test the "large head forces saturation or large weights" hypothesis of Section 2. NaP links norm growth to effective-learning-rate decay.

**Heuristic flags (my own, not from the literature; calibrate against the split critic):**
- more than about 10% fully saturated units in a layer;
- layer-2 centred srank_0.01 falling while the split nets' srank stays flat;
- head-correlation erank below 1.5;
- f_k > 0.9 for every head.

**Experiment that separates bootstrapping (H3) from scale dominance (H1/H4)** (my own proposal; it reuses the existing offline fixed-target pipeline):
- Fit the shared critic to frozen targets taken from the split critic, which are decorrelated, and track head correlation and the metrics above. Under a fixed target, Kumar's result says rank should not collapse.
  - If the heads stay distinct, the collapse needs the bootstrapping loop.
  - If they collapse anyway, scale dominance in the trunk is sufficient on its own.
- Then repeat with LayerNorm before FAST_TANH, or with unit-ball penultimate features, to see whether they change saturation, f_k and collapse.

### Gaps
- No published threshold says what srank, saturation fraction or head-correlation rank counts as "collapsed" for a 64-unit continuous-control critic.
- Lyle 2022's ε and Dohare's tanh ε were not given in the passages I read.
- I found no published per-head effective-rank metric for multi-head critics. Items 3a-3b are my proposals.
