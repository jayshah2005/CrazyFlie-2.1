# Critic action-gradient (dQ/da) quality in DPG / DDPG / TD3: theory, measurement, and fixes for our multi-head TD3 critic

Scope: does an accurate critic value imply an accurate action-gradient? How is dQ/da quality measured, and which methods train critics for good dQ/da? Applied to the 4-head TD3 critic in `l2f/docs/shared_critic_research_prompt.md` (hypothesis H2: accurate values, wrong slopes).
Reading status, given per source: **[FT]** full text processed (HTML, or text extracted from the PDF, mostly through a summarising fetch tool); **[PT]** part of the full text; **[ABS]** abstract, landing page or search snippet only. "venue†" means the venue comes from the standard citation and was not re-checked on a proceedings page in this session. Everything under "Inferences" is my own reasoning or proposal, unless a source is named.

## 1. Theory: what condition must the critic's action-gradient satisfy (compatible function approximation), and what happens when it does not?

### Takeaway
The deterministic policy gradient uses the critic only through ∇_a Q at a = μ_θ(s). Silver et al.'s compatibility conditions are stated entirely in terms of the critic's action-gradient, and the critic's state-value part is arbitrary. MAGE turns this into a bound: the policy-gradient error is at most L_π/(1−γ) times the critic's expected action-gradient error. So value accuracy matters only through its action-derivative. When compatibility fails, the update need not be an ascent direction at all.

### Cited Findings
- **DPG theorem** (Silver et al., ICML 2014, PMLR v32): ∇_θ J(μ_θ) = E_{s∼ρ^μ}[ ∇_θ μ_θ(s) ∇_a Q^μ(s,a)|_{a=μ_θ(s)} ]. The off-policy form takes the expectation over s ∼ ρ^β. ∇_θ μ_θ(s) is the n×m Jacobian (n policy parameters, m action dimensions). — [Silver et al. 2014, §3-4.3, PDF text extracted, §4.3 read in full [PT]](https://proceedings.mlr.press/v32/silver14.pdf)
- **Theorem 3, near-verbatim:** "A function approximator Q^w(s,a) is compatible with a deterministic policy μ_θ(s), ∇_θ J_β(θ) = E[∇_θ μ_θ(s) ∇_a Q^w(s,a)|_{a=μ_θ(s)}], if (1) ∇_a Q^w(s,a)|_{a=μ_θ(s)} = ∇_θ μ_θ(s)^T w and (2) w minimises the mean-squared error MSE(θ,w) = E[ε(s;θ,w)^T ε(s;θ,w)] where ε(s;θ,w) = ∇_a Q^w(s,a)|_{a=μ_θ(s)} − ∇_a Q^μ(s,a)|_{a=μ_θ(s)}." The proof needs only E[∇_θ μ_θ(s) ε(s;θ,w)] = 0. — [Silver et al. 2014 [PT]](https://proceedings.mlr.press/v32/silver14.pdf)
- **Compatible form:** for any deterministic policy there is a compatible critic Q^w(s,a) = (a − μ_θ(s))^T ∇_θ μ_θ(s)^T w + V^v(s), "where V^v(s) may be any differentiable baseline function that is independent of the action a". The first term is interpreted as the advantage A^w(s,a) = φ(s,a)^T w, with features φ(s,a) = ∇_θ μ_θ(s)(a − μ_θ(s)). — [Silver et al. 2014 [PT]](https://proceedings.mlr.press/v32/silver14.pdf)
- **What happens when it does not hold** (quote): "In general, substituting an approximate Q^w(s,a) into the deterministic policy gradient will not necessarily follow the true gradient (nor indeed will it necessarily be an ascent direction at all)." — [Silver et al. 2014, §4.3 [PT]](https://proceedings.mlr.press/v32/silver14.pdf)
- **Local critic:** a linear critic "is not very useful for predicting action-values globally, since the action-value diverges to ±∞ for large actions. However, it can still be highly effective as a local critic". It "is sufficient to select the direction in which the actor should adjust its policy parameters." Condition 2 is a linear regression from the features φ to the targets ∇_a Q^μ, but "acquiring unbiased samples of the true gradient is difficult". In practice the authors learn w by policy evaluation (COPDAC-Q: δ_t = r_t + γQ^w(s_{t+1}, μ_θ(s_{t+1})) − Q^w(s_t,a_t); θ ← θ + α_θ ∇_θ μ_θ(s_t)(∇_θ μ_θ(s_t)^T w); w ← w + α_w δ_t φ(s_t,a_t); a gradient-TD variant is COPDAC-GQ). — [Silver et al. 2014, §4.3, Eqs. 19-21 [PT]](https://proceedings.mlr.press/v32/silver14.pdf)
- **MAGE Proposition 3.1** (D'Oro & Jaśkowski, NeurIPS 2020†). For L_π-Lipschitz differentiable deterministic policies and any p-norm: ‖∇_θ J(θ) − ∇̂_θ J(θ)‖ ≤ (L_π/(1−γ)) E_{s∼d^π_μ} ‖∇_a δ^{π,Q̂}(s,a)|_{a=π(s)}‖, where δ^{π,Q̂} = Q^π − Q̂ is the evaluation error. The value error itself does not appear, only its action-gradient. MAGE then uses the TD error as a stand-in for the evaluation error. — [MAGE, §3.1 [FT, ar5iv]](https://ar5iv.labs.arxiv.org/html/2004.14309)
- MAGE §3.1: "minimizing the TD-error does not guarantee that the critic will be effective for the goal of solving the control problem". §1: in standard DPG "the critic learns action-value gradients implicitly". — [MAGE [FT]](https://ar5iv.labs.arxiv.org/html/2004.14309)
- Saglam & Kalogerias (arXiv 2024, revised 2025) argue that classical compatibility needs the critic's gradient to be linear in the policy Jacobian, which deep critics violate. This leaves "modern successors to DPG technically unsound", and they state that "accurately approximating function values" does not necessarily imply accurate gradient approximation. — [Compatible Gradient Approximations for Actor-Critic Algorithms [FT]](https://arxiv.org/html/2409.01477)
- **Time-step effect** (Tallec, Blier & Ollivier, ICML 2019†), Theorem 2: Q^π_δt(s,a) = V^π_δt(s) + O(δt). The rescaled advantage A_δt = (Q_δt − V_δt)/δt has a non-degenerate limit. With function approximation, the error of Q must therefore be much smaller than O(δt) to rank actions correctly (the fetch tool's paraphrase of the paper's argument). — [Tallec et al., §3.2 [PT, first 100k of 111k chars]](https://ar5iv.labs.arxiv.org/html/1901.09732)
- Fujimoto et al. (ICML 2022) find that the Bellman error "is a poor proxy for the accuracy of the value function". Errors can cancel, and with finite data "the Bellman error can be minimized without improving the accuracy of the value function". — [Why Should I Trust You, Bellman? [ABS]](https://arxiv.org/abs/2201.12417)

### Inferences
- **Value accuracy is neither necessary nor sufficient for a correct DPG step.** It is not necessary because V^v(s) is arbitrary in Theorem 3 and absent from MAGE's bound. It is not sufficient because two critics with the same values on the data can differ arbitrarily in ∇_a Q at π(s).
- **Multi-objective extension.** Each group k has its own true Q^π_k (the same π, reward r_k), so the DPG theorem and MAGE's bound apply per head. With g_k = E_s[∇_θ π(s) ∇_a Q̂_k(s,π(s))] and slope error ε_k(s) = ∇_a Q̂_k − ∇_a Q^π_k:
  - ‖ĝ_k − g_k‖ ≤ (L_π/(1−γ)) E_s‖ε_k(s)‖.
  - The **sum** aggregator needs only E_s[∇_θ π(s) Σ_k ε_k(s)] ≈ 0, so per-head slope errors that cancel across heads do no harm. This is the compatibility condition for the scalar reward.
  - **Non-linear aggregators** (UPGrad, PCGrad, CAGrad, MGDA, Aligned-MTL, SVD/OS-NMF-PGA) need each g_k to be accurate. Scale-invariant ones (PGA) need each g_k's *direction*: sin∠(ĝ_k, g_k) ≤ ‖ĝ_k − g_k‖/‖g_k‖.
  - So a critic can be good enough for "sum" yet misleading for every other aggregator. Small groups (time, precision) are hit hardest: their ‖g_k‖ is small, the trunk loss is dominated by safety (100-3000×), and PGA gives them the largest weight. This matches "PGA with total loss: per-group slopes −0.26 to 0.60 (some reversed)", and it would explain why per-head critic changes hurt the non-sum aggregators most.
- **R² ≈ 0.99 is nearly blind to slopes in our regime.** Write Var(y_k) = Var_s E[y_k|s] + E_s Var(y_k|s). A critic that ignores the action entirely loses only E_s Var_{a∼replay}[Q_k(s,a)] of explained variance. To first order that is ≈ σ_a² E‖∇_a Q_k‖², where σ_a is the within-state action spread in replay (exploration 0.1, decaying ×0.9 per 100k from 500k to ≈0.02 by 2M).
  - At σ_a = 0.02, the action-dependent part stays below 1 − R² = 0.01 of Var(y_k) whenever ‖∇_a Q_k‖ < 5·std(y_k) per unit action. For safety (std 152) that is a slope below ~760 per unit action.
  - Put differently: at R² = 0.99 the residual RMS is 0.1·std(y_k), i.e. ≈ 0.3 (time), 0.8 (precision), 1.7 (efficiency) and 15 (safety, about 4% of the 400-point crash penalty). Within one state, the critic's action-dependent variation over the replay's action spread can be smaller than that.
  - The R² values reported in the problem are against each head's own TD target (`critic_snapshot.py` docstring), which is a Bellman-error-type fit. By Fujimoto et al. 2022, that is not even proof of value accuracy against true per-group returns.
- **Slopes are weakly identified by off-policy data.** In COPDAC the slope parameters w are learned only from the action deviations (a − μ(s)), i.e. from exploration (Silver et al., Eq. 21 features). The slope information per transition therefore scales with σ_a. At σ_a ≈ 0.02, ∇_a Q at π(s) is set mostly by the inductive bias of the network (shared features) rather than by the data. This fits the frozen-data result that even "sum" changed the slopes substantially (cosine 0.74-0.79) while R² stayed at 0.985-0.996: there is a large set of slope changes that leave the values unchanged.
- **Small action effect per step.** By Tallec et al., the action's effect on Q in one control step is O(δt). The action-dependent signal each head must learn is a small residual on top of large state-dependent values. In a shared trunk, those state-dependent values are dominated by safety (std 152).

### Gaps
- I did not read Silver et al.'s supplementary proofs or the COPDAC-GQ convergence discussion beyond §4.3.
- No paper found states a compatibility result for vector-valued critics or for non-linear gradient aggregators. The per-head extension above is my derivation.
- MAGE's venue (NeurIPS 2020) and Tallec et al.'s (ICML 2019) are from the standard citations. The fetched pages did not state them.

## 2. Methods that train or regularise the critic for good action-gradients, and which fit our 4-head TD3 critic

### Takeaway
Three families exist:
1. **Supervise the slope with a model:** MAGE's action-gradient-of-TD-error loss, Sobolev training, value-gradient learning, GProp's gradient network.
2. **Avoid needing the slope:** Gaussian-smoothed or zeroth-order actor gradients (CPG, Soft-DPG), and advantage decompositions that isolate the action-dependent part (DAU).
3. **Regularise the critic's action geometry:** gradient penalties, spectral normalisation, PAVE.

Plain magnitude penalties and spectral normalisation either did little or collapsed TD3 in the one comparison found. MAGE shows that penalising slope error without a value term degenerates to ∇_a Q ≈ 0.

### Cited Findings
- **MAGE critic loss** (penalised form, Eq. 8): minimise ‖∇_a δ̂^{π_θ,Q_φ}(s,a,ŝ)|_{a=π_θ(s)}‖ + λ|δ̂^{π_θ,Q_φ}(s,a,ŝ)|, where δ̂(s,a,ŝ) = r(s,a) + γ Q̃(ŝ, π(ŝ)) − Q̃(s,a) and ŝ ∼ p̂(·|s,π(s)) comes from a learned Gaussian ensemble model (8 members), differentiated by reparameterisation. The reward is assumed known and differentiable. The second-order gradients are with respect to actions. λ = 0.2 (§4.1) versus λ = 0.05 (App. D.2); the paper is internally inconsistent here. — [MAGE, §3.1-3.2, App. C-D [FT]](https://ar5iv.labs.arxiv.org/html/2004.14309)
- MAGE results: "at least as fast as all the baselines" on MuJoCo/Gym tasks in the first 10^5 steps, with 5 runs. Dyna-TD3 (same model, standard TD loss) showed no consistent gain, which isolates the gain to the gradient loss. On HalfCheetah, MAGE-TD3 reaches 9660.79 ± 2821.31 at 3·10^6 steps versus TD3 9636.95 ± 859.07 at 10^7 steps. — [MAGE, §4.1, App. B.5 [FT]](https://ar5iv.labs.arxiv.org/html/2004.14309)
- **MAGE warnings:** with λ = 0 (slope loss only), training "rapidly" reaches a degenerate solution with ‖∇_a Q̂‖ ≈ 0 (App. B.1). The second-order gradient roughly doubles training time compared with Dyna-style policy gradients (§6). The model adds bias (§3.1). — [MAGE [FT]](https://ar5iv.labs.arxiv.org/html/2004.14309)
- **Sobolev training** (Czarnecki et al., NeurIPS 2017†): loss Σ_i [ℓ(m(x_i|θ), f(x_i)) + Σ_j ℓ_j(D^j_x m(x_i|θ), D^j_x f(x_i))]. Jacobians are matched through projections onto random unit vectors. Theorem 2: if the target derivative g ≠ f′, no C¹ function can be close to both f and g. Experiments: regression benchmarks (lower test error on 6 of 7 functions), Atari policy distillation (matching projections of ∇_s log π) and synthetic gradients. There was **no RL-critic experiment, and no separate derivative-error metric**. — [Sobolev Training [FT, ar5iv]](https://ar5iv.labs.arxiv.org/html/1706.04859)
- **Value-gradient learning** (Fairbank & Alonso, arXiv 2011): learning the value *gradient* along a greedy trajectory is "a sufficient condition for the trajectory to be locally extremal". Policy-gradient learning on a greedy policy gives a weight update equivalent to a value-gradient update. The paper targets deterministic continuous control with smooth function approximators. — [Fairbank & Alonso 2011 [ABS]](https://arxiv.org/abs/1101.0428)
- **GProp / deviator-actor-critic** (Balduzzi & Ghifary, arXiv 2015): "a temporal-difference based method for learning the gradient of the value-function". Three networks estimate the value, its gradient and the policy. GProp was competitive with supervised methods on a contextual bandit task and achieved the best result to date on the octopus arm. — [Balduzzi & Ghifary 2015 [ABS]](https://arxiv.org/abs/1509.03005)
- **VaGraM** (Voelcker et al., ICLR 2022†) weights *model* errors by the value function's state-gradient to fix the "objective mismatch" between model fit and control. It is about value-aware model learning, not about critic action-gradients. — [VaGraM [ABS]](https://arxiv.org/abs/2204.01464)
- **Deep Advantage Updating** (Tallec et al.): Q_Θ(s,a) = V_θ(s) + δt·A_ψ(s,a), with A_ψ(s,a) = Ā_ψ(s,a) − Ā_ψ(s,π(s)) so that A_ψ(s,π(s)) = 0. V and A get separate updates scaled by δQ/δt, and learning rates η = α·δt^β with β = 1. DDPG was "not at all robust to changes in time discretization" on Pendulum, Cartpole, BipedalWalker, Ant and Half-Cheetah. — [Tallec et al., §4-5 [PT]](https://ar5iv.labs.arxiv.org/html/1901.09732)
- **Compatible Policy Gradient (oCPG)** (Saglam & Kalogerias) replaces ∇_a Q̂ with a two-point Gaussian-smoothing estimate: ∇̂J = (1/(1−γ)) E_{s,u}[∇_θ π_θ(s) · ((Q_ψ(s,a+μu) − Q_ψ(s,a))/μ) · u]|_{a=π_θ(s)}, with u ∼ N(0,I), μ = 0.1, and the first TD3 critic.
  - Theorem 1 bound: B/(1−γ)·[(ε_{μ,ψ}/μ)√p + E‖∇Q_μ − ∇Q‖]. For G-smooth Q this becomes B√p/(1−γ)·[ε/μ + Gμ]. Here ε is the critic's *value* error at perturbed actions.
  - Results (10 seeds): Ant 4225 → 5285 and HalfCheetah 6745 → 7957 vs TD3; statistically best on 3 of 6 tasks.
  - The authors "do not explicitly claim" more accurate gradients; there is no alignment measurement. — [CPG [FT]](https://arxiv.org/html/2409.01477)
- **Soft-DPG** (Na & Lee, arXiv May 2026): a σ-smoothed Bellman operator Q_σ(s,a) = R(s,a) + γE[Q_σ(s′, π(s′)+σw)], which is a γ-contraction, with actor gradient E[(1/σ²)∇_θ π_θ(s)(ã − π_θ(s))Q_σ(s,ã)], ã ∼ N(π_θ(s), σ²I). No dQ/da is used. Bias bound: ‖Q^π − Q_σ^π‖_∞ ≤ γL_Qσ√m/(1−γ).
  - Mixed results: large gains on discrete-reward variants and on continuous Ant and InvertedDoublePendulum, but large losses on continuous HalfCheetah (4275.58 → 188.81) and Humanoid (2496.36 → 437.65).
  - Its only gradient diagnostic is qualitative. — [Soft-DPG [PT, first 100k of 128k chars]](https://arxiv.org/html/2605.06228)
- **PAVE** (Lee, Kwak, Kim, Hwang, arXiv Jan 2026): critic-only losses L = L_TD + λ1·L_MPR + λ2·L_VFC + λ3·L_Curv.
  - L_MPR = E‖∇_a Q(s+ε,a) − ∇_a Q(s,a)‖² with ε ∼ N(0,σ²I); its expectation is σ²‖∇²_sa Q‖²_F.
  - L_VFC = E‖∇_a Q(s_t,a_t) − ∇_a Q(s_{t+1},a_t)‖².
  - L_Curv = E[max(0, vᵀ∇²_aa Q v + δ)] with Rademacher probes v and δ = 1.0.
  - Requires C² critics (SiLU).
  - TD3 results: the best smoothness on all 6 tasks (e.g. Hopper 2.715 → 0.950), return kept or improved on 5 of 6 (Walker 4834 → 5563, Reacher −3.55 → −4.02). — [PAVE [FT]](https://arxiv.org/html/2601.22970)
- **PAVE appendix, critic-side baselines on TD3** (Lunar/Walker):
  - Spectral normalisation of the critic "collapsed the policy": return −970 and 14.
  - A plain gradient penalty λ‖∇_a Q‖² ("GP-Critic") gave 202/5011 vs Base 228/4834. No λ in 0.01-10 beat Base smoothness on Lunar, and λ = 10 collapsed Walker (return 484).
  - PAVE: 264/5563. — [PAVE, appendix [FT, offset 100k]](https://arxiv.org/html/2601.22970)
- **Fisher-BRC** (Kostrikov et al., ICML 2021†, offline RL): the critic is log μ_behaviour(a|s) + O(s,a), with a gradient penalty on the offset that is shown to equal Fisher-divergence regularisation. The exact penalty form, e.g. whether it is ‖∇_a O‖² at policy actions, was not confirmed from the abstract. — [Fisher-BRC [ABS]](https://arxiv.org/abs/2103.08050)
- Wen et al. (ICML 2021) derive "the exact adjustment" that turns the actor-critic update into the true policy gradient, and propose Residual and Stackelberg actor-critic corrections that improve sample efficiency and final performance. The form of the correction was not retrieved. — [Wen et al. 2021 [ABS]](https://proceedings.mlr.press/v139/wen21b.html)

### Inferences
Ranked candidate fixes for our multi-head TD3 critic. None changes the deployed actor; all are my proposals built on the papers named. **Run the §6 diagnostics first:** they decide whether slope quality is the problem at all.

1. **Gaussian-smoothed, zeroth-order per-head actor gradients** (CPG; Soft-DPG theory). This is the cheapest decisive test of H2.
   - *Mechanism:* replace the point slope ∇_a Q̂_k(s,π(s)) by a local finite difference at scale μ. The actor then needs correct *values within ±μ of π(s)*, not correct derivatives (CPG bound ε/μ + Gμ).
   - *Applied to our critic:* in `moo.py`, for each actor batch draw M directions (4 coordinate axes ±, or N(0,I)). Compute c_{k,b,j} = [Q̂_k(s_b, clip(a_b+μu_j)) − Q̂_k(s_b, clip(a_b−μu_j))]/(2μ) without gradient to θ. Use the surrogate L_k(θ) = −mean_b (1/M)Σ_j c_{k,b,j} u_jᵀ π_θ(s_b) for each head, and feed the 4 gradients to the aggregator unchanged. Start with μ = 0.05-0.1 (CPG used 0.1, our exploration std at the start).
   - *Hypothesis:* H2, the rugged or noisy local-slope part.
   - *Expected effect:* if a shared critic plus aggregator starts to fly like split plus aggregator, H2 is confirmed. If not, the within-state values themselves are wrong, which points to H1/H3/H4.
   - *Cost:* about 20-30 lines and 8 extra critic forward passes per actor update.
   - *Risk:* bias O(Gμ), clipping bias near |a| = 1 (tanh saturation), and evidence from single-objective MuJoCo only.
2. **Advantage-decomposed heads** (compatible form V(s) + A(s,a); Deep Advantage Updating).
   - *Mechanism:* Q_k(s,a) = V_k(s) + Ā_k(s,a) − Ā_k(s,π(s)), so A_k(s,π(s)) = 0 (Tallec et al. Eq. 27). The large, state-dependent variance (safety std 152, crash risk) goes into V; the actor only sees ∇_a Ā_k. This separates "fit the value" from "fit the slope", which is exactly the separation Theorem 3 says matters.
   - *Applied:* a shared state-only V-trunk (privileged obs → 64 → 64 → 4 heads, ≈6.5k params) plus 4 small per-group advantage nets ((obs, a) → 32 → 32 → 1, ≈2.2k each). That is ≈15k params in total, below the 26k of the split critic. The target uses the same decomposition with π′. Tallec et al. also scale A by δt and its learning rate by δt; the analogous knob here is a separate output scale or learning rate for the A-part (untested).
   - *Hypotheses:* H2 and H4 (state-value interference can no longer reshape the action features), and in part H1 (safety's state-value dominance moves into V).
   - *Cost:* about 60-100 lines in `networks.py`/`moo.py`, plus one actor forward pass per critic batch.
   - *Risk:* no published test with TD3 multi-head critics. Advantage extrapolation off the replay actions stays as hard as before.
3. **Exact gradients for the reward terms that depend directly on the action.**
   - *Mechanism:* MAGE assumes a known, differentiable reward and uses ∇_a r. Here r_k(s,a) is known analytically for the efficiency terms (action magnitude, action rate). Let the critic learn only the remainder F_k(s,a) = Q_k − r_k^imm(s,a), with target y_k − r_k^imm. The actor's per-group slope is then ∇_a r_k^imm (exact) + ∇_a F̂_k.
   - *Hypothesis:* H2, for the efficiency head, which is in conflict with precision in 82% of updates on the split critic. This removes a learned-slope error source exactly where the conflict signal lives.
   - *Cost:* about 30-50 lines; the reward formula must match the C++ reward (cf. `reward_check.py`).
   - *Risk:* only part of the reward is immediate, and the action-rate term needs the previous action, which is part of the state.
4. **Keep TD3's target-smoothing noise from decaying in MOO runs** (σ = 0.2, c = 0.5 fixed). See §5. This is a one-flag, 2-seed screen. *Risk:* variant E works with the decay, and Soft-DPG's bias bound grows with σ/(1−γ).
5. **MAGE / Sobolev slope supervision with a differentiable nominal quadrotor model.**
   - *Mechanism:* per head, ‖∇_a δ̂_k(s,π(s),ŝ′)‖ + λ|δ̂_k| with λ ≈ 0.05-0.2 (never λ = 0). ŝ′ = f_nominal(s,a) is a PyTorch rigid-body plus motor-lag model built from the measured parameters (mass, thrust coefficient, motor lag 0.06 s), optionally fed the domain-randomised parameters from the privileged observation.
   - *Hypotheses:* H2 directly, and H1 (keeps the safety slope accurate).
   - *Cost:* high: 200-300 lines of model code, about 2× critic compute (MAGE §6), and double-backward through the custom `FastTanhFunction` (whose backward uses the saved output y) still has to be checked.
   - *Risk:* model bias under domain randomisation (motor delay up to 20 ms, uneven gains), and the crash event is discontinuous.
6. **PAVE-style field-consistency terms** (L_MPR, L_VFC; skip L_Curv, which needs C² activations and our FAST_TANH is clipped). These target jitter, not crashes, and the evidence is one 2026 preprint. Second priority, if flight jitter matters.
- **Not recommended:**
  - A plain gradient-magnitude penalty λ‖∇_a Q‖²: marginal at best and collapses at λ = 10 (PAVE appendix). It also shrinks the actor signal, and differently per head, which distorts any aggregator.
  - Spectral normalisation of the critic: TD3 collapsed in PAVE's appendix.
  - Slope-only losses without a value term: MAGE's degenerate ∇_a Q ≈ 0 with λ = 0.
- **Offline Sobolev fit as a capacity test, not a fix.** On frozen data, add Σ_k‖∇_a Q̂_k(s,π(s)) − ĝ_k^MC(s)‖² (MC slopes from §6, D2) to the TD loss. If the shared 64-64 trunk can fit values and MC slopes of all 4 heads at once, the failure comes from training dynamics or targets (H2/H3), not representation (H4).

### Gaps
- No method in the literature was found that supervises or regularises *per-head* action-gradients in a vector-valued critic.
- The exact GProp deviator update, the Fisher-BRC penalty form, Fairbank & Alonso's IJCNN 2012 "Value-gradient learning" paper and Wen et al.'s correction term were not read (abstracts only).
- PAVE and Soft-DPG are 2026 preprints, not peer reviewed. CPG is an arXiv preprint with no stated venue.
- Taylor-TD-style and SVG-style value-gradient methods were not covered in this session.

## 3. Evidence that value accuracy and policy-gradient quality diverge, and how action-gradient accuracy has been measured

### Takeaway
The divergence is documented in two settings:
- **On-policy:** a value network can fit its loss and generalise while being off by about 50% from the true value, and gradient estimates at practical batch sizes correlate weakly (sometimes zero or negatively) with the true gradient (Ilyas et al.).
- **Off-policy:** the only direct measurement found is MAGE's Pendulum study, where a TD-trained critic "completely fails" to predict the true action-gradient.

Measurement methods in the literature:
- ground-truth gradients from a differentiable oracle model (MAGE);
- cosine to a "true" gradient estimated with very many samples, plus pairwise cosine between independent estimates as a noise measure (Ilyas);
- critic-geometry statistics (PAVE).

### Cited Findings
- **Ilyas et al.** (ICLR 2020†, on-policy PPO/TRPO):
  - Gradient estimates "poorly correlate with the 'true' gradient", and the authors "sometimes observe a zero or even negative correlation" at ~2K state-action pairs per estimate.
  - "True" gradient: ≈10M state-action pairs (Humanoid-v2), ≈11M (Walker2d-v2, Hopper-v2); bootstrap CIs from 500 samples.
  - Variance metric: average pairwise cosine of 10 independent estimates. — [A Closer Look at Deep Policy Gradients [FT, ar5iv]](https://ar5iv.labs.arxiv.org/html/1811.02553)
- Ilyas et al., value network: it "does succeed at both fitting the given loss function and generalizing to unseen data", yet "the learned value function is off by about 50%" from the true value function (true value fitted to returns from 5·10^6 on-policy pairs). The true value function gives "a much lower-variance estimate of the gradient". Surrogate and true reward landscapes diverge late in training: "directions that increase the surrogate objective lead to a decrease of the true reward". — [Ilyas et al. [FT]](https://ar5iv.labs.arxiv.org/html/1811.02553)
- **MAGE §4.2, off-policy action-gradient accuracy:**
  - Setup: Pendulum-v0, fixed random actor, differentiable oracle model. Each critic's ∇_a Q̂ is compared with the gradient of the empirical discounted return, computed by differentiating through the oracle model.
  - Metric: discounted average ℓ1 norm of the gradient difference over 10 trajectories (App. D.2, Eq. 14); 4 runs, 95% CIs.
  - Result: the TD-trained critic "completely fails in predicting it", while MAGE's critic "progressively learns an accurate estimate". The authors say this undermines the "common assumption" that "minimizing the TD-error yields also a minimization of the error on the gradients". — [MAGE [FT]](https://ar5iv.labs.arxiv.org/html/2004.14309)
- Bellman/TD error is a poor proxy for value error (cancellation; infinitely many finite-data solutions). — [Fujimoto et al., ICML 2022 [ABS]](https://arxiv.org/abs/2201.12417)
- DDPG learning "can get stuck into a fixed point corresponding to a poor solution" in deterministic sparse-reward environments (an elementary control problem). — [Matheron, Perrin & Sigaud 2019 (Springer DOI 10.1007/978-3-030-61616-8_25 listed on the arXiv page) [ABS]](https://arxiv.org/abs/1911.11679)
- **PAVE critic-geometry statistics on unregularised TD3/SAC critics:**
  - The action-Hessian ∇²_aa Q is negative definite in only 14-47% of visited states; their regulariser raised this to 64-100%.
  - "Cosine flip rates" of ∇_a Q also fell (e.g. Pendulum 25.4% → 2.3%).
  - They bound the Lipschitz constant of the optimal-action map, ∂a*/∂s = −[∇²_aa Q]^{-1}∇²_sa Q, by M/μ (mixed-partial norm over curvature). The exact flip-rate procedure was not in the parts retrieved. — [PAVE [FT]](https://arxiv.org/html/2601.22970)
- Neither CPG nor Soft-DPG measures gradient alignment. CPG infers better gradients only from returns, and Soft-DPG shows only a qualitative toy-landscape figure. — [CPG [FT]](https://arxiv.org/html/2409.01477); [Soft-DPG [PT]](https://arxiv.org/html/2605.06228)
- Soft-DPG states that DDPG's TD objective "does not explicitly constrain the correctness or smoothness of its action-gradient". — [Soft-DPG [PT]](https://arxiv.org/html/2605.06228)

### Inferences
- Our "repeated observation" (R² ≈ 0.99, crashes 18-95%) is the off-policy, per-objective analogue of Ilyas et al.'s finding: fitting the regression target says little about the quantity the actor consumes.
- Two literature measurement styles transfer directly to us:
  - **MAGE-style ground truth.** Our C++ simulator is not differentiable, so use finite differences with common random numbers instead of autodiff (§6, D2).
  - **Ilyas-style cosine to a high-sample reference, with a noise ceiling** (the cosine between two independent MC estimates bounds the achievable agreement).
- PAVE's geometry statistics (flip rate, curvature sign) are cheap to compute on our critics and need no ground truth, but they measure roughness, not correctness.

### Gaps
- No large-scale off-policy (DDPG/TD3/SAC) study that replicates Ilyas et al.'s "true gradient" methodology was found. MAGE's single-environment Pendulum study is the closest.
- No study was found that measures action-gradient accuracy per reward component, or for multi-head critics.
- PAVE's exact definitions of "cosine flip rate" and M_sup were not retrieved.

## 4. Multi-output critics: do shared features give correlated or noisy per-objective action-gradients?

### Takeaway
**Nothing found** that measures per-head action-gradient quality, or slope correlation between heads, in a shared-trunk multi-head (decomposed, multi-objective) critic. The closest evidence is value-level: in multi-task actor-critic, a single shared critic causes "negative interference" between task values, and separate per-task critics improve performance (MultiCriticAL, ICLR 2022). Multi-head DDPG critics exist, but the papers found do not analyse interference. For our novelty claim, this suggests the slope-level failure mode is unreported. The search was limited, though, and an adjacent researcher covers multi-head collapse in general.

### Cited Findings
- **MultiCriticAL** (Mysore, Cheng, Zhao, Saenko, Wu; ICLR 2022): "Using a single value function (critic) shared over multiple tasks in Actor-Critic multi-task reinforcement learning (MTRL) can result in negative interference between tasks". The method keeps "separate critics for each task being trained while training a single multi-task actor". This "mitigates interference between task-value estimates". The authors suspected that similar game-play states shared across styles interfered with each style's value optimisation. Reported gains are "up to 56%" (ICLR/MIT-IBM summaries) vs "up to 45%" (Papers with Code listing); the versions conflict. — [ICLR 2022 poster page [ABS]](https://iclr.cc/virtual/2022/poster/6568); [project page [ABS]](https://ai.bu.edu/MultiCriticAL)
- Huang et al. (arXiv 2021) apply "a multi-head critic to learn a separate value function for each reward component", with a "hybrid policy gradient" and dynamic weights for bipedal locomotion. The abstract does not discuss shared-feature interference. — [Reward-Adaptive RL [ABS]](https://arxiv.org/abs/2107.01908)
- A search for shared-trunk, multi-head, decomposed-reward critics with deterministic policy gradients returned no paper analysing action-gradient interference. Hits included a robot-manipulation critic with per-reward heads on a shared trunk and parameter-sharing multi-agent DDPG; these were not read. — [Efficient Real-World Online RL ... Critic Decomposition (arXiv 2608.09762), not read](https://arxiv.org/pdf/2608.09762); [Parameter Sharing DDPG (arXiv 1710.00336), not read](https://arxiv.org/pdf/1710.00336)
- **Our own code**, for the mechanism below:
  - The shared critic is (obs, a) → Linear 64 → FAST_TANH → Linear 64 → FAST_TANH → Linear k. FAST_TANH clamps inputs to [−3, 3] and its backward is grad·(1 − y²), which is 0 where |x| ≥ 3. — [networks.py](/Users/khushpatel/code/CrazyFlie-2.1/l2f/py/networks.py)
  - The split critic is k separate 64-64 networks.
  - The tower critic shares only the first layer.

### Inferences
- **Mechanism: values can be distinct while slopes are not.** With trunk output h(s,a) ∈ R^64 and heads Q̂_k = c·(w_kᵀh + b_k), the slope of every head is ∇_a Q̂_k(s,a) = c·M(s,a) w_k, where M(s,a) = W1aᵀ D1 W2ᵀ D2 ∈ R^{4×64}.
  - W1a is the action columns of the first layer; D_i = diag(FAST_TANH′(z_i)) is 0 for saturated units.
  - All four heads' slope fields are linear combinations of the same 64 "slope basis fields" (the columns of M), which are shaped mostly by the dominant safety loss.
  - Head *values* can still differ through the 64 state-dependent features, which do not enter the slopes once saturated or action-insensitive. So distinct, accurate values (e.g. the PopArt-like run: values −0.98..+0.93, R² 0.97-0.99) can coexist with slopes that are correlated or wrong.
  - The split critic gives each head its own M_k(s,a). This is a first-principles explanation of H2 and H4; it is testable with §6, D6.
- **Saturation removes slope capacity, not value capacity.** A unit with |z| ≥ 3 still contributes a state-dependent constant to the values but contributes nothing to ∇_a Q. Gradient re-weighting (aggregators, normalisation) changes which units saturate. Proposal: count saturated units at (s, π(s)) in shared and split critics.
- **The aggregator amplifies whatever slope errors are shared.** If M(s) aligns the heads' slopes (slope-level collapse), the actor's 4 gradients look parallel and "every aggregator reduces to the sum", which matches §3 of the problem statement. If instead normalisation or aggregation makes the small heads' slopes noisy, scale-invariant aggregators amplify that noise (§1 inferences).

### Gaps
- No paper reports collapse or interference *of action-gradients* between critic heads. I cannot rule out work under other names (e.g. explainable-RL reward decomposition, or the multi-agent decomposed critics in DOP) that measures it. MultiCriticAL's per-style critics are separate networks; whether it measured gradient conflict was not checked.

## 5. Policy smoothness regularisers (CAPS) and TD3 target policy smoothing as action-gradient regularisers

### Takeaway
TD3's target policy smoothing is an explicit action-space regulariser on the critic's *bootstrap target* ("similar actions should have similar value"), and it measurably helps TD3. Soft-DPG and CPG formalise the same idea as a Gaussian-smoothed Q whose gradient can be estimated without dQ/da. CAPS acts only on the actor and leaves the critic untouched. PAVE argues that jitter comes from critic geometry, and it beats CAPS on smoothness with TD3.

### Cited Findings
- **TD3 target policy smoothing** (Fujimoto, van Hoof & Meger, ICML 2018†, §5.3):
  - Target: y = r + γ Q_θ′(s′, π_φ′(s′) + ε), with ε ∼ clip(N(0,σ), −c, c); defaults σ = 0.2, c = 0.5.
  - Motivation: "deterministic policies ... can overfit to narrow peaks in the value estimate"; "our approach enforces the notion that similar actions should have similar value". It "mimics the learning update from SARSA" and is "reminiscent of Expected SARSA".
  - Ablation (TD3 vs TD3 without smoothing, 1M steps, 10 trials): HalfCheetah 9532.99/8987.69, Hopper 3304.75/2392.59, Walker2d 4565.24/4033.67, Ant 4185.06/4155.24. — [TD3 [PT, §5.3 and Table 2, ar5iv]](https://ar5iv.labs.arxiv.org/html/1802.09477)
- Soft-DPG's smoothed operator, its zeroth-order gradient and its bias bounds: Lemma 5.5, ‖Q^π − Q_σ^π‖_∞ ≤ γL_Qσ√m/(1−γ); Lemma 5.3 is a similar bound for V. — [Soft-DPG [PT]](https://arxiv.org/html/2605.06228)
- CPG's smoothing Q_μ(s,a) = E_u[Q(s,a+μu)], with μ set equal to TD3's exploration std (0.1). — [CPG [FT]](https://arxiv.org/html/2409.01477)
- **CAPS** (Mysore et al., ICRA 2021†):
  - Objective: J^CAPS = J − λ_T L_T − λ_S L_S, with L_T = D_T(π(s_t), π(s_{t+1})) and L_S = D_S(π(s_t), π(s̄_t)), s̄_t ∼ N(s_t, σ); D is the Euclidean distance. The page gives no σ or λ values.
  - CAPS does not change how value functions are learned. The authors note that surrogate value functions have representational limits, and leave improving value estimation to future work.
  - On-quadrotor results: current draw 4.86 A vs Neuroflight's 22.87 A; smoothness Sm 0.16×10^-3 vs 4.3×10^-3. Sm = (2/(n f_s)) Σ_i M_i f_i (FFT-amplitude-weighted mean frequency). — [CAPS [FT, ar5iv]](https://ar5iv.labs.arxiv.org/html/2012.06644)
- PAVE frames CAPS as treating "the symptom rather than the cause" and moves the regularisation to the critic's action-gradient field. With TD3, PAVE had better smoothness than CAPS on all 6 tasks; CAPS was a baseline in every table. — [PAVE [FT]](https://arxiv.org/html/2601.22970)

### Inferences
- **What TD3's smoothing does and does not regularise.** It smooths the critic's *target* as a function of the next action. It regularises ∇_a Q(s,a) at the current action only indirectly: Q(s,a) inherits the smoothness of E[Q′(s′, π′(s′)+ε)] through s′ = f(s,a). It does not supervise the slope.
- **Our schedule leaves little of it late in training.** Our TD3 decays σ_target and the exploration std by ×0.9 every 100k steps from 500k. After 15-16 decays that leaves σ_target ≈ 0.04 (clip ≈ 0.1) and exploration ≈ 0.02 by 2M. So late-training critics get far less action-space smoothing, and far less action variation in the data (§1), than TD3's defaults. Variant E tolerates this; for heads read individually by non-linear aggregators it may matter more. Hence the cheap screen (fix 4 in §2).
- **CAPS filters slope noise but cannot correct bias.** As an actor-side penalty, CAPS low-pass filters the effect of state-to-state slope noise, but it cannot correct a biased per-head slope. If used with MOO aggregators, add it *after* aggregation as a regulariser, not as a 5th "objective"; otherwise PGA-type aggregators would weight it scale-invariantly. The same logic supports fix 3 in §2: give the actor exact gradients for terms that are known functions of the action (action magnitude and rate), instead of routing them through learned slopes.

### Gaps
- No paper was found that analyses target policy smoothing specifically as a regulariser of ∇_a Q, or its interaction with noise decay schedules.
- CAPS's λ and σ values for the hardware runs are on the project website, which was not fetched.
- No work combines CAPS or PAVE with multi-objective gradient aggregation.

## 6. Cheap diagnostics to test H2 (accurate values, wrong slopes) on our runs — exact procedures

### Takeaway
Run three free checks first, using `critic_snapshot.py` reruns:
1. how much of each head's R² is action-dependent at all (D0);
2. twin-critic slope agreement (D1);
3. a shared-trunk "slope basis" and saturation test (D6).

Then run the decisive check, D2: finite-difference Monte Carlo slopes per group with common random numbers, compared with each critic's ∇_a Q̂_k. D2 needs a small simulator binding to clone or set env states, and critic weights saved alongside the actor checkpoints.

### Cited Findings
- **Our code** (from the local source):
  - `critic_snapshot.py` reruns one training deterministically and records, at chosen steps, the group-value correlation, the top-direction share and each group's R² against **its own TD target**. — [critic_snapshot.py](/Users/khushpatel/code/CrazyFlie-2.1/l2f/sim/critic_snapshot.py)
  - Only actor checkpoints (.h5, every 100k) are saved, not critics. — [td3.py header](/Users/khushpatel/code/CrazyFlie-2.1/l2f/py/td3.py); [train.py](/Users/khushpatel/code/CrazyFlie-2.1/l2f/py/train.py)
  - The Python sim binding exposes reset/observe/step/states (a getter) but no state setter. — [sim_binding.cpp](/Users/khushpatel/code/CrazyFlie-2.1/l2f/py/sim_binding.cpp)
- **Measurement precedents:**
  - Cosine to a high-sample "true" gradient, and pairwise cosine between independent estimates as a noise ceiling. — [Ilyas et al. [FT]](https://ar5iv.labs.arxiv.org/html/1811.02553)
  - Discounted ℓ1 action-gradient error against an oracle-model gradient. — [MAGE App. D.2 [FT]](https://ar5iv.labs.arxiv.org/html/2004.14309)
  - Negative-definiteness rate and cosine flip rate of ∇_a Q. — [PAVE [FT]](https://arxiv.org/html/2601.22970)
  - Gaussian-smoothing (Nesterov-Spokoiny-type) gradient estimates, with the ε/μ + Gμ trade-off. — [CPG [FT]](https://arxiv.org/html/2409.01477)

### Inferences
The procedures below are my proposals; precedents are named where they exist. Run each on four critics: split + OS-NMF-PGA (works); plain shared + sum (≈5% crashes); and two failing shared critics, `--critic-norm-by target` + sum (28%) and `--critic-agg` OS-NMF-PGA (64-95%). Use 2 seeds and snapshot steps 600k, 1.2M and 2M. States: N = 1024 replay states, as the snapshot tool does, plus N = 256 states from deterministic test episodes. The second set matters because the DPG theorem integrates over d^π.

- **D0: Can R² see slopes at all?** Free; add it to `critic_snapshot.py`.
  1. For each state s_i, draw J = 32 actions a_ij = clip(π(s_i) + σu_ij, −1, 1), u ∼ N(0, I_4), at σ ∈ {current exploration std, 0.05, 0.1}.
  2. For each head k, compute v_k = mean_i Var_j[Q̂_k(s_i, a_ij)] / Var_i(y_k) and the slope proxy mean_i‖∇_a Q̂_k(s_i, π(s_i))‖·σ / RMSE_k.
  3. **Decision rule:** if v_k ≪ 1 − R²_k (e.g. < 0.003 when R² = 0.99), the reported R² cannot detect even a total failure of head k's action-dependence, and the value-accuracy evidence against H2 is void.
  4. Also compute R² against MC per-group returns (D2, step 2) instead of TD targets. A head may fit its TD target but not its true return (Fujimoto et al. 2022), which is relevant to H3.
- **D1: Twin-critic slope agreement.** Free; TD3 already has 2 critics trained on the same targets from different initialisations.
  1. Per head k at (s_i, π(s_i)), compute the value agreement corr_i(Q̂¹_k, Q̂²_k), the slope agreement median_i cos(∇_a Q̂¹_k, ∇_a Q̂²_k), the per-dimension sign agreement, and the magnitude ratio ‖∇_a Q̂¹_k‖/‖∇_a Q̂²_k‖.
  2. **Prediction under H2:** failing critics show value agreement ≈ 1 but low slope agreement on at least one head, especially safety (H1 + H2), or on the small heads that PGA weights most.
  3. **Caveat:** both twins regress onto the same per-sample target, so agreement measures function-approximation variance, not bias. High agreement does not prove correct slopes.
- **D2: Monte Carlo finite-difference slopes per group.** The decisive test; Ilyas/MAGE-style ground truth adapted to a non-differentiable simulator.
  1. **Infrastructure.** (a) Save critic and critic-target weights (`torch.save`) in `critic_snapshot.py` at the `--at` steps, chosen as multiples of 100k so they pair with saved actor checkpoints. (b) Add a `clone_states(src, dst)` or `set_states` binding to `sim_binding.cpp`. It must copy the full per-env state: rigid body, rotor speeds, motor-delay buffers, domain-randomised parameters, mission/target, step counter, any observation-side state such as the leaky integral and RPM history, and the RNG state, so branches share sensor noise (common random numbers). My estimate is 20-40 lines, to be checked against the env struct.
  2. **Per-group MC returns.** For a state s (env snapshot) and first action a, run one step with a, then follow π deterministically (no exploration noise) for H = 300 steps, or to the episode end. Record the discounted group returns G_k(s,a) = Σ_t γ^t r_{k,t} (γ^300 ≈ 0.05). Use only states with ≥ 300 steps left before the 500-step time limit, or bootstrap the tail with the critic.
  3. **Slopes.** Use central coordinate differences, ĝ_{k,d}(s) = [G_k(s, π(s)+h e_d) − G_k(s, π(s)−h e_d)]/(2h), for d = 1..4 and h ∈ {0.02, 0.05}: 8-16 rollouts per state. Skip states with |π_d(s)| > 1 − 2h. For states near a crash boundary, also compute the Gaussian-smoothed version at σ = 0.1 with M = 16 antithetic pairs, (1/(2σM))Σ_j[G_k(s,π+σu_j) − G_k(s,π−σu_j)]u_j. Run a second independent set of directions or seeds to get the **noise ceiling** (the cosine between two MC estimates).
  4. **Metrics per head.** median_i cos(∇_a Q̂_k(s_i,π(s_i)), ĝ_k(s_i)); sign agreement per dimension; relative error ‖∇_a Q̂_k − ĝ_k‖/‖ĝ_k‖. Then push both through the actor: g_k^MC = mean_i ∇_θπ(s_i)ᵀ ĝ_k(s_i) versus g_k^critic, with the cosine in θ-space. Finally the metric that matters most: cos(A(g^MC_1..4), A(g^critic_1..4)) for each aggregator A. It answers "would the aggregated step have pointed the right way?"
  5. **Cost.** 256 states × 16 rollouts × 300 steps ≈ 1.2M simulator steps per checkpoint, vectorised as 4096 envs. That is minutes of compute plus an estimated 1-2 hours of binding work.
  6. **Caveats.** The 400-point terminal crash penalty makes G_safe discontinuous in a. Report the share of states whose ± branches disagree on crashing, and report the safety head with medians. The critic learns Q under target-smoothing noise (σ ≈ 0.04-0.2), not exactly Q^π, so repeat a subset with π + N(0, σ_target²) continuation noise (common random numbers) as a sensitivity check.
  7. **Decision rule.** H2 is supported if failing critics have much lower slope cosine or sign agreement than split and plain-sum critics while having similar MC-value R². It is H1 + H2 if the degradation is concentrated in the safety head.
- **D3: Reference-critic comparison.** Cheaper per state once built.
  1. Train a large split reference critic (e.g. 4 × 256-256) by policy evaluation of the frozen π on fresh data, using MC group returns as targets.
  2. **Include perturbed first actions** (a = π(s) + N(0, 0.05²-0.2²), with their branched MC returns from D2). On-policy data alone has a = π(s), so it cannot identify ∇_a Q.
  3. Validate the reference on D2 states (noise-ceiling-level cosine). Then compare every training critic's slopes with the reference on thousands of states.
- **D4: Slope stability and flip rate.** Free in snapshot reruns.
  1. (a) **Churn:** cos(∇_a Q̂_k at step t, at step t + 10k updates) at fixed states, measured during RL rather than on frozen data. This extends the existing offline drift test.
  2. (b) **PAVE-like flip rate:** along deterministic test trajectories, the share of consecutive steps with cos(∇_a Q̂_k(s_t,π(s_t)), ∇_a Q̂_k(s_{t+1},π(s_{t+1}))) < 0. This is my definition; PAVE's exact one was not retrieved.
  3. (c) **Curvature sign:** the share of states where vᵀ∇²_aa Q̂_k v < 0 for Rademacher v. Computable despite FAST_TANH via a finite difference of ∇_a.
  4. High churn or flip rates mean the actor follows a moving or rugged slope field; they link to the jitter metrics measured on the flight day.
- **D5: Predicted vs realised per-group improvement.** The end-to-end test; uses the D2 infrastructure.
  1. From a checkpoint, compute the aggregated step d = A(g_1..4) from critic slopes. Apply a virtual actor update θ′ = θ + ηd, with η equal to 1, 10 and 100 actor steps' worth.
  2. Compare the critic-predicted first-order change ΔĴ_k = η g_kᵀd with the realised change in MC group returns J_k(θ′) − J_k(θ) on the same test episodes (same seeds and randomisation).
  3. Report sign agreement and rank correlation across checkpoints and aggregators. This is the per-objective analogue of Ilyas et al.'s surrogate-vs-true landscape check.
- **D6: Shared-trunk slope basis and saturation.** Free; this is the test of the §4 mechanism.
  1. For each state, compute the trunk action-Jacobian J_i = ∂h/∂a ∈ R^{64×4} at (s_i, π(s_i)) with `torch.func.jacrev`.
  2. Report: (a) the fraction of FAST_TANH units with |z| ≥ 3 (zero derivative), per layer; (b) the singular-value spectrum and effective rank of the stacked J_i; (c) the inter-head slope cosines cos(∇_a Q̂_i, ∇_a Q̂_j), compared with the same statistic for the split critic and for MC slopes (D2). Slope-level collapse shows up as cosines near 1 where MC shows conflict.
  3. Fit the **optimal head weights for slopes:** w_k* = argmin_w Σ_i‖c J_iᵀw − ĝ_k^ref(s_i)‖² + α‖w‖². Here ĝ^ref comes from D2 or D3; the split critic's slopes are a stand-in until those exist. Report the slope R²_k, and compare it with the value R² from the same features, h(s_i, π(s_i)) regressed on the MC values.
  4. **Interpretation:** high value R² with low *optimal* slope R² means the shared features cannot express head k's slope field (representational H2/H4). High optimal slope R² with poor *actual* slopes means the head training or targets are at fault (H2/H3).
- **Order and cost.** D0, D1, D4(a-b) and D6(a-c) can go into one `critic_snapshot.py` extension (about 100 lines, no new runs beyond the existing snapshot reruns). Next come D2 and D5 (binding plus about 150 lines), then D3 only if D2's per-state cost limits the number of states. If D0 shows v_k ≪ 1 − R² and D2 shows slope accuracy separating working from failing critics, H2 is established. The fixes in §2 then follow in the stated order.

### Gaps
- Whether rl_tools' per-env state, including domain-randomisation parameters, RNG and observation history, can be cloned cleanly was not checked. The 20-40 line estimate for the binding is unverified.
- Which twin drives the actor loss in `moo.py` was not checked; D1 and D2 should use that twin.
- Whether `FastTanhFunction` supports double backward (needed for D4(c) via autograd, and for fixes 5-6) was not checked. It is not needed for D0-D3, D5 or D6.
- No published study gives expected values for these diagnostics in TD3 multi-head critics, so the thresholds above are heuristics.
