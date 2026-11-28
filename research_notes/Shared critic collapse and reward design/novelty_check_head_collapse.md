# Novelty check: head collapse in shared multi-head (decomposed, multi-objective, multi-task) critics

Scope (2026-11-27). This note closes the three gaps left open by `decomposed_critic_collapse.md` and `critic_representation_collapse.md`, which it does not repeat:
1. TOPPO's "co-linear collapse", read in full.
2. MultiCriticAL's full text.
3. A forward-citation sweep of HRA, SAC-D and GCR-PPO.

It also adds a direct phrase sweep (web search and the arXiv API) and ends with a verdict and a safe wording for the novelty claim.

**Our finding** (from `l2f/docs/shared_critic_research_prompt.md` §3):
- **Setup.** A TD3 critic has one shared 64-64 trunk and 4 heads (time, precision, efficiency, safety). Every transition trains every head, each with its own reward group and its own bootstrapped TD target.
- **Collapse.** Late in training the 4 group values become near-copies: correlation 0.93-1.00 at 2M steps, and 1.00 at 200k with head scaling. The TD targets are just as correlated.
- **Effect on the actor.** The 4 actor gradients become near-parallel, so every aggregator reduces to the sum.
- **Controls.** A parameter-matched wider shared critic (25,990 params) still collapses (0.75-0.97). Separate critics do not collapse (+0.30..+0.75).

**Reading status:**
- **[full]**: the full text was downloaded and extracted locally (PDFs with macOS PDFKit, HTML with a parser), then read section by section and searched. Quotes from [full] sources are verbatim. PDF extraction garbled some math symbols; where I restored them in plain notation the quote is marked [math restored].
- **[partial]**: some sections read, or a full-text keyword search plus reading of the matching passages.
- **[abstract]**: abstract only, from the Semantic Scholar (S2), OpenAlex or arXiv APIs.
- **[title]**: title or bibliographic record only.

## Q1. TOPPO: what exactly is its "co-linear collapse", in what architecture, how was it measured, and how close is it to our finding?

### Takeaway
TOPPO exists and is the closest precedent found: Li, Lin, Qu, Miao, arXiv 2605.11473; v1 12 May 2026, v3 7 Nov 2026.

What its "co-linear collapse" is:
- **What it measures.** The mean absolute cosine between the per-task critic-loss gradients on the shared critic parameters (the off-diagonal entries of the gradient Gram matrix). It is 0.34 early in training without LayerNorm and 0.20 with it; over a full training trace the median falls from 0.24 to 0.10. It is gradient-level and critic-side.
- **Architecture.** A single-output, task-conditioned critic: one-hot task input, a shared `Linear(400,1)` output, and a 2-parameter PopArt affine per task.
- **Routing.** Each task's loss is computed on that task's own rollout samples.
- **Learner.** On-policy PPO with GAE and no target network.

What it does not do:
- It does not measure head values, value correlation or TD-target correlation.
- It does not link critic co-linearity to the actor-side aggregator.
- It has no shared-vs-separate critic comparison.

So TOPPO pre-empts the generic claim that "per-objective gradients collapse in a shared critic", and it owns the term "co-linear collapse". It does not pre-empt our value-level, dense-head, bootstrapped, actor-side finding.

### Cited Findings
**Identity** [full; v3 PDF, 54 pages]
- Title: "TOPPO: Rethinking PPO for Multi-Task Reinforcement Learning with Critic Balancing".
- Authors: Yuanpeng Li (UC Irvine), Gefei Lin (George Washington University), Annie Qu (UC Santa Barbara), Rui Miao (affiliation not captured in the extraction).
- arXiv 2605.11473 [cs.AI]. v1 12 May 2026, v2 3 Nov 2026, v3 7 Nov 2026.
- A preprint: the PDF names no venue. Code: github.com/NAIVlab/TOPPO.

— [arXiv abs](https://arxiv.org/abs/2605.11473); [v3 PDF](https://arxiv.org/pdf/2605.11473v3)

**Architecture: what is shared and what is per task**
- Table B.1 gives:
  - critic hidden sizes [400,400,400];
  - "critic output projection shared Linear(400,1) for V";
  - "PopArt affine (Wi,bi) 2 params per task";
  - "task identity input one-hot, length K = 10" on MT10 (K = 50 on MT50);
  - ReLU activations;
  - LayerNorm placement "critic hidden layers only".

  — [v3 PDF, App. B](https://arxiv.org/pdf/2605.11473v3)
- From §4.1:
  - "the shared critic outputs a single normalized scalar z_θc(s,i) ∈ R, and a per-task scale-and-shift (σi, µi) maps it back to the raw value range" [math restored];
  - "the underlying critic remains a single shared MLP".

  — [v3 PDF](https://arxiv.org/pdf/2605.11473v3)
- From App. A:
  - "We keep a fully task-shared [400×3] MLP backbone with no task-specific subnetworks, experts, or routers";
  - "architecturally TOPPO adds only critic LayerNorm and a per-task PopArt affine head (Table B.1)".

  — [v3 PDF](https://arxiv.org/pdf/2605.11473v3)
- The critic is a state-value function V_θc(s, i). Each task's value loss is L^c_i(θc) = E_t (V_θc(s_t, i) − V̂^(i),targ_t)² over task i's own rollouts, where "Et denotes the empirical average over steps in the rollout batch" [math restored].
- The targets come from GAE with λ = 0.97 and γ = 0.99 (Table B.1). — [v3 PDF, §2](https://arxiv.org/pdf/2605.11473v3)
- There is no target network: "For our PPO this is just actor + critic (no target network)." — [v3 PDF, App. B.3](https://arxiv.org/pdf/2605.11473v3)
- "Our reported architecture also uses separate actor and critic networks." — [v3 PDF, §6](https://arxiv.org/pdf/2605.11473v3)

**What collapses, and how it is measured**
- **The gradients.** "we capture the per-task gradients g^a_i := ∇θa L^a_i and g^c_i := ∇θc L^c_i on the shared parameters for actor and critic, respectively". The Gram matrix is G_ij = ⟨g_i, g_j⟩, "where the diagonal G_ii = ∥g_i∥² records per-task scale, and the off-diagonal cosines … record pairwise direction conflict" [math restored]. — [v3 PDF, §2](https://arxiv.org/pdf/2605.11473v3)
- **The definition**, verbatim apart from the symbols [math restored]: "(D2) Co-linear collapse. A high mean off-diagonal |cos| = |G^c_ij| / sqrt(G^c_ii G^c_jj) for i ≠ j indicates that per-task critic gradients have collapsed toward a small set of shared-feature directions, which leaves little task-specific direction for any aggregator to act on and makes G^c near-singular." — [v3 PDF, §3](https://arxiv.org/pdf/2605.11473v3)
- **The numbers.**
  - §4.2: "In early-stage critic Gram diagnostics, critic LN reduces the mean off-diagonal |cos| from 0.34 to 0.20 while roughly preserving diagonal magnitude".
  - App. D: "over the full trace of the matched FG-trained dump in Figure D.6, the median falls from 0.24 to 0.10) while leaving the diagonal magnitude palette roughly intact — a representation-level intervention, not a target-side fix."

  — [v3 PDF](https://arxiv.org/pdf/2605.11473v3)
- **The metric's meaning.** Fig. D.3 caption: "Absolute cosine measures co-linearity rather than negative-conflict frequency." — [v3 PDF, App. D](https://arxiv.org/pdf/2605.11473v3)
- **The protocol.**
  - The instrumented run writes "per-task flat actor and critic gradients … to disk every 5 collects (∼0.5M env steps, ∼27 MB of uncompressed fp32 gradients per snapshot) for offline replay".
  - Snapshots are taken on MT10 at three stages: EARLY 0.5M, MID 4.5M and LATE 7.5M steps.
  - The "raw-unit" numbers come from PopArt-trained runs: the dumped gradients are rescaled "by σ²_i at fixed network weights".

  — [v3 PDF, App. D.2](https://arxiv.org/pdf/2605.11473v3)
- **Claimed mechanism of the fix** (Table 1, LN-c): "Regularizes the critic feature Jacobian and lowers critic-Gram off-diagonal cosine collapse while leaving scale structure intact." — [v3 PDF](https://arxiv.org/pdf/2605.11473v3)

**The 497x scale spread**
- "Empirically, per-task critic gradient norms span ∼497× across tasks, while per-task actor gradient norms span only ∼4.1× after per-task advantage normalization (Figure 2)." — [v3 PDF, §1](https://arxiv.org/pdf/2605.11473v3)
- The 497× figure is a median over 0.5-8M steps on MT10. It is read from the PopArt + LN-c run and reconstructed in raw loss units. — [v3 PDF, App. D.2](https://arxiv.org/pdf/2605.11473v3)
- Without LN, the spread rises "up to ∼10,000× at early training, decaying to ∼200× by mid-training and rebounding to ∼700× by late training". — [v3 PDF, App. D](https://arxiv.org/pdf/2605.11473v3)
- Early-stage badges in Fig. 3: vanilla 9864×, +PopArt 27×, +LN 5314×, +PopArt+LN 16×. — [v3 PDF](https://arxiv.org/pdf/2605.11473v3)

**A self-reinforcing loop, with a different mechanism from ours** (§1):
- "we trace the remaining gap to a self-reinforcing trap on the critic. Early in training, a few easier tasks learn fast and produce large critic gradients, pulling the shared critic network toward features that reduce their TD errors. Harder tasks that contribute little to this aggregate update must then fit their value functions through shared features optimized primarily for other tasks, which biases their advantage estimates rather than simply making them vanish."
- The loop closes through the data: "the resulting policies collect worse or less informative rollouts".

— [v3 PDF](https://arxiv.org/pdf/2605.11473v3)

**The actor side**
- Actor Gram spread is 2.7-4.6×, and "off-diagonals shift from early inter-task conflict to late near-orthogonal alignment" (Fig. D.5 caption).
- Actor conflict is handled by PCGrad on the actor (PCGrad-a).
- I found no statement linking critic co-linearity to near-parallel actor gradients.

— [v3 PDF](https://arxiv.org/pdf/2605.11473v3)

**Scope statements**
- Future work: "Our study uses compact shared MLPs, where per-task critic gradients show large scale spread and strong co-linearity. In much larger networks, per-task gradients are closer to orthogonal, so cross-task interference, and with it the benefit of critic-side balancing, may shrink." In the parts read, no experiment backs the claim about larger networks. — [v3 PDF, §6](https://arxiv.org/pdf/2605.11473v3)
- Limitations: "Porting the critic-balancing to MTSAC gives no reliable final gain (Appendix E.1), so our claims are limited to the on-policy PPO setting." — [v3 PDF, §6](https://arxiv.org/pdf/2605.11473v3)
- Multi-objective use: "The same critic-side ill-conditioning can arise in other PPO settings where one shared critic fits value targets of different scales, such as multi-objective control." — [v3 PDF, §6](https://arxiv.org/pdf/2605.11473v3)
- **App. E.5, the AMP humanoid.** This is TOPPO's only reward-decomposed shared critic.
  - Setup: "combining task-tracking and motion-style rewards through a shared critic trunk", with an arm named "Two-component critic + LN + PopArt".
  - PPO result: the full PPO stack "reduces mean style FID by 76.0% relative to the base policy", but "the full-stack policy walks more slowly".
  - SAC result: the full SAC stack lowers style FID by 19.8%, but the effect is "not statistically reliable at n=10 (p=0.078…)".
  - A text search of E.5 finds no Gram, head-value or correlation diagnostics for this two-component critic.

  — [v3 PDF](https://arxiv.org/pdf/2605.11473v3)

**Results context**
- With 717K parameters, TOPPO "reaches 90.9 ±1.6% final-checkpoint mean success on MT50 and 56.5 ±7.8% on the worst-10 tail tasks".
- "In the MT50 cumulative ablation ladder, LN-c gives the largest single-step gain."

— [v3 PDF](https://arxiv.org/pdf/2605.11473v3)

**Correction to `critic_representation_collapse.md`.** That note, sourced through a web summariser, described TOPPO as a "shared 400×3 MLP with per-task PopArt heads". The per-task "heads" are only 2-parameter affine rescalings of one shared scalar output, and task identity enters through a one-hot input. The full text confirms that the Gram matrix is built from per-task critic-loss gradients on the shared critic parameters. — [v3 PDF](https://arxiv.org/pdf/2605.11473v3)

### Inferences
**Side-by-side comparison** (my synthesis of the cited text and our prompt):

| Dimension | Our finding | TOPPO (D2) |
|---|---|---|
| What is measured | Correlation of head **values** across states, and of TD targets; parallelism of the 4 actor gradients | Mean \|cos\| between per-task critic-**loss gradients** on the shared critic parameters |
| Size of effect | Near-copies (0.93-1.00) | Moderate alignment (0.34 early → 0.20 with LN; median 0.24 → 0.10) |
| Critic structure | One trunk with 4 output heads; every head sees the same (s, a) input | One trunk with one output; task enters as a one-hot input; 2-parameter PopArt affine per task |
| Sample routing | Dense: every transition trains all 4 heads | Routed: each task's loss uses its own rollout slice |
| Targets | 1-step TD from target networks (TD3, Polyak 0.995) | GAE (λ 0.97) returns; no target network |
| Learner and domain | Off-policy TD3, quadrotor continuous control | On-policy PPO, Meta-World (manipulation) |
| Harm | The actor-side aggregator sees parallel gradients and reduces to the sum, so the conflict is hidden | The critic-side aggregator has "little task-specific direction"; tail tasks get biased advantages |
| Capacity | A parameter-matched wider shared critic still collapses | Asserted that larger nets have near-orthogonal gradients; no test found |
| Separate critics | Do not collapse | Not tested |
| Fix studied | Target-std normalisation removes the collapse but hurts control | LayerNorm lowers \|cos\|; PopArt + FairGrad |

- **Closeness.** TOPPO is the nearest neighbour on four points:
  1. gradient geometry of a shared critic across tasks or objectives;
  2. order-of-magnitude scale spread (497× against our 100-3000×);
  3. reading the collapse as gradients pulled "toward a small set of shared-feature directions";
  4. a self-reinforcing loop.

  It differs on what collapses (gradients, not values), on how strongly, on architecture (input-conditioned single output against output heads), on routing (routed against dense), on bootstrapping (none against a target-network TD loop) and on the actor-side consequence. My rating: the closest precedent, with partial overlap in mechanism and vocabulary, and no overlap with our value-level and actor-side observations.
- **The nearest published phrasing of "the critic hides the conflict".** TOPPO's D2 sentence ("leaves little task-specific direction for any aggregator to act on") comes closest to our "a collapsed critic hides the conflict from the aggregator". But it concerns the critic-gradient aggregator (FairGrad on the critic loss), not the actor-gradient aggregator downstream of the critic. Cite TOPPO for the critic-side version.
- **Why value-level collapse is the stronger claim** (my reasoning). TOPPO's tasks differ in data and in input (one-hot), so their gradients can align only through shared features. In our dense setting every head shares the same inputs and the same samples and differs only in reward and output weights. A collapse to near-identical values there is stronger than gradient co-linearity.
- **A cheap, directly comparable test** (my proposal):
  - Add pre-activation LayerNorm (TOPPO's LN-c) to our 64-64 trunk.
  - Report TOPPO's D2 metric, the mean off-diagonal |cos| of the per-head trunk-gradients, for the shared, wide-shared and split critics, next to our value correlations. Reviewers can then place our numbers against TOPPO's 0.34 and 0.20.

### Gaps
- **Appendix E.1 (MTSAC port) was not read in full.** So I do not know whether TOPPO ran co-linearity diagnostics on an off-policy critic.
- **TOPPO's code and W&B logs were not inspected.** Whether the cosine pools all critic parameters or is computed per layer is taken from the text.
- **The "larger networks give near-orthogonal gradients" statement is unsupported in the parts read.** No experiment backs it there.

## Q2. MultiCriticAL (ICLR 2022): full text. What did it measure about interference between task values in a shared critic, and with what architecture?

### Takeaway
I read MultiCriticAL in full (21 pages). OpenReview blocks automated access, so the PDF came from a Wayback Machine snapshot of the OpenReview PDF.

The paper compares three critics:
- a single task-conditioned critic with one output (MTPPO and MTSAC, with a one-hot task input);
- a multi-headed critic on a shared backbone (MH, one output node per style);
- separate networks per style (MN).

The heads are selected by the style weights ω_i, which are one-hot during training, so they are routed, not dense.

It measures performance only (reward, win rate, play time, style rewards). Interference is stated as a hypothesis and never measured: there is no analysis of value accuracy, value similarity or gradients.

Its relevant result is that MN beat MH "on all the tested tasks", and both beat the single critic. MN has 2-17× the critic parameters of MH, so that comparison is not parameter-matched. The paper is a precedent for "separate critics beat a shared multi-head critic", but not for head collapse.

### Cited Findings
**Identity**
- Title: "Multi-Critic Actor Learning: Teaching RL Policies to Act with Style".
- Authors: Siddharth Mysore (Boston University & Electronic Arts), George Cheng & Yunqi Zhao (Electronic Arts), Kate Saenko (Boston University & MIT-IBM Watson AI Lab), Meng Wu (Electronic Arts).
- The header reads "Published as a conference paper at ICLR 2022". OpenReview id rJvY_5OzoI.

— [OpenReview forum](https://openreview.net/forum?id=rJvY_5OzoI); full text read from the [Wayback copy of the OpenReview PDF (snapshot 2026-03-06)](http://web.archive.org/web/20260306144854/https://openreview.net/pdf?id=rJvY_5OzoI) [full]

**Claims in the abstract**
- "Using a single value function (critic) shared over multiple tasks in Actor-Critic multi-task reinforcement learning (MTRL) can result in negative interference between tasks, which can compromise learning performance."
- "Explicitly distinguishing between tasks also eliminates the need for critics to learn to do so and mitigates interference between task-value estimates."
- The method "yields up to 56% performance gains over the single-critic baselines".

— [PDF](http://web.archive.org/web/20260306144854/https://openreview.net/pdf?id=rJvY_5OzoI)

**Interference is stated as a hypothesis**
- §1: "We suspected that the similarity of game-play states visited between each trained style interfered with each style's value optimization."
- §3: "We hypothesize that negative interference more strongly impacts value-function learning in multi-style tasks as task similarity may make specific style-values harder to distinguish."

— [PDF](http://web.archive.org/web/20260306144854/https://openreview.net/pdf?id=rJvY_5OzoI)

**Why a single critic is a problem** (§4): "Practically, the value-functions are learned by a single network with a single output node, regardless of the number of tasks. A single learned value-function representing multiple values may enforce continuity between the learned values, which may not necessarily reflect the true value functions and could compromise the quality of learned values and resulting policies." — [PDF](http://web.archive.org/web/20260306144854/https://openreview.net/pdf?id=rJvY_5OzoI)

**Architecture**
- §4: "Practically, multi-valued critics may be represented by separate independent critic networks – multi-network (MN) MultiCriticAL – or a single base critic network with multiple heads, where each head represents a different learned value – multi-headed (MH) MultiCriticAL. While the latter allows for some learned representations to be shared between the value functions, the former may be more beneficial when sharing is likely to lead to undesirable interference."
- App. C: "the MH configuration allows value functions to share a common backbone and instead utilizes different network heads (i.e. output nodes) for each learned value."

— [PDF](http://web.archive.org/web/20260306144854/https://openreview.net/pdf?id=rJvY_5OzoI)

**Routing**
- The critic loss is Σ_i ω_i (Q_φi(s,a) − (r_i(s,a) + Q_φi(s′, π(s̄′))))², "where, for M styles, the task/style encoding, Ω = [ω0,...,ωM] and ωi ∈ [0,1]" [math restored, Eqs. 5-6].
- Multi-style training uses "a one-hot style encoder and randomly uniformly sampling either defensive or aggressive play for each new training episode" (§5.2). Each transition therefore updates only its own style's head.
- An "explicit style interpolation" variant trains with fractional ω (Table 2).

— [PDF](http://web.archive.org/web/20260306144854/https://openreview.net/pdf?id=rJvY_5OzoI)

**Network sizes and parameter counts**
- App. E lists the hidden layers as "Path following: PPO [64,64], SAC [8,8]", "Pong: PPO [64,64], SAC [32,32]" and "Sonic the Hedgehog 1 and 2: PPO [256,256]". For Sonic, "the CNN feature extractor is shared between the actors and all critics".
- The main text instead says path-following PPO used "two hidden layers with 32 neurons each". App. E's parameter count (4,545 for single-style PPO) matches [64,64].
- Table 7, "Trainable Critic parameters per task":

  | Setting | Single critic (MT) | MH | MN |
  |---|---|---|---|
  | PPO Pong | 76,545 | 76,802 | 153,090 |
  | PPO Sonic | 201,729 | 205,841 | 3,429,393 |
  | SAC path following | 161 | 163 | 483 |

  MH adds 0.3-2.7% over the single critic, while MN has 2-17× the critic parameters of MH.

— [PDF](http://web.archive.org/web/20260306144854/https://openreview.net/pdf?id=rJvY_5OzoI)

**Setup**
- PPO and SAC, built on OpenAI Spinning Up.
- Baselines are MTPPO and MTSAC with a one-hot task input.
- "performance is averaged over 15 random seeds" (§5).

— [PDF](http://web.archive.org/web/20260306144854/https://openreview.net/pdf?id=rJvY_5OzoI)

**What the shared scalar critic did in Pong** (sparse aggressive reward against dense defensive reward): "MTSAC agents often defaulted to highly defensive play (likely due to the over-representation of the dense defensive rewards in the replay buffers) while MTPPO agents tended to learn intermediate behavior that was neither aggressive nor defensive." (§5.2) — [PDF](http://web.archive.org/web/20260306144854/https://openreview.net/pdf?id=rJvY_5OzoI)

**Sonic**: "35% and 20% improvements on Sonic the Hedgehog and 56% and 40% improvement on Sonic the Hedgehog 2 of MN-MultiCriticAL and MH-MultiCriticAL over MTPPO respectively." (§5.3) — [PDF](http://web.archive.org/web/20260306144854/https://openreview.net/pdf?id=rJvY_5OzoI)

**MN vs MH**
- §5.5: "Curiously, MN-MultiCriticAL consistently outperformed MH-MultiCriticAL on all the tested tasks, which calls into question the idea that shared representation is important and/or useful (Dewangan et al., 2018; Yang et al., 2017) and is an aspect that may warrant further study."
- App. C is softer: MN "appears to offer superior performance in most of the tested cases".

— [PDF](http://web.archive.org/web/20260306144854/https://openreview.net/pdf?id=rJvY_5OzoI)

**Cost and capacity** (§5.5)
- "MH-MultiCriticAL allows for a bounded increase to computational load (<3% as tested - see Appendix E) with an acceptable compromise in relative performance while still outperforming single-critic value learning."
- On capacity: "Increasing the network capacity of a single-critic value-function may improve the efficacy of the critic, but this is not guaranteed". This is stated, not tested.

— [PDF](http://web.archive.org/web/20260306144854/https://openreview.net/pdf?id=rJvY_5OzoI)

**What the text does not contain.** A full-text search finds no occurrence of "collapse" or "correlat". "Interference" appears only as motivation, hypothesis or conclusion, and in App. A about composing separately trained policies. — [PDF](http://web.archive.org/web/20260306144854/https://openreview.net/pdf?id=rJvY_5OzoI)

### Inferences
- **What it already establishes.** At the performance level and with routed heads, MultiCriticAL shows: separate critics > multi-head shared critic > single task-conditioned critic. Our split-vs-shared result must cite it. That "separate per-objective critics beat a shared multi-head critic" is not new in general.
- **Why it cannot pre-empt head collapse.**
  - Its heads train on disjoint samples, so there is no every-sample coupling to push them toward identical values.
  - It measured nothing about value similarity.
  - Dense heads, where every transition trains all heads as in our critic, are the regime it did not test.
- **Our control goes further.** Its MN-vs-MH comparison is confounded by 2-17× more critic parameters in MN. Our parameter-matched wide shared critic separates sharing from capacity, which MultiCriticAL did not do.
- **One anecdote matches H1/H4.** The MTSAC agents' default to "highly defensive play" is a behaviour-level case of one dense reward dominating a shared critic. That fits the dominance hypotheses (H1/H4) in our prompt; the authors attribute it to the dense rewards being over-represented in replay.

### Gaps
- The supplementary code (a zip on OpenReview) was not inspected.
- The PDF extraction scrambles the Table 1-3 columns, so only numbers stated in the text, plus the checked Table 7 parameter counts, are quoted.
- The OpenReview reviews and discussion were not read, because of the bot challenge.

## Q3. Forward-citation sweep: papers citing HRA (van Seijen et al. 2017), SAC-D (MacGlashan et al., NeurIPS 2022) and GCR-PPO (arXiv 2509.14816)

### Takeaway
I swept two independent citation indexes, Semantic Scholar and OpenAlex. Google Scholar returned a page with no parsable results.

The sweep covered 302 papers citing HRA, 13-14 citing SAC-D and 12 citing GCR-PPO. None reports collapse, correlation or loss of distinctness between the heads of a decomposed or multi-head critic. Every "collapse" in this literature means something else: scalarisation collapsing priorities, reward or behaviour collapse, or action-gap collapse.

Three citing papers cannot be fully excluded, because I could not access their full text:
- an IROS 2026 role-based reward decomposition study comparing single-critic and multi-critic designs;
- a 2025 Applied Intelligence paper on SAC with a multi-head critic;
- a 2026 paper on safe RL for batch processes with multi-head critics.

### Cited Findings
**Coverage**
- **HRA** (arXiv 1706.04208).
  - Semantic Scholar lists 302 citing papers, and all 302 were retrieved. 69 have no abstract and were screened by title. — [S2 API](https://api.semanticscholar.org/graph/v1/paper/arXiv:1706.04208/citations)
  - OpenAlex (W2624731731) gives cited_by_count 309. The first 200 were retrieved before cursor paging returned HTTP 401. 29 of them are not in S2, and those were screened. — [OpenAlex](https://api.openalex.org/works?filter=cites:W2624731731)
- **SAC-D** (arXiv 2206.13901).
  - Semantic Scholar lists 13 citing papers. — [S2 API](https://api.semanticscholar.org/graph/v1/paper/arXiv:2206.13901/citations)
  - OpenAlex (W7133192336, the NeurIPS DOI 10.52202/068431-0872) gives cited_by_count 14; 3 are not in S2. — [OpenAlex](https://api.openalex.org/works?filter=cites:W7133192336)
- **GCR-PPO**, now ICRA 2026 (DOI 10.1109/ICRA57385.2026.11695973; DBLP conf/icra/MunnTBGH26).
  - Semantic Scholar lists 12 citing papers. — [S2 API](https://api.semanticscholar.org/graph/v1/paper/arXiv:2509.14816/citations)
  - OpenAlex (W4417077596) also gives 12; 3 are not in S2. — [OpenAlex](https://api.openalex.org/works?filter=cites:W4417077596)

**Screening method**
- A paper was flagged when its title or abstract contained both:
  - an architecture term: multi-head, heads, decomposed, vector-valued, reward component, multi-critic, shared or separate critic, hybrid reward;
  - an issue term: collapse, co-linear, correlat, interfer, rank, similar, redundan, homogen, conflict, entangle, negative transfer, dominat.
- This flagged 14 HRA citers, 2 SAC-D citers and 1 GCR-PPO citer.
- I read all flagged abstracts, all SAC-D and GCR-PPO citer titles, and the abstracts of the candidates below.

— [S2 API](https://api.semanticscholar.org/graph/v1/paper/arXiv:1706.04208/citations)

**Candidates, closest first**
- **"Role-Based Reward Decomposition for Legged Locomotion Reinforcement Learning"**, IROS 2026 [abstract; no arXiv version found].
  - "This coupling can introduce value interference and destabilize learning."
  - "we compare single-critic, multi-critic, and role-specialized multi-agent realizations under a protocol varying only the value and actor architecture".
  - "Scalarized training then exploits a dominant minority objective while behavior collapses, per-stream multi-critic normalization preserves the minority objective".

  Here "value interference" refers to the single scalar critic and "collapses" to behaviour. The abstract does not say whether the multi-critic shares a trunk. — [DOI 10.1109/IROS61738.2026.11776619](https://doi.org/10.1109/IROS61738.2026.11776619); abstract via the [S2 API](https://api.semanticscholar.org/graph/v1/paper/arXiv:1706.04208/citations)
- **Fan et al., "Quadruped robot locomotion via soft actor-critic with muti-head critic and dynamic policy gradient"**, Applied Intelligence 55(8): 720 (2025) [title and bibliographic record only]. It is an off-policy SAC with a multi-head critic, and the citing paper most likely to have met our phenomenon. — [dblp](https://dblp1.uni-trier.de/rec/journals/apin/FanPSLGT25.html)
- **"Efficient Safe Reinforcement Learning for Batch Processes with Multi-Head Critics"** (2026) [title only]. S2 has no abstract, and a web search did not find it. — [S2 API](https://api.semanticscholar.org/graph/v1/paper/arXiv:1706.04208/citations)
- **MD3QN, "Distributional Reinforcement Learning for Multi-Dimensional Reward Functions"** (NeurIPS 2021, arXiv 2110.13578) [abstract]. It models "the rich reward correlation between the randomness of different sources". Correlation is treated as signal to model, not as a pathology. — [arXiv](https://arxiv.org/abs/2110.13578)
- **Lehnert, Laroche and van Seijen, "On Value Function Representation of Long Horizon Problems"** (AAAI 2018) [abstract]. Action gaps "can collapse if a long planning horizon is used", which is a different sense of collapse. The authors suggest value functions "should be decomposed along bottlenecks of the MDP's transition dynamics". — [AAAI DOI](https://doi.org/10.1609/aaai.v32i1.11646)
- **"About the attractor phenomenon in decomposed reinforcement learning"** (2018) [title only]. — [S2 API](https://api.semanticscholar.org/graph/v1/paper/arXiv:1706.04208/citations)
- **SAC-D citers.**
  - "Multi-Discounting Reinforcement Learning Based on Reward Decomposition" (IEEE TETC 2026) [abstract]: "MDMDP allows practitioners to set separate discount factors for different reward components". The abstract has no interference or collapse analysis.
  - "DreamerVFD" (AIAA SciTech 2026, DOI 10.2514/6.2026-0348) [abstract excerpt only]: interpretable model-based planning with VFD.
  - The remaining citers are unrelated: pitch estimation, XR codec adaptation, offline-RL explanation, AGV scheduling, multi-agent team credo.

  — [S2 API](https://api.semanticscholar.org/graph/v1/paper/arXiv:2206.13901/citations); [OpenAlex](https://api.openalex.org/works?filter=cites:W7133192336)
- **"Beyond Scalar Rewards: Distributional RL with Preordered Objectives…"** (ICRA 2026, arXiv 2603.20230) cites both HRA and SAC-D [abstract]. Its "collapse" is in the scalarisation sense: "weighted summation, which collapses their relative priorities". — [arXiv](https://arxiv.org/abs/2603.20230)
- **GCR-PPO citers** [abstracts]:
  - the AdvA tokamak controller: "reward scalarisation collapses objective-specific temporal credit" — [arXiv 2608.20834](https://arxiv.org/abs/2608.20834);
  - micro-swarm MO-MARL: "Without PCGrad, energy and smoothness rewards collapse during training" — [arXiv 2605.25025](https://arxiv.org/abs/2605.25025);
  - UAV MADDPG-TDec (EITCE 2026): "For each subtask, we design dedicated reward functions and independent Critic networks" — [S2 API](https://api.semanticscholar.org/graph/v1/paper/arXiv:2509.14816/citations);
  - PROMO (2610.01260), DAMPER (2609.38903) and DODGER (2609.38873) contain no critic-head analysis in their abstracts — [S2 API](https://api.semanticscholar.org/graph/v1/paper/arXiv:2509.14816/citations).
- **Google Scholar.** The title query for SAC-D returned an 83 KB page, but no "Cited by" entries could be parsed, so it was not used. — [Google Scholar query](https://scholar.google.com/scholar?q=%22Value+Function+Decomposition+for+Iterative+Design+of+Reinforcement+Learning+Agents%22)

### Inferences
- **The citing literature is silent on head collapse.** Across the papers that cite the three decomposed-critic anchors, no paper analyses whether the heads stay distinct.
- **The SAC-D line is thin.** SAC-D has 13-14 citations in both indexes, mostly applications and explainability work.
- **What remains open.** The forward-citation evidence cannot exclude the inaccessible Fan et al. (2025) and the IROS 2026 role-based paper. It does reduce the risk to those two papers plus the untraced batch-process paper.

### Gaps
- Fan et al. 2025 (Springer, paywalled), the IROS 2026 role-based paper (IEEE) and the 2026 batch-process paper were not read. All three should be checked before submission.
- 69 HRA citers without abstracts were screened by title only.
- The OpenAlex HRA list stopped at 200 of 309 records. Semantic Scholar's list was complete (302).
- Citation indexes lag, so papers posted in the last few weeks may be missing.

## Q4. Direct phrase searches, 2017-2026

### Takeaway
I ran three extended web searches with different phrasings and a systematic arXiv API sweep of 18 abstract queries (268 unique results). None found a paper reporting near-copy value heads in a decomposed or multi-head critic. The nearest items are:
- TOPPO (Q1).
- CDE, where disagreement between bootstrapped critic heads shrinks over training. These are ensemble heads trained on a common target, and the authors treat the shrinking as desired.
- Joshi et al. 2025, where gradient conflict in multi-task PPO is concentrated in the critic. That is the opposite of collapse.
- PASTA, where early scalarisation "destroys information about gradient conflict".

The bare term "head collapse" is already used in other fields.

### Cited Findings
**The arXiv API queries.** All searched the abstract field, sorted by date:
- "multi-head critic"; "multi-headed critic"; "decomposed critic"; "vector-valued critic"; "multi-head value"; "vector critic"; "multi-objective critic"; "per-objective critic" (no results).
- "value heads" AND critic; "reward decomposition" AND critic; "value decomposition" AND "reward components".
- "head collapse"; critic AND "co-linear"; critic AND "collinear" (200 results, mostly physics noise).
- "shared critic" AND (interference OR collapse); "multi-critic" AND (shared OR interference).
- "decomposed value function" AND (collapse OR interference OR correlated); "reward components" AND critic AND (collapse OR interference OR correlated).

— [arXiv API, e.g. abs:"multi-head critic"](http://export.arxiv.org/api/query?search_query=abs:%22multi-head%20critic%22&max_results=200)

**CDE: "Curiosity-Driven Exploration for Efficient Reinforcement Learning in Large Language Models"** (arXiv 2509.09675) [partial, text search]
- The exploration bonus is the standard deviation across K bootstrapped value heads.
- "this average decreases steadily as training progresses. The decline reflects that, with more training, similar trajectories are revisited more frequently, leading to reduced disagreement among critic heads."
- These heads are ensemble members of one value function, and the decline is treated as desired, like a count-based bonus.

— [arXiv HTML](https://arxiv.org/html/2509.09675v1)

**Joshi, Xu, Liu, Stone and Zhang, "Benchmarking Massively Parallelized Multi-Task Reinforcement Learning for Robotics Tasks"** (RLJ / RLC 2025, arXiv 2507.23172) [partial]
- "Multi-task RL struggles primarily with value estimation rather than policy learning, as gradient conflicts mostly impact the critic function."
- Fig. 6 shows "The average gradient cosine similarity across all task pairs in MT10-rand for both actor and critic networks".
- "demonstrating that gradient conflicts also arise when learning the critic network in MT-PPO. However, we do not observe similar conflicts in policy optimization."

— [arXiv HTML](https://arxiv.org/html/2507.23172)

**PASTA: "Adaptive Smooth Tchebycheff Attention for Multi-Objective Policy Optimization"** (arXiv 2605.12771) [partial]: "Rather than scalarizing Â immediately, which effectively destroys information about gradient conflict, we use this vector during the actor's optimization." — [arXiv HTML](https://arxiv.org/html/2605.12771)

**VaDar: "Implicit Goal Conditioning via Value Disaggregation"** (arXiv 2610.26629) [partial]
- "VaDar supervises all subgoals on each transition, updating a shared critic representation. We hypothesize that this enables an implicit Bellman backup that improves final-goal value estimates."
- Its task-and-safety variant is a 2-output vector critic, "a 4-layer MLP of width 512 with layer norm, ensembled over 10 members".
- A text search finds no head-similarity analysis.

— [arXiv HTML](https://arxiv.org/html/2610.26629)

**RIDGE** (arXiv 2611.05601, Nov 2026) [partial]
- It uses "a multi-head critic that trains one value head per persona on its own stationary signal".
- "multi-head critic isolates each persona's value estimate from the shifting mixture".
- A text search finds no head-similarity analysis.

— [arXiv HTML](https://arxiv.org/html/2611.05601)

**LOCU** (arXiv 2610.26256) [partial]
- "interactions among constraint gradients often yield ill-conditioned update directions, causing numerically fragile steps and unintended cancellation or redundant overlap in constraint corrections".
- The near-collinear directions are induced by design ("redundancy variants … induce near-collinear constraint directions") and handled on the policy side, not traced to a shared cost critic.

— [arXiv HTML](https://arxiv.org/html/2610.26256)

**The term "head collapse" is already in use elsewhere**
- Multiple-hypothesis learning: "preventing the head collapse observed in conventional multi-head architectures" (Compote, arXiv 2610.37005).
- Mixtures: "reducing head collapse" (GEM-FI, arXiv 2605.03750).
- Transformer attention: "the head collapse phenomenon observed in prior work, where only a fixed subset of attention heads contributes to generation" (arXiv 2602.01203).

"Value flattening" is also in use, for a scalar PPO critic in LLM training (arXiv 2609.18708; title and keyword search only).

— [arXiv API, abs:"head collapse"](http://export.arxiv.org/api/query?search_query=abs:%22head%20collapse%22&max_results=200)

**The web searches** surfaced only works already covered in `decomposed_critic_collapse.md`: HARC (2608.09762), Explicit Critic Guidance (2605.27736), Shared Actors Need Not Share Critics (2608.26481), D3PO, MOMA-AC and GCR-PPO. — [HARC](https://arxiv.org/abs/2608.09762); [Liang et al.](https://arxiv.org/abs/2605.27736); [Liu et al.](https://arxiv.org/abs/2608.26481)

### Inferences
- **What the literature does describe.** Across the 2017-2026 phrasings:
  1. heads or critics separated to avoid interference;
  2. gradient conflict in shared critics;
  3. gradient co-linearity in a shared multi-task critic (TOPPO);
  4. diversity loss in ensemble heads trained on a common target.

  What it does not describe is reward-decomposed heads with distinct targets converging to the same values.
- **Naming.** Avoid these terms, which are already taken:
  - bare "head collapse" (multiple-hypothesis learning, mixtures, transformers);
  - "co-linear collapse" (TOPPO);
  - "value function collapse" (MED-RL, for ensembles);
  - "value function interference" (Vamplew et al.).

  Use "value-head collapse in reward-decomposed critics", or "group-value collapse". Define it operationally as the mean pairwise correlation of head outputs on replay states, compared against separate critics and against per-group Monte Carlo returns.

### Gaps
- The arXiv API searches titles and abstracts only. A collapse observation buried in the body or appendix of a paper whose abstract does not use these terms would be missed.
- Non-arXiv venues (IEEE and Springer journals) were reached only through S2 and OpenAlex abstracts.

## Q5. Verdict: which parts of our finding are already published, which appear new, and a safe wording for the novelty claim

### Takeaway
**Already published:**
1. Shared critics across tasks or styles interfere, and separate critics can do better: MultiCriticAL (routed heads, performance only), Liang et al. 2026, Liu et al. 2026.
2. Per-task critic gradients on a shared critic become co-linear, with about 497× scale spread, leaving an aggregator little to act on: TOPPO (on-policy, routed, gradient-level, critic-side).
3. Scale dominance and rare-termination losses in shared multi-head critics: multi-task PopArt, SAC-D.
4. Diversity loss among ensemble heads on a shared torso: Bootstrapped DQN, MED-RL, CDE.

**Found nowhere:**
1. Value-level, near-copy collapse of separately supervised, densely trained reward-decomposed heads in an off-policy TD critic.
2. The same correlation in the TD targets.
3. The resulting near-parallel per-objective actor gradients, which make actor-side multi-objective aggregators reduce to the sum.
4. The parameter-matched control showing the collapse is not a capacity effect.

### Cited Findings
| # | Part of our finding | Status | Closest prior work |
|---|---|---|---|
| 1 | A shared critic across tasks or styles interferes; separate critics beat a multi-head shared critic | **Published** (routed heads, performance only, not parameter-matched) | [MultiCriticAL §5.5](http://web.archive.org/web/20260306144854/https://openreview.net/pdf?id=rJvY_5OzoI); separate heads beat a shared head in [Liang et al. 2026](https://arxiv.org/html/2605.27736); routed multihead in [Liu et al. 2026](https://arxiv.org/html/2608.26481v1) |
| 2 | Per-objective gradients on a shared critic align or "collapse toward shared-feature directions" | **Published** for per-task critic-loss gradients in a task-conditioned, routed, on-policy critic (mean \|cos\| 0.34 → 0.20 with LN) | [TOPPO §3 (D2), §4.2](https://arxiv.org/pdf/2605.11473v3) |
| 3 | Per-objective gradient scale on a shared critic spans orders of magnitude | **Published** (497× critic against 4.1× actor) | [TOPPO §1](https://arxiv.org/pdf/2605.11473v3); [multi-task PopArt (Hessel et al. 2019)](https://ar5iv.labs.arxiv.org/html/1809.04474) |
| 4 | Rare terminations cause very large component losses and gradient conflict in a shared multi-head critic | **Published** | [SAC-D App. E](https://ar5iv.labs.arxiv.org/html/2206.13901) |
| 5 | A self-reinforcing loop in a shared critic | **Published, other mechanism** (a PPO data-collection loop, not bootstrapping) | [TOPPO §1](https://arxiv.org/pdf/2605.11473v3) |
| 6 | Heads on a shared torso lose diversity when they share a target | **Published** (ensembles) | [Bootstrapped DQN](https://ar5iv.labs.arxiv.org/html/1602.04621); [MED-RL](https://ar5iv.labs.arxiv.org/html/2006.13823); [CDE](https://arxiv.org/html/2509.09675v1) |
| 7 | Value-level collapse (corr 0.93-1.00) of separately supervised, dense reward-group heads in an off-policy TD critic | **Not found** | Nearest: TOPPO (gradient-level, routed); MultiCriticAL (no measurement) |
| 8 | The collapse also appears in the TD targets (propagated through the target network) | **Not found** | Theory for single and auxiliary TD representations only: [Kumar et al. 2021](https://arxiv.org/abs/2010.14498); [Lyle et al. 2021](https://ar5iv.labs.arxiv.org/html/2102.13089) |
| 9 | The collapsed critic makes per-objective **actor** gradients near-parallel, so actor-side aggregators reduce to the sum and the conflict is hidden | **Not found** | Nearest wording: TOPPO D2, "little task-specific direction for any aggregator" (critic side); [PASTA](https://arxiv.org/html/2605.12771) (scalarisation destroys conflict information) |
| 10 | Not a capacity effect: a parameter-matched wider shared critic collapses, separate critics do not | **Not found** | TOPPO asserts larger nets give near-orthogonal gradients, untested in the parts read; MultiCriticAL calls capacity "not guaranteed" and does not test it (its MN has 2-17× MH's parameters); [Kim et al. 2024](https://arxiv.org/html/2308.12517v4) matches parameters the other way (reduced separate critics) |
| 11 | Per-group target normalisation removes the collapse but worsens control | **Related only** | PopArt equalises task influence ([Hessel et al.](https://ar5iv.labs.arxiv.org/html/1809.04474); [TOPPO](https://arxiv.org/pdf/2605.11473v3)); PopArt lowered gains in [Liu et al.](https://arxiv.org/html/2608.26481v1). No paper reports distinct heads together with worse control |

### Inferences
**Suggested novelty wording** (my proposal; safe given the evidence above):

> In a reward-decomposed TD3 critic whose heads share one trunk, the per-objective heads converge to near-copies of each other, even though each head is trained on every transition with its own reward and its own bootstrapped TD target. Their pairwise correlation reaches 0.93–1.00 late in training, against +0.30 to +0.75 for separate per-objective critics trained on the same data, and the TD targets inherit the same correlation. Because the actor's per-objective gradients pass through these heads, they become near-parallel, and every multi-objective gradient aggregator we tested reduces to the plain sum: the shared critic hides the conflict that separate critics expose. A wider shared critic with the same parameter count as the separate critics still collapses. To our knowledge, neither this value-level collapse nor its effect on actor-side gradient aggregation has been reported. It differs from the co-linearity of per-task critic-loss gradients in multi-task PPO (Li et al., 2026) and from the performance advantage of separate over multi-headed critics in task-routed multi-style RL (Mysore et al., 2022).

Add the hedge "in the literature we surveyed up to 27 November 2026 (arXiv, Semantic Scholar and OpenAlex, including forward citations of HRA, SAC-D and GCR-PPO)".

**Do not claim** any of the following:
- the first to observe collapse or co-linearity in a shared multi-objective or multi-task critic (TOPPO);
- the first to show that separate critics beat a shared multi-head critic (MultiCriticAL; Liang et al. 2026);
- the first to show scale imbalance or dominance in shared critics (TOPPO, PopArt, SAC-D);
- the bare terms "head collapse" or "co-linear collapse".

**To make the claim reviewer-proof** (my proposals):
1. Report the correlation of per-group discounted Monte Carlo returns as the reference level (see `decomposed_critic_collapse.md`, Q1 Inferences). This shows the shared critic exceeds the true value correlation.
2. Report TOPPO's D2 metric (mean off-diagonal |cos| of the per-head trunk gradients) for the shared, wide-shared and split critics, so readers can compare directly with TOPPO's 0.34 and 0.20.
3. Include critic LayerNorm (TOPPO's LN-c) as the literature-baseline fix, and SAC-D's recipe as the decomposed-critic baseline.

### Gaps
- **Candidates I could not verify:**
  - Fan et al. 2025 (SAC with a multi-head critic, Applied Intelligence);
  - the IROS 2026 role-based reward decomposition paper (single-critic vs multi-critic);
  - the 2026 batch-process multi-head-critic paper;
  - TOPPO App. E.1, its off-policy MTSAC port.
- **Body-text reports.** A report inside a paper whose abstract does not mention heads, collapse or correlation cannot be excluded by abstract-level sweeps and citation screening.
