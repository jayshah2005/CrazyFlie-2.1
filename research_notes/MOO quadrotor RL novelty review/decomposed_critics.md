# Decomposed Value Functions and Multi-Head Critics: Prior Work and Reported Failure Modes

Scope: reward/value decomposition, multi-head critics (shared trunk vs separate networks), and whether prior work reports (a) per-head value collapse, (b) interference between heads through shared layers, (c) a shared critic hiding objective/gradient conflict, (d) effects of output scaling / head normalization. Search date: 2026-11-25. About 20 tool calls. Primary PDFs were read for SAC-D and MOPPO, and HTML for D3PO, HRA (ar5iv) and the dual/unified humanoid paper. Other items come from abstracts or listing pages, as marked.

## Q1. Inventory: reward/value decomposition and multi-head critic work

### Takeaway
The shared-trunk, summed-heads critic is an established design. It goes from HRA (2017) through SAC-D (NeurIPS 2022) to 2026 robotics (HARC) and LLM (reward-wise value estimation) work. Almost all of these papers present it as beneficial and give little failure analysis. The main exception is SAC-D, which says naive multi-head training underperforms and needs a conflict-averse gradient fix.

### Cited Findings
**Foundational value-based decomposition**
- HRA (van Seijen et al., NeurIPS 2017): the reward is split into n components, each with its own head and target y_k = R_k + γ max Q_k(s′,a′). The loss sums the squared errors over heads, and Q_HRA = Σ_k Q_k. "In general, different agents can share multiple lower-level layers of a deep Q-network." In fruit collection, one shared 250-unit hidden layer feeds 10 heads. — [HRA, ar5iv](https://ar5iv.labs.arxiv.org/html/1706.04208); [arXiv 1706.04208](https://arxiv.org/abs/1706.04208v2); [NeurIPS PDF](https://proceedings.nips.cc/paper_files/paper/2017/file/1264a061d82a2edae1574b07249800d6-Paper.pdf)
- HRA Ms. Pac-Man heads are tabular GVFs ("small enough to be represented without any function approximation"). The headline Pac-Man result therefore does not test a shared deep trunk. — [HRA, ar5iv](https://ar5iv.labs.arxiv.org/html/1706.04208)
- Juozapaitis et al. (2019) proposed reward decomposition as a local explanation method that builds on HRA-style per-component Q_c. It introduced Reward Difference eXplanations (RDX) and Minimal Sufficient eXplanations (MSX). I did not read the original paper (an IJCAI 2019 XAI workshop paper, venue from memory and not verified). The description here comes from secondary sources. — [Septon et al. 2022, arXiv 2210.11825](https://arxiv.org/abs/2210.11825); [XRL survey, arXiv 2202.08434](https://arxiv.org/pdf/2202.08434)

**Actor-critic decomposition**
- SAC-D (MacGlashan, Archer, Devlic, Seno, Sherstan, Wurman, Stone; Sony AI / UT Austin; NeurIPS 2022). Each reward component (plus entropy as an m+1'th component) gets its own Q head, and the composite Q is the sum. — [arXiv 2206.13901](https://arxiv.org/abs/2206.13901v2); [NeurIPS 2022 PDF](https://papers.neurips.cc/paper_files/paper/2022/file/4eb2c0adafbe71269f3a772c130f9e53-Paper-Conference.pdf)
- SAC-D introduces a per-component "influence" metric (each component's effect on decisions). It is used to diagnose a reward bug where the agent stands still in a local minimum. — [NeurIPS 2022 PDF](https://papers.neurips.cc/paper_files/paper/2022/file/4eb2c0adafbe71269f3a772c130f9e53-Paper-Conference.pdf)
- Reward-adaptive RL for bipedal locomotion (arXiv 2107.01908): a multi-head critic with one head per sub-reward. The overall Q is the sum of the Q_k, and the heads "share multiple lower-level layers of a critic network." — [arXiv 2107.01908](https://arxiv.org/pdf/2107.01908)
- MultiCriticAL (ICLR 2022; Mysore et al., authors from memory). It contrasts separate independent critic networks with "a single base critic network with multiple heads." It says sharing representations can help, but separate networks "may be more beneficial when sharing is likely to lead to undesirable interference." — [OpenReview PDF](https://openreview.net/pdf?id=rJvY_5OzoI) (read through a search snippet only)
- SLIM, Skill Learning with Multiple Critics (arXiv 2402.00823): one dedicated critic per reward component, motivated by avoiding interference between components that "can stall convergence." — [arXiv 2402.00823](https://arxiv.org/html/2402.00823) (snippet only)

**Robotics, 2026**
- HARC (Li, Zhang, Zhang, Torielli, Gasperini, Laurenzi, Muratore, Ajoudani, Tsagarakis; arXiv 2608.09762, 10 Aug 2026). CTDE combined with HRA: the arm and gripper actors share one multi-head critic with a sparse-task head and a potential-based grasp head, each with its own TD target. The authors report that the critic "exhibits substantially larger gradient norms when trained on real-world observations." Success rates rise from 60% to 80% (tennis ball), 60% to 90% (banana) and 25% to 95% (simulated block relocation) against a SOTA baseline. — [arXiv abs 2608.09762](https://arxiv.org/abs/2608.09762) (abstract only; the full PDF was too large to fetch)
- "Critic Architecture Matters: Dual vs. Unified Critics for Humanoid Loco-Manipulation" (Yardımcı, arXiv 2606.11891v3, Sep 2026). Fully separate critics plus actors (dual) against one critic on the summed reward (unified). Reach rate 65.2% vs 53.8%, time-to-reach 6.5 vs 22.6 steps. The author warns that critic architecture and update rule change together ("cannot separate the critic from the update rule"), that each configuration is a single unseeded run, and that the reward tables differ. The unified critic is a scalar critic, not a multi-head one. — [arXiv HTML 2606.11891](https://arxiv.org/html/2606.11891)

**Multi-objective RL (preference-conditioned)**
- D3PO (Ambadkar et al., arXiv 2602.07764, Feb 2026). A PPO critic with a shared body over (s, ω) and d heads, each predicting "the unweighted value of a single objective." Weighting is applied late. Its "representational collapse"/"mode collapse" is a policy failure ("nearly identical behaviors across wide regions of the preference simplex"), attributed to "destructive gradient interference caused by premature scalarization." There is no multi-head vs scalar critic ablation and no per-objective critic normalization. — [arXiv HTML 2602.07764v1](https://arxiv.org/html/2602.07764v1)
- MOPPO / MOA2C (Terekhov & Gulcehre, EPFL, arXiv 2407.16807, 2024). Uses multi-task PopArt per objective, so the advantage is α⊺(Q−V)/σ with component-wise σ. Here "shared vs non-shared trunk" means actor-critic trunk sharing: "Non-shared trunk architectures consistently perform slightly worse than their shared counterparts". Multi-body networks perform best. The paper argues that competing rewards' gradients "may differ, resulting in canceling each other in the shared trunk's parameters." This is stated as motivation and not measured. — [arXiv 2407.16807](https://arxiv.org/pdf/2407.16807)

**LLM / RLHF multi-reward critics, 2025-2026**
- "Reward-Wise Value Estimation for Multi-Reward Optimization in Large Language Models" (Wei, Wang, Li, Chen, Frunza, Hong; ICML 2026 workshop poster). A multi-head critic with reward-wise values in one model. The paper claims "lower irreducible error than the standard single-head critic loss" and reports gains in PPO on agentic search and math. It frames separate per-reward critics as "computationally inefficient". The listing shows no interference analysis. — [ICML 2026 listing](https://icml.cc/virtual/2026/83684) (abstract only; no arXiv link found)
- SMOPD (arXiv 2608.03092) is a specialize-and-merge alternative for multi-reward RL. It appeared in search results and was not read. — [arXiv 2608.03092](https://arxiv.org/pdf/2608.03092)

**Multi-task normalization**
- PopArt (van Hasselt et al. 2016) adaptively normalizes targets while "preserving outputs precisely." Multi-task PopArt (Hessel et al., AAAI 2019) keeps a mean and variance for each task so that "all tasks have a similar impact on the learning dynamics." PopArt-IMPALA reached superhuman median performance on 57 Atari games and SOTA on DMLab-30. — [arXiv 1809.04474](https://arxiv.org/pdf/1809.04474); [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/4266); [DeepMind blog](https://deepmind.google/discover/blog/preserving-outputs-precisely-while-adaptively-rescaling-targets/)

### Inferences
- Reward-component heads (HRA, SAC-D, HARC, reward-wise LLM critic) and multi-task PopArt heads are structurally the same thing: K output heads on a shared torso, one per return stream. The multi-task literature (PopArt, PCGrad/CAGrad) is therefore directly applicable to decomposed critics.

### Gaps
- Distributional critics with multiple heads per reward component: not searched within the budget. No source found.
- The original Juozapaitis et al. 2019 PDF was not retrieved. Its venue and title come from secondary sources and memory.
- The full HARC PDF was not read (it exceeded the size limit). Head normalization details and any monolithic-critic ablation are unknown.

## Q2. Do prior works report (a) collapse of per-head values into copies of one?

### Takeaway
I found no paper that reports per-head critic values collapsing into copies of one another. The "collapse" in D3PO and MOPPO is policy/behaviour collapse across preferences, not critic heads becoming identical. This looks like an unreported failure mode as far as this search reached.

### Cited Findings
- D3PO's "representational collapse" means the policy produces "nearly identical behaviors across wide regions of the preference simplex". It is a policy failure fixed by a diversity regularizer, not critic-head collapse. — [arXiv 2602.07764v1](https://arxiv.org/html/2602.07764v1)
- MOPPO's "collapse" refers to entropy collapse: "All 12 considered hyperparameter configurations lead to collapse if we fix λ=10⁻³." — [arXiv 2407.16807](https://arxiv.org/pdf/2407.16807)
- SAC-D discusses underestimation bias from per-component twin minimums and recommends taking all component predictions from the twin network with the lower composite Q. It does not report heads becoming copies. — [NeurIPS 2022 PDF](https://papers.neurips.cc/paper_files/paper/2022/file/4eb2c0adafbe71269f3a772c130f9e53-Paper-Conference.pdf)

### Inferences
- The closest adjacent literature is general representation collapse in value networks: implicit under-parameterization (Kumar et al. 2020, arXiv 2010.14498) and capacity loss (Lyle et al. 2022, arXiv 2204.09560). These are cited from memory and were not fetched this session; they concern single-value networks, not head-to-head similarity. A finding of K heads converging to near-copies would be novel relative to everything found here.

### Gaps
- No source measures similarity or correlation between per-head value estimates in a shared-trunk critic.

## Q3. (b) Interference between value heads through shared hidden layers

### Takeaway
Interference through shared layers is widely asserted as motivation (MultiCriticAL, SLIM, MOPPO, D3PO). Only SAC-D gives empirical evidence specific to a multi-head critic: naive multi-head SAC-D "significantly underperforms SAC," and on Ant, termination-induced Q errors cause "catastrophic gradient conflicts." That evidence is about task performance and is not a direct measurement of head-to-head interference.

### Cited Findings
- SAC-D: "simultaneous optimization of all Q_i components may introduce training problems common in multi-objective optimization: conflicting gradients, high curvature and large differences in gradient magnitudes". The authors therefore apply CAGrad with each component treated as a task. "SAC-D-Naive significantly underperforms SAC," and "In Ant, SAC-D (without CAGrad) underperforms all other methods… infrequent environment termination causes large Q-function errors and leads to catastrophic gradient conflicts." With CAGrad, "SAC-D meets or exceeds SAC's performance in all environments we tested." Further analysis is in their Appendices E-F, which I did not read. — [NeurIPS 2022 PDF](https://papers.neurips.cc/paper_files/paper/2022/file/4eb2c0adafbe71269f3a772c130f9e53-Paper-Conference.pdf)
- SAC-D also notes the upside: "Multi-headed prediction can also improve representation learning, as in work on auxiliary tasks." On BWH, SAC-D-CAGrad significantly outperforms SAC. — same source
- CAGrad itself (Liu, Liu, Jin, Stone, Liu, NeurIPS 2021) was designed for multi-task RL gradient conflict. — cited as ref [24] in [SAC-D](https://papers.neurips.cc/paper_files/paper/2022/file/4eb2c0adafbe71269f3a772c130f9e53-Paper-Conference.pdf); arXiv 2110.14048 (ID from memory, not fetched)
- MultiCriticAL: separate critics "may be more beneficial when sharing is likely to lead to undesirable interference." — [OpenReview](https://openreview.net/pdf?id=rJvY_5OzoI) (snippet)
- SLIM motivates per-component critics as avoiding interference that "can stall convergence." — [arXiv 2402.00823](https://arxiv.org/html/2402.00823) (snippet)
- MOPPO: competing objectives' gradient signs "may differ, resulting in canceling each other in the shared trunk's parameters." Yet it found shared actor-critic trunks slightly better than non-shared ones. — [arXiv 2407.16807](https://arxiv.org/pdf/2407.16807)
- HRA describes aggregator-level dominance rather than parameter interference: "when only a few pellets remain on the screen, the aggregator gets overwhelmed by the ghost avoider values." — [HRA ar5iv](https://ar5iv.labs.arxiv.org/html/1706.04208)
- The dual vs unified humanoid paper raises the hypothesis that a unified critic "lets locomotion reward suppress arm action magnitudes," but its own data do not support it. — [arXiv 2606.11891](https://arxiv.org/html/2606.11891)

### Inferences
- SAC-D is the closest prior to a "shared multi-head critic has gradient-conflict problems" claim and should be cited as such. It fixes the problem with CAGrad and does not diagnose it with per-head value analysis or a shared-vs-separate comparison.

### Gaps
- No source found that compares a shared-trunk multi-head critic against separate per-objective critic networks while holding the actor/update rule fixed, and reports differences in the learned per-objective values or measured gradient cosine. The humanoid paper compares separate critics with a scalar critic and confounds the update rule. MultiCriticAL may contain such an ablation, but its full text was not read.
- A paper surfaced as "Shared Actors Need Not Share Critics: Effects of Value Mismatch in Parallel RL" (alphaxiv URL with a non-standard ID). It could not be verified and should be treated as unconfirmed.

## Q4. (c) Can a shared critic hide objective/gradient conflict?

### Takeaway
Partially reported. D3PO and MOPPO argue that early scalarization (one scalar critic/advantage) causes destructive gradient interference that decomposition exposes or avoids. No paper found explicitly shows that a shared multi-head critic masks conflict that separate critics reveal, for example by measuring lower inter-objective gradient conflict under sharing.

### Cited Findings
- D3PO: mode collapse is caused by "destructive gradient interference caused by premature scalarization." Per-objective heads keep advantage signals separate until late weighting. — [arXiv 2602.07764v1](https://arxiv.org/html/2602.07764v1)
- The reward-wise LLM critic claims a single scalar critic "limit[s] flexibility across reward components." It claims a lower irreducible error for multi-head critics. — [ICML 2026](https://icml.cc/virtual/2026/83684)
- SAC-D: "Even though the scalar values of the composite Q-function are identical to those used in SAC," per-component optimization introduces conflicts. So decomposition creates or surfaces conflicts that are invisible in the scalar critic. — [SAC-D PDF](https://papers.neurips.cc/paper_files/paper/2022/file/4eb2c0adafbe71269f3a772c130f9e53-Paper-Conference.pdf)
- Humanoid dual/unified: with a shared critic, a locomotion reward "never part of the imitation objective" would enter the arm update. With separate critics, each reward reaches only its branch. — [arXiv 2606.11891](https://arxiv.org/html/2606.11891)

### Inferences
- The specific claim that a shared trunk couples heads, so that measured per-objective gradient conflict looks smaller than with independent critics, appears unaddressed in this literature. It could be a defensible novelty point if supported by controlled measurements.

### Gaps
- No direct measurement of gradient cosine or conflict, shared vs separate critics, found.

## Q5. (d) Output scaling / head normalization, and summed heads' effective step size

### Takeaway
Head normalization is known: HRA rescaled Pac-Man heads to [0,1], multi-task PopArt normalizes per task or objective, and MOPPO applies per-objective PopArt. No source found analyses how summing K heads changes the effective step size of the total value, or proposes 1/K output scaling for a decomposed critic.

### Cited Findings
- HRA Ms. Pac-Man: the additive aggregator was modified by "normalisation over the score heads between 0 and 1," and the ghost multiplier was changed to −10 to fit. The fruit-collection text says the final layer computes "the mean of all nodes across heads" but also uses fixed weights of 1 (a sum). The paper is internally inconsistent, which leaves the sum vs mean (1/K) choice unclear. — [HRA ar5iv](https://ar5iv.labs.arxiv.org/html/1706.04208)
- Multi-task PopArt: per-task mean and variance normalization so that larger-scale tasks do not dominate. — [arXiv 1809.04474](https://arxiv.org/pdf/1809.04474)
- MOPPO: without normalization the target "will unfairly favor the objectives of larger scales." It uses multi-task PopArt per objective and divides the advantages component-wise by σ. — [arXiv 2407.16807](https://arxiv.org/pdf/2407.16807)
- The dual/unified humanoid paper normalizes advantages per reward stream in the dual setup and once on the summed advantage in the unified setup. — [arXiv 2606.11891](https://arxiv.org/html/2606.11891)
- SAC-D identifies "large differences in gradient magnitudes" across components as a problem and handles it with CAGrad, not output scaling. — [SAC-D PDF](https://papers.neurips.cc/paper_files/paper/2022/file/4eb2c0adafbe71269f3a772c130f9e53-Paper-Conference.pdf)
- HARC reports large critic gradient norms on real data, from the abstract only. — [arXiv 2608.09762](https://arxiv.org/abs/2608.09762)
- A targeted search on summed-head effective learning rate and 1/K scaling returned only MARL additive decomposition (VDN: gradients "backpropagat[ed]… through the summation"), with no step-size analysis. — [VDN arXiv 1706.05296](https://arxiv.org/pdf/1706.05296)

### Inferences
- Mechanism, my own reasoning and not from any source: with Q_total = Σ_k Q_k on a shared trunk, each head's TD loss sends gradient into the trunk. A single trunk update therefore moves every head's output, and changes to the total add up across heads, roughly K-fold in the worst, fully aligned case. That is consistent with the user's observed ~2.3x step relative to TD3. Dividing by K (a mean aggregator) or per-head PopArt would counteract it. I found no published analysis of this, so the observation looks novel. It should still be framed against HRA's sum/mean ambiguity and PopArt's scale-invariance motivation.

### Gaps
- No source found that measures the per-step change in composite Q for a K-head critic against a scalar critic, or that evaluates 1/K output scaling.
- Possibly relevant work not checked: "Ensemble/REDQ-style" critic averaging and multi-task loss-weighting (e.g. GradNorm, uncertainty weighting). These were not searched.
