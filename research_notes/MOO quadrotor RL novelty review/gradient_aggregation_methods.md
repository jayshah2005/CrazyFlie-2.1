# Gradient Aggregation / Multi-Task and Multi-Objective Optimization Methods: Competitive Landscape for SVD- and Orthogonal-NMF-based "Principal Gradient" Aggregation

Notation used below: T task/objective gradients g_1..g_T stacked as columns of G (n x T; n = number of parameters). Gramian M = G^T G (T x T). Thin SVD G = U S V^T, with U (n x r) the parameter-space principal directions, V (T x r) the task-space eigenvectors of M, and S the singular values.

Provenance markers:
- [verified] means the claim was checked this session against the linked page.
- [prior knowledge] means the claim comes from the researcher's training knowledge. The link is the canonical arXiv/proceedings ID but was not re-opened this session. Spot-check these before citing in a paper.

## Q1. Inventory of core gradient aggregation methods (math, year, venue, evaluation, weaknesses)

### Takeaway
The field holds about 25 established aggregators. Each one reduces the Jacobian G to a single update direction in one of these ways:
- a min-norm point in the convex hull (MGDA family)
- pairwise projection or "surgery" (PCGrad, GradVac, GradOPS)
- a constrained or game-theoretic optimum (CAGrad, Nash-MTL, FairGrad, PCD)
- equal-projection or equal-angle rules (IMTL-G, ConFIG, UPGrad/DualProj)
- whitening of the Gramian spectrum (Aligned-MTL)
- loss-level reweighting (GradNorm, UW, DWA, RLW, FAMO, ExcessMTL)

TorchJD (23 aggregators) and LibMTL (roughly 15 or more weighting strategies) are the reference implementations a reviewer will expect a baseline comparison against.

### Cited Findings

**Reference libraries**
- TorchJD lists 23 aggregators: UPGrad, Aligned-MTL, CAGrad, ConFIG, Constant, CR-MOGM, DualProj, ExcessMTL, FairGrad, GradDrop, GradVac, IMTL-G, Krum, Mean, MGDA, MoDo, Nash-MTL, PCD, PCGrad, Random, SDMGrad, Sum, Trimmed Mean. An aggregator "combines the rows of a Jacobian matrix into a single vector". Some aggregators also expose a `Weighting` interface for the Gramian-based "autogram" engine. [verified] — [TorchJD aggregation docs](https://torchjd.org/stable/docs/aggregation/)
- LibMTL weighting modules include Aligned_MTL, CAGrad, DB_MTL, DWA, EW, GLS, GradDrop, GradNorm, GradVac, IMTL, MGDA, MoCo, Nash_MTL, PCGrad, RLW, UW, and possibly others; the list was truncated in the search result. EW sets every weight to 1/T. [verified, partial] — [LibMTL module index](https://libmtl.readthedocs.io/en/latest/py-modindex.html); [LibMTL weighting API](https://libmtl.readthedocs.io/en/latest/docs/_autoapi/LibMTL/weighting/index.html); [LibMTL paper, JMLR 2023](https://baijiong-lin.github.io/files/jmlr2023.pdf); [arXiv 2203.14338](https://arxiv.org/abs/2203.14338)
- A survey of gradient-based multi-objective deep learning, updated August 2025, is a good entry point for taxonomy. [verified, existence only] — [arXiv 2501.10945](https://arxiv.org/abs/2501.10945)

**MGDA family (min-norm / Pareto-stationary direction)**
- **MGDA** (Désidéri 2012; Sener & Koltun, NeurIPS 2018)
  - Math: d = G α*, with α* = argmin over the simplex of ||G α||². The negative of d is a common descent direction. d = 0 if and only if the point is Pareto-stationary.
  - Sener & Koltun added MGDA-UB, which uses an upper bound on the shared-representation gradients so that only one backward pass is needed.
  - Weaknesses: the direction is dominated by the smallest-norm gradient, which can stall progress; it gives no preference control. [prior knowledge] — [Sener & Koltun, arXiv 1810.04650](https://arxiv.org/abs/1810.04650); Désidéri, Comptes Rendus Math. 2012, DOI 10.1016/j.crma.2012.03.014.
- **CR-MOGM** (Zhou et al., NeurIPS 2022)
  - Analyses why MGDA, PCGrad and CAGrad can fail to converge with stochastic gradients.
  - Fix: exponentially average the composite weights λ over iterations.
  - Complexity O(ε⁻²) with O(1) batch size; the analysis assumes bounded function values. [verified via search snippet] — [NeurIPS 2022 paper](https://papers.nips.cc/paper_files/paper/2022/file/f91bd64a3620aad8e70a27ad9cb3ca57-Paper-Conference.pdf); [survey 2501.10945](https://arxiv.org/pdf/2501.10945)
- **MoCo** (Fernando et al., ICLR 2023): a stochastic MGDA with gradient tracking that corrects bias. [prior knowledge] — [arXiv 2210.12624](https://arxiv.org/abs/2210.12624)
- **MoDo** (Chen et al., NeurIPS 2023, "Three-Way Trade-Off in Multi-Objective Learning: Optimization, Generalization and Conflict-Avoidance")
  - A double-sampling stochastic MGDA.
  - Removes the bounded-function-value assumption of CR-MOGM and MoCo. [verified via search snippet] — [arXiv 2305.20057](https://arxiv.org/abs/2305.20057)
- **SDMGrad** (Xiao, Ban & Ji, NeurIPS 2023): a stochastic direction-oriented MGDA. It regularises the direction toward a target such as the average gradient, which interpolates between MGDA and mean. [prior knowledge] — [arXiv 2305.18409](https://arxiv.org/abs/2305.18409)
- **Capped MGDA** (Hu, Ho & Yu, arXiv May 2026)
  - Comes from a unified framework built on a "sufficient alignment condition". In that framework, a non-conflicting direction in the convex hull is sufficient for (optimal-rate) convergence to Pareto stationarity, and projection onto the dual cone ensures feasibility.
  - Capped MGDA is derived from CVaR and is robust in adversarial federated learning.
  - The abstract does not mention SVD or low-rank aggregators. [verified] — [arXiv 2605.30452](https://arxiv.org/abs/2605.30452)

**Projection / "gradient surgery" family**
- **PCGrad** (Yu et al., NeurIPS 2020)
  - Math: for each g_i, for each g_j in random order with g_i·g_j < 0, set g_i ← g_i − (g_i·g_j / ||g_j||²) g_j; the update is the sum of the projected gradients.
  - Evaluated on MultiMNIST, CityScapes, NYUv2, CelebA, and Meta-World MT10/MT50 (SAC).
  - Weaknesses: depends on the random order; does not converge to Pareto-stationary points in general; scales poorly as O(T²). [prior knowledge] — [arXiv 2001.06782](https://arxiv.org/abs/2001.06782)
- **GradVac** (Wang et al., ICLR 2021, multilingual NMT)
  - Like PCGrad, but it aligns pairwise cosine similarity toward an EMA target ϕ̂_ij rather than only to zero. [prior knowledge] — [arXiv 2010.05874](https://arxiv.org/abs/2010.05874)
- **GradDrop** (Chen et al., NeurIPS 2020)
  - A stochastic sign-consistency mask applied per coordinate of the summed gradient (on the activation-layer gradients).
  - Does not target Pareto stationarity. [prior knowledge] — [arXiv 2010.06808](https://arxiv.org/abs/2010.06808)
- **GradOPS** (Zhu et al., WSDM 2025)
  - Projects each gradient onto the orthogonal complement of the subspace spanned by the other tasks' gradients.
  - Claims to resolve all pairwise conflicts regardless of order and to converge to Pareto-stationary points.
  - Adds reweighting to explore trade-offs. [verified, abstract only] — [arXiv 2503.03438](https://arxiv.org/abs/2503.03438)
- **GDOD** (Dong et al., CIKM 2022): uses an orthogonal basis of the span of all task gradients. Covered in Q2 because it is SVD-adjacent. [verified] — [arXiv 2301.13465](https://arxiv.org/abs/2301.13465)

**Constrained / game-theoretic optimum**
- **CAGrad** (Liu et al., NeurIPS 2021)
  - Math: maximise the worst-case local improvement min_i g_iᵀd subject to ||d − g_0|| ≤ c||g_0||, where g_0 is the average gradient. It is solved via the dual over the simplex.
  - c = 0 recovers GD; c → ∞ approaches MGDA. Converges to the minimum of the average loss when c < 1.
  - Evaluated on NYUv2, CityScapes, and Meta-World MT10/MT50.
  - Weakness: the hyperparameter c needs tuning. [prior knowledge] — [arXiv 2110.14048](https://arxiv.org/abs/2110.14048)
- **Nash-MTL** (Navon et al., ICML 2022)
  - Math: a Nash bargaining solution. Find d = Gα with α > 0 satisfying (GᵀG)α = 1/α elementwise; it is solved by a sequence of convex programs.
  - Claims scale invariance (the solution does not change when individual losses are rescaled) and Pareto optimality.
  - Evaluated on QM9 (11 tasks), NYUv2, CityScapes, and MT10.
  - Weakness: the iterative solver is costly; the paper updates the weights only every few steps. [prior knowledge] — [arXiv 2202.01017](https://arxiv.org/abs/2202.01017)
- **FairGrad** (Ban & Ji, ICML 2024): α-fair utility maximisation over the per-task improvement rates gᵢᵀd. It generalises Nash-MTL (α = 1) and moves toward max-min fairness as α grows. [prior knowledge] — [arXiv 2402.15638](https://arxiv.org/abs/2402.15638)
- **PCD — Priority-Constrained Descent** (TMLR 2026)
  - Math: min ½||d − g̃_1||² subject to g̃_jᵀd ≥ τ||g̃_j||² for every secondary objective j. Each g̃ is RMS-normalised using an EMA of the squared gradient norm. The output is rescaled to ||g_1||.
  - Hierarchical: the primary objective is followed closely while each secondary objective gets guaranteed progress. [verified] — [TorchJD PCD doc](https://torchjd.org/stable/docs/aggregation/pcd/); [OpenReview HT01yGHLEt](https://openreview.net/forum?id=HT01yGHLEt); [arXiv 2606.29521](https://arxiv.org/abs/2606.29521)
- **ConicGrad** (arXiv Feb 2025, "Fantastic Multi-Task Gradient Updates and How to Find Them In a Cone")
  - Restricts the update to a cone around a reference direction.
  - Its toy experiments report that only ConicGrad consistently reached the global minimum, while FAMO reached the Pareto front but not the global minimum. [verified, search snippet] — [arXiv 2502.00217](https://arxiv.org/abs/2502.00217)

**Equal-projection / equal-angle rules**
- **IMTL** (Liu et al., ICLR 2021, "Towards Impartial Multi-task Learning")
  - IMTL-G is a closed-form rule: it picks the weights so that the aggregate has equal projection onto every unit task gradient.
  - IMTL-L learns loss scales. [prior knowledge] — [OpenReview, ICLR 2021](https://openreview.net/forum?id=IMPnRXEWpvr) (URL ID not re-verified)
- **ConFIG** (Liu, Chu & Thuerey, ICLR 2025 Spotlight)
  - Math: d = (Σ_i ||g_i||-weighted) · normalize(Ĝ⁺ᵀ 1), the pseudo-inverse of the matrix of unit gradients applied to the ones vector. This gives equal cosine similarity with every loss gradient and a positive dot product with each one, which the paper calls "conflict-free".
  - The update length is scaled adaptively to the degree of conflict.
  - M-ConFIG uses EMA momenta so that one loss can be backpropagated per step.
  - Evaluated on PINNs and on MTL benchmarks.
  - Weakness: needs a per-loss gradient every iteration. [verified] — [ICLR 2025 proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/94e85561a342de88b559b72c9b29f638-Abstract-Conference.html); [arXiv 2408.11104](https://arxiv.org/abs/2408.11104); [project page](https://tum-pbs.github.io/ConFIG)
- **UPGrad / DualProj** (Quinton & Rey, "Jacobian Descent for Multi-Objective Optimization"; arXiv v1 June 2024, v4 Oct 2026, listed as accepted at NeurIPS 2026)
  - Introduces Jacobian descent and the TorchJD library.
  - Early experiments use instance-wise risk minimisation (IWRM) for image classification. [verified: title, authors, versions, venue comment] — [arXiv 2406.16232](https://arxiv.org/abs/2406.16232)
  - [prior knowledge] UPGrad projects each row g_i onto the dual cone of the Jacobian rows (the set {d : g_jᵀd ≥ 0 ∀j}) and averages the projections. DualProj projects the mean gradient onto that dual cone. The paper's stated properties are non-conflicting, linear under scaling, and weighted (the output lies in the row span with data-dependent weights).

**Spectral / whitening**
- **Aligned-MTL** (Senushkin et al., CVPR 2023): SVD/eigen-based. Details are in Q2. [verified] — [CVPR 2023 paper](https://openaccess.thecvf.com/content/CVPR2023/html/Senushkin_Independent_Component_Alignment_for_Multi-Task_Learning_CVPR_2023_paper.html); [arXiv 2305.19000](https://arxiv.org/abs/2305.19000); code at github.com/SamsungLabs/MTL.

**Loss-level weighting (operates on losses, not on G directly)**
- **GradNorm** (Chen et al., ICML 2018): learns task weights so that the gradient norms ||∇_W w_i L_i|| on the last shared layer follow target ratios set by relative training rates. [prior knowledge] — [arXiv 1711.02257](https://arxiv.org/abs/1711.02257)
- **Uncertainty weighting** (Kendall et al., CVPR 2018) — [arXiv 1705.07115](https://arxiv.org/abs/1705.07115). **DWA** (Liu et al., CVPR 2019, MTAN) — [arXiv 1803.10704](https://arxiv.org/abs/1803.10704). [prior knowledge]
- **RLW** (Lin et al., TMLR 2022): random loss weights sampled each step, for example a softmax of Gaussian samples. It is surprisingly competitive, which supports the critiques in Q3. [prior knowledge] — [arXiv 2111.10603](https://arxiv.org/abs/2111.10603)
- **FAMO** (Liu et al., NeurIPS 2023): O(1)-memory/compute dynamic weighting that balances the rates of log-loss decrease without computing all task gradients. Evaluated on CelebA (40 tasks), QM9, NYUv2, CityScapes, and MT10. [prior knowledge] — [arXiv 2306.03792](https://arxiv.org/abs/2306.03792)
- **ExcessMTL** (He et al., ICML 2024, "Robust Multi-Task Learning with Excess Risks"): weights tasks by estimated excess risk rather than raw loss, for robustness to label noise. [prior knowledge] — [arXiv 2402.02009](https://arxiv.org/abs/2402.02009)
- **BayesAgg-MTL** (Achituve et al., ICML 2024): Bayesian uncertainty over task gradients, applied as per-dimension uncertainty weighting during aggregation. [verified, existence] — [arXiv 2402.04005](https://arxiv.org/abs/2402.04005)

**Robust aggregators borrowed from federated learning**
- TorchJD also ships Krum and Trimmed Mean, which are Byzantine-robust aggregators. [verified, names only] — [TorchJD docs](https://torchjd.org/stable/docs/aggregation/)

### Inferences
- The newest entries (Capped MGDA 2026, PCD 2026, the unified framework 2026, UPGrad accepted at NeurIPS 2026) concentrate on two themes: dual-cone feasibility and convergence theory. None of them is spectral. A principal-component family would therefore sit in a comparatively uncrowded corner, with Aligned-MTL as its only mainstream representative.
- Because TorchJD contains 23 aggregators, a new method paper is likely to be asked to compare against at least these baselines: Mean/Sum, MGDA, PCGrad, CAGrad, Nash-MTL, Aligned-MTL, UPGrad/DualProj, ConFIG, FAMO, and FairGrad.

### Gaps
- The exact IMTL OpenReview ID, and the venue/evaluation details for SDMGrad, FairGrad, ExcessMTL, FAMO and RLW, were not re-verified this session.
- The authors of PCD were not shown on the TorchJD page.
- GradOPS and ConicGrad update equations were not read from the full text.
- I did not do an exhaustive sweep of 2025–2026 arXiv. Other recent aggregators may exist, for example ones published at ICLR 2026 or ICML 2026, that I did not find.

## Q2. Methods using SVD / eigenvectors / principal components / low-rank structure / NMF of the gradient matrix or Gramian (overlap check)

### Takeaway
The closest prior art is Aligned-MTL. It already computes the full eigendecomposition of GᵀG and returns a sum of all principal directions with equalised singular values. Its update can be rewritten as σ_min · Σ_i u_i (v_iᵀ w). A "principal gradient" aggregator that sums or averages SVD directions must therefore differentiate itself explicitly, on these points:
- truncation to the top-k directions
- weighting by σ_i rather than equalising
- the sign/orientation rule
- the absence of whitening

Two other close methods exist:
- **GDOD** (CIKM 2022) uses an orthogonal basis of span(G) to split shared and conflicting components.
- **FedLD** (MICCAI 2024, federated learning) explicitly uses "principal gradients", the eigenvectors of GᵀG, with sign calibration.

No prior gradient aggregation method based on NMF or orthogonal NMF was found.

### Cited Findings

**Aligned-MTL ("Independent Component Alignment for Multi-Task Learning", Senushkin, Patakin, Kuznetsov, Konushin, CVPR 2023, pp. 20083–20093)** [verified from full text]
- Stability criterion: the condition number κ(G) = σ_max/σ_min. κ = 1 if and only if the gradients are orthogonal and have equal norms. The paper argues that κ captures both conflicting gradients and dominating gradients. — [ar5iv full text](https://ar5iv.labs.arxiv.org/html/2305.19000)
- Algorithm:
  1. Form M = GᵀG (T x T).
  2. Eigendecompose with `eigh`: M = VΛVᵀ.
  3. Set Σ⁻¹ = diag(1/√λ_i).
  4. Compute the aligned matrix Ĝ = σ G V Σ⁻¹ Vᵀ = σ U Vᵀ, where σ = σ_min, the smallest nonzero singular value.
  5. The update is ĝ_0 = Ĝ w = G α, with α = √λ_R V Σ⁻¹ Vᵀ w, where w are pre-set task weights (uniform by default).
  In words, every principal component is rescaled to the smallest singular value. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2305.19000)
- Interpretation: Ĝ is the closest orthogonal matrix to G in Frobenius norm (a Procrustes/polar factor), computed via SVD. — [search summary citing the paper](https://arxiv.org/pdf/2305.19000)
- Aligned-MTL-UB aligns the gradients Z with respect to the shared representation, using the bound ||G − Ĝ||²_F ≤ ||∂H/∂θ||²_F ||Z − Ẑ||²_F. This makes its cost O(1) in T. For non-encoder-decoder models such as MTAN it is a heuristic. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2305.19000)
- Theory: with Λ-Lipschitz gradients and step size ≤ 1/Λ, it converges to a Pareto-stationary point at which ∇L_0 = 0 for the pre-set weights w. The authors stress this as preference control that other methods lack. The appendix gives the convergence rate inconsistently, as both "linear" and O(1/T). — [ar5iv](https://ar5iv.labs.arxiv.org/html/2305.19000)
- Benchmarks:
  - a synthetic two-task problem
  - NYUv2 (segmentation, depth, normals) with MTAN and PSPNet
  - CityScapes two-task with MTAN
  - CityScapes three-task (instance, semantic, disparity) with PSPNet
  - Meta-World MT10 with SAC
  — [ar5iv](https://ar5iv.labs.arxiv.org/html/2305.19000)
- Available as an aggregator in both TorchJD and LibMTL. — [TorchJD](https://torchjd.org/stable/docs/aggregation/); [LibMTL index](https://libmtl.readthedocs.io/en/latest/py-modindex.html)

**GDOD (Dong et al., "Effective Gradient Descent using Orthogonal Decomposition for Multi-Task Learning", CIKM 2022, pp. 386–395; arXiv Jan 2023)**
- "Manipulates gradients of each task using an orthogonal basis decomposed from the span of all task gradients". It splits each gradient into task-shared and task-conflict components and updates along the shared components.
- Convergence is proven under convex and non-convex assumptions.
- Evaluated on recommendation-style datasets with AUC/Logloss metrics. [verified, abstract] — [arXiv 2301.13465](https://arxiv.org/abs/2301.13465)
- A search summary says the basis is the right-singular vectors of the stacked task gradients. The abstract does not name SVD explicitly, so this is unconfirmed. — [arXiv PDF 2301.13465](https://arxiv.org/pdf/2301.13465)

**FedLD — principal-gradient server aggregation (Zeng et al., "Tackling Data Heterogeneity in Federated Learning via Loss Decomposition", MICCAI 2024)**
- Abstract: "a principal gradient-based server aggregation strategy to reduce the aggregation loss". [verified] — [arXiv 2408.12300](https://arxiv.org/abs/2408.12300)
- Mechanism, from search snippets of the full text and not read directly:
  - Stack the client gradients into G and take the eigenvectors of GᵀG as a coordinate system.
  - Project and revise each local gradient in that system before aggregating.
  - The eigenvectors are unoriented, so their sign is calibrated against a reference gradient.
  — [arXiv PDF](https://arxiv.org/pdf/2408.12300); [HTML v1](https://arxiv.org/html/2408.12300v1)
- This is the most direct prior use of the term "principal gradient" for SVD-of-gradients aggregation, although it is in federated learning, not MTL/MOO.

**Continual learning (SVD of activations or gradients, used for projection, not for aggregation)**
- **GPM** (Saha, Garg & Roy, ICLR 2021) runs SVD on layer representation (activation) matrices to build a memory of important subspaces. New-task gradients are then projected orthogonal to that memory. It is not a multi-gradient aggregator. [prior knowledge] — [arXiv 2103.09762](https://arxiv.org/abs/2103.09762)
- Parameter-efficient gradient projection (arXiv 2405.13383) and GORP (ACL 2025) project gradients onto low-rank singular-vector subspaces. — [arXiv 2405.13383](https://arxiv.org/html/2405.13383v1); [ACL 2025](https://aclanthology.org/2025.acl-long.721.pdf)
- PRISM (arXiv 2605.01061, federated multimodal continual learning) keeps per-expert gradient subspace bases from the top eigenpairs of a combined covariance, and shows that server averaging preserves the orthogonality of client projections. — [arXiv 2605.01061](https://arxiv.org/pdf/2605.01061)
- A task-free continual learning paper (arXiv 2608.27070) uses the leading eigenvectors of K-FAC gradient/activation covariance factors to measure task similarity. — [arXiv 2608.27070](https://arxiv.org/pdf/2608.27070)

**Model merging (spectral, but on weights or task vectors rather than training gradients)**
- Closed-form spectral regularisation for multi-task model merging (arXiv 2606.07289, 2026) applies a per-eigendirection filter with a top-K truncation of small-eigenvalue directions. It uses one symmetric eigendecomposition per layer. — [arXiv 2606.07289](https://arxiv.org/abs/2606.07289)
- Adaptive projective gradient descent for model merging (arXiv 2501.01230) uses SVD-based shared subspaces. — [arXiv 2501.01230](https://arxiv.org/pdf/2501.01230)

**NMF**
- A targeted search found no MTL/MOO gradient aggregation method based on NMF or orthogonal NMF. The results were only deep-NMF clustering and representation learning, plus NMF for multi-objective ranking. — e.g. [Deep Approximately Orthogonal NMF for clustering, arXiv 1711.07437](https://arxiv.org/pdf/1711.07437); [NMF-based multi-objective ranking (PMC)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10101413/)
- A spectral search over 2025–2026 also found no Gramian-eigendecomposition aggregator other than Aligned-MTL. — search results including [arXiv 2606.07289](https://arxiv.org/abs/2606.07289) and [BayesAgg-MTL](https://arxiv.org/pdf/2402.04005)

### Inferences
These are the researcher's derivations from the Aligned-MTL formulas above.

**How Aligned-MTL relates to averaging principal directions**
- From G = U S Vᵀ, the Aligned-MTL update is ĝ_0 = σ_min U Vᵀ w = σ_min Σ_{i=1..r} (v_iᵀ w) u_i.
- So Aligned-MTL is already a weighted sum of all r left principal directions u_i, with these properties:
  - (a) Equal magnitude per component: each singular value is replaced by σ_min, which is whitening.
  - (b) Coefficients v_iᵀ w come from the task-space eigenvectors.
  - (c) No truncation.
  - (d) No sign ambiguity, because u_i v_iᵀ is invariant to a joint sign flip.

**What a "principal gradient" aggregator must do differently**
- An SVD-based aggregator that uses the top-k u_i (or u_1 alone), weighted by σ_i, differs from Aligned-MTL in three ways:
  - Truncation discards low-variance or conflict directions, whereas Aligned-MTL amplifies them relatively by equalising.
  - σ_i weighting keeps the dominant shared direction, whereas Aligned-MTL removes dominance on purpose.
  - The aggregator needs an explicit orientation rule: orient each u_i so that it has a positive dot product with the mean gradient, or with v_iᵀ1 > 0, as FedLD does with its reference gradient.
- These three choices are the novelty axes to state explicitly. The paper should also show an ablation in which the method reduces to Aligned-MTL (k = r, equal weights).
- With σ_i weighting and k = r, Σ σ_i (v_iᵀ w) u_i = G w, which is the plain weighted mean. The SVD family therefore interpolates between the mean (k = r, σ-weighted) and Aligned-MTL (k = r, whitened). Top-1 is a third pole, related to the dominant eigenvector of the Gramian.

**Perron–Frobenius link**
- When all pairwise cosines are non-negative, GᵀG is entrywise non-negative. Perron–Frobenius then makes v_1 non-negative, so u_1 ∝ G v_1 is a convex-cone combination of the gradients and is non-conflicting.
- When gradients conflict, u_1 can conflict with some g_i. Pareto-stationarity or non-conflict guarantees would need an extra projection step, for example onto the dual cone as in UPGrad or DualProj.

**NMF**
- Gradient entries are signed, so plain NMF of G is ill-posed. An orthogonal-NMF aggregator presumably factorises either the non-negative part of the Gramian or a semi-NMF of G (G ≈ F Hᵀ with H ≥ 0).
- A non-negative H forces each "principal gradient" to be a conic combination of the task gradients, which addresses the sign ambiguity of SVD.
- No prior aggregation method of this kind was found, so this looks like the more novel half of the proposed family. A semi-/symmetric-NMF framing of clustering the task gradients is close to "task grouping" work, which was not surveyed here.

**Wording risk**
- GDOD's "task-shared component from an orthogonal basis of span(G)" is close in spirit to projecting onto the top principal subspace. The novelty claim should cite GDOD and contrast with it.
- "Principal gradient" is already used by FedLD (federated learning). Cite FedLD to pre-empt a reviewer.

### Gaps
- GDOD's exact basis construction (SVD vs Gram–Schmidt) and its update rule were not verified from the full text.
- FedLD's exact revision and aggregation formula, and whether it truncates or weights by eigenvalue, were not verified from the full text.
- I could not rule out workshop papers, Chinese-language venues, or 2026 preprints that use "PCA gradient" or "eigengradient" aggregation in MTL. Searches for "principal gradient" combined with multi-task, and for "spectral gradient aggregation", returned nothing beyond the items above.
- Task-grouping methods that cluster gradient affinity matrices (e.g. TAG, Fifty et al. 2021) were not surveyed. They may be relevant to the NMF variant.

## Q3. Benchmarks and RL evaluations; critiques showing that MTL optimizers do not beat tuned scalarization

### Takeaway
The de facto benchmark suite is:
- NYUv2 (3 tasks)
- CityScapes (2–3 tasks)
- QM9 (11 regression tasks)
- CelebA (40 tasks)
- MultiMNIST
- Meta-World MT10/MT50 with SAC for RL

The results are usually reported as Δm%, the mean relative drop versus single-task baselines. Two NeurIPS 2022 papers show that tuned or unitary scalarization matches most specialised optimizers in both supervised learning and RL. Any new aggregator must therefore include a well-tuned linear-scalarization baseline.

### Cited Findings
- Aligned-MTL evaluation: NYUv2 (MTAN, PSPNet), CityScapes 2-task and 3-task, Meta-World MT10 with SAC. [verified] — [ar5iv 2305.19000](https://ar5iv.labs.arxiv.org/html/2305.19000)
- [prior knowledge] Typical evaluations in other papers:
  - PCGrad: MultiMNIST, CityScapes, NYUv2, CelebA, MT10/MT50.
  - CAGrad: NYUv2, CityScapes, MT10/MT50.
  - Nash-MTL: QM9, NYUv2, CityScapes, MT10.
  - FAMO: CelebA, QM9, NYUv2, CityScapes, MT10.
  - Links: [PCGrad](https://arxiv.org/abs/2001.06782); [CAGrad](https://arxiv.org/abs/2110.14048); [Nash-MTL](https://arxiv.org/abs/2202.01017); [FAMO](https://arxiv.org/abs/2306.03792)
- ConFIG is evaluated mainly on PINNs (physics losses), plus MTL benchmarks. [verified] — [arXiv 2408.11104](https://arxiv.org/abs/2408.11104)
- Jacobian descent (UPGrad) is evaluated on instance-wise risk minimisation for image classification. [verified] — [arXiv 2406.16232](https://arxiv.org/abs/2406.16232)
- Xin et al., "Do Current Multi-Task Optimization Methods in Deep Learning Even Help?" (Google Research, NeurIPS 2022):
  - In large-scale language and vision experiments, multi-task optimization methods gave no gains over traditional approaches.
  - Scalarization with appropriate weights matches their optimization and generalization behaviour.
  - Caveat: the grid search used to trace the scalarization Pareto front is expensive. [verified] — [arXiv 2209.11379](https://arxiv.org/abs/2209.11379); [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/hash/580c4ec4738ff61d5862a122cdf139b6-Abstract.html)
- Kurin et al., "In Defense of the Unitary Scalarization for Deep Multi-Task Learning" (NeurIPS 2022):
  - Unitary scalarization means minimising the plain sum of losses.
  - Combined with standard regularisation and stabilisation, it matches or beats complex MTL optimizers in both supervised learning and RL.
  - The authors argue that many specialised optimizers act as regularisers. [verified] — [arXiv 2201.04122](https://arxiv.org/abs/2201.04122); [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/hash/4f301ae934f396086bfefd1139039dbd-Abstract.html)
  - [prior knowledge] Their RL experiments use Meta-World MT10/MT50 with SAC, comparing against PCGrad, CAGrad and others.
- RLW (random weighting) is competitive with sophisticated methods, which supports the same critique. [prior knowledge] — [arXiv 2111.10603](https://arxiv.org/abs/2111.10603)

### Inferences
- In a quadrotor RL MOO setting where reward terms or tasks are treated as objectives, the expected reviewer baselines are: (i) unitary sum, (ii) tuned linear weights, (iii) Aligned-MTL as the closest spectral method, (iv) UPGrad or DualProj, plus PCGrad and CAGrad. These are the methods used on Meta-World.
- Meta-World MT10 with SAC is the standard RL check in this literature, so a SAC/TD3 continuous-control evaluation is reasonably aligned with it.
- Reporting multiple seeds with confidence intervals matters, because the scalarization critiques hinge on variance and tuning budget.

### Gaps
- I found no 2025–2026 large-scale re-benchmark that updates Xin/Kurin to include Aligned-MTL, UPGrad, FAMO or ConFIG. Such a study may exist but did not surface.
- Exact Δm% numbers per method were not collected.

## Q4. Theoretical properties emphasised

### Takeaway
Papers sell aggregators on a small set of properties:
- convergence to Pareto stationarity (MGDA, CAGrad, Nash-MTL, Aligned-MTL, GradOPS, UPGrad, the unified framework)
- being conflict-free or non-conflicting, meaning gᵢᵀd ≥ 0 for every i (UPGrad, DualProj, ConFIG, MGDA)
- scale invariance or "linearity under scaling" (Nash-MTL, UPGrad)
- preference control through pre-set weights (Aligned-MTL, PCD)
- stochastic convergence rates without bias (CR-MOGM, MoCo, MoDo, SDMGrad)
- cost in T (MGDA-UB, Aligned-MTL-UB, FAMO, M-ConFIG)

### Cited Findings
- Aligned-MTL: converges to a Pareto-stationary point that is also optimal for the pre-set weights w (∇L_0 = 0). κ(G) serves as the stability metric. The UB variant gives O(1) cost in T. [verified] — [ar5iv 2305.19000](https://ar5iv.labs.arxiv.org/html/2305.19000)
- Unified framework (2026):
  - A sufficient alignment condition shows that any non-conflicting direction in the convex hull converges to Pareto stationarity at (optimal) rates.
  - Dual-cone projection makes feasibility hold more widely.
  - A primal view unifies existing algorithms. [verified] — [arXiv 2605.30452](https://arxiv.org/abs/2605.30452)
- ConFIG: the update has a positive dot product with every loss gradient, and convergence is proven. [verified] — [ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/94e85561a342de88b559b72c9b29f638-Abstract-Conference.html)
- GradOPS: resolves all pairwise conflicts regardless of order and converges to Pareto-stationary points. [verified] — [arXiv 2503.03438](https://arxiv.org/abs/2503.03438)
- PCD: gives each secondary objective a guaranteed fraction τ of normalised first-order progress. [verified] — [TorchJD PCD](https://torchjd.org/stable/docs/aggregation/pcd/)
- CR-MOGM / MoDo: O(ε⁻²) stochastic complexity. MoDo removes the bounded-loss assumption. [verified, snippet] — [survey 2501.10945](https://arxiv.org/pdf/2501.10945); [arXiv 2305.20057](https://arxiv.org/abs/2305.20057)
- [prior knowledge] Nash-MTL claims scale invariance and Pareto optimality — [arXiv 2202.01017](https://arxiv.org/abs/2202.01017). UPGrad claims non-conflicting, linear-under-scaling and weighted properties — [arXiv 2406.16232](https://arxiv.org/abs/2406.16232).

### Inferences
- **Scale invariance:** a raw SVD of G is not invariant to per-task loss scaling, because a task with a large norm dominates u_1. Normalising the gradients before the SVD (as IMTL-G and ConFIG do with unit gradients) would fix this. Otherwise the paper should state openly that the method is not scale-invariant. Aligned-MTL's whitening is partly a fix for exactly this kind of dominance.
- **Non-conflict:** not guaranteed for top-k SVD directions when gradients conflict. Adding a dual-cone projection, as in UPGrad, or a sign/orientation rule plus clipping would give a provable non-conflict property. The 2026 unified framework then supplies a ready-made convergence argument, provided the final direction lies in the convex hull or dual cone and satisfies its alignment condition.
- **Orthogonal NMF:** the non-negative coefficients put each factor in the conic hull of the gradients. This makes a "weighted"/conic-combination property easy to state. Whether a factor is conflict-free still depends on the angles between the gradients.

### Gaps
- I did not verify the formal definitions and proofs of UPGrad's properties from the paper text this session.
- I did not check whether any paper proves properties specifically for truncated-SVD aggregation. None was found.
