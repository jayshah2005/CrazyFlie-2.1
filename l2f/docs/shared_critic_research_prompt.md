# Research task: why a shared multi-objective critic fails, and what could make it work

You are a research agent. Search the literature (arXiv, NeurIPS/ICML/ICLR/CoRL/RA-L/IROS/ICRA,
AAMAS, JMLR, IEEE T-RO/T-NNLS) for work that explains or fixes the problem below, then recommend
directions. Cite only real, verifiable papers (authors, year, venue, link). Mark anything you did
not read in full. If the literature has no answer, say so plainly and propose the most promising
directions from first principles, labelled as your own proposals.

## 1. Setting

- Task: end-to-end low-level quadrotor control (Crazyflie 2.1 Brushless, Learning-to-Fly-style,
  Eschmann et al. 2024, arXiv 2311.13081). Continuous state and action; the policy outputs the 4
  motor commands directly. Missions: hover, take-off, landing, circle tracking, waypoint jumps
  (0.5-0.8 m). Trained in simulation with domain randomisation (motor delay up to 20 ms, uneven
  motor gains, drag), then flown on real hardware.
- Algorithm: TD3 with an asymmetric critic.
  - Actor: 25 inputs (position error 3, rotation matrix 9, linear velocity 3, angular velocity 3,
    measured rotor speeds 4, leaky position-error integral 3) -> 64 -> 64 -> 4, tanh output.
  - Critic: 31-dim privileged observation + 4-dim action -> 64 -> 64 -> outputs; activation is a
    rational tanh approximation clipped at +-3 (rl_tools FAST_TANH).
  - Twin critics with target networks (Polyak 0.995); twin chosen per sample by the smaller
    summed value; target smoothing noise std 0.2 (clip 0.5); exploration std 0.1; both decay x0.9
    every 100k steps from 500k.
  - Adam (AdamW form) lr 1e-3, betas (0.9, 0.999), eps 1e-8, weight decay 1e-4 on weights only;
    batch 256; gamma 0.99; replay 700k; critic update every 10 env steps, actor every 20; warm-up
    15k (critic) / 30k (actor) steps; 2M env steps; episodes up to 500 steps; a curriculum ramps
    the mission and some reward weights over the first 800k steps.
- Multi-objective formulation: the hand-tuned scalar reward (our baseline, "variant E") is split
  into 4 groups whose sum is exactly the scalar reward:
  - time (time-to-target cost, weight 0.5);
  - precision (position and velocity error);
  - efficiency (action magnitude, action rate, angular acceleration, power);
  - safety (survival term, a 400-point crash penalty on the terminal step, orientation, landing
    speed).
  The critic predicts one value per group (Q_time, Q_prec, Q_eff, Q_safe); each head learns its own
  TD target r_k + gamma * Q'_k(s', a'). The actor receives 4 gradients, one per group
  (g_k = grad_theta of -mean_b Q_k(s_b, pi(s_b))), which a multi-task gradient aggregator combines
  into one step (TorchJD implementations: sum, UPGrad, DualProj, Aligned-MTL, PCGrad, CAGrad,
  MGDA, and our own methods SVD-PGA and OS-NMF-PGA, which average the principal directions of the
  gradient matrix, normalised to unit length, i.e. scale-invariant). With "sum" the step equals
  plain TD3 on the scalar reward.
- Scales (seed 70, 600k steps, real units): spread (std) of each group's TD target: time 3.2,
  precision 7.8, efficiency 17, safety 152. Per-step reward correlations between groups are low
  (-0.08 to +0.39). On the shared critic's hidden layers, the safety group's loss gradient is
  100-3000x larger than time's.

## 2. What works (reference points)

All numbers: 2 seeds unless noted, sim test with domain randomisation, 6 ms estimation delay;
"circle" = tracking error on a circle, "uneven" = same with mismatched motors, "crashed" = share
of test episodes, "waypoint" = share of 0.8 m jumps arrived within 10 cm.

| Critic | Params | Group-value corr at 2M | Circle | Uneven | Crashed | Waypoint |
|---|---|---|---|---|---|---|
| Variant E (scalar TD3 critic) | 6.5k | - | 5.2 cm | 8.9 cm | 0.6 % | 72 % |
| Split critic (4 separate 64-64 nets), actor OS-NMF-PGA | 26.1k | +0.30..+0.60 | 4.5 cm | 11.9 cm | 2.4 % | 68 % |

The split critic shows genuine conflict between the groups' actor gradients (precision vs
efficiency opposed in 82 % of updates, precision vs safety 76 %, time vs precision 68 %). With it,
the reweighting aggregators cut crashes vs plain sum (8 seeds: SVD-PGA 1.7 %, OS-NMF-PGA 2.5 %, vs
sum 4.5 %). On the real drone (28 flights), split-critic PGA policies beat variant E on circle
tracking, fast circles and waypoint settling.

## 3. The problem: a shared critic (one trunk, 4 heads) does not work

We want a shared critic (lower complexity; concern that reviewers will question 4 separate
networks). Every shared variant is clearly worse than the split critic.

- Unscaled 4-head critic: the summed value moves about 2.3x TD3's step per update (each head gets
  the full correction of the total error). Crashes 4.5-11 %, circle 9.1-10.6 cm, waypoint 29-37 %.
- Head scaling (outputs x 1/k, initial output layer x sqrt(k)) gives TD3's step size, but the
  heads learn about 4x slower than the trunk and the 4 group values collapse into near-copies
  (correlation 1.00 at 200k, 0.98-1.00 at 2M). Crashes 15 %.
- Collapse is not limited to head scaling: the unscaled shared critic also collapses late
  (correlation 0.93-0.99 at 2M), although the groups' per-step rewards are nearly uncorrelated.
  The TD targets are equally correlated (0.93-0.99), so the collapse seems self-reinforcing through
  bootstrapping (targets come from the same collapsed target network). The split critic tracks the
  reward structure instead (values +0.30..+0.75).
- A collapsed critic hides the objective conflict from the actor's aggregator: the 4 actor
  gradients become near-parallel and every aggregator reduces to the sum.

## 4. Everything we tried on the shared critic (all failed to match the split critic)

A. MOO inside the critic (`--critic-agg`): the per-group critic losses (optionally plus a 5th
   "total value" loss, mean (sum_k e_k)^2) give one gradient each on the shared hidden layers,
   combined by an aggregator; output layer gets the plain sum. Verified: with "sum" it reproduces
   the plain shared-critic gradient to float precision; a full training with "sum" flies like the
   plain shared critic (5.5 % crashes); applied steps match the aggregator's output to 1e-7.
   Results (crashes): UPGrad 35 %; Aligned-MTL 64 %; SVD-PGA 85-93 %; OS-NMF-PGA 64-95 % (with or
   without the total loss, with or without head scaling); PCGrad 7.7 % but circle 15 cm; CAGrad
   (c = 0.5) 12.3 % with the best shared tracking (circle 7.1 cm, waypoint 49 %, lowest jitter) but
   values still collapsed (0.89-0.98).
   Mechanism found: PGA-type aggregators are scale-invariant (unit-norm step averaging principal
   directions); with gradient sizes 100-3000x apart they give the tiny time gradient the largest
   weight and nearly zero weight to safety.

B. Normalisation of the groups before aggregating:
   1. `--norm`: each head learns value / s_k, s_k = RMS(group reward) / (1 - gamma), measured once
      at 15k steps from 20k replay samples. Scales came out time 49, precision 26, efficiency 298,
      safety 6389 (inflated by early crashes, 2.5 % of steps). It overestimated each group's value
      spread by 3-42x and flipped the gradient gap (safety 1e5x smaller than precision). Norm + sum
      crashed 55 %; with PGA aggregators 28-30 %.
   2. `--critic-norm` (running error scale): each group's TD error divided by the running RMS of
      that error (EMA 0.01 per update). Gradients balanced (within 2-10x) but every group's loss is
      ~1 regardless of fit quality (size-blind). Sum 53 %, SVD-PGA 51 %, UPGrad 27 %, OS-NMF-PGA 21 %
      crashes. With PCGrad 30 %, CAGrad 25 % (PCGrad's group values distinct, -0.32..+0.82, R2 0.99).
   3. `--critic-norm-by target` (PopArt-like): each group's TD error divided by the running std of
      that group's TD targets. Collapse fixed: group values distinct (-0.98..+0.93), every head
      R2 0.97-0.99. But crashes: sum 28 %, UPGrad 18 %, CAGrad 37 %, OS-NMF-PGA 56 % (plain shared
      critic: 4.5 %).

C. Architecture:
   1. Wider shared critic (hidden 142, 25,990 params, matched to the split critic): still collapses
      (0.75-0.97 at 2M); circle 9.3 cm, crashes 6.6 %, waypoint 34 %. So it is not capacity.
   2. Partial sharing ("towers": shared first layer 64, then a separate 64-unit second layer and
      head per group; 19,204 params), actor OS-NMF-PGA, 6 seeds: seed 71 learned to crash (100 %);
      seeds 73 and 75 crash 13-15 %; seeds 72-75 together: circle 7.0 cm, uneven 14.2 cm, crashes
      8.9 %, waypoint 60 % (split critic on the same seeds: 4.8 cm, 10.4 cm, 2.1 %, 74 %). Values
      distinct on seed 70 (-0.31..+0.89) but its efficiency head stopped fitting (R2 0.005 at 2M).

D. Diagnostics run:
   - Offline, fixed targets (critic warm-started from a 600k run, frozen replay set and TD targets,
     15k updates, real training code): most aggregators keep the fit as good as sum (total R2
     0.985-0.996, groups 0.86-0.99). Exception: SVD-PGA / OS-NMF-PGA with the total-value loss
     wreck the time and precision heads (R2 -4.5 / -2.8 and 0.25-0.27).
   - Action-slope drift in the same offline setup (cosine between dQ/da before and after 15k
     updates, at the policy's actions; 1 = unchanged): sum 0.74 total / 0.76-0.79 per group; PGA
     with total loss: per-group slopes -0.26 to 0.60 (some reversed); PGA without total 0.40-0.64;
     UPGrad without total 0.73-0.80 (yet it crashed 55 % in RL).
   - Float32 precision of the aggregators at our gradient ratios: step direction unchanged
     (cosine >= 0.994).

Repeated observation (4 separate times): critics with distinct, accurate group values (R2 ~ 0.99)
still produce policies that crash 18-95 %. Value accuracy on replay data does not predict control
quality.

## 5. Our current hypotheses

- H1 (crash-value priority): in real units the safety group (crash penalty) dominates the shared
  layers' gradient; this is what keeps the crash value and its action-gradient accurate. Any
  rebalancing (normalisation or aggregation) shrinks safety's share of the shared features and
  degrades exactly the quantity the policy depends on. The split critic avoids the trade-off.
- H2 (action-gradient quality): the actor learns only from dQ_k/da. Shared features fitted to
  several heads may give accurate values but wrong or noisy slopes; R2 does not measure this.
- H3 (bootstrap collapse loop): shared features + bootstrapped targets from the same network make
  head collapse self-reinforcing; loss-level changes do not break the loop.
- H4 (negative transfer in the trunk): classic multi-task interference, made worse by the
  100-3000x scale gap and the rare large crash penalty.

## 6. Questions to answer

1. Has anyone reported collapse, rank loss or interference between heads of a multi-head /
   decomposed / vector-valued critic (shared torso), in RL generally or in continuous control?
   How did they detect and fix it? Specifically check: Hybrid Reward Architecture (van Seijen et
   al. 2017); SAC-D / decomposed critics (NeurIPS 2022); MOAC (arXiv 2405.03082); CoMOGA; GCR-PPO
   (arXiv 2509.14816); MOMA-AC (arXiv 2511.18181, shared vector-valued, preference-conditioned
   critic); multi-critic multi-task quadrotor RL (arXiv 2412.12442); PopArt multi-task (Hessel et
   al. 2019); Distral; reward-decomposition and explainable-RL critics; value factorisation in
   MARL (VDN, QMIX) where relevant.
2. Representation-collapse literature for critics: implicit under-parameterisation / feature-rank
   collapse (Kumar et al. 2021), DR3, capacity loss / plasticity loss, primacy bias, spectral
   normalisation of critics, layer norm in critics. Does any of it explain multi-head collapse
   under bootstrapping, and do its fixes (resets, rank regularisers, normalisation layers) apply?
3. Multi-task architectures that reduce negative transfer in a shared network: Multi-gate
   Mixture-of-Experts (MMoE), Progressive Layered Extraction (PLE), cross-stitch networks,
   soft parameter sharing, task-specific adapters, hypernetworks. Any applied to critics or value
   functions? Which would most plausibly keep 4 value heads distinct with fewer parameters than 4
   separate networks?
4. Loss balancing that keeps error-size information: GradNorm, uncertainty weighting (Kendall et
   al. 2018), dynamic weight averaging, FAMO, IMTL, Nash-MTL. Any evidence for or against using
   them inside a TD-learned critic?
5. Critic action-gradient quality in DDPG/TD3: gradient penalties on dQ/da, Sobolev training,
   smoothness regularisation, action-gradient consistency, critics trained to match target
   gradients. Any work showing value accuracy without action-gradient accuracy, or methods that
   regularise dQ/da directly?
6. Large, rare penalties (crash/termination) in value learning: reward clipping, survival bonus vs
   crash penalty, termination-aware critics, separate termination/safety critics (safe RL with a
   cost critic), distributional critics for rare events. Is giving safety its own network (while
   sharing the rest) a known design?
7. Bootstrapping-specific fixes: per-head target networks, different Polyak rates or learning rates
   per head, stop-gradient between heads, auxiliary decorrelation losses.
8. Preference-conditioned critics (critic takes objective weights as input, policy trained over
   varying weights): does varying preferences keep per-objective heads distinct?

## 7. Deliverable

1. A ranked list of candidate directions (best first). For each: the papers behind it, the
   mechanism, exactly how it would apply to our 4-head TD3 critic, which hypothesis (H1-H4) it
   addresses, the expected effect, implementation cost (lines of code, compute), and the risk.
2. Any paper that reports multi-head/decomposed-critic collapse or interference, quoted precisely:
   this affects our novelty claim, so flag it prominently.
3. Direct answers to questions 1-8 (say "nothing found" where applicable).
4. A short "if nothing in the literature works" section: the 2-3 most promising untested
   directions from first principles, and the minimal experiment for each (we can run 2-seed
   screens of about 30-45 minutes each).
5. Constraints for any recommendation: the deployed actor must stay the same size (only the
   critic may change); the baseline to beat is variant E, the original hand-tuned scalar TD3;
   prefer methods a reviewer would accept as principled over ad hoc tuning.
