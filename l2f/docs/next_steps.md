# Next steps (2026-11-25)

Where the MOO work stands and what could come next. Results so far:
[moo_findings.md](moo_findings.md) (sections 12-13) and [results/README.md](results/README.md).

## Short-term goal

Show that PGA and its variants work: on the drone application they should beat linear
scalarization (MOO sum) and the other aggregators, and reach or beat the baseline, variant E (TD3,
E's hand-tuned reward). The drone is one application in the PGA paper, so the evaluation needs to
be fair and honest, not exhaustive:

- the same training budget, critic (`--split-critic --total 1`) and reward for every aggregator;
- 8 seeds per arm, reported as median or mean with spread or an interval;
- a few metrics chosen before the runs and reported for every method;
- claims no stronger than the numbers.

### Candidate experiments for that goal

Every arm trains on variant E's own reward (inspired by L2F), split into 4 groups; only the
gradient combination differs. Untuned TD3 is not a baseline.

1. **Score the methods on the objectives.** Evaluate each final policy, variant E's included, on a
   fixed set of test episodes and record the 4 group returns (time, precision, efficiency, safety).
   Plot the seeds in objective space: does PGA dominate variant E (better on some objectives, no
   worse on the others)? Compare hypervolume. Evaluation only, on the existing runs.

Not in the short-term plan: the untuned reward on split critics (every arm, PGA included, on equal
group shares, against tuned variant E). It would be a fair but harder test with a different claim
("PGA without tuning reaches hand-tuned quality").

### Where the evidence stands (8 seeds, `results/split_agg/stats.csv`)

Per-seed medians, Mann-Whitney test, Holm correction over 48 comparisons per baseline:

- **Holds after correction:** sum, DualProj, UPGrad and Aligned-MTL have smoother motor commands
  than variant E (jitter 0.066-0.078 against 0.118).
- **p < 0.05 before correction:** every MOO arm reaches waypoints 0.15-0.37 s sooner than variant E;
  SVD-PGA crashes less than sum (1.35 against 3.05%, p 0.018); sum crashes more than variant E.
- **No clear difference:** circle error for any arm against variant E; waypoint arrival share;
  SVD-PGA's crash rate against variant E (p 0.34).
- With 8 seeds the 95% interval on circle error is about +-2 cm, so 1-2 cm differences cannot be
  shown.

## Later: a fuller drone evaluation (not decided)

Modelled on the experiments of Eschmann et al., "Learning to Fly in Seconds" (RA-L 2024,
arXiv 2311.13081): training speed over many seeds, ablations, real position control (success out of
10, position error), and real figure-eight tracking at 15 s, 5.5 s and 3.5 s cycles (up to 3 m/s)
against PID, geometric, nonlinear, INDI and two RL controllers.

| Idea | What it tests | Cost |
| --- | --- | --- |
| Agility: fast circles (`L2F_MISSION_R`, `L2F_MISSION_P`) and the paper's figure-eight at 3 speeds | speed and precision against safety, where E's slow-circle weights may be a poor trade-off | evaluation only first (figure-eight needs a small evaluator addition); then training with the existing `L2F_P_EIGHT` and `L2F_P_FAST_CIRCLE` episodes |
| Training speed | which arm becomes crash-free first; worst seeds | evaluation only: the saved checkpoints from 100k to 1M steps |
| Robustness sweeps | sensor noise, delay 0-40 ms, disturbance force, motor mismatch, Lighthouse glitches | evaluation only |
| Gust rejection | precision against safety under disturbance (`L2F_P_GUST` episodes) | training study |
| Real flights | the paper's protocol: success out of 10 and tracking error | flight sessions |
| Remaining confounds | critic size (split critics have 4x variant E's critic parameters); "just a reweighting" (TD3 with smoothing weights matched to SVD-PGA's effective shares); why sum on split critics crashes more than variant E | one TD3 arm each, about 1 h |

For claims of 1-2 cm, plan 12-16 seeds and fix the primary metrics before running.

## Other open items

- Preference vectors against a grid of TD3 weights (a Pareto front).
- A task with designed interference (target jump against a Lighthouse glitch); camera tasks later.
- Tools: `l2f/sim/study_export.py` (CSV export), `l2f/sim/study_stats.py` (per-seed tests).
