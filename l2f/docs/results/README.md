# Results: separate critics and aggregators (2026-11-24/25)

Baseline: **variant E** (`td3`: plain TD3, E's hand-tuned reward). Every MOO arm: 4 reward groups
(time, precision, efficiency, safety), separate critic per group (`--split-critic`), `--total 1`.
Task: the time 0.5 recipe (mission + 0.3-0.5 m waypoint episodes, Huber 0.2, crash 400, simulator
options). 2M steps per run. Discussion: [moo_findings.md](../moo_findings.md), sections 12-13.

Mission test: plain world, 12 ms delay; medians over seeds x the 9 checkpoints from 1.2M to 2.0M.
Waypoint tests: 3.5 s legs; wp05L jumps 0.3-0.5 m (as trained), wp08L 0.3-0.8 m.

## 8 seeds (70-77), all arms

| Arm | Circle | Uneven motors | Crashed | Crash-free checkpoints | Jitter | 10 cm, 0.5 m jumps (time) | 10 cm, 0.8 m jumps | Overshoot 0.5 m |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **td3 (variant E)** | 6.0 cm | **9.7 cm** | **1.4%** | **18/72** | 0.121 | 79.5% (1.73 s) | 69.0% | 9.6 cm |
| sum | 5.0 cm | 11.2 cm | 4.5% | 7/72 | **0.065** | 78.0% (1.55 s) | 71.5% | 7.8 cm |
| DualProj | 5.3 cm | 11.3 cm | 3.9% | 9/72 | 0.066 | 77.0% (1.60 s) | 68.0% | 8.9 cm |
| UPGrad | 5.0 cm | 10.1 cm | 2.7% | 7/72 | 0.069 | **84.0%** (1.54 s) | **76.5%** | 7.8 cm |
| Aligned-MTL | 5.0 cm | 10.4 cm | 2.4% | 17/72 | 0.079 | 82.5% (**1.48 s**) | 76.0% | 7.8 cm |
| **SVD-PGA** | 4.9 cm | 10.4 cm | 1.7% | 14/72 | 0.106 | 83.5% (1.53 s) | 76.0% | 7.2 cm |
| OS-NMF-PGA | **4.5 cm** | 10.2 cm | 2.5% | 16/72 | 0.081 | 79.5% (**1.48 s**) | 73.0% | **7.1 cm** |

Per-seed numbers: `split_agg/per_seed_8seeds.csv`. The 4-seed run of every aggregator (seeds 70-73)
is in `split_agg/summary_4seeds.csv`.

## The critic variants (MOO sum, 4 seeds)

| Arm | Circle | Uneven motors | Crashed | Jitter | 10 cm, 0.5 / 0.8 m jumps | Group values distinct? |
| --- | --- | --- | --- | --- | --- | --- |
| td3 (variant E) | 4.9 cm | 8.6 cm | 0.8% | 0.124 | 84.0 / 73.5% | (one value) |
| shared critic, unscaled | 7.4 cm | 11.9 cm | 2.5% | 0.105 | 61.0 / 54.0% | yes, but unstable total |
| shared critic, head-scaled | 4.9 cm | 11.1 cm | 10.1% | 0.121 | 75.0 / 69.5% | no (copies of one) |
| separate critics, `--total 0` | 7.4 cm | 11.4 cm | 1.9% | 0.114 | 66.5 / 56.5% | yes |
| separate critics, `--total 1` | 4.9 cm | 11.3 cm | 3.3% | 0.065 | 78.0 / 71.0% | yes |

## Files

| Folder | Study | Files |
| --- | --- | --- |
| `split/` | `ablation_split.txt` (critic variants) | `summary.csv`, `per_seed.csv`, `curves.csv` |
| `split_agg/` | `ablation_split_agg.txt` (all 7 arms, seeds 70-77) | `summary_4seeds.csv`, `summary_8seeds.csv`, `per_seed_*.csv`, `curves.csv` |

`curves.csv` (long format: arm, seed, step, metric, value) holds, per run: the training return every
100k steps; the checkpoint tests from 1.0M to 2.0M (`clean_` / `randomized_` / `jumps_` x `crash`,
`circle`, `jitter`); and the MOO gradient log averaged per 50k steps (`cos_<a>_<b>`, `weight_<g>`,
`norm_<g>`, `step_norm`). Regenerate with
`python l2f/sim/study_export.py --study-dir <ablation output folder> --out <folder>`.
