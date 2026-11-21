"""Check the C++ gradient aggregators (l2f/learning_to_fly/src/moo/aggregators.h) against TorchJD.

Builds src/moo/check_aggregators.cpp (with WSL g++ on Windows), feeds it Gram matrices and
compares its weights with TorchJD's weightings on the same matrices (float64, preference
vector all ones). Test matrices: random gradients, conflicting gradients, rank-deficient and
badly scaled ones, and, with --log, the Gram matrices dumped by a MOO training run
(L2F_MOO_GRAM_LOG / moo_gram_seed*.csv).

Two errors per aggregator, relative to TorchJD:
  weights    max |w_cpp - w_torchjd| / max |w_torchjd|
  direction  |J^T (w_cpp - w_torchjd)| / |J^T w_torchjd|   (from the Gram matrix)

With --agg NAME, the weights the training run itself logged next to each Gram matrix are
also compared with TorchJD's NAME weighting (end-to-end check of the run's aggregator).

Run with a Python that has torchjd (pip install torchjd[quadprog_projector]):
  python l2f/sim/check_aggregators.py [--log path/to/moo_gram_seed0.csv --agg upgrad]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import run_log  # noqa: E402

from torchjd.aggregation import AlignedMTLWeighting, UPGradWeighting  # noqa: E402

log = run_log.setup(__file__)
SRC = ROOT / 'l2f' / 'learning_to_fly' / 'src' / 'moo'

REFERENCE = {  # TorchJD weightings with an all-ones preference vector
    'sum': lambda G: torch.ones(len(G), dtype=torch.float64),
    'upgrad': lambda G: UPGradWeighting(pref_vector=torch.ones(len(G), dtype=torch.float64))(G),
    'aligned_mtl': lambda G: AlignedMTLWeighting(pref_vector=torch.ones(len(G), dtype=torch.float64))(G),
}
try:  # SVDPGA / OSNMFPGA: only in the rkhosrowshahi/torchjd fork (branch feat/pga-aggregators)
    from torchjd.aggregation._pga import OSNMFPGAWeighting, SVDPGAWeighting
    REFERENCE['svd_pga'] = SVDPGAWeighting()
    REFERENCE['osnmf_pga'] = OSNMFPGAWeighting()
except ImportError:
    log.warning('torchjd without the PGA weightings: svd_pga / osnmf_pga not checked')


def wsl_path(p: Path) -> str:
    s = str(p).replace('\\', '/')
    return f'/mnt/{s[0].lower()}{s[2:]}' if s[1] == ':' else s


def run_harness(grams: np.ndarray) -> dict[tuple[int, str], np.ndarray]:
    text = '\n'.join(' '.join(f'{x:.17g}' for x in G.ravel()) for G in grams) + '\n'
    src, exe = SRC / 'check_aggregators.cpp', '$HOME/.cache/l2f_check_aggregators'
    M = grams.shape[1]
    script = f'mkdir -p $HOME/.cache && g++ -O2 -std=c++17 -o {exe} "{wsl_path(src)}" && {exe} {M}'
    cmd = ['wsl', '-e', 'bash', '-c', script] if sys.platform == 'win32' else ['bash', '-c', script]
    out = subprocess.run(cmd, input=text, capture_output=True, text=True, check=True).stdout
    result = {}
    for line in out.splitlines():
        parts = line.split()
        result[(int(parts[0]), parts[1])] = np.array([float(x) for x in parts[2:]])
    return result


def test_grams(rng: np.random.Generator, M: int) -> list[tuple[str, np.ndarray]]:
    cases = []
    n = 200
    for i in range(20):
        J = rng.standard_normal((M, n))
        cases.append(('random', J @ J.T))
    for i in range(40):  # shared direction plus conflicts, like the logged training gradients
        base = rng.standard_normal(n)
        J = rng.uniform(-0.6, 1.0, (M, 1)) * base + 0.6 * rng.standard_normal((M, n))
        J *= np.exp(rng.uniform(-2, 2, (M, 1)))  # norms spread over ~2 decades
        cases.append(('conflict', J @ J.T))
    for r in (1, 3, 5):
        for i in range(5):
            J = rng.standard_normal((M, r)) @ rng.standard_normal((r, n))
            cases.append((f'rank{r}', J @ J.T))
    for i in range(5):
        J = rng.standard_normal((M, n)) * np.logspace(-4, 2, M)[:, None]
        cases.append(('scaled', J @ J.T))
    return cases


def logged_grams(path: Path) -> list[tuple[str, np.ndarray, np.ndarray]]:
    header = open(path).readline().strip().split(',')
    n_tri = sum(c.startswith('g_') for c in header)
    M = int(round((np.sqrt(8 * n_tri + 1) - 1) / 2))
    data = np.loadtxt(path, delimiter=',', skiprows=1, ndmin=2)  # columns: step, upper triangle row-major, then the weights
    iu = np.triu_indices(M)
    cases = []
    for row in data:
        G = np.zeros((M, M))
        G[iu] = row[1:1 + n_tri]
        G = G + G.T - np.diag(np.diag(G))
        cases.append(('logged', G, row[1 + n_tri:1 + n_tri + M]))
    return cases


def errors(w: np.ndarray, wt: np.ndarray, G: np.ndarray) -> tuple[float, float]:
    d = w - wt
    err_w = np.abs(d).max() / max(np.abs(wt).max(), 1e-300)
    err_d = np.sqrt(max(d @ G @ d, 0)) / max(np.sqrt(max(wt @ G @ wt, 0)), 1e-300)
    return err_w, err_d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--log', type=Path, action='append', default=[], help='moo_gram_seed*.csv from a run')
    ap.add_argument('--agg', help='aggregator the logged run used (compares its logged weights)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--size', type=int, default=8, help='objectives for the test matrices (5, 8 or 9)')
    args = ap.parse_args()

    logged = [c for p in args.log for c in logged_grams(p)]
    M = logged[0][1].shape[0] if logged else args.size
    cases = test_grams(np.random.default_rng(args.seed), M)
    cases += [(k, G) for k, G, _ in logged]
    grams = np.stack([G for _, G in cases])
    cpp = run_harness(grams)

    worst: dict[tuple[str, str], tuple[float, float]] = {}
    for i, (kind, G) in enumerate(cases):
        Gt = torch.tensor(G, dtype=torch.float64)
        for name, ref in REFERENCE.items():
            wt = ref(Gt).numpy()
            err_w, err_d = errors(cpp[(i, name)], wt, G)
            a, b = worst.get((name, kind), (0.0, 0.0))
            worst[(name, kind)] = (max(a, err_w), max(b, err_d))

    kinds = list(dict.fromkeys(k for k, _ in cases))
    log.info('%d Gram matrices of size %d; worst relative error vs TorchJD (weights / direction)', len(cases), M)
    log.info('%-12s %s', 'aggregator', ' '.join(f'{k:>20s}' for k in kinds))
    for name in REFERENCE:
        cells = ' '.join(f'{worst[(name, k)][0]:9.1e} /{worst[(name, k)][1]:9.1e}' for k in kinds)
        log.info('%-12s %s', name, cells)
    if args.agg and logged:
        # the run's logged weights went through float printing at 17 digits: exact
        worst_w = worst_d = 0.0
        ref = REFERENCE[args.agg]
        for _, G, w_logged in logged:
            wt = ref(torch.tensor(G, dtype=torch.float64)).numpy()
            e_w, e_d = errors(w_logged, wt, G)
            worst_w, worst_d = max(worst_w, e_w), max(worst_d, e_d)
        log.info('weights logged during training (%d updates) vs TorchJD %s: %.1e / %.1e',
                 len(logged), args.agg, worst_w, worst_d)


if __name__ == '__main__':
    main()
