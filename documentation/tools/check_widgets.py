"""Check the math inside the documentation's widgets against trusted references.

The widgets (documentation/assets/widgets/*.js) compute what the pages explain. This tool runs
their JavaScript with Node and compares:
  - aggregator-playground: sum, mean, PCGrad, MGDA, UPGrad, DualProj, Aligned-MTL and SVD-PGA on
    random 2- and 3-objective cases in the plane, against TorchJD (the library train.py uses,
    all-ones preference vectors as in l2f/py/moo.py). Skipped if torchjd does not import.
  - crash-incentive: the discounted return of a perfect waypoint leg, against the table in
    l2f/docs/reward_studies.md section 4 (squared, Huber 0.2, progress 20; weight 40, time 0.5).
  - seed-compare: the exact Mann-Whitney p value, against scipy (skipped without scipy).

Usage (from the repository root): python3 documentation/tools/check_widgets.py
Needs node on the PATH.
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

DOCS = Path(__file__).resolve().parents[1]
ROOT = DOCS.parent
sys.path.insert(0, str(ROOT))  # run_log
import run_log  # noqa: E402

log = run_log.setup(__file__)

HARNESS = r"""
global.window = global;
const fs = require('fs');
const load = (f) => { eval(fs.readFileSync(f, 'utf8')); };
const dir = process.argv[2];
load(dir + '/widgets-core.js');
for (const f of ['aggregators.js', 'crash_incentive.js', 'seeds.js']) load(dir + '/widgets/' + f);
const W = window.DocWidgets;
const input = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const out = { agg: input.agg.map((G) => Object.fromEntries(Object.keys(W.aggregate).map((k) => [k, W.aggregate[k](G)]))),
  leg: input.leg.map((p) => W.legCost(p).G0),
  mw: input.mw.map(([a, b]) => W.stats.mannWhitney(a, b).p) };
console.log(JSON.stringify(out));
"""

# l2f/docs/reward_studies.md, section 4: "Perfect flight of one waypoint leg ..."
LEG_TABLE = {('squared', 0.5): -122, ('squared', 1.0): -612, ('squared', 1.5): -1652, ('squared', 3.0): -9200,
             ('huber', 0.5): -92, ('huber', 1.0): -273, ('huber', 1.5): -514, ('huber', 3.0): -1461,
             ('progress', 0.5): -32, ('progress', 1.0): -41, ('progress', 1.5): -43, ('progress', 3.0): -33}


def torchjd_cases(cases):
    """TorchJD's aggregated vector for every case and method, or None without torchjd."""
    try:
        import torch
        from torchjd import aggregation as A
    except Exception as e:  # noqa: BLE001
        log.warning('torchjd not importable (%s): aggregator check skipped', e)
        return None
    try:  # TorchJD's dual-cone QP adds reg_eps = 1e-4 to the normalized Gram matrix; near 0 it is exact
        from torchjd._linalg._dual_cone import QuadprogProjector
        exact = QuadprogProjector(norm_eps=1e-12, reg_eps=1e-12)
    except Exception:  # noqa: BLE001 - private API; fall back to the default projector
        exact = None
    out = []
    for G in cases:
        J = torch.tensor(G, dtype=torch.float64)
        m = J.shape[0]
        ones = torch.ones(m, dtype=torch.float64)
        proj = exact if (exact is not None and m == 2) else None  # 3 gradients in a plane: singular Gram, keep the default
        aggs = {'sum': A.Sum(), 'mean': A.Mean(), 'mgda': A.MGDA(), 'upgrad': A.UPGrad(pref_vector=ones, projector=proj),
                'dualproj': A.DualProj(pref_vector=ones, projector=proj), 'aligned_mtl': A.AlignedMTL(pref_vector=ones), 'svd_pga': A.SVDPGA()}
        if m == 2:
            aggs['pcgrad'] = A.PCGrad()  # TorchJD visits the other objectives in random order; with 2 the order does not matter
        out.append({k: a(J).tolist() for k, a in aggs.items()})
    return out


def exact_cases(cases):
    """UPGrad, DualProj and MGDA solved exactly with scipy (SLSQP), for the 3-objective cases where
    TorchJD's regularized solvers are not exact (3 gradients in a plane: a singular Gram matrix)."""
    try:
        from scipy.optimize import minimize
    except ImportError:
        return None

    def proj(v, G):
        r = minimize(lambda x: ((x - v) ** 2).sum(), np.zeros(2), jac=lambda x: 2 * (x - v),
                     constraints=[{'type': 'ineq', 'fun': lambda x: G @ x, 'jac': lambda x: G}], method='SLSQP',
                     options={'ftol': 1e-14, 'maxiter': 500})
        return r.x

    def mgda(G):
        m = len(G)
        r = minimize(lambda w: ((G.T @ w) ** 2).sum(), np.full(m, 1 / m), jac=lambda w: 2 * G @ (G.T @ w),
                     bounds=[(0, None)] * m, constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - 1}], method='SLSQP',
                     options={'ftol': 1e-15, 'maxiter': 500})
        return G.T @ r.x

    out = []
    for G in cases:
        G = np.array(G)
        out.append({'upgrad': sum(proj(g, G) for g in G).tolist(), 'dualproj': proj(G.sum(0), G).tolist(), 'mgda': mgda(G).tolist()})
    return out


def main():
    node = shutil.which('node')
    if not node:
        log.error('node is not on the PATH')
        sys.exit(1)
    rng = np.random.default_rng(1)
    cases = [rng.normal(size=(m, 2)).tolist() for m in (2, 3) for _ in range(150)]
    cases += [[[1.6, 0.3], [-0.26, 0.05]], [[1.2, 0.0], [0.0, 0.8]], [[1.0, 0.0], [-1.0, 0.01]]]
    legs = [{'D': d, 'shape': s, 'cap': 0.2, 'wPos': 40, 'wTime': 0.5, 'K': 20, 'gamma': 0.99} for (s, d) in LEG_TABLE]
    mw = [(rng.normal(size=8).round(3).tolist(), (rng.normal(size=8) + shift).round(3).tolist()) for shift in (0, 0.5, 1, 2)]
    with tempfile.TemporaryDirectory() as tmp:
        harness, data = Path(tmp) / 'harness.js', Path(tmp) / 'in.json'
        harness.write_text(HARNESS)
        data.write_text(json.dumps({'agg': cases, 'leg': legs, 'mw': mw}))
        res = subprocess.run([node, str(harness), str(DOCS / 'assets'), str(data)], capture_output=True, text=True)
    if res.returncode:
        log.error('node failed:\n%s', res.stderr)
        sys.exit(1)
    js = json.loads(res.stdout)
    failures = 0

    ref = torchjd_cases(cases)
    exact = exact_cases(cases)
    if ref is not None:
        # Reference: TorchJD, except for MGDA (its Frank-Wolfe iterations stop at epsilon 1e-3) and for
        # UPGrad / DualProj with 3 gradients in a plane (TorchJD regularizes the singular Gram matrix,
        # which can leave its answer slightly outside the dual cone): there the exact scipy solution.
        # Error relative to the size of the gradients (a result of 0 has no relative error of its own).
        worst = {}
        for i, (G, got, want) in enumerate(zip(cases, js['agg'], ref)):
            size = np.mean([np.linalg.norm(g) for g in G])
            for k, w in want.items():
                src = 'torchjd'
                if exact is not None and (k == 'mgda' or (k in ('upgrad', 'dualproj') and len(G) == 3)):
                    w, src = exact[i][k], 'scipy'
                g, w = np.array(got[k]), np.array(w)
                err = np.linalg.norm(g - w) / max(np.linalg.norm(w), size)
                key = (k, len(G), src)
                worst[key] = max(worst.get(key, 0), err)
        for (k, m, src), err in sorted(worst.items()):
            tol = 1e-6 if src == 'torchjd' else 1e-5
            ok = err <= tol
            failures += not ok
            log.info('aggregator %-12s %d objectives: worst error %.2e against %s %s', k, m, err, src, 'ok' if ok else f'FAIL (> {tol:g})')

    for (shape, d), got in zip(LEG_TABLE, js['leg']):
        want = LEG_TABLE[(shape, d)]
        ok = abs(got - want) <= max(3, 0.03 * abs(want))
        failures += not ok
        log.info('leg %-8s %.1f m: widget %8.1f, reward_studies.md %6d %s', shape, d, got, want, 'ok' if ok else 'FAIL')

    try:
        from scipy.stats import mannwhitneyu
        for (a, b), got in zip(mw, js['mw']):
            want = mannwhitneyu(a, b, alternative='two-sided', method='exact').pvalue
            ok = abs(got - want) < 1e-9
            failures += not ok
            log.info('Mann-Whitney p: widget %.6f, scipy %.6f %s', got, want, 'ok' if ok else 'FAIL')
    except ImportError:
        log.warning('scipy not installed: Mann-Whitney check skipped')

    if failures:
        log.error('%d check(s) failed', failures)
        sys.exit(1)
    log.info('all widget checks passed')


if __name__ == '__main__':
    main()
