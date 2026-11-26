/* Widget "seed-compare": per-seed results of two training arms and what they can and cannot show
 * (12-experiments.md, 13-issues.md). The unit is the seed, as in l2f/sim/study_stats.py.
 *   median difference B - A, with a bootstrap 95% interval (seeds resampled with replacement)
 *   Mann-Whitney U, exact two-sided permutation p value (midranks for ties; normal approximation
 *   when there are more than 200,000 splits)
 *   Cliff's delta = (#{b > a} - #{b < a}) / (n_A n_B), from -1 to 1
 * Data come from the page: data-arms="name:v1,v2,...;name2:...", data-metric, data-a, data-b. */
(function () {
  'use strict';
  const W = window.DocWidgets;
  const median = (a) => { const s = a.slice().sort((x, y) => x - y); const n = s.length; return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2; };
  function midranks(v) {
    const idx = v.map((x, i) => [x, i]).sort((p, q) => p[0] - q[0]);
    const r = new Array(v.length);
    for (let i = 0; i < idx.length;) {
      let j = i; while (j + 1 < idx.length && idx[j + 1][0] === idx[i][0]) j++;
      for (let k = i; k <= j; k++) r[idx[k][1]] = (i + j) / 2 + 1;
      i = j + 1;
    }
    return r;
  }
  function choose(n, k) { let c = 1; for (let i = 1; i <= k; i++) c = (c * (n - k + i)) / i; return c; }
  function mannWhitney(a, b) {
    const v = a.concat(b), na = a.length, nb = b.length, N = v.length;
    const r = midranks(v);
    const Ua = r.slice(0, na).reduce((s, x) => s + x, 0) - (na * (na + 1)) / 2;
    const mu = (na * nb) / 2, obs = Math.abs(Ua - mu) - 1e-9;
    if (choose(N, na) > 200000) {
      const sd = Math.sqrt((na * nb * (N + 1)) / 12);
      const z = Math.abs(Ua - mu) / sd;
      const p = 2 * (1 - 0.5 * (1 + erf(z / Math.SQRT2)));
      return { U: Ua, p, exact: false };
    }
    let count = 0, total = 0;
    const pick = (start, left, sum) => {
      if (left === 0) { total++; if (Math.abs(sum - (na * (na + 1)) / 2 - mu) >= obs) count++; return; }
      for (let i = start; i <= N - left; i++) pick(i + 1, left - 1, sum + r[i]);
    };
    pick(0, na, 0);
    return { U: Ua, p: count / total, exact: true };
  }
  function erf(x) { const t = 1 / (1 + 0.3275911 * Math.abs(x)); const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x); return x >= 0 ? y : -y; }
  function cliff(a, b) { let g = 0, l = 0; for (const x of a) for (const y of b) { if (y > x) g++; else if (y < x) l++; } return (g - l) / (a.length * b.length); }
  function bootstrap(a, b, n = 4000, seed = 11) {
    const r = W.rng(seed), diffs = [];
    const res = (x) => x.map(() => x[Math.floor(r() * x.length)]);
    for (let i = 0; i < n; i++) diffs.push(median(res(b)) - median(res(a)));
    diffs.sort((x, y) => x - y);
    return [diffs[Math.floor(0.025 * n)], diffs[Math.floor(0.975 * n)]];
  }
  W.stats = { median, mannWhitney, cliff, bootstrap };

  W.register('seed-compare', (root, opts) => {
    const arms = (opts.arms || '').split(';').filter(Boolean).map((s) => { const [name, vals] = s.split(':'); return { name: name.trim(), vals: vals.split(',').map(Number) }; });
    if (arms.length < 2) { root.innerHTML = '<p class="widget-error">seed-compare needs data-arms with two or more arms.</p>'; return; }
    const metric = opts.metric || 'value';
    W.header(root, `Per-seed results: ${metric}`, 'Each dot is one training seed (the median over its late checkpoints); the tick is the arm\'s median. Pick two arms to compare them the way <code>l2f/sim/study_stats.py</code> does.');
    const st = { a: opts.a || arms[0].name, b: opts.b || arms[1].name, four: false };
    const ctl = W.controls(root);
    const opt = arms.map((x) => [x.name, x.name]);
    W.select(ctl, { label: 'Arm A (baseline)', options: opt, value: st.a, onChange: (v) => { st.a = v; draw(); } });
    W.select(ctl, { label: 'Arm B', options: opt, value: st.b, onChange: (v) => { st.b = v; draw(); } });
    const cb = W.h('div', { class: 'checks', style: 'align-self:end' });
    W.checkbox(cb, { label: 'Only the first 4 seeds', value: false, onChange: (v) => { st.four = v; draw(); } });
    ctl.append(cb);
    const box = W.h('div'); const out = W.h('div', { class: 'readout' });
    root.append(box, out);
    function draw() {
      const data = arms.map((x) => ({ name: x.name, vals: st.four ? x.vals.slice(0, 4) : x.vals }));
      const all = data.flatMap((x) => x.vals);
      const lo = Math.min(0, ...all), hi = Math.max(...all) * 1.08 || 1;
      const Wd = 640, rowH = 26, ml = 120, mr = 20, Hd = data.length * rowH + 34;
      const xs = (v) => ml + ((v - lo) / (hi - lo)) * (Wd - ml - mr);
      const svg = W.svgEl('svg', { viewBox: `0 0 ${Wd} ${Hd}`, role: 'img', 'aria-label': `Per-seed ${metric}` });
      for (const t of W.niceTicks(lo, hi, 6)) {
        svg.append(W.svgEl('line', { x1: xs(t), x2: xs(t), y1: 6, y2: Hd - 24, style: 'stroke:var(--grid)' }));
        const tx = W.svgEl('text', { x: xs(t), y: Hd - 8, 'text-anchor': 'middle' }); tx.textContent = W.fmt(t, 1); svg.append(tx);
      }
      data.forEach((arm, i) => {
        const y = 18 + i * rowH;
        const sel = arm.name === st.a || arm.name === st.b;
        const col = sel ? W.color(arms.findIndex((x) => x.name === arm.name)) : 'var(--axis)';
        const lab = W.svgEl('text', { x: ml - 10, y: y + 4, 'text-anchor': 'end', class: sel ? 'lbl-ink' : '' }); lab.textContent = arm.name; svg.append(lab);
        arm.vals.forEach((v, k) => svg.append(W.svgEl('circle', { cx: xs(v), cy: y + ((k % 3) - 1) * 4, r: 4.5, style: `fill:${col};stroke:var(--chart-surface);stroke-width:2` })));
        const m = xs(median(arm.vals));
        svg.append(W.svgEl('line', { x1: m, x2: m, y1: y - 9, y2: y + 9, style: `stroke:${sel ? 'var(--ink)' : 'var(--ink-2)'};stroke-width:2` }));
      });
      box.innerHTML = ''; box.append(svg);
      const A = data.find((x) => x.name === st.a).vals, B = data.find((x) => x.name === st.b).vals;
      if (st.a === st.b) { out.innerHTML = 'Pick two different arms.'; return; }
      const mw = W.stats.mannWhitney(A, B), ci = bootstrap(A, B), d = cliff(A, B);
      const diff = median(B) - median(A);
      out.innerHTML = `<b>${st.b}</b> − <b>${st.a}</b>: median difference <b>${W.fmt(diff, 2)}</b> (bootstrap 95% interval ${W.fmt(ci[0], 2)} to ${W.fmt(ci[1], 2)}), Mann-Whitney ${mw.exact ? 'exact' : 'approximate'} p = <b>${W.fmt(mw.p, 3)}</b>, Cliff's δ = <b>${W.fmt(d, 2)}</b> (n = ${A.length} and ${B.length} seeds). ` +
        (mw.p < 0.05 ? 'Below 0.05 for this one comparison; a study with many comparisons must correct for them (Holm), which the stats tool does over every arm and metric.' : 'Not significant: with this many seeds, a difference this size is within what seed-to-seed spread alone produces.') +
        ' Tick "Only the first 4 seeds" to see how a 4-seed study can point the other way.';
    }
    draw();
  });
})();
