/* Widget "overestimation": why TD3 takes the smaller of two critics (02-rl-foundations.md, 07-td3.md).
 * Every action has true value 0. Two critics estimate each value with independent noise N(0, s^2).
 * The actor climbs critic 1, so it ends up at a* = argmax_a Q1(a), an action where Q1 errs high.
 *   one critic:          target uses Q1(a*)          -> biased up (E[max of K noisy values] > 0)
 *   second critic's view: Q2(a*)                    -> unbiased (Q2's error is independent of a*)
 *   TD3 (clipped double Q): min(Q1(a*), Q2(a*))      -> slightly below 0: a small, safer bias
 * The bias of a target feeds into the next target through bootstrapping, so it compounds. */
(function () {
  'use strict';
  const W = window.DocWidgets;
  W.register('overestimation', (root) => {
    W.header(root, 'Overestimation and the clipped double-Q target', 'All actions are equally good (true value 0). The critics only have noisy estimates, and the actor picks the action critic 1 likes best.');
    const st = { K: 10, s: 1, seed: 7 };
    const ctl = W.controls(root);
    W.slider(ctl, { label: 'Actions to choose from K', min: 2, max: 40, step: 1, value: st.K, onInput: (v) => { st.K = v; draw(); } });
    W.slider(ctl, { label: 'Critic error σ', min: 0.1, max: 2, step: 0.1, value: st.s, fmt: (v) => v.toFixed(1), onInput: (v) => { st.s = v; draw(); } });
    const row = W.h('div', { class: 'row', style: 'align-self:end' });
    row.append(W.h('button', { class: 'w-btn', type: 'button', onclick: () => { st.seed += 1; draw(); } }, 'New sample'));
    ctl.append(row);
    const box = W.h('div'); const out = W.h('div', { class: 'readout' });
    root.append(box, out);
    function draw() {
      const r = W.rng(st.seed);
      const N = 4000;
      const vals = { one: [], other: [], min: [] };
      for (let i = 0; i < N; i++) {
        let best = -Infinity, bestK = 0; const q2 = [];
        for (let k = 0; k < st.K; k++) { const q1 = st.s * r.normal(); q2.push(st.s * r.normal()); if (q1 > best) { best = q1; bestK = k; } }
        vals.one.push(best); vals.other.push(q2[bestK]); vals.min.push(Math.min(best, q2[bestK]));
      }
      const stats = (a) => { const s = a.slice().sort((x, y) => x - y); const mean = a.reduce((p, q) => p + q, 0) / a.length; return { mean, p10: s[Math.floor(0.1 * s.length)], p90: s[Math.floor(0.9 * s.length)] }; };
      const rows = [['One critic: Q1(a*)', stats(vals.one), 1], ['Second critic: Q2(a*)', stats(vals.other), 3], ['TD3: min(Q1(a*), Q2(a*))', stats(vals.min), 2]];
      const lo = Math.min(...rows.map((x) => x[1].p10)) - 0.2 * st.s, hi = Math.max(...rows.map((x) => x[1].p90)) + 0.2 * st.s;
      const Wd = 640, Hd = 150, ml = 190, mr = 84; // right margin leaves room for the "mean" labels
      const xs = (v) => ml + ((v - lo) / (hi - lo)) * (Wd - ml - mr);
      const svg = W.svgEl('svg', { viewBox: `0 0 ${Wd} ${Hd}`, role: 'img', 'aria-label': 'Bias of the three targets' });
      for (const t of W.niceTicks(lo, hi, 6)) {
        svg.append(W.svgEl('line', { x1: xs(t), x2: xs(t), y1: 10, y2: Hd - 24, style: `stroke:${t === 0 ? 'var(--axis)' : 'var(--grid)'};stroke-width:${t === 0 ? 1.5 : 1}` }));
        const tx = W.svgEl('text', { x: xs(t), y: Hd - 8, 'text-anchor': 'middle' }); tx.textContent = W.fmt(t, 1); svg.append(tx);
      }
      rows.forEach(([name, s, c], i) => {
        const y = 30 + i * 38;
        const lab = W.svgEl('text', { x: ml - 10, y: y + 4, 'text-anchor': 'end', class: 'lbl-ink' }); lab.textContent = name; svg.append(lab);
        svg.append(W.svgEl('line', { x1: xs(s.p10), x2: xs(s.p90), y1: y, y2: y, style: `stroke:var(--series-${c});stroke-width:2;stroke-linecap:round` }));
        svg.append(W.svgEl('circle', { cx: xs(s.mean), cy: y, r: 5.5, style: `fill:var(--series-${c});stroke:var(--chart-surface);stroke-width:2` }));
        const v = W.svgEl('text', { x: xs(s.p90) + 8, y: y + 4 }); v.textContent = `mean ${W.fmt(s.mean, 2)}`; svg.append(v);
      });
      box.innerHTML = ''; box.append(svg);
      out.innerHTML = `Dots: mean target value over ${N} draws; lines: 10th to 90th percentile; the true value is 0. One critic overestimates by <b>${W.fmt(rows[0][1].mean, 2)}</b>, and the error grows with more actions to pick from and noisier critics. ` +
        'In TD learning each target is built from the next estimate, so an upward bias feeds itself. TD3 takes the smaller of two critics (its "clipped double-Q" target): slightly pessimistic, but stable.';
    }
    draw();
  });
})();
