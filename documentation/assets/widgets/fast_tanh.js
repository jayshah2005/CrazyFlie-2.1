/* Widget "fast-tanh": rl_tools' FAST_TANH activation and its gradient (07-td3.md).
 *   y = x (27 + x^2) / (27 + 9 x^2) for x clipped to [-3, 3]  (y = +-1 at the clip)
 * rl_tools back-propagates dy/dx = 1 - y^2 (the tanh rule, not the rational function's own
 * derivative); l2f/py/networks.py copies that. Beyond |x| = 3 the gradient is exactly 0. */
(function () {
  'use strict';
  const W = window.DocWidgets;
  const ft = (x) => { const c = Math.max(-3, Math.min(3, x)); return (c * (27 + c * c)) / (27 + 9 * c * c); };
  const exact = (x) => { if (Math.abs(x) >= 3) return 0; const n = (27 + 3 * x * x) * (27 + 9 * x * x) - 18 * x * x * (27 + x * x); return n / Math.pow(27 + 9 * x * x, 2); };
  W.register('fast-tanh', (root) => {
    W.header(root, 'FAST_TANH: the activation of every hidden layer', 'A cheap rational stand-in for tanh, clipped at ±3. Left: the value. Right: the slope that training uses.');
    const st = { x: 2.5 };
    const ctl = W.controls(root);
    W.slider(ctl, { label: 'Pre-activation x', min: -4, max: 4, step: 0.05, value: st.x, fmt: (v) => v.toFixed(2), onInput: (v) => { st.x = v; draw(); } });
    const grid = W.h('div', { style: 'display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px' });
    const a = W.h('div'), b = W.h('div'); grid.append(a, b);
    const out = W.h('div', { class: 'readout' });
    root.append(grid, out);
    function draw() {
      const xsr = []; for (let x = -4; x <= 4.0001; x += 0.02) xsr.push(x);
      W.linePlot(a, { width: 420, height: 220, x: { min: -4, max: 4, label: 'x', fmt: (v) => v.toFixed(0) }, y: { min: -1.1, max: 1.1, label: 'y', fmt: (v) => v.toFixed(1) },
        series: [{ name: 'tanh', points: xsr.map((x) => [x, Math.tanh(x)]), color: 'var(--ink-2)', width: 1.5 }, { name: 'FAST_TANH', points: xsr.map((x) => [x, ft(x)]) }],
        markers: [{ x: st.x, y: ft(st.x) }] });
      W.linePlot(b, { width: 420, height: 220, x: { min: -4, max: 4, label: 'x', fmt: (v) => v.toFixed(0) }, y: { min: 0, max: 1.15, label: 'dy/dx', fmt: (v) => v.toFixed(1) },
        series: [{ name: 'used in training: 1 − y²', points: xsr.map((x) => [x, 1 - ft(x) * ft(x)]) }, { name: 'true slope of the rational', points: xsr.map((x) => [x, exact(x)]), color: 'var(--series-2)', width: 1.5 }],
        markers: [{ x: st.x, y: 1 - ft(st.x) * ft(st.x) }] });
      const y = ft(st.x);
      out.innerHTML = `At x = ${st.x.toFixed(2)}: y = <b>${y.toFixed(4)}</b> (tanh ${Math.tanh(st.x).toFixed(4)}), training slope <b>${(1 - y * y).toFixed(4)}</b>. ` +
        'Beyond ±3 the unit is saturated: its slope is exactly 0 and no gradient passes through it. With the untuned reward (every weight 1, values near −10⁵) the units saturated, and from about 0.4M steps the logged actor gradient was exactly 0 for every objective (l2f/docs/results.md, "Untuned weights"); the actor\'s output layer uses FAST_TANH too, so a saturated output stops learning.';
    }
    draw();
  });
})();
