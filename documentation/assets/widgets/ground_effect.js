/* Widget "ground-effect": the simulator's ground-effect thrust factor (04-simulator.md).
 * Cheeseman-Bennett: thrust x 1 / (1 - k (R / 4h)^2), with R = 0.0275 m (55 mm propellers),
 * h = rotor height above the floor = height + 0.02 m, k ~ U(0, 2) per episode (not identified on
 * this drone), capped at 1.4 (Mission::ge_*). */
(function () {
  'use strict';
  const W = window.DocWidgets;
  const factor = (z, k) => { const h = z + 0.02; const q = 1 - k * Math.pow(0.0275 / (4 * h), 2); return q <= 1 / 1.4 ? 1.4 : Math.min(1.4, 1 / q); };
  W.register('ground-effect', (root) => {
    W.header(root, 'Ground effect near the floor', 'Close to the floor the rotors push against it and get extra thrust. Training draws the strength k per episode because it was never measured on this drone.');
    const st = { k: 1.0 };
    const ctl = W.controls(root);
    W.slider(ctl, { label: 'Strength k', min: 0, max: 2, step: 0.1, value: st.k, fmt: (v) => v.toFixed(1), onInput: (v) => { st.k = v; draw(); } });
    const plot = W.h('div'); const out = W.h('div', { class: 'readout' });
    root.append(plot, out);
    function draw() {
      const pts = []; for (let z = 0; z <= 0.3001; z += 0.002) pts.push([z * 100, factor(z, st.k)]);
      const maxk = []; for (let z = 0; z <= 0.3001; z += 0.002) maxk.push([z * 100, factor(z, 2)]);
      W.linePlot(plot, { height: 220, x: { min: 0, max: 30, label: 'height of the drone above the floor (cm)', fmt: (v) => v.toFixed(0) }, y: { min: 0.98, max: 1.45, label: 'thrust factor', fmt: (v) => v.toFixed(2) },
        series: [{ name: `k = ${st.k.toFixed(1)}`, points: pts }, { name: 'k = 2 (largest in training)', points: maxk, color: 'var(--ink-2)', width: 1.5 }] });
      out.innerHTML = `Thrust factor at 0 cm: <b>${factor(0, st.k).toFixed(3)}</b>, 2 cm: <b>${factor(0.02, st.k).toFixed(3)}</b>, 5 cm: <b>${factor(0.05, st.k).toFixed(3)}</b>, 10 cm: <b>${factor(0.1, st.k).toFixed(3)}</b>. It matters for take-off and landing only; above about 10 cm it is gone.`;
    }
    draw();
  });
})();
