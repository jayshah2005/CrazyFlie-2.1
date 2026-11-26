/* Widget "discount-horizon": how far ahead does TD3 look (02-rl-foundations.md)?
 * Weight of a reward t steps ahead: gamma^t. Effective horizon 1 / (1 - gamma) steps; half-life
 * ln(0.5) / ln(gamma) steps; a cost c paid on every step forever is worth c / (1 - gamma).
 * One step is dt = 10 ms (the simulator and the onboard policy run at 100 Hz). */
(function () {
  'use strict';
  const W = window.DocWidgets;
  W.register('discount-horizon', (root, opts) => {
    W.header(root, 'Discount and horizon', 'A reward <i>t</i> steps ahead counts γ<sup>t</sup> as much as one now. The policy runs at 100 Hz, so 100 steps are 1 second.');
    const st = { gamma: +opts.gamma || 0.99, bonus: 2 };
    const ctl = W.controls(root);
    W.slider(ctl, { label: 'Discount γ', min: 0.9, max: 0.999, step: 0.001, value: st.gamma, fmt: (v) => v.toFixed(3), onInput: (v) => { st.gamma = v; draw(); } });
    W.slider(ctl, { label: 'Constant reward per step b', min: 0.5, max: 4, step: 0.5, value: st.bonus, fmt: (v) => v.toFixed(1), onInput: (v) => { st.bonus = v; draw(); } });
    const plot = W.h('div'); const out = W.h('div', { class: 'readout' });
    root.append(plot, out);
    function draw() {
      const g = st.gamma;
      const horizon = 1 / (1 - g);
      const tmax = Math.min(10, Math.max(1, (5 * horizon) / 100));
      const w = [], c = [];
      for (let t = 0; t <= tmax * 100; t += Math.max(1, Math.round(tmax))) { w.push([t / 100, Math.pow(g, t)]); c.push([t / 100, 1 - Math.pow(g, t + 1)]); }
      W.linePlot(plot, { height: 230, x: { min: 0, max: tmax, label: 'time ahead (s)', fmt: (v) => v.toFixed(1) }, y: { min: 0, max: 1, label: 'share', fmt: (v) => v.toFixed(1) },
        series: [{ name: 'weight γᵗ of a reward t steps ahead', points: w }, { name: 'share of the whole return collected by then', points: c }],
        vlines: [{ x: horizon / 100, label: `horizon ${(horizon / 100).toFixed(2)} s` }] });
      const half = Math.log(0.5) / Math.log(g);
      out.innerHTML = `Effective horizon 1/(1−γ) = <b>${W.fmt(horizon, 0)} steps = ${(horizon / 100).toFixed(2)} s</b>; a reward ${W.fmt(half, 0)} steps (${(half / 100).toFixed(2)} s) ahead counts half. ` +
        `A reward b on every step forever is worth b/(1−γ) = <b>${W.fmt(st.bonus * horizon, 0)}</b>: with the paper's survival bonus b = 2 and γ = 0.99 that is 200, which is why a crash penalty of 200 is "the same incentive" in theory (06-reward.md).`;
    }
    draw();
  });
})();
