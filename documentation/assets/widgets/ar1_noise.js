/* Widget "ar1-noise": white against correlated (AR(1)) gyro noise (08-sensor-path.md).
 *   x_t = rho x_(t-1) + sigma sqrt(1 - rho^2) e_t,  e_t ~ N(0, 1)
 * Both series have the same standard deviation sigma. The mean of N correlated samples has variance
 *   sigma^2 / N * (1 + 2 sum_{k=1}^{N-1} (1 - k/N) rho^k),
 * so correlated noise does not average out over a few steps. The hover survey measured lag-1
 * correlations of 0.54-0.66 (x), 0.24-0.35 (y), 0.23-0.39 (z). */
(function () {
  'use strict';
  const W = window.DocWidgets;
  W.register('ar1-noise', (root) => {
    W.header(root, 'White noise against correlated (AR(1)) noise', 'Same spread, different memory. Correlated noise drifts for several steps, so a policy reads it as real rotation and reacts.');
    const st = { rho: 0.6, seed: 3 };
    const ctl = W.controls(root);
    W.slider(ctl, { label: 'Lag-1 correlation ρ', min: 0, max: 0.95, step: 0.05, value: st.rho, fmt: (v) => v.toFixed(2), onInput: (v) => { st.rho = v; draw(); } });
    const row = W.h('div', { class: 'row', style: 'align-self:end' });
    row.append(W.h('button', { class: 'w-btn', type: 'button', onclick: () => { st.seed += 1; draw(); } }, 'New sample'));
    ctl.append(row);
    const plot = W.h('div'); const out = W.h('div', { class: 'readout' });
    root.append(plot, out);
    function draw() {
      const r = W.rng(st.seed), r2 = W.rng(st.seed + 1000);
      const white = [], corr = [];
      let x = r2.normal();
      for (let t = 0; t < 100; t++) {
        white.push([t * 10, r.normal()]);
        x = st.rho * x + Math.sqrt(1 - st.rho * st.rho) * r2.normal();
        corr.push([t * 10, x]);
      }
      W.linePlot(plot, { height: 230, x: { min: 0, max: 990, label: 'time (ms)', fmt: (v) => v.toFixed(0) }, y: { min: -3.5, max: 3.5, label: 'noise (σ = 1)', fmt: (v) => v.toFixed(0) },
        series: [{ name: 'white (what the simulator had at first)', points: white, width: 1.5 }, { name: `AR(1), ρ = ${st.rho.toFixed(2)} (like the real gyro)`, points: corr }] });
      const N = 10;
      let sum = 1; for (let k = 1; k < N; k++) sum += 2 * (1 - k / N) * Math.pow(st.rho, k);
      const sdMean = Math.sqrt(sum / N);
      out.innerHTML = `The average of ${N} samples (100 ms) keeps a spread of <b>${sdMean.toFixed(2)} σ</b> with this correlation, against <b>${(1 / Math.sqrt(N)).toFixed(2)} σ</b> for white noise. ` +
        'Policies trained on white gyro noise saw the real drone\'s slow, correlated noise as motion and shook (first sensor-path flight, 2026-11-22); calibration 1 added per-axis AR(1) noise to the simulator.';
    }
    draw();
  });
})();
