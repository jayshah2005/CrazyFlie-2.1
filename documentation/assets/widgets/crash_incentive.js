/* Widget "crash-incentive": is flying a waypoint leg worth more than crashing (06-reward.md)?
 *
 * The drone flies a perfect leg of length D: accelerate at 5 m/s^2, cruise at 1.5 m/s, brake at
 * 2 m/s^2 on the curve v = sqrt(2 * 2 * d) (the fastest flight with no speed-limit cost), and the
 * reward of every 10 ms step is
 *   r = -0.5 * (w_pos * f(d) + w_vel * v^2 [where it applies] + w_time * [not arrived])
 *       (+ K (d_t - gamma d_t+1) for the progress reward)
 * with f(d) the position cost shape (squared, Huber, flat cap, progress = flat beyond the cap).
 * G_t = sum_j gamma^j r_t+j over a 3 s window from the jump (l2f/docs/reward_studies.md section 4
 * uses the same profile and window). Crashing at step t is worth -P (and -K d_t with progress),
 * so states with G_t below that are "crash-better" (what l2f/py/reward_check.py counts).
 * Only position, velocity and time costs are counted: tilt, action, smoothing and angular
 * acceleration add more in real flight, so real legs cost more than this.
 */
(function () {
  'use strict';
  const W = window.DocWidgets;

  function shapeCost(shape, d, c) {
    if (shape === 'squared') return d * d;
    if (shape === 'huber') return d <= c ? d * d : c * c + 2 * c * (d - c);
    return Math.min(d * d, c * c); // flat cap, and the progress reward's flat far field
  }

  /** The leg and its discounted costs; exported for tools/check_widgets.py. */
  function legCost(p) {
    const o = Object.assign({ D: 1, shape: 'huber', cap: 0.2, wPos: 40, wVel: 1, includeVel: false, wTime: 0.5, K: 20, P: 200, gamma: 0.99, a: 5, vmax: 1.5, brake: 2, window: 3, dt: 0.01, scale: 0.5 }, p);
    const n = Math.round(o.window / o.dt);
    let x = 0, v = 0;
    const steps = [];
    for (let k = 0; k < n; k++) {
      const d = Math.max(0, o.D - x);
      const arrived = d < 0.05 && v < 0.2;
      const pos = o.scale * o.wPos * shapeCost(o.shape, d, o.cap);
      const velOn = o.includeVel && (o.shape === 'squared' || d <= o.cap);
      const vel = velOn ? o.scale * o.wVel * v * v : 0;
      const time = arrived ? 0 : o.scale * o.wTime;
      // motion over the step
      const vb = Math.sqrt(2 * o.brake * d);
      v = Math.min(v + o.a * o.dt, o.vmax, vb);
      x += v * o.dt;
      const dNext = Math.max(0, o.D - x);
      const prog = o.shape === 'progress' ? o.K * (d - o.gamma * dNext) : 0;
      steps.push({ t: k * o.dt, d, v, arrived, pos: -pos, vel: -vel, time: -time, prog, r: -pos - vel - time + prog });
    }
    const G = new Array(n).fill(0);
    const parts = { pos: 0, vel: 0, time: 0, prog: 0 };
    for (let k = n - 1; k >= 0; k--) G[k] = steps[k].r + o.gamma * (k + 1 < n ? G[k + 1] : 0);
    for (let k = 0; k < n; k++) { const f = Math.pow(o.gamma, k); for (const key of Object.keys(parts)) parts[key] += f * steps[k][key]; }
    const crashValue = steps.map((s) => -o.P - (o.shape === 'progress' ? o.K * s.d : 0));
    const firstArrival = steps.findIndex((s) => s.arrived);
    const until = firstArrival < 0 ? n : Math.max(1, firstArrival);
    let better = 0;
    for (let k = 0; k < until; k++) if (G[k] < crashValue[k]) better++;
    return { steps, G, G0: G[0], parts, crashValue, crashBetterShare: better / until, arrivalTime: firstArrival < 0 ? null : firstArrival * o.dt, opts: o };
  }
  W.legCost = legCost;

  W.register('crash-incentive', (root, opts) => {
    W.header(root, 'Crash-incentive calculator: one waypoint leg', 'A perfect flight to a target <b>D</b> metres away, scored with the training reward at its <b>final</b> weights. If the discounted return of flying on is below the crash penalty, the reward teaches the policy to crash.');
    const st = {
      D: +opts.jump || 1.0, shape: opts.shape || 'huber', cap: 0.2, wPos: 40, wTime: +opts.time || 0.5, P: +opts.penalty || 200,
      K: 20, gamma: 0.99, includeVel: false,
    };
    const ctl = W.controls(root);
    W.slider(ctl, { label: 'Jump D (m)', min: 0.1, max: 3, step: 0.05, value: st.D, fmt: (v) => v.toFixed(2), onInput: (v) => { st.D = v; draw(); } });
    W.select(ctl, { label: 'Position cost shape', value: st.shape, options: [['squared', 'Squared d² (original)'], ['huber', 'Huber (quadratic to the cap, then linear)'], ['flat', 'Flat cap (min(d², c²))'], ['progress', 'Progress reward K(dₜ − γdₜ₊₁), flat far away']], onChange: (v) => { st.shape = v; draw(); } });
    W.slider(ctl, { label: 'Cap c (m)', min: 0.05, max: 0.5, step: 0.05, value: st.cap, fmt: (v) => v.toFixed(2), onInput: (v) => { st.cap = v; draw(); } });
    W.slider(ctl, { label: 'Position weight w_pos', min: 5, max: 40, step: 1, value: st.wPos, onInput: (v) => { st.wPos = v; draw(); } });
    W.slider(ctl, { label: 'Crash penalty P', min: 100, max: 800, step: 50, value: st.P, onInput: (v) => { st.P = v; draw(); } });
    W.slider(ctl, { label: 'Time weight w_time', min: 0, max: 2, step: 0.1, value: st.wTime, fmt: (v) => v.toFixed(1), onInput: (v) => { st.wTime = v; draw(); } });
    W.slider(ctl, { label: 'Progress K (per metre)', min: 0, max: 40, step: 1, value: st.K, onInput: (v) => { st.K = v; draw(); } });
    W.slider(ctl, { label: 'Discount γ', min: 0.95, max: 0.999, step: 0.001, value: st.gamma, fmt: (v) => v.toFixed(3), onInput: (v) => { st.gamma = v; draw(); } });
    const cb = W.h('div', { class: 'checks', style: 'align-self:end' });
    W.checkbox(cb, { label: 'Add the velocity cost (w = 1)', value: st.includeVel, onChange: (v) => { st.includeVel = v; draw(); } });
    ctl.append(cb);
    const plotA = W.h('div');
    const plotB = W.h('div');
    const out = W.h('div', { class: 'readout' });
    root.append(W.h('div', { class: 'w-sub', style: 'margin-top:6px' }, 'Value of flying on from each moment of the leg (G_t), against the value of crashing at that moment:'), plotA, out,
      W.h('div', { class: 'w-sub', style: 'margin-top:14px' }, 'Position reward per 10 ms step, by distance to the target, at this weight:'), plotB);

    function draw() {
      const res = legCost(st);
      const pts = res.steps.map((s, k) => [s.t, res.G[k]]);
      const crash = res.steps.map((s, k) => [s.t, res.crashValue[k]]);
      const ymin = Math.min(-st.P * 1.25, ...res.G, ...res.crashValue) * 1.05;
      const ymax = Math.max(0, ...res.G) * 1.1 + 1; // the progress reward can make the return positive
      const bands = [];
      let start = null;
      res.steps.forEach((s, k) => {
        const bad = res.G[k] < res.crashValue[k];
        if (bad && start === null) start = s.t;
        if ((!bad || k === res.steps.length - 1) && start !== null) { bands.push({ x0: start, x1: s.t, color: 'var(--bad)', opacity: 0.12 }); start = null; }
      });
      W.linePlot(plotA, {
        height: 250, x: { min: 0, max: 3, label: 'time after the jump (s)', fmt: (v) => v.toFixed(1) }, y: { min: ymin, max: ymax, label: 'return', fmt: (v) => W.fmt(v, 0) },
        series: [{ name: 'G_t: fly on (perfect flight)', points: pts }, { name: st.shape === 'progress' ? 'crash now: −P − K·d' : 'crash now: −P', points: crash, color: 'var(--series-8)' }],
        bands,
        vlines: res.arrivalTime !== null ? [{ x: res.arrivalTime, label: 'arrived (5 cm, 0.2 m/s)' }] : [],
      });
      const ds = []; for (let d = 0; d <= 3.0001; d += 0.02) ds.push(d);
      // the flat cap and the progress reward's far field are the same curve: one series for both
      const shapes = [['squared', 'squared', ['squared']], ['huber', 'Huber', ['huber']], ['flat', 'flat cap (also the progress reward\'s position cost)', ['flat', 'progress']]];
      W.linePlot(plotB, {
        height: 220, x: { min: 0, max: 3, label: 'distance to the target d (m)', fmt: (v) => v.toFixed(1) }, y: { min: 0, max: Math.min(0.5 * st.wPos * 9, Math.max(1, 0.5 * st.wPos * shapeCost('huber', 3, st.cap) * 1.3)), label: 'cost per step', fmt: (v) => W.fmt(v, 1) },
        series: shapes.map(([k, name, members], i) => ({ name: name + (members.includes(st.shape) ? ' (selected)' : ''), points: ds.map((d) => [d, 0.5 * st.wPos * shapeCost(k, d, st.cap)]), width: members.includes(st.shape) ? 3 : 1.5, color: W.color(i) })),
      });
      const better = res.G0 < res.crashValue[0];
      const p = res.parts;
      out.innerHTML = `Return of a perfect ${st.D.toFixed(2)} m leg: <b>G₀ = ${W.fmt(res.G0, 0)}</b> (position ${W.fmt(p.pos, 0)}, ${st.includeVel ? `velocity ${W.fmt(p.vel, 0)}, ` : ''}time ${W.fmt(p.time, 0)}${st.shape === 'progress' ? `, progress +${W.fmt(p.prog, 0)}` : ''}); arrival after ${res.arrivalTime === null ? 'more than 3 s' : res.arrivalTime.toFixed(2) + ' s'}. ` +
        `Crashing at the jump: <b>${W.fmt(res.crashValue[0], 0)}</b>. Crash-better states before arrival: <b>${Math.round(100 * res.crashBetterShare)}%</b>.` +
        `<div class="verdict ${better ? 'bad' : 'good'}">${better ? '✗ <b>Crashing pays more than flying.</b> TD3 maximizes return, so this reward pushes the policy toward crashing on such legs.' : '✓ <b>Flying pays more than crashing</b> on this leg (other costs and imperfect flight still lower the margin).'}</div>`;
    }
    draw();
  });
})();
