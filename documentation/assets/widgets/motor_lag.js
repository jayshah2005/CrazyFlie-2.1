/* Widget "motor-lag": first-order motor response (04-simulator.md).
 * The simulator's rotor speed follows the command with time constant tau:
 *   d omega / dt = (omega_cmd - omega) / tau, so omega(t) = omega_cmd + (omega_0 - omega_cmd) e^(-t/tau)
 * Thrust is c * omega^2 per rotor. Optional 10 ms dead time (L2F_ACT_DELAY_P): the command reaches
 * the motor one policy step later. Measured: tau 0.045 s at 3.66 V, 0.057 s at 3.84 V (model 0.060);
 * training range 0.045-0.09 s (L2F_MOTOR_TAU_MIN/MAX). */
(function () {
  'use strict';
  const W = window.DocWidgets;
  W.register('motor-lag', (root) => {
    W.header(root, 'Motor lag: how fast can a command change the thrust?', 'Step the command of one motor from hover (15,470 rpm) up by the chosen amount and watch the rotor speed and thrust follow.');
    const st = { tau: 0.06, step: 3000, dead: false };
    const ctl = W.controls(root);
    W.slider(ctl, { label: 'Time constant τ (s)', min: 0.02, max: 0.2, step: 0.005, value: st.tau, fmt: (v) => v.toFixed(3), onInput: (v) => { st.tau = v; draw(); } });
    W.slider(ctl, { label: 'Command step (rpm)', min: 500, max: 8000, step: 250, value: st.step, onInput: (v) => { st.step = v; draw(); } });
    const cb = W.h('div', { class: 'checks', style: 'align-self:end' });
    W.checkbox(cb, { label: '10 ms actuator dead time', value: st.dead, onChange: (v) => { st.dead = v; draw(); } });
    ctl.append(cb);
    const plot = W.h('div'); const out = W.h('div', { class: 'readout' });
    root.append(plot, out);
    function draw() {
      const w0 = 15470, w1 = Math.min(24000, w0 + st.step), delay = st.dead ? 0.01 : 0;
      const rpm = [], thr = [], cmd = [];
      for (let ms = 0; ms <= 300; ms += 1) {
        const t = ms / 1000;
        const te = Math.max(0, t - delay);
        const w = w1 + (w0 - w1) * Math.exp(-te / st.tau);
        rpm.push([ms, (w - w0) / (w1 - w0)]);
        thr.push([ms, (w * w - w0 * w0) / (w1 * w1 - w0 * w0)]);
        cmd.push([ms, ms >= 0 ? 1 : 0]);
      }
      const policySteps = []; for (let k = 10; k <= 300; k += 10) policySteps.push(k);
      W.linePlot(plot, { height: 240, x: { min: 0, max: 300, label: 'time after the command (ms)', fmt: (v) => v.toFixed(0) }, y: { min: 0, max: 1.05, label: 'share of the change', fmt: (v) => v.toFixed(1) },
        series: [{ name: 'command', points: cmd, color: 'var(--ink-2)', width: 1.5 }, { name: 'rotor speed', points: rpm }, { name: 'thrust (∝ rpm²)', points: thr }],
        vlines: [{ x: (delay + st.tau) * 1000, label: 't = τ: 63%' }] });
      const after = (ms) => 1 - Math.exp(-Math.max(0, ms / 1000 - delay) / st.tau);
      out.innerHTML = `After one policy step (10 ms) the rotor has made <b>${Math.round(100 * after(10))}%</b> of the change, after 3 steps <b>${Math.round(100 * after(30))}%</b>, after τ = ${Math.round(st.tau * 1000)} ms 63%, after 3τ 95%. ` +
        'The lag filters fast command changes: a policy that flips its command every step moves the rotors only a little each time, but the frame still vibrates. That was the chatter of the first flights (commands jumping between about 0% and 100%) and is why the reward has an action-rate cost. With the dead time on, the first 10 ms of every response are lost; together with lags up to 0.15 s it made every run of the time-to-target study collapse (runs 2-3).';
    }
    draw();
  });
})();
