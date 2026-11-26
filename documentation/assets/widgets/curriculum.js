/* Widget "curriculum": the schedules that change during training (07-td3.md, 06-reward.md).
 * From l2f/py/td3.py and train.py (same as the C++ recipe):
 *   every 100k steps (not at 0): position weight x1.2 (cap 40), velocity and action weights x1.4
 *   (cap 1), starting from 5, 0.01, 0.01 (brushless.h);
 *   from 500k, every 100k: exploration std, target-noise std and clip x0.9 (from 0.1, 0.2, 0.5);
 *   mission progress k = min(1, t / 800k), updated every 10k (episode mix and randomization ranges
 *   grow with k); critic updates start after 15k steps, actor updates after 30k. */
(function () {
  'use strict';
  const W = window.DocWidgets;
  function at(t) {
    const n = Math.floor(t / 100000);
    const m = t >= 500000 ? Math.floor((t - 500000) / 100000) + 1 : 0;
    return {
      pos: Math.min(40, 5 * Math.pow(1.2, n)), vel: Math.min(1, 0.01 * Math.pow(1.4, n)), act: Math.min(1, 0.01 * Math.pow(1.4, n)),
      explore: 0.1 * Math.pow(0.9, m), tnoise: 0.2 * Math.pow(0.9, m), tclip: 0.5 * Math.pow(0.9, m),
      k: Math.min(1, Math.floor(t / 10000) * 10000 / 800000),
    };
  }
  W.curriculumAt = at;
  W.register('curriculum', (root) => {
    W.header(root, 'Training schedules', 'What changes while TD3 trains: reward weights grow, exploration noise shrinks, and the mission episodes and randomization ramp in. Move the step slider.');
    const st = { t: 1200000 };
    const ctl = W.controls(root);
    W.slider(ctl, { label: 'Training step', min: 0, max: 3000000, step: 10000, value: st.t, fmt: (v) => (v / 1e6).toFixed(2) + 'M', onInput: (v) => { st.t = v; draw(); } });
    const plot = W.h('div'); const out = W.h('div', { class: 'readout' });
    root.append(plot, out);
    const ts = []; for (let t = 0; t <= 3000000; t += 10000) ts.push(t);
    function draw() {
      const s = (f) => ts.map((t) => [t / 1e6, f(at(t))]);
      W.linePlot(plot, {
        height: 250, x: { min: 0, max: 3, label: 'training step (millions)', fmt: (v) => v.toFixed(1) + 'M' }, y: { min: 0, max: 1.05, label: 'share of full value', fmt: (v) => v.toFixed(1) },
        series: [
          { name: 'position weight / 40', points: s((a) => a.pos / 40) },
          { name: 'velocity & action weight / 1', points: s((a) => a.vel) },
          { name: 'exploration noise / 0.1', points: s((a) => a.explore / 0.1) },
          { name: 'mission progress k', points: s((a) => a.k) },
        ],
        vlines: [{ x: st.t / 1e6, label: 'now' }, { x: 2, label: 'recipe end (2M)' }],
      });
      const a = at(st.t);
      out.innerHTML = `At step <b>${(st.t / 1e6).toFixed(2)}M</b>: position weight <b>${W.fmt(a.pos, 1)}</b>, velocity weight <b>${W.fmt(a.vel, 3)}</b>, action weight <b>${W.fmt(a.act, 3)}</b>; exploration std <b>${W.fmt(a.explore, 4)}</b>, target noise std <b>${W.fmt(a.tnoise, 3)}</b> (clip ${W.fmt(a.tclip, 3)}); mission progress k = <b>${a.k.toFixed(2)}</b>. ` +
        (st.t <= 15000 ? 'No updates yet (the buffer fills with random flights until 15k steps).' : st.t <= 30000 ? 'Critics are training; the actor starts at 30k.' : 'Critics update every 10 steps, the actor every 20.') +
        ' Position reaches its cap of 40 at 1.2M and velocity/action reach 1 at 1.4M: a reward check must use these final weights, not the starting ones.';
    }
    draw();
  });
})();
