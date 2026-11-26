/* Widget framework for the documentation viewer.
 *
 * A page embeds a widget with an HTML block in its Markdown:
 *
 *   <div data-widget="crash-incentive" data-jump="1.0">
 *
 *   > **Interactive:** ... (fallback text shown on GitHub, replaced in the viewer)
 *
 *   </div>
 *
 * Each widget lives in its own file in assets/widgets/ and registers itself:
 *
 *   DocWidgets.register('crash-incentive', (root, opts) => { ... });
 *
 * `root` is the div (emptied before the call), `opts` its data-* attributes (dataset). Widgets
 * draw with the helpers below and use CSS variables for every color, so they follow the light /
 * dark theme without redrawing. Add the new file to index.html; tools/build_docs.py checks that
 * every data-widget name used in a page is registered and loaded.
 */
window.DocWidgets = (function () {
  'use strict';
  const registry = {};
  const SVGNS = 'http://www.w3.org/2000/svg';

  function register(name, fn) { registry[name] = fn; }

  function mountAll(root) {
    root.querySelectorAll('[data-widget]').forEach((el) => {
      const name = el.dataset.widget;
      const fn = registry[name];
      if (!fn) { el.insertAdjacentHTML('beforeend', `<p class="widget-error">Unknown widget "${name}" (is its file listed in index.html?)</p>`); return; }
      el.innerHTML = '';
      el.classList.add('widget');
      try { fn(el, Object.assign({}, el.dataset)); } catch (e) { console.error(e); el.innerHTML = `<p class="widget-error">Widget "${name}" failed: ${e.message}</p>`; }
    });
  }

  // ---------------------------------------------------------------- DOM helpers
  function h(tag, attrs, ...children) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (k === 'class') el.className = v;
      else if (k === 'html') el.innerHTML = v;
      else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
      else el.setAttribute(k, v);
    }
    for (const c of children) if (c != null) el.append(c);
    return el;
  }
  function header(root, title, sub) {
    root.append(h('p', { class: 'w-title' }, title));
    if (sub) root.append(h('p', { class: 'w-sub', html: sub }));
  }
  function controls(root) { const c = h('div', { class: 'controls' }); root.append(c); return c; }

  /** Range slider with a live value: {label, min, max, step, value, fmt(v), onInput(v)}. */
  function slider(parent, o) {
    const val = h('span', { class: 'val' });
    const input = h('input', { type: 'range', min: o.min, max: o.max, step: o.step, value: o.value, 'aria-label': o.label });
    const lab = h('label', { class: 'ctl' }, o.label, val, input);
    const show = () => { val.textContent = o.fmt ? o.fmt(+input.value) : input.value; };
    input.addEventListener('input', () => { show(); o.onInput && o.onInput(+input.value); });
    show();
    parent.append(lab);
    return { get value() { return +input.value; }, set value(v) { input.value = v; show(); }, input };
  }
  /** Drop-down: {label, options: [[value, text], ...], value, onChange(v)}. */
  function select(parent, o) {
    const sel = h('select', { 'aria-label': o.label });
    for (const [v, t] of o.options) sel.append(h('option', { value: v }, t));
    sel.value = o.value;
    sel.addEventListener('change', () => o.onChange && o.onChange(sel.value));
    parent.append(h('label', { class: 'ctl' }, o.label, h('br'), sel));
    return { get value() { return sel.value; }, set value(v) { sel.value = v; }, el: sel };
  }
  function checkbox(parent, o) {
    const input = h('input', { type: 'checkbox' });
    input.checked = !!o.value;
    input.addEventListener('change', () => o.onChange && o.onChange(input.checked));
    const lab = h('label', {}, input, o.swatch ? h('span', { class: 'swatch', style: `background:${o.swatch}` }) : null, ' ' + o.label);
    parent.append(lab);
    return { get value() { return input.checked; }, input };
  }

  // ---------------------------------------------------------------- numbers
  function fmt(v, d = 2) {
    if (v === null || v === undefined || !isFinite(v)) return '–';
    const a = Math.abs(v);
    if (a !== 0 && (a >= 1e5 || a < 1e-3)) return v.toExponential(1);
    return v.toLocaleString('en-US', { maximumFractionDigits: d, minimumFractionDigits: 0 });
  }
  function niceTicks(min, max, count = 5) {
    const span = max - min;
    if (!(span > 0)) return [min];
    const raw = span / count;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const norm = raw / mag;
    const step = (norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1) * mag;
    const start = Math.ceil(min / step - 1e-9) * step;
    const out = [];
    for (let v = start; v <= max + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
    return out;
  }
  /** Seeded random numbers (mulberry32) so a widget shows the same sample on every load. */
  function rng(seed = 1) {
    let a = seed >>> 0;
    const next = () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
    let spare = null;
    next.normal = () => {
      if (spare !== null) { const s = spare; spare = null; return s; }
      let u = 0, v = 0; while (u === 0) u = next(); while (v === 0) v = next();
      const r = Math.sqrt(-2 * Math.log(u));
      spare = r * Math.sin(2 * Math.PI * v);
      return r * Math.cos(2 * Math.PI * v);
    };
    return next;
  }

  // ---------------------------------------------------------------- linear algebra (small, dense)
  const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
  const add = (a, b) => a.map((x, i) => x + b[i]);
  const scale = (a, k) => a.map((x) => x * k);
  const norm = (a) => Math.sqrt(dot(a, a));
  /** Eigen-decomposition of a small symmetric matrix by Jacobi rotations: {values, vectors}
   *  with vectors[k] the k-th eigenvector, sorted by decreasing eigenvalue. */
  function symEig(M) {
    const n = M.length;
    const A = M.map((r) => r.slice());
    const V = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
    for (let sweep = 0; sweep < 60; sweep++) {
      let off = 0;
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += A[i][j] * A[i][j];
      if (off < 1e-24) break;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
        if (Math.abs(A[p][q]) < 1e-300) continue;
        const theta = (A[q][q] - A[p][p]) / (2 * A[p][q]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = A[k][p], akq = A[k][q];
          A[k][p] = c * akp - s * akq; A[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = A[p][k], aqk = A[q][k];
          A[p][k] = c * apk - s * aqk; A[q][k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = V[k][p], vkq = V[k][q];
          V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq;
        }
      }
    }
    const order = Array.from({ length: n }, (_, i) => i).sort((i, j) => A[j][j] - A[i][i]);
    return { values: order.map((i) => A[i][i]), vectors: order.map((i) => V.map((row) => row[i])) };
  }

  // ---------------------------------------------------------------- SVG plotting
  function svgEl(tag, attrs) {
    const el = document.createElementNS(SVGNS, tag);
    for (const [k, v] of Object.entries(attrs || {})) el.setAttribute(k, v);
    return el;
  }
  const color = (i) => `var(--series-${(i % 8) + 1})`;

  /**
   * Line plot into `box` (replaces its content). cfg = {
   *   width, height, x: {min, max, label, fmt}, y: {min, max, label, fmt},
   *   series: [{name, color, points: [[x, y], ...], dash, width}],
   *   hlines: [{y, label, color}], vlines: [{x, label, color}],
   *   bands: [{x0, x1, color, opacity}], markers: [{x, y, color, label}],
   *   tooltip: true, legend: true }
   * Hovering shows a crosshair and every series' value at that x (nearest point).
   */
  function linePlot(box, cfg) {
    box.innerHTML = '';
    box.classList.add('plot-box');
    const W = cfg.width || 640, H = cfg.height || 260;
    const m = { l: 52, r: 16, t: 12, b: 38 };
    const pw = W - m.l - m.r, ph = H - m.t - m.b;
    const xs = (v) => m.l + ((v - cfg.x.min) / (cfg.x.max - cfg.x.min)) * pw;
    const ys = (v) => m.t + ph - ((v - cfg.y.min) / (cfg.y.max - cfg.y.min)) * ph;
    const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': cfg.aria || cfg.y.label || 'plot' });
    const fx = cfg.x.fmt || ((v) => fmt(v)), fy = cfg.y.fmt || ((v) => fmt(v));
    for (const t of niceTicks(cfg.y.min, cfg.y.max, cfg.y.ticks || 5)) {
      svg.append(svgEl('line', { x1: m.l, x2: W - m.r, y1: ys(t), y2: ys(t), style: 'stroke:var(--grid);stroke-width:1' }));
      const tx = svgEl('text', { x: m.l - 6, y: ys(t) + 4, 'text-anchor': 'end' }); tx.textContent = fy(t); svg.append(tx);
    }
    for (const t of niceTicks(cfg.x.min, cfg.x.max, cfg.x.ticks || 6)) {
      svg.append(svgEl('line', { x1: xs(t), x2: xs(t), y1: H - m.b, y2: H - m.b + 4, style: 'stroke:var(--axis)' }));
      const tx = svgEl('text', { x: xs(t), y: H - m.b + 16, 'text-anchor': 'middle' }); tx.textContent = fx(t); svg.append(tx);
    }
    svg.append(svgEl('line', { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, style: 'stroke:var(--axis)' }));
    if (cfg.x.label) { const t = svgEl('text', { x: m.l + pw / 2, y: H - 4, 'text-anchor': 'middle' }); t.textContent = cfg.x.label; svg.append(t); }
    if (cfg.y.label) { const t = svgEl('text', { x: 12, y: m.t + ph / 2, 'text-anchor': 'middle', transform: `rotate(-90 12 ${m.t + ph / 2})` }); t.textContent = cfg.y.label; svg.append(t); }
    const clipId = 'clip' + Math.random().toString(36).slice(2);
    const defs = svgEl('defs'); const cp = svgEl('clipPath', { id: clipId }); cp.append(svgEl('rect', { x: m.l, y: m.t - 2, width: pw, height: ph + 4 })); defs.append(cp); svg.append(defs);
    const g = svgEl('g', { 'clip-path': `url(#${clipId})` }); svg.append(g);
    for (const b of cfg.bands || []) {
      const x0 = xs(Math.max(cfg.x.min, b.x0)), x1 = xs(Math.min(cfg.x.max, b.x1));
      if (x1 > x0) g.append(svgEl('rect', { x: x0, y: m.t, width: x1 - x0, height: ph, style: `fill:${b.color || 'var(--bad)'};opacity:${b.opacity || 0.12}` }));
    }
    for (const l of cfg.hlines || []) {
      g.append(svgEl('line', { x1: m.l, x2: W - m.r, y1: ys(l.y), y2: ys(l.y), style: `stroke:${l.color || 'var(--ink-2)'};stroke-width:1.5` }));
      if (l.label) { const t = svgEl('text', { x: W - m.r - 4, y: ys(l.y) - 5, 'text-anchor': 'end', class: 'lbl-ink' }); t.textContent = l.label; svg.append(t); }
    }
    for (const l of cfg.vlines || []) {
      g.append(svgEl('line', { x1: xs(l.x), x2: xs(l.x), y1: m.t, y2: H - m.b, style: `stroke:${l.color || 'var(--ink-2)'};stroke-width:1` }));
      if (l.label) { const t = svgEl('text', { x: xs(l.x) + 4, y: m.t + 11 }); t.textContent = l.label; svg.append(t); }
    }
    cfg.series.forEach((s, i) => {
      if (!s.points.length) return;
      const d = s.points.filter((p) => isFinite(p[1])).map((p, k) => `${k ? 'L' : 'M'}${xs(p[0]).toFixed(1)},${ys(p[1]).toFixed(1)}`).join('');
      g.append(svgEl('path', { d, style: `fill:none;stroke:${s.color || color(i)};stroke-width:${s.width || 2};stroke-linejoin:round;stroke-linecap:round${s.dash ? ';stroke-dasharray:' + s.dash : ''}` }));
    });
    for (const p of cfg.markers || []) {
      g.append(svgEl('circle', { cx: xs(p.x), cy: ys(p.y), r: 4.5, style: `fill:${p.color || 'var(--ink)'};stroke:var(--chart-surface);stroke-width:2` }));
      if (p.label) { const t = svgEl('text', { x: xs(p.x) + 7, y: ys(p.y) - 7, class: 'lbl-ink' }); t.textContent = p.label; svg.append(t); }
    }
    box.append(svg);
    if (cfg.legend !== false && cfg.series.length > 1) {
      const lg = h('div', { class: 'legend' });
      cfg.series.forEach((s, i) => lg.append(h('span', {}, h('span', { class: 'swatch', style: `background:${s.color || color(i)}` }), ' ' + s.name)));
      box.append(lg);
    }
    if (cfg.tooltip !== false) {
      const tip = h('div', { class: 'tip' }); box.append(tip);
      const cross = svgEl('line', { y1: m.t, y2: H - m.b, style: 'stroke:var(--axis);stroke-width:1;display:none' }); svg.append(cross);
      const hit = svgEl('rect', { x: m.l, y: m.t, width: pw, height: ph, style: 'fill:transparent' }); svg.append(hit);
      const nearest = (pts, x) => { let lo = 0, hi = pts.length - 1; while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (pts[mid][0] < x) lo = mid; else hi = mid; } return Math.abs(pts[lo][0] - x) < Math.abs(pts[hi][0] - x) ? pts[lo] : pts[hi]; };
      const move = (ev) => {
        const pt = svg.createSVGPoint(); pt.x = ev.clientX; pt.y = ev.clientY;
        const loc = pt.matrixTransform(svg.getScreenCTM().inverse());
        const x = cfg.x.min + ((loc.x - m.l) / pw) * (cfg.x.max - cfg.x.min);
        if (x < cfg.x.min || x > cfg.x.max) { tip.style.display = 'none'; cross.style.display = 'none'; return; }
        cross.setAttribute('x1', xs(x)); cross.setAttribute('x2', xs(x)); cross.style.display = '';
        const rows = cfg.series.filter((s) => s.points.length && !s.noTip).map((s, i) => { const p = nearest(s.points, x); return `<div><span class="swatch" style="background:${s.color || color(i)}"></span> ${s.name}: <b>${fy(p[1])}</b></div>`; });
        tip.innerHTML = `<div>${cfg.x.label ? cfg.x.label.replace(/\s*\(.*\)/, '') + ' ' : ''}<b>${fx(x)}</b></div>` + rows.join('');
        const r = box.getBoundingClientRect();
        const left = ev.clientX - r.left + 14;
        tip.style.display = 'block';
        tip.style.left = Math.min(left, r.width - tip.offsetWidth - 4) + 'px';
        tip.style.top = Math.max(0, ev.clientY - r.top - tip.offsetHeight - 10) + 'px';
      };
      hit.addEventListener('pointermove', move);
      hit.addEventListener('pointerleave', () => { tip.style.display = 'none'; cross.style.display = 'none'; });
    }
    return { svg, xs, ys };
  }

  return { register, mountAll, h, header, controls, slider, select, checkbox, fmt, niceTicks, rng, dot, add, scale, norm, symEig, svgEl, color, linePlot, registry };
})();
