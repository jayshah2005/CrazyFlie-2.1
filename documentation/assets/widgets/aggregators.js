/* Widget "aggregator-playground": drag two or three objective gradients in the plane and see the
 * update each gradient aggregator builds from them (09-moo.md).
 *
 * The formulas follow TorchJD (the library train.py uses), with all-ones preference vectors as in
 * l2f/py/moo.py:
 *   sum        d = sum_i g_i                      mean  d = sum_i g_i / m
 *   PCGrad     each g_i loses its component along every g_j it conflicts with (g_i . g_j < 0),
 *              then the results are summed (TorchJD uses a random order; here the fixed order 1..m)
 *   MGDA       the shortest vector in the convex hull of the g_i (weights >= 0, summing to 1)
 *   UPGrad     d = sum_i proj_C(g_i), C = {x : g_j . x >= 0 for all j} (the dual cone)
 *   DualProj   d = proj_C(sum_i g_i)
 *   Aligned-MTL w = sqrt(lambda_min) V Lambda^-1/2 V^T 1 from the Gram matrix M = V Lambda V^T, d = sum_i w_i g_i
 *   SVD-PGA    w = mean_j (o_j v_j / sqrt(lambda_j)) over the Gram matrix's eigenpairs, each
 *              eigenvector v_j flipped (o_j = +-1) so most of its squared mass is positive
 * DocWidgets.aggregate (exported below) is what tools/test_widgets.mjs checks against TorchJD.
 */
(function () {
  'use strict';
  const W = window.DocWidgets;
  const { dot, add, scale, norm, symEig } = W;
  const EPS = 1e-12;

  function gram(G) { return G.map((a) => G.map((b) => dot(a, b))); }
  function combine(G, w) { return G.reduce((d, g, i) => add(d, scale(g, w[i])), G[0].map(() => 0)); }

  /** Projection of v onto the dual cone {x : g . x >= 0 for every g in G}, in the plane. */
  function projCone(v, G) {
    const feasible = (x) => G.every((g) => dot(g, x) >= -1e-10 * norm(g) * (norm(x) + 1));
    if (feasible(v)) return v.slice();
    let best = v.map(() => 0), bestD = norm(v);
    for (const g of G) {
      const gg = dot(g, g);
      if (gg < EPS) continue;
      const x = add(v, scale(g, -dot(g, v) / gg));
      const dist = norm(add(x, scale(v, -1)));
      if (feasible(x) && dist < bestD) { best = x; bestD = dist; }
    }
    return best;
  }
  /** Shortest point of the segment a-b. */
  function minNormSegment(a, b) {
    const ab = add(a, scale(b, -1));
    const den = dot(ab, ab);
    const t = den < EPS ? 1 : Math.min(1, Math.max(0, dot(b, scale(ab, -1)) / den));
    return add(scale(a, t), scale(b, 1 - t));
  }
  /** Shortest point of the convex hull of 1-3 points in the plane. */
  function minNormHull(G) {
    if (G.length === 1) return G[0].slice();
    if (G.length === 2) return minNormSegment(G[0], G[1]);
    const [a, b, c] = G;
    const cross = (p, q) => p[0] * q[1] - p[1] * q[0];
    const s1 = cross(add(b, scale(a, -1)), scale(a, -1)), s2 = cross(add(c, scale(b, -1)), scale(b, -1)), s3 = cross(add(a, scale(c, -1)), scale(c, -1));
    if ((s1 >= 0 && s2 >= 0 && s3 >= 0) || (s1 <= 0 && s2 <= 0 && s3 <= 0)) return [0, 0];
    return [minNormSegment(a, b), minNormSegment(b, c), minNormSegment(c, a)].reduce((p, q) => (norm(q) < norm(p) ? q : p));
  }
  function alignedMtl(G) {
    const { values, vectors } = symEig(gram(G));
    const tol = Math.max(...values) * G.length * 2.2e-16;
    const keep = values.map((l, k) => k).filter((k) => values[k] > tol);
    if (!keep.length) return G[0].map(() => 0);
    const lmin = values[keep[keep.length - 1]];
    const w = G.map(() => 0);
    for (const k of keep) {
      const v = vectors[k];
      const proj = v.reduce((s, x) => s + x, 0) / Math.sqrt(values[k]);
      for (let i = 0; i < w.length; i++) w[i] += Math.sqrt(lmin) * v[i] * proj;
    }
    return combine(G, w);
  }
  function svdPga(G) {
    const { values, vectors } = symEig(gram(G));
    const keep = values.map((l, k) => k).filter((k) => values[k] > 1e-10 * values[0] && values[k] > 0);
    if (!keep.length) return G[0].map(() => 0);
    const w = G.map(() => 0);
    for (const k of keep) {
      let v = vectors[k];
      const pos = v.reduce((s, x) => s + (x > 0 ? x * x : 0), 0), neg = v.reduce((s, x) => s + (x < 0 ? x * x : 0), 0);
      if (neg > pos) v = scale(v, -1);
      for (let i = 0; i < w.length; i++) w[i] += v[i] / Math.sqrt(values[k]) / keep.length;
    }
    return combine(G, w);
  }
  function pcgrad(G) {
    let total = G[0].map(() => 0);
    G.forEach((gi, i) => {
      let v = gi.slice();
      G.forEach((gj, j) => {
        if (j === i) return;
        const p = dot(v, gj);
        if (p < 0) v = add(v, scale(gj, -p / dot(gj, gj)));
      });
      total = add(total, v);
    });
    return total;
  }
  const AGG = {
    sum: (G) => G.reduce(add),
    mean: (G) => scale(G.reduce(add), 1 / G.length),
    pcgrad,
    mgda: minNormHull,
    upgrad: (G) => G.map((g) => projCone(g, G)).reduce(add),
    dualproj: (G) => projCone(G.reduce(add), G),
    aligned_mtl: alignedMtl,
    svd_pga: svdPga,
  };
  W.aggregate = AGG;

  const METHODS = [
    ['sum', 'Sum (TD3)'], ['upgrad', 'UPGrad'], ['dualproj', 'DualProj'], ['svd_pga', 'SVD-PGA'],
    ['aligned_mtl', 'Aligned-MTL'], ['pcgrad', 'PCGrad'], ['mgda', 'MGDA'], ['mean', 'Mean'],
  ];
  const PRESETS = {
    opposed: { label: 'Big vs small, opposed (crash vs smoothing)', g: [[1.6, 0.3], [-0.26, 0.05]], names: ['crash', 'smoothing'] },
    aligned: { label: 'Aligned (no conflict)', g: [[1.2, 0.4], [0.9, 0.9]], names: ['g1', 'g2'] },
    orthogonal: { label: 'Orthogonal', g: [[1.2, 0], [0, 0.8]], names: ['g1', 'g2'] },
    three: { label: 'Three groups', g: [[1.4, 0.2], [-0.3, 0.7], [0.5, -0.6]], names: ['safety', 'precision', 'efficiency'] },
  };

  W.register('aggregator-playground', (root, opts) => {
    W.header(root, 'Aggregator playground', 'Each gray arrow is one objective\'s gradient (the direction that improves it). Drag the dots (or focus one and use the arrow keys). Colored arrows are the updates the aggregators build. The shaded wedge is the <b>dual cone</b>: updates there improve every objective at once.');
    let preset = opts.preset && PRESETS[opts.preset] ? opts.preset : 'opposed';
    let G = PRESETS[preset].g.map((v) => v.slice());
    let names = PRESETS[preset].names.slice();
    const shown = { sum: true, upgrad: true, dualproj: true, svd_pga: true };
    let unit = false;

    const ctl = W.controls(root);
    W.select(ctl, { label: 'Start from', options: Object.entries(PRESETS).map(([k, p]) => [k, p.label]), value: preset, onChange: (v) => { preset = v; G = PRESETS[v].g.map((x) => x.slice()); names = PRESETS[v].names.slice(); draw(); } });
    const lenBox = W.h('div', { class: 'checks', style: 'align-self:end' });
    W.checkbox(lenBox, { label: 'Directions only (same length)', value: unit, onChange: (v) => { unit = v; draw(); } });
    ctl.append(lenBox);
    const checks = W.h('div', { class: 'checks' });
    METHODS.forEach(([k, label], i) => W.checkbox(checks, { label, value: !!shown[k], swatch: W.color(i), onChange: (v) => { shown[k] = v; draw(); } }));
    root.append(checks);

    const R = 2;
    const S = 420;
    const toPx = (v) => [S / 2 + (v[0] / R) * (S / 2), S / 2 - (v[1] / R) * (S / 2)];
    const svg = W.svgEl('svg', { viewBox: `0 0 ${S} ${S}`, style: 'max-width:440px;margin:8px auto;overflow:hidden;border-radius:8px', role: 'img', 'aria-label': 'Gradients and aggregated updates' });
    root.append(svg);
    const table = W.h('div', { class: 'readout' });
    root.append(table);

    function arrow(parent, v, colorCss, widthPx, label) {
      const [x, y] = toPx(v);
      const [ox, oy] = toPx([0, 0]);
      const len = Math.hypot(x - ox, y - oy);
      parent.append(W.svgEl('line', { x1: ox, y1: oy, x2: x, y2: y, style: `stroke:${colorCss};stroke-width:${widthPx};stroke-linecap:round` }));
      if (len > 6) {
        const ux = (x - ox) / len, uy = (y - oy) / len, s = 7 + widthPx;
        const p = `${x},${y} ${x - s * ux - s * 0.45 * uy},${y - s * uy + s * 0.45 * ux} ${x - s * ux + s * 0.45 * uy},${y - s * uy - s * 0.45 * ux}`;
        parent.append(W.svgEl('polygon', { points: p, style: `fill:${colorCss}` }));
      }
      if (label) { const t = W.svgEl('text', { x: x + 8, y: y - 8, class: 'lbl-ink' }); t.textContent = label; parent.append(t); }
    }
    function conePolygon() {
      let poly = [[-R, -R], [R, -R], [R, R], [-R, R]];
      for (const g of G) {
        if (norm(g) < EPS) continue;
        const out = [];
        for (let i = 0; i < poly.length; i++) {
          const a = poly[i], b = poly[(i + 1) % poly.length];
          const da = dot(g, a), db = dot(g, b);
          if (da >= 0) out.push(a);
          if ((da >= 0) !== (db >= 0)) { const t = da / (da - db); out.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]); }
        }
        poly = out;
        if (!poly.length) break;
      }
      return poly;
    }

    function draw() {
      svg.innerHTML = '';
      svg.append(W.svgEl('rect', { x: 0, y: 0, width: S, height: S, style: 'fill:var(--chart-surface)' }));
      for (let k = -R; k <= R; k += 0.5) {
        const [x] = toPx([k, 0]); const [, y] = toPx([0, k]);
        svg.append(W.svgEl('line', { x1: x, x2: x, y1: 0, y2: S, style: `stroke:var(--grid);stroke-width:${k === 0 ? 1.5 : 1}` }));
        svg.append(W.svgEl('line', { x1: 0, x2: S, y1: y, y2: y, style: `stroke:var(--grid);stroke-width:${k === 0 ? 1.5 : 1}` }));
      }
      const poly = conePolygon();
      if (poly.length > 2) svg.append(W.svgEl('polygon', { points: poly.map((p) => toPx(p).join(',')).join(' '), style: 'fill:var(--good);opacity:0.10' }));
      const results = [];
      METHODS.forEach(([k, label], i) => {
        if (!shown[k]) return;
        const d = AGG[k](G);
        results.push({ k, label, d, color: W.color(i) });
        const n = norm(d);
        const v = unit && n > EPS ? scale(d, 1.6 / n) : d;
        arrow(svg, v, W.color(i), 2.5, '');
      });
      G.forEach((g, i) => {
        arrow(svg, g, 'var(--ink-2)', 2, names[i] || `g${i + 1}`);
        const [x, y] = toPx(g);
        const hnd = W.svgEl('circle', { cx: x, cy: y, r: 9, tabindex: 0, style: 'fill:var(--ink-2);fill-opacity:0.25;stroke:var(--ink-2);stroke-width:1.5;cursor:grab', 'aria-label': `Gradient ${names[i] || i + 1}: arrow keys rotate and resize` });
        // capture on the svg: the handle itself is redrawn on every move
        hnd.addEventListener('pointerdown', (e) => { e.preventDefault(); svg.setPointerCapture(e.pointerId); dragging = i; });
        hnd.addEventListener('keydown', (e) => {
          const a = Math.atan2(g[1], g[0]), r = norm(g);
          const step = { ArrowLeft: [0.087, 0], ArrowRight: [-0.087, 0], ArrowUp: [0, 0.05], ArrowDown: [0, -0.05] }[e.key];
          if (!step) return;
          e.preventDefault();
          const nr = Math.min(1.9, Math.max(0.05, r + step[1])), na = a + step[0];
          G[i] = [nr * Math.cos(na), nr * Math.sin(na)];
          focusAfter = i; draw();
        });
        svg.append(hnd);
        if (focusAfter === i) { hnd.focus(); }
      });
      focusAfter = -1;
      renderTable(results);
    }
    let dragging = -1, focusAfter = -1;
    svg.addEventListener('pointermove', (e) => {
      if (dragging < 0) return;
      const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
      const loc = pt.matrixTransform(svg.getScreenCTM().inverse());
      let v = [((loc.x - S / 2) / (S / 2)) * R, (-(loc.y - S / 2) / (S / 2)) * R];
      const n = norm(v);
      if (n > 1.9) v = scale(v, 1.9 / n);
      G[dragging] = v;
      draw();
    });
    const stop = () => { dragging = -1; };
    svg.addEventListener('pointerup', stop);
    svg.addEventListener('pointercancel', stop);

    function weightsFor(d) {
      if (G.length !== 2) return null;
      const M = gram(G);
      const det = M[0][0] * M[1][1] - M[0][1] * M[1][0];
      if (Math.abs(det) < 1e-9) return null;
      const b = [dot(G[0], d), dot(G[1], d)];
      return [(M[1][1] * b[0] - M[0][1] * b[1]) / det, (M[0][0] * b[1] - M[1][0] * b[0]) / det];
    }
    function renderTable(results) {
      const cosG = G.length >= 2 ? dot(G[0], G[1]) / (norm(G[0]) * norm(G[1]) + EPS) : 1;
      const head = `<th>Method</th><th>|d|</th>` + G.map((_, i) => `<th>effect on ${names[i] || 'g' + (i + 1)} (g·d)</th>`).join('') + (G.length === 2 ? '<th>weights w</th>' : '');
      const rows = results.map((r) => {
        const effects = G.map((g) => {
          const e = dot(g, r.d);
          const zero = Math.abs(e) < 1e-9 * (norm(g) * norm(r.d) + 1); // rounding, not a real effect
          return `<td>${zero ? '0' : W.fmt(e, 3)} ${zero ? '· on the edge' : e < 0 ? '<span style="color:var(--bad)">✗ hurts</span>' : '<span style="color:var(--good)">✓</span>'}</td>`;
        }).join('');
        const w = weightsFor(r.d);
        return `<tr><td><span class="swatch" style="background:${r.color}"></span> ${r.label}</td><td>${W.fmt(norm(r.d), 3)}</td>${effects}${G.length === 2 ? `<td>${w ? w.map((x) => W.fmt(x, 2)).join(' / ') : '–'}</td>` : ''}</tr>`;
      }).join('');
      table.innerHTML = `<div>Gradient norms: ${G.map((g, i) => `${names[i] || 'g' + (i + 1)} <b>${W.fmt(norm(g), 2)}</b>`).join(', ')}${G.length >= 2 ? ` · cos(${names[0]}, ${names[1]}) = <b>${W.fmt(cosG, 2)}</b>` : ''}</div>` +
        `<div class="table-wrap" style="margin-top:6px"><table><thead><tr>${head}</tr></thead><tbody>${rows}</tbody></table></div>` +
        '<div class="w-sub" style="margin-top:6px">g·d &gt; 0: the update improves that objective (to first order). Weights: d = w₁g₁ + w₂g₂ (sum has 1 / 1). With Adam only the direction of d matters much, so compare directions more than lengths.</div>';
    }
    draw();
  });
})();
