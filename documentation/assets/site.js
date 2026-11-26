/* Documentation viewer (documentation/index.html).
 *
 * Renders the Markdown pages of documentation/ in the browser: Markdown with marked, math with
 * KaTeX ($...$ and $$...$$, the GitHub syntax), diagrams with Mermaid (```mermaid blocks), and the
 * interactive widgets of assets/widgets/ (<div data-widget="name">). Adds the page list, search,
 * an "on this page" list, sortable and filterable tables, copy buttons and a light/dark switch.
 *
 * Page text comes from assets/bundle.js (written by tools/build_docs.py), so the viewer works
 * when index.html is opened straight from disk (file://). Over http (python3 -m http.server) it
 * fetches each .md file fresh, so edits show without a rebuild.
 *
 * URLs: index.html#/<page id>/<heading id>, where the page id is the file name without .md and
 * the heading id is GitHub's slug of the heading, so links work the same here and on GitHub.
 */
(function () {
  'use strict';

  const BUNDLE = window.DOC_BUNDLE || { pages: [], files: {} };
  const PAGES = BUNDLE.pages;
  const HOME = 'README';
  const state = { id: null, md: {}, index: null, renderToken: 0 };

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const pageById = (id) => PAGES.find((p) => p.id === id);

  // ---------------------------------------------------------------- slugs (GitHub's rule)
  // Lower case, drop punctuation (keep letters, digits, marks, "_", "-", spaces), each space -> "-";
  // repeats get -1, -2, ... in document order.
  function slugBase(text) {
    return text.trim().toLowerCase().replace(/[^\p{L}\p{M}\p{N}\p{Pc}\- ]/gu, '').replace(/ /g, '-');
  }
  function makeSlugger() {
    const seen = {};
    return (text) => {
      const base = slugBase(text);
      if (seen[base] === undefined) { seen[base] = 0; return base; }
      seen[base] += 1;
      return `${base}-${seen[base]}`;
    };
  }

  // ---------------------------------------------------------------- math protection
  // marked would read "_" and "*" inside formulas as emphasis, so formulas are swapped for tokens
  // before parsing and rendered with KaTeX afterwards. Code (fenced blocks and `spans`) is skipped.
  function protectMath(src) {
    const store = [];
    const out = [];
    const lines = src.split('\n');
    let fence = null;
    let buf = [];
    const flush = () => { if (buf.length) { out.push(scanText(buf.join('\n'), store)); buf = []; } };
    for (const line of lines) {
      const m = line.match(/^\s{0,3}(`{3,}|~{3,})/);
      if (fence) {
        out.push(line);
        if (m && m[1][0] === fence[0] && m[1].length >= fence.length && line.trim() === m[1]) fence = null;
        continue;
      }
      if (m) { flush(); fence = m[1]; out.push(line); continue; }
      buf.push(line);
    }
    flush();
    return { text: out.join('\n'), store };
  }
  function scanText(text, store) {
    let res = '';
    let i = 0;
    const n = text.length;
    while (i < n) {
      const c = text[i];
      if (c === '\\' && text[i + 1] === '$') { res += '\\$'; i += 2; continue; }
      if (c === '`') {
        let j = i; while (text[j] === '`') j++;
        const ticks = text.slice(i, j);
        const end = text.indexOf(ticks, j);
        if (end < 0) { res += ticks; i = j; continue; }
        res += text.slice(i, end + ticks.length); i = end + ticks.length; continue;
      }
      if (c === '$' && text[i + 1] === '$') {
        const end = text.indexOf('$$', i + 2);
        if (end < 0) { res += '$$'; i += 2; continue; }
        store.push({ tex: text.slice(i + 2, end), display: true });
        res += `\n\nMATHTOK${store.length - 1}X\n\n`; i = end + 2; continue;
      }
      if (c === '$') {
        // inline: no space right after the opening $ or before the closing $; stays in its paragraph
        if (text[i + 1] && !/\s/.test(text[i + 1])) {
          let j = i + 1;
          let found = -1;
          while (j < n) {
            if (text[j] === '\\') { j += 2; continue; }
            if (text[j] === '\n' && text[j + 1] === '\n') break;
            if (text[j] === '$') { if (!/\s/.test(text[j - 1])) found = j; break; }
            j++;
          }
          if (found > 0) {
            store.push({ tex: text.slice(i + 1, found), display: false });
            res += `MATHTOK${store.length - 1}X`; i = found + 1; continue;
          }
        }
      }
      res += c; i++;
    }
    return res;
  }
  function renderTex(tex, display) {
    if (window.katex) {
      try {
        return window.katex.renderToString(tex, { displayMode: display, throwOnError: false, strict: 'ignore', output: 'html' });
      } catch (e) { /* fall through to the plain text */ }
    }
    return display ? `<code class="math-fallback math-block-fallback">${escapeHtml(tex)}</code>` : `<code class="math-fallback">${escapeHtml(tex)}</code>`;
  }
  function restoreMath(html, store) {
    html = html.replace(/<p>\s*MATHTOK(\d+)X\s*<\/p>/g, (_, k) => renderTex(store[+k].tex, true));
    return html.replace(/MATHTOK(\d+)X/g, (_, k) => renderTex(store[+k].tex, store[+k].display));
  }

  // ---------------------------------------------------------------- loading pages
  async function getMarkdown(page) {
    if (state.md[page.id] !== undefined) return state.md[page.id];
    let text = BUNDLE.files[page.file];
    if (location.protocol.startsWith('http')) {
      try {
        const r = await fetch(page.file, { cache: 'no-store' });
        if (r.ok) text = await r.text();
      } catch (e) { /* keep the bundled copy */ }
    }
    state.md[page.id] = text === undefined ? null : text;
    return state.md[page.id];
  }

  // ---------------------------------------------------------------- navigation
  function buildSidebar() {
    const nav = $('#sidebar');
    let html = '';
    let group = null;
    for (const p of PAGES) {
      if (p.group !== group) { group = p.group; html += `<div class="group">${escapeHtml(group || '')}</div>`; }
      const num = p.id.match(/^(\d+)-/);
      html += `<a href="#/${p.id}" data-id="${p.id}">${num ? `<span class="num">${num[1]}</span>` : ''}${escapeHtml(p.title)}</a>`;
    }
    nav.innerHTML = html;
  }
  function markSidebar(id) {
    $$('#sidebar a').forEach((a) => a.classList.toggle('current', a.dataset.id === id));
  }
  function parseHash() {
    const h = decodeURIComponent(location.hash.replace(/^#\/?/, ''));
    if (!h) return { id: HOME, anchor: '' };
    const k = h.indexOf('/');
    return k < 0 ? { id: h, anchor: '' } : { id: h.slice(0, k), anchor: h.slice(k + 1) };
  }
  function linkFor(href) {
    // .md link inside documentation/ -> viewer route; everything else is left alone
    if (/^[a-z]+:/i.test(href) || href.startsWith('//')) return null;
    if (href.startsWith('#')) return `#/${state.id}/${href.slice(1)}`;
    const m = href.match(/^(?:\.\/)?([^/#?]+)\.md(?:#(.*))?$/);
    if (m && pageById(m[1])) return `#/${m[1]}${m[2] ? '/' + m[2] : ''}`;
    return null;
  }

  // ---------------------------------------------------------------- rendering
  async function route() {
    const { id, anchor } = parseHash();
    const page = pageById(id) || pageById(HOME) || PAGES[0];
    if (!page) { $('#content').innerHTML = '<p class="notice">No pages found: run <code>python3 documentation/tools/build_docs.py</code>.</p>'; return; }
    if (state.id !== page.id) await render(page);
    scrollToAnchor(anchor);
    $('#sidebar').classList.remove('open');
  }
  function scrollToAnchor(anchor) {
    if (!anchor) { window.scrollTo(0, 0); return; }
    const el = document.getElementById(anchor);
    if (el) el.scrollIntoView({ block: 'start' }); else window.scrollTo(0, 0);
  }

  async function render(page) {
    const token = ++state.renderToken;
    state.id = page.id;
    markSidebar(page.id);
    document.title = `${page.title} · Crazyflie learned controller docs`;
    const md = await getMarkdown(page);
    if (token !== state.renderToken) return;
    const content = $('#content');
    if (md == null) { content.innerHTML = `<p class="notice">Page ${escapeHtml(page.file)} is missing. Rebuild the bundle.</p>`; return; }
    let html;
    const body = md.replace(/^\s*<!--[\s\S]*?-->\s*/, '');
    if (window.marked) {
      const { text, store } = protectMath(body);
      html = restoreMath(window.marked.parse(text, { gfm: true }), store);
    } else {
      html = '<p class="notice">The Markdown renderer (marked, loaded from cdn.jsdelivr.net) is not available, so the page is shown as plain text. Connect to the internet, or read the .md files directly (they also render on GitHub).</p>' +
        `<pre class="raw">${escapeHtml(body)}</pre>`;
    }
    content.innerHTML = html;
    decorate(content, page);
    buildToc(content);
    buildPager(content, page);
    await renderDiagrams(content);
    if (window.DocWidgets) window.DocWidgets.mountAll(content);
  }

  function decorate(root, page) {
    // heading ids (GitHub slugs) and hover anchors
    const slug = makeSlugger();
    $$('h1, h2, h3, h4, h5, h6', root).forEach((h) => {
      const id = slug(h.textContent);
      h.id = id;
      if (h.tagName !== 'H1') {
        const a = document.createElement('a');
        a.className = 'anchor'; a.href = `#/${page.id}/${id}`; a.textContent = '#'; a.setAttribute('aria-label', 'Link to this section');
        h.appendChild(a);
      }
    });
    // links
    $$('a[href]', root).forEach((a) => {
      if (a.classList.contains('anchor')) return;
      const href = a.getAttribute('href');
      const routed = linkFor(href);
      if (routed) { a.setAttribute('href', routed); return; }
      if (/^https?:/i.test(href)) { a.target = '_blank'; a.rel = 'noopener'; a.classList.add('ext'); }
      else if (!href.startsWith('#')) { a.target = '_blank'; a.rel = 'noopener'; }
    });
    // GitHub alerts: > [!NOTE] / [!TIP] / [!IMPORTANT] / [!WARNING] / [!CAUTION]
    $$('blockquote', root).forEach((q) => {
      const p = q.querySelector('p');
      if (!p) return;
      const m = p.innerHTML.match(/^\s*\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]\s*(<br\s*\/?>)?/i);
      if (!m) return;
      const kind = m[1].toLowerCase();
      p.innerHTML = p.innerHTML.slice(m[0].length);
      q.classList.add('callout', `callout-${kind}`);
      const t = document.createElement('div');
      t.className = 'callout-title';
      t.textContent = { note: 'Note', tip: 'Tip', important: 'Important', warning: 'Warning', caution: 'Caution' }[kind];
      q.prepend(t);
      if (!p.textContent.trim() && !p.querySelector('*')) p.remove();
    });
    // code: math blocks (```math), copy buttons
    $$('pre > code', root).forEach((code) => {
      const pre = code.parentElement;
      if (code.classList.contains('language-math')) {
        const div = document.createElement('div');
        div.innerHTML = renderTex(code.textContent, true);
        pre.replaceWith(div);
        return;
      }
      if (code.classList.contains('language-mermaid')) return;
      const b = document.createElement('button');
      b.className = 'copy-btn'; b.type = 'button'; b.textContent = 'Copy';
      b.addEventListener('click', () => {
        const done = () => { b.textContent = 'Copied'; setTimeout(() => { b.textContent = 'Copy'; }, 1200); };
        if (navigator.clipboard) navigator.clipboard.writeText(code.textContent).then(done, () => {});
      });
      pre.appendChild(b);
    });
    // tables: scroll wrapper, sorting, a filter box on long tables
    $$('table', root).forEach((table) => enhanceTable(table));
    // page footer
    const meta = document.createElement('div');
    meta.className = 'page-meta';
    meta.innerHTML = `<span>Source: <code>documentation/${escapeHtml(page.file)}</code></span>` +
      (page.reviewed ? `<span>Last reviewed: ${escapeHtml(page.reviewed)}</span>` : '') +
      (BUNDLE.generated ? `<span>Bundle built: ${escapeHtml(BUNDLE.generated)}</span>` : '');
    root.appendChild(meta);
  }

  function cellValue(td) {
    // the first number in the cell ("4.9 cm", "−0.41", "~1.0%"), or null for text
    const t = td.textContent.trim().replace(/−/g, '-').replace(/,/g, '');
    const m = t.match(/^[~≈<>+]?\s*(-?\d+(?:\.\d+)?)/);
    return m ? parseFloat(m[1]) : null;
  }
  function enhanceTable(table) {
    const wrap = document.createElement('div');
    wrap.className = 'table-wrap';
    table.parentNode.insertBefore(wrap, table);
    wrap.appendChild(table);
    const body = table.tBodies[0];
    if (!body) return;
    const rows = Array.from(body.rows);
    $$('thead th', table).forEach((th, col) => {
      th.classList.add('sortable');
      th.title = 'Sort by this column';
      let dir = 0;
      th.addEventListener('click', () => {
        dir = dir === 1 ? -1 : 1;
        $$('thead th', table).forEach((o) => o.classList.remove('sorted-asc', 'sorted-desc'));
        th.classList.add(dir === 1 ? 'sorted-asc' : 'sorted-desc');
        const sorted = rows.slice().sort((a, b) => {
          const ca = a.cells[col], cb = b.cells[col];
          if (!ca || !cb) return 0;
          const va = cellValue(ca), vb = cellValue(cb);
          if (va !== null && vb !== null) return (va - vb) * dir;
          if (va !== null) return -1;
          if (vb !== null) return 1;
          return ca.textContent.localeCompare(cb.textContent) * dir;
        });
        sorted.forEach((r) => body.appendChild(r));
      });
    });
    if (rows.length >= 8) {
      const tools = document.createElement('div');
      tools.className = 'table-tools';
      tools.innerHTML = '<input type="search" placeholder="Filter rows…" aria-label="Filter table rows"><span class="count"></span>';
      wrap.parentNode.insertBefore(tools, wrap);
      const input = tools.querySelector('input');
      const count = tools.querySelector('.count');
      const update = () => {
        const terms = input.value.toLowerCase().split(/\s+/).filter(Boolean);
        let shown = 0;
        rows.forEach((r) => {
          const t = r.textContent.toLowerCase();
          const ok = terms.every((w) => t.includes(w));
          r.style.display = ok ? '' : 'none';
          if (ok) shown++;
        });
        count.textContent = terms.length ? `${shown} of ${rows.length} rows` : `${rows.length} rows`;
      };
      input.addEventListener('input', update);
      update();
    }
  }

  function buildToc(root) {
    const toc = $('#toc');
    const hs = $$('h2, h3', root);
    if (hs.length < 2) { toc.innerHTML = ''; return; }
    toc.innerHTML = '<div class="title">On this page</div>' + hs.map((h) => {
      const label = escapeHtml(h.textContent.replace(/#$/, ''));
      return `<a class="${h.tagName.toLowerCase()}" href="#/${state.id}/${h.id}" data-target="${h.id}">${label}</a>`;
    }).join('');
    if (state.observer) state.observer.disconnect();
    const links = $$('a', toc);
    state.observer = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) links.forEach((a) => a.classList.toggle('active', a.dataset.target === e.target.id));
      });
    }, { rootMargin: '-60px 0px -70% 0px' });
    hs.forEach((h) => state.observer.observe(h));
  }

  function buildPager(root, page) {
    const i = PAGES.indexOf(page);
    const prev = PAGES[i - 1], next = PAGES[i + 1];
    if (!prev && !next) return;
    const div = document.createElement('nav');
    div.className = 'pager';
    div.innerHTML = (prev ? `<a href="#/${prev.id}"><span class="lbl">← Previous</span>${escapeHtml(prev.title)}</a>` : '<span></span>') +
      (next ? `<a class="next" href="#/${next.id}"><span class="lbl">Next →</span>${escapeHtml(next.title)}</a>` : '');
    root.appendChild(div);
  }

  // ---------------------------------------------------------------- diagrams
  function isDark() {
    const t = document.documentElement.dataset.theme;
    if (t) return t === 'dark';
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  }
  async function renderDiagrams(root) {
    const codes = $$('pre > code.language-mermaid', root);
    if (!codes.length) return;
    const nodes = codes.map((code) => {
      const div = document.createElement('div');
      div.className = 'mermaid';
      div.textContent = code.textContent;
      div.dataset.src = code.textContent;
      code.parentElement.replaceWith(div);
      return div;
    });
    if (!window.mermaid) {
      nodes.forEach((d) => { d.innerHTML = `<p class="diagram-error">Diagram renderer (Mermaid) not loaded; source:</p><pre>${escapeHtml(d.dataset.src)}</pre>`; });
      return;
    }
    try {
      window.mermaid.initialize({ startOnLoad: false, securityLevel: 'strict', theme: isDark() ? 'dark' : 'default', fontFamily: getComputedStyle(document.body).fontFamily, flowchart: { htmlLabels: true, curve: 'basis' } });
      await window.mermaid.run({ nodes, suppressErrors: true });
    } catch (e) {
      console.error(e);
    }
    nodes.forEach((d) => {
      const svg = d.querySelector('svg');
      if (!svg) { d.innerHTML = `<p class="diagram-error">This diagram did not render; source:</p><pre>${escapeHtml(d.dataset.src)}</pre>`; return; }
      // a wide diagram (a timeline) shrunk to the page would have unreadable text: keep at least
      // 75% of its natural size and let the box scroll sideways instead
      const vb = svg.viewBox && svg.viewBox.baseVal;
      if (vb && vb.width && vb.width * 0.75 > d.clientWidth - 26) {
        svg.style.maxWidth = 'none';
        svg.setAttribute('width', String(Math.round(vb.width * 0.75)));
        svg.removeAttribute('height');
      }
    });
  }

  // ---------------------------------------------------------------- search
  // Heading text as the browser shows it (the same rules as tools/build_docs.py plain()), so a
  // search result's anchor matches the id decorate() gives the heading.
  function plainHeading(md) {
    return md.replace(/!?\[([^\]]*)\]\([^)]*\)/g, '$1').replace(/<[^>]+>/g, '').replace(/(`+)(.+?)\1/g, '$2')
      .replace(/(\*\*|__)(.+?)\1/g, '$2').replace(/(^|[^\w*])([*_])(.+?)\2(?![\w*])/g, '$1$3')
      .replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>').trim();
  }
  function stripMd(s) {
    return s.replace(/```[\s\S]*?```/g, ' ').replace(/<[^>]+>/g, ' ').replace(/!?\[([^\]]*)\]\([^)]*\)/g, '$1')
      .replace(/[`*_>#|]/g, ' ').replace(/\$\$?([^$]*)\$\$?/g, '$1').replace(/\s+/g, ' ').trim();
  }
  async function buildIndex() {
    const index = [];
    for (const p of PAGES) {
      const md = (await getMarkdown(p)) || '';
      const slug = makeSlugger();
      let heading = p.title, anchor = '', buf = [];
      let inFence = false;
      const push = () => { const text = stripMd(buf.join('\n')); if (text || anchor) index.push({ page: p, heading, anchor, text, lower: text.toLowerCase() }); buf = []; };
      for (const line of md.split('\n')) {
        if (/^\s{0,3}(```|~~~)/.test(line)) inFence = !inFence;
        const m = !inFence && line.match(/^(#{1,6})\s+(.*)$/);
        if (m) {
          push();
          const shown = plainHeading(m[2].replace(/\s*#+\s*$/, ''));
          const id = slug(shown);
          heading = shown; anchor = m[1].length === 1 ? '' : id;
        } else buf.push(line);
      }
      push();
    }
    state.index = index;
  }
  function searchFor(q) {
    const terms = q.toLowerCase().split(/\s+/).filter((t) => t.length > 1);
    if (!terms.length) return [];
    const hits = [];
    for (const s of state.index) {
      const hl = s.heading.toLowerCase();
      const tl = s.page.title.toLowerCase();
      let score = 0, ok = true;
      for (const t of terms) {
        const inText = s.lower.split(t).length - 1;
        const inHead = hl.includes(t) ? 1 : 0;
        const inTitle = tl.includes(t) ? 1 : 0;
        if (!inText && !inHead && !inTitle) { ok = false; break; }
        score += Math.min(inText, 6) + 8 * inHead + 3 * inTitle;
      }
      if (ok) hits.push({ s, score });
    }
    return hits.sort((a, b) => b.score - a.score).slice(0, 14).map((h) => h.s);
  }
  function snippet(s, terms) {
    const t = terms[0] || '';
    const k = s.lower.indexOf(t);
    const start = Math.max(0, k - 60);
    let text = (start > 0 ? '… ' : '') + s.text.slice(start, start + 170) + (start + 170 < s.text.length ? ' …' : '');
    text = escapeHtml(text);
    for (const w of terms) text = text.replace(new RegExp(`(${w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi'), '<mark>$1</mark>');
    return text;
  }
  function setupSearch() {
    const input = $('#search');
    const box = $('#search-results');
    let active = -1;
    const show = async () => {
      if (!state.index) await buildIndex();
      const q = input.value.trim();
      if (!q) { box.classList.remove('open'); return; }
      const terms = q.toLowerCase().split(/\s+/).filter((t) => t.length > 1);
      const res = searchFor(q);
      active = -1;
      box.innerHTML = res.length ? res.map((s) => `<a href="#/${s.page.id}${s.anchor ? '/' + s.anchor : ''}"><div class="where">${escapeHtml(s.page.title)}${s.anchor ? ' › ' + escapeHtml(s.heading) : ''}</div><div class="snip">${snippet(s, terms)}</div></a>`).join('')
        : '<div class="empty">No match. Try fewer or shorter words.</div>';
      box.classList.add('open');
    };
    input.addEventListener('input', show);
    input.addEventListener('focus', () => { if (input.value.trim()) show(); });
    input.addEventListener('keydown', (e) => {
      const links = $$('a', box);
      if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        e.preventDefault();
        if (!links.length) return;
        active = (active + (e.key === 'ArrowDown' ? 1 : -1) + links.length) % links.length;
        links.forEach((a, i) => a.classList.toggle('active', i === active));
        links[active].scrollIntoView({ block: 'nearest' });
      } else if (e.key === 'Enter') {
        const target = links[active >= 0 ? active : 0];
        if (target) { location.hash = target.getAttribute('href'); box.classList.remove('open'); input.blur(); }
      } else if (e.key === 'Escape') { box.classList.remove('open'); input.blur(); }
    });
    box.addEventListener('click', (e) => { if (e.target.closest('a')) box.classList.remove('open'); });
    document.addEventListener('click', (e) => { if (!e.target.closest('.search')) box.classList.remove('open'); });
    document.addEventListener('keydown', (e) => {
      if (e.key === '/' && document.activeElement !== input && !/input|textarea|select/i.test(document.activeElement.tagName)) { e.preventDefault(); input.focus(); }
    });
  }

  // ---------------------------------------------------------------- theme
  function setupTheme() {
    const btn = $('#theme-btn');
    let saved = null;
    try { saved = localStorage.getItem('doc-theme'); } catch (e) { /* private mode */ }
    if (saved === 'light' || saved === 'dark') document.documentElement.dataset.theme = saved;
    const label = () => { btn.innerHTML = isDark() ? '☀︎<span class="lbl"> Light</span>' : '☾<span class="lbl"> Dark</span>'; };
    label();
    btn.addEventListener('click', () => {
      const next = isDark() ? 'light' : 'dark';
      document.documentElement.dataset.theme = next;
      try { localStorage.setItem('doc-theme', next); } catch (e) { /* private mode */ }
      label();
      rerender();
    });
    if (window.matchMedia) window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { label(); rerender(); });
  }
  function rerender() {
    // diagrams take their colors at render time; widgets use CSS variables and follow by themselves
    const page = pageById(state.id);
    if (page && $$('.mermaid').length) { const y = window.scrollY; state.id = null; render(page).then(() => window.scrollTo(0, y)); }
  }

  // ---------------------------------------------------------------- start
  function start() {
    buildSidebar();
    setupSearch();
    setupTheme();
    $('#menu-btn').addEventListener('click', () => $('#sidebar').classList.toggle('open'));
    window.addEventListener('hashchange', route);
    route();
  }
  // exposed for debugging and for checks of the viewer itself
  window.DocViewer = { slugBase, makeSlugger, protectMath, restoreMath, plainHeading, searchIndex: async () => { if (!state.index) await buildIndex(); return state.index; } };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start); else start();
})();
