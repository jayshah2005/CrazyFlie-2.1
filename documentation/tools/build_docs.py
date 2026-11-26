"""Build and check the project documentation (documentation/).

The pages are the Markdown files in documentation/ (README.md first, then NN-name.md in order,
then the other .md files). This tool:

  1. writes assets/bundle.js: every page's text and its title, group and review date, so that
     index.html works when opened straight from disk (file://). It is only rewritten when a page
     changed, so the git diff stays clean.
  2. checks the pages and reports every problem:
       - each page starts with a metadata comment  <!-- group: Name | reviewed: YYYY-MM-DD -->
         and has one "# Title" line;
       - every relative link resolves: other pages (with #anchors, using GitHub's heading slugs)
         and files elsewhere in the repository;
       - every <div data-widget="name"> names a registered widget, and index.html loads every
         widget file in assets/widgets/;
       - $$ display-math delimiters are balanced, and no inline formula inside a table uses "|"
         (GitHub would split the table cell there; write \\lvert / \\rvert instead).
  With --check-external it also requests every http(s) link once (slow; needs the internet).

Usage (from the repository root):
  python3 documentation/tools/build_docs.py            # rebuild the bundle and check
  python3 documentation/tools/build_docs.py --check    # check only; fails if the bundle is stale
  python3 documentation/tools/build_docs.py --check-external
"""
import argparse
import hashlib
import json
import re
import sys
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1]
ROOT = DOCS.parent
sys.path.insert(0, str(ROOT))  # run_log
import run_log  # noqa: E402

log = run_log.setup(__file__)

META = re.compile(r'^\s*<!--\s*(.*?)\s*-->', re.S)
FENCE = re.compile(r'^\s{0,3}(`{3,}|~{3,})')
HEADING = re.compile(r'^(#{1,6})\s+(.+?)\s*#*\s*$')
# link target: no spaces, parentheses allowed when balanced one level deep (Wikipedia_(topic))
LINK = re.compile(r'(?<!\!)\[([^\]]*)\]\(((?:[^()\s]|\([^()\s]*\))+)(?:\s+"[^"]*")?\)')
WIDGET_USE = re.compile(r'data-widget="([^"]+)"')
WIDGET_REG = re.compile(r"register\('([^']+)'")


def page_files():
    """README.md, then NN-*.md in order, then the other .md files."""
    files = sorted(DOCS.glob('*.md'))
    readme = [f for f in files if f.name == 'README.md']
    numbered = sorted(f for f in files if re.match(r'^\d{2}-', f.name))
    rest = sorted(f for f in files if f not in readme and f not in numbered)
    return readme + numbered + rest


def strip_code(text):
    """Text with fenced code blocks blanked (line count kept) and inline code spans removed."""
    out, fence = [], None
    for line in text.split('\n'):
        m = FENCE.match(line)
        if fence:
            out.append('')
            if m and m.group(1)[0] == fence[0] and len(m.group(1)) >= len(fence) and line.strip() == m.group(1):
                fence = None
            continue
        if m:
            fence = m.group(1)
            out.append('')
            continue
        out.append(re.sub(r'(`+)(.+?)\1', ' ', line))
    return '\n'.join(out)


def plain(md):
    """Heading text as a browser shows it: links, emphasis, code marks and tags removed."""
    s = re.sub(r'!?\[([^\]]*)\]\([^)]*\)', r'\1', md)
    s = re.sub(r'<[^>]+>', '', s)
    s = re.sub(r'(`+)(.+?)\1', r'\2', s)
    s = re.sub(r'(\*\*|__)(.+?)\1', r'\2', s)
    s = re.sub(r'(?<![\w*])([*_])(.+?)\1(?![\w*])', r'\2', s)
    s = s.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    return s.strip()


def slug_base(text):
    """GitHub's heading slug: lower case, keep letters, marks, digits, "_", "-" and spaces."""
    s = text.strip().lower()
    s = ''.join(c for c in s if c in '-_ ' or unicodedata.category(c)[0] in 'LMN')
    return s.replace(' ', '-')


def heading_slugs(text):
    """All heading slugs of a Markdown text, with GitHub's -1, -2 suffixes for repeats."""
    seen, slugs, fence = {}, set(), None
    for line in text.split('\n'):
        m = FENCE.match(line)
        if fence:
            if m and m.group(1)[0] == fence[0] and line.strip() == m.group(1):
                fence = None
            continue
        if m:
            fence = m.group(1)
            continue
        h = HEADING.match(line)
        if not h:
            continue
        base = slug_base(plain(h.group(2)))
        if base in seen:
            seen[base] += 1
            slugs.add(f'{base}-{seen[base]}')
        else:
            seen[base] = 0
            slugs.add(base)
    return slugs


def parse_page(path):
    """{id, file, title, group, reviewed} and a list of problems."""
    text = path.read_text(encoding='utf-8')
    problems = []
    meta = {}
    m = META.match(text)
    if m:
        for part in m.group(1).split('|'):
            key, _, val = part.partition(':')
            if val:
                meta[key.strip()] = val.strip()
    if 'group' not in meta or 'reviewed' not in meta:
        problems.append('missing the first-line comment <!-- group: ... | reviewed: YYYY-MM-DD -->')
    elif not re.fullmatch(r'\d{4}-\d{2}-\d{2}', meta['reviewed']):
        problems.append(f'reviewed date {meta["reviewed"]!r} is not YYYY-MM-DD')
    titles = [line[2:].strip() for line in strip_code(text).split('\n') if line.startswith('# ')]
    if len(titles) != 1:
        problems.append(f'needs exactly one "# Title" line (found {len(titles)})')
    page = {'id': path.stem, 'file': path.name, 'title': plain(titles[0]) if titles else path.stem,
            'group': meta.get('group', 'Other'), 'reviewed': meta.get('reviewed', '')}
    return page, text, problems


def check_links(path, text, slugs_of):
    """Problems with the relative links of one page."""
    problems = []
    body = strip_code(text)
    for n, line in enumerate(body.split('\n'), 1):
        for _, target in LINK.findall(line):
            if re.match(r'^[a-z][a-z0-9+.-]*:', target, re.I):
                continue
            file_part, _, anchor = target.partition('#')
            if not file_part:
                dest = path
            else:
                dest = (path.parent / urllib.parse.unquote(file_part)).resolve()
                if not dest.exists():
                    problems.append(f'line {n}: link to a missing file: {target}')
                    continue
            if anchor and dest.suffix == '.md':
                if anchor not in slugs_of(dest):
                    problems.append(f'line {n}: no heading "#{anchor}" in {dest.relative_to(ROOT)}')
    return problems


def check_math(text):
    """Problems with math delimiters (outside code)."""
    problems = []
    body = strip_code(text)
    if body.count('$$') % 2:
        problems.append('odd number of $$ (an unclosed display formula)')
    for n, line in enumerate(body.split('\n'), 1):
        if line.lstrip().startswith('|'):
            for formula in re.findall(r'(?<![\\$])\$([^$\n]+?)\$', line):
                if '|' in formula:
                    problems.append(f'line {n}: "|" inside a formula in a table row: use \\lvert / \\rvert')
    return problems


def check_widgets(texts):
    """Problems with widget names and the widget files index.html loads."""
    problems = []
    files = sorted((DOCS / 'assets' / 'widgets').glob('*.js'))
    registered = {}
    for f in files:
        for name in WIDGET_REG.findall(f.read_text(encoding='utf-8')):
            registered[name] = f.name
    index = (DOCS / 'index.html').read_text(encoding='utf-8')
    for f in files:
        if f'assets/widgets/{f.name}' not in index:
            problems.append(f'index.html does not load assets/widgets/{f.name}')
    for name, text in texts.items():
        for w in WIDGET_USE.findall(strip_code(text)):
            if w not in registered:
                problems.append(f'{name}: widget "{w}" is not registered by any file in assets/widgets/')
    return problems, registered


def check_external(texts):
    """Request every http(s) link once; problems for the ones that fail."""
    urls = sorted({t for text in texts.values() for _, t in LINK.findall(strip_code(text)) if t.startswith('http')})
    problems = []
    for url in urls:
        req = urllib.request.Request(url, method='GET', headers={'User-Agent': 'docs-link-check/1.0'})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                if r.status >= 400:
                    problems.append(f'{url}: HTTP {r.status}')
        except urllib.error.HTTPError as e:
            if e.code not in (403, 429):  # some sites refuse scripts; check those by hand
                problems.append(f'{url}: HTTP {e.code}')
            else:
                log.warning('%s: HTTP %d (site refuses scripts; check by hand)', url, e.code)
        except Exception as e:  # noqa: BLE001 - report every failure, keep going
            problems.append(f'{url}: {e.__class__.__name__}: {e}')
    log.info('external links checked: %d', len(urls))
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--check', action='store_true', help='do not write the bundle; fail if it is stale')
    ap.add_argument('--check-external', action='store_true', help='also request every http(s) link')
    args = ap.parse_args()

    pages, texts, problems = [], {}, []
    for f in page_files():
        page, text, probs = parse_page(f)
        pages.append(page)
        texts[f.name] = text
        problems += [f'{f.name}: {p}' for p in probs]

    cache = {}

    def slugs_of(path):
        if path not in cache:
            cache[path] = heading_slugs(path.read_text(encoding='utf-8')) if path.exists() else set()
        return cache[path]

    for f in page_files():
        problems += [f'{f.name}: {p}' for p in check_links(f, texts[f.name], slugs_of)]
        problems += [f'{f.name}: {p}' for p in check_math(texts[f.name])]
    wprobs, registered = check_widgets(texts)
    problems += wprobs
    if args.check_external:
        problems += check_external(texts)

    payload = {'pages': pages, 'files': texts}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
    bundle = DOCS / 'assets' / 'bundle.js'
    old = bundle.read_text(encoding='utf-8') if bundle.exists() else ''
    stale = f'"hash": "{digest}"' not in old
    if args.check:
        if stale:
            problems.append('assets/bundle.js is stale: run python3 documentation/tools/build_docs.py')
    elif stale:
        payload = {'generated': datetime.now().astimezone().strftime('%Y-%m-%d %H:%M'), 'hash': digest, **payload}
        bundle.write_text('// GENERATED by documentation/tools/build_docs.py from the .md files in documentation/.\n'
                          '// Do not edit; run the tool after changing a page.\n'
                          'window.DOC_BUNDLE = ' + json.dumps(payload, ensure_ascii=False, indent=1) + ';\n',
                          encoding='utf-8')
        log.info('wrote %s (%d pages)', bundle.relative_to(ROOT), len(pages))
    else:
        log.info('bundle up to date (%d pages)', len(pages))

    n_math = sum(strip_code(t).count('$$') // 2 for t in texts.values())
    n_mermaid = sum(len(re.findall(r'^```mermaid', t, re.M)) for t in texts.values())
    n_widgets = sum(len(WIDGET_USE.findall(t)) for t in texts.values())
    log.info('%d pages, %d display formulas, %d diagrams, %d widget uses (%d widgets registered)',
             len(pages), n_math, n_mermaid, n_widgets, len(registered))
    for p in problems:
        log.error('%s', p)
    if problems:
        log.error('%d problem(s)', len(problems))
        sys.exit(1)
    log.info('all checks passed')


if __name__ == '__main__':
    main()
