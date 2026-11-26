<!-- group: Reference | reviewed: 2026-11-26 -->
# Conventions

How these pages are written, checked and extended, for people and for agents. Keep to these rules
and the pages stay consistent, searchable and readable both on GitHub and in the viewer.

## Principles

1. **These pages explain; the records prove.** Detailed tables and logs stay in `l2f/docs/` (the
   lab notebook). A page here gives the headline numbers, explains them, and links the record.
   Every number must come from a record, the code or a measurement, never from memory.
2. **Explain why**, not only what: why something works, why it failed, what it implies.
3. **Give the math** ([AGENTS.md](../AGENTS.md)): the formula, every symbol with its units, what it
   does in words, the real values with the file they come from, and a worked example where it
   helps. Link a review of any math concept the first time a page uses it.
4. **Date what can change.** Status sections say "as of" a date; each page carries the date it was
   last checked against the code and the records.
5. **One topic per page, one place per fact.** Link instead of repeating; when a fact must appear
   twice (a headline number), keep it identical.

## Files

| File | Role |
| --- | --- |
| `README.md` | the start page: how to open, reading paths, page list, status |
| `NN-name.md` | the pages, in reading order (`01-overview.md` … `15-glossary.md`) |
| `CONVENTIONS.md` | this page |
| `index.html` | the viewer |
| `assets/site.css`, `assets/site.js` | viewer style and code |
| `assets/widgets-core.js`, `assets/widgets/*.js` | widget framework and one file per widget |
| `assets/bundle.js` | **generated** by `tools/build_docs.py`; never edit by hand |
| `tools/build_docs.py` | rebuilds the bundle and checks the pages |
| `tools/check_widgets.py` | checks the widgets' math against TorchJD, scipy and the records |

The viewer orders pages as: `README.md`, then `NN-*.md` by number, then the other `.md` files. A
new page needs a free number; renumbering breaks links, so prefer adding at the end of a group.

## Page template

```markdown
<!-- group: The system | reviewed: 2026-11-26 -->
# Page title

One paragraph: what this page covers and who needs it.

## First section
...

## Where this is in the code
| What | File |
```

- The first line is the metadata comment: `group` (the sidebar heading) and `reviewed`
  (YYYY-MM-DD, the last time the page was checked against the code and the records). Groups in
  use: Start, Foundations, The system, Research, Flying, Results, Reference.
- Exactly one `# Title`. Sections are `##`, subsections `###`; the "on this page" list shows both.
- No math in headings (it breaks anchors).

## Writing style

- Plain, direct English; short sentences; active voice; one idea per sentence where possible.
- Define a term on first use or link the [Glossary](15-glossary.md); add new terms there.
- Use the project's names exactly: variant E, old path, sensor path, split critic, the time 0.5
  recipe, old3.
- Status words: **adopted**, **failed**, **superseded**, **open**, **untested**, **partly fixed**,
  **watch**. Say what superseded a result and link it.
- Numbers with units (`4.9 cm`, `0.06 s`, `15,469 rpm`), percentages with `%`, seeds as "8 seeds".

## Math

Formulas use the GitHub syntax, which the viewer renders with KaTeX:

- Inline: `$Q(s, a)$`, with no space right after the opening `$` or before the closing one.
- Display: `$$` on its own line, with blank lines before and after the block.
- Multi-line: `\begin{aligned} … \end{aligned}` inside `$$`; not `align`.
- Inside a table row, write `\lvert x \rvert` and `\lVert x \rVert`, never `|` (GitHub splits
  the cell there; the build tool flags it).
- A literal dollar sign is `\$`.

## Diagrams

Use Mermaid code blocks (```` ```mermaid ````); GitHub renders them too. Put labels with spaces or
symbols in quotes, use `<br/>` for line breaks, keep diagrams small (they must fit a laptop screen
and a phone). Flowcharts for structure, sequence diagrams for loops, state diagrams for procedures.

## Callouts

GitHub alert syntax, styled by the viewer:

```markdown
> [!NOTE]      background worth knowing
> [!TIP]       a useful shortcut
> [!IMPORTANT] something that changes results if ignored
> [!WARNING]   a common mistake
> [!CAUTION]   risk to the drone or to data
```

## Links

- Between pages: `[text](06-reward.md#the-crash-incentive)`. Anchors are GitHub's heading slugs:
  lower case, punctuation removed, spaces to `-`, repeats get `-1`, `-2`.
- To code and records: relative paths from this folder, for example
  `[train.py](../l2f/py/train.py)` or `[record](../l2f/docs/moo_findings.md#13-finding-9-...)`.
  Spaces in paths are `%20`.
- External: full `https://` links; prefer stable sources (arXiv abstract pages, Wikipedia,
  official documentation).
- `tools/build_docs.py` checks every relative link and anchor, including anchors in `l2f/docs/`.

## Tables

The viewer makes every table sortable and adds a filter box from 8 rows. Put the number first in
a cell (`4.9 cm`, not `about 4.9 cm`) so numeric sorting works; mark the best value in **bold**
only when the comparison is the point of the table.

## Widgets

A widget is an interactive figure, written once in JavaScript and embedded in any page:

```markdown
<div data-widget="crash-incentive" data-jump="1.0">

> **Interactive:** open this page in the viewer to use the calculator. (fallback text for GitHub)

</div>
```

The blank lines matter: they let the fallback text render as Markdown. `data-*` attributes reach
the widget as options. Available widgets:

| Name | File | Shows |
| --- | --- | --- |
| `aggregator-playground` | `aggregators.js` | draggable objective gradients and every aggregator's update |
| `crash-incentive` | `crash_incentive.js` | the return of a perfect waypoint leg against the crash penalty |
| `discount-horizon` | `discount.js` | discount weights and the effective horizon |
| `curriculum` | `curriculum.js` | reward weights, noise and mission progress over training |
| `motor-lag` | `motor_lag.js` | first-order motor response and dead time |
| `ar1-noise` | `ar1_noise.js` | white against correlated noise |
| `ground-effect` | `ground_effect.js` | the ground-effect thrust factor |
| `overestimation` | `overestimation.js` | maximization bias and the clipped double-Q target |
| `fast-tanh` | `fast_tanh.js` | FAST_TANH and its training gradient |
| `seed-compare` | `seeds.js` | per-seed results with a bootstrap interval and an exact Mann-Whitney test (data in `data-arms`) |

To add one:

1. Create `assets/widgets/<name>.js`; start it with a comment that states the math it computes and
   where the numbers come from.
2. Register it: `DocWidgets.register('<name>', (root, opts) => { ... })`. Use the helpers of
   `widgets-core.js` (`header`, `controls`, `slider`, `select`, `checkbox`, `linePlot`, `rng`,
   `symEig`, `fmt`).
3. Color only through CSS variables (`var(--series-1)` … `var(--series-8)` in fixed order,
   `var(--ink)`, `var(--ink-2)`, `var(--grid)`, `var(--good)`, `var(--bad)`), so light and dark
   mode work without redrawing. Text uses the ink colors, never a series color.
4. Show the numbers in text next to the plot (a readout), so nothing is only in a tooltip.
5. Add a `<script defer src="assets/widgets/<name>.js">` line to `index.html`.
6. If the math is not trivial, expose a pure function (as `DocWidgets.aggregate`,
   `DocWidgets.legCost`) and add a reference check to `tools/check_widgets.py`.

## Adding an experiment

1. Write the full record in `l2f/docs/` (as the existing studies are).
2. Add a row to the table in [12-experiments.md](12-experiments.md) with the next ID, and a short
   entry under its period:

```markdown
**E35 Short name.** The question in one sentence. What was run (arms, seeds, the test). The
headline result with numbers. What it changed or why it matters.
→ [record](../l2f/docs/<record>.md#<section>)
```

3. If it changes a conclusion, update the topic page (and mark the old statement superseded with a
   link), [Issues](13-issues.md) and the status table in [Overview](01-overview.md) and
   [README](README.md).
4. Update the `reviewed` date of every page you checked.

## Build and check

```bash
python3 documentation/tools/build_docs.py          # rebuild the bundle, run every check
python3 documentation/tools/build_docs.py --check  # check only; fails on a stale bundle
python3 documentation/tools/build_docs.py --check-external   # also request every web link (slow)
python3 documentation/tools/check_widgets.py       # widget math (needs node; torchjd and scipy if present)
```

The build checks the metadata line and the single title of every page, every relative link and
anchor, every widget name against the registered widgets and the scripts `index.html` loads, and
the math delimiters. Run it before every commit that touches this folder; commit `assets/bundle.js`
with the pages so the viewer works from disk.

## For agents

- Read the page you change and the record it cites before editing; keep numbers identical to the
  record.
- Never edit `assets/bundle.js`; run the build tool.
- Do not rename headings that other pages link to without fixing those links (the build tool will
  list them).
- Follow the repository's rules: `run_log.py` for scripts, the git skill for commits, commit only
  when asked.
