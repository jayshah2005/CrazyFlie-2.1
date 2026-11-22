# Agent notes

## Git

Commit, stage, branch, and push only when the user asks. Follow the project git skill:

- Canonical: [`.agents/skills/git/SKILL.md`](.agents/skills/git/SKILL.md)
- Cursor: `.cursor/skills/git` → same skill; always-on rule in `.cursor/rules/git.mdc`
- Claude Code: `.claude/skills/git` → same skill

Message format: `type[/scope[/subscope]][+scope]: summary`  
Types: `feat` · `fix` · `refactor` · `chore`  
Scopes: see the skill allowlist — do not invent new ones without asking.

## Logging (`run_log`)

All scripts must use [`run_log.py`](run_log.py) for console and CSV logs — do not reinvent logging.

- Canonical: [`.agents/skills/run-log/SKILL.md`](.agents/skills/run-log/SKILL.md)
- Cursor: `.cursor/skills/run-log` → same skill; always-on rule in `.cursor/rules/run-log.mdc`
- Claude Code: `.claude/skills/run-log` → same skill

Pattern: `log = run_log.setup(__file__)`; CSVs via `run_log.csv_path` + `run_log.write_csv`.

## Explanations: give the math

Whenever something can be written as a formula (a reward, a cost, dynamics, an update rule, a
metric, a schedule), give the formula and explain it:

- Write the formula itself, not only a description of it.
- Say what every symbol means and its units.
- Say in words what the formula does and why it has that shape.
- Use the real values from the code (weights, limits, rates) and name the file they come from.
- Where it helps, work one example with numbers.
