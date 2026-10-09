---
name: git
description: >-
  Commit/branch message convention — always follow when committing, staging,
  or creating branches. Use when the user asks to commit, stage, push, create a
  branch, write a commit message, or name a branch.
---

# Git commit & branch

Only commit / push / create branches when the user explicitly asks. Never commit proactively.

## Message format

```
type[/scope[/subscope]][+scope]: summary
```

### Types (only these)

`feat` · `fix` · `refactor` · `chore`

- No `style` or `docs` type — visual polish is `refactor`; documentation is `chore/docs`
- **Scope** — from the allowlist below. Omit only for genuinely extension-wide changes.
- **Never invent a new scope** — if none fit, ask the user before adding one.
- Nested subscope with `/` (e.g. `engine/workday`); combined scopes with `+` when one commit truly spans two (e.g. `feat/sidepanel+auth: …`)
- **Summary** — lowercase first word, imperative, no trailing period, ≤ ~50 chars (hard ~60)
- Single-line subject by default; body only when the why isn’t obvious (blank line before body; `git commit -F` preferred)

## Scope allowlist

| Scope | Use for |
| --- | --- |
| `preview` | Right column |
| `edit` | full screen edit mode |
| `header` | header in the full screen mode |
| `docs` | README, store listing copy, publisher/setup docs (`chore/docs`) |
| `backend` | Other backend related changes |
| `engine` | Image Extraction engine |
| `tests` | Jest / test files |
| `agents` | `.cursor/agents`, `.cursor/skills`, `.cursor/rules` |
| `gitignore` | `.gitignore` |

Need a new scope? Ask first, then add it here.

## Examples

```
chore/gitignore: ignore python caches and flight logs
chore/agents: add git commit convention skill
feat/engine: add flip apex detection from imu
fix/backend: recover when radio link drops mid-flip
refactor/edit: simplify setpoint helper naming
chore/docs: document lighthouse geometry setup
feat/preview+edit: sync flip params across views
fix/engine/workday: clamp thrust after brake overshoot
```

## Branches

`type/scope[-short-slug]` — same type/scope as the primary commit. Prefer over ad-hoc `cursor/…` unless the user requires that prefix.

Examples:

```
chore/gitignore
chore/agents-git-skill
feat/engine-flip-apex
fix/backend-radio-drop
```

## Atomic commits

One coherent change per commit. Split unrelated concerns; stage precise paths (`git add <paths>`), not blind `git add .`.

## Safety

- No history rewrite / force-push / `--no-verify` / discarding unfamiliar files unless explicitly asked.
- Never commit secrets (`.env*`, client secrets).
- Don’t push unless asked.
- Don’t update git config.

## Commit workflow

When the user explicitly asks to commit:

1. Run in parallel: `git status`, `git diff` (staged + unstaged), `git log` (recent subjects for style).
2. Draft a subject matching `type[/scope[/subscope]][+scope]: summary`.
3. Stage precise paths only.
4. Commit with a HEREDOC (`git commit -m "$(cat <<'EOF' … EOF)"` or `-F`).
5. Verify with `git status`.
