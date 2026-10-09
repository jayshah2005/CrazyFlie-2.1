# Agent notes

## Git

Commit, stage, branch, and push only when the user asks. Follow the project git skill:

- Canonical: [`.agents/skills/git/SKILL.md`](.agents/skills/git/SKILL.md)
- Cursor: `.cursor/skills/git` → same skill; always-on rule in `.cursor/rules/git.mdc`
- Claude Code: `.claude/skills/git` → same skill

Message format: `type[/scope[/subscope]][+scope]: summary`  
Types: `feat` · `fix` · `refactor` · `chore`  
Scopes: see the skill allowlist — do not invent new ones without asking.
