# Project rules for Claude Code

These apply to every session working in this repo, not just the one that wrote them.

- **Always use the `ponytail` skill.** Keep implementations minimal — no speculative abstractions, no unrequested flexibility, shortest working diff.
- **Always use the `superpowers` skills** for their intended workflows (brainstorming before new features, subagent-driven-development or executing-plans for multi-step work, systematic-debugging for bugs, requesting-code-review before merging, etc.).
- **All new features and functions follow TDD** (`superpowers:test-driven-development`): write the failing test first, confirm it fails for the stated reason, then implement.

These rules stand until the user says otherwise.
