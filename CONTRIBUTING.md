# Contributing

This repository backs a challenge that may become a paper. History is part of
the deliverable, so it is kept readable and attributable.

## Authorship

**Commits and pull requests carry no AI attribution.** No `Co-Authored-By`
trailer naming an assistant, no "generated with" line, no tool footer. Commits
are authored by the people accountable for the work and nothing else.

This applies regardless of how a change was produced. If a tool adds such a
line automatically, amend the commit before pushing.

## Commits

Conventional Commits, one logical change each:

```
<type>(<scope>): <imperative summary under 72 chars>

<body: what changed and why, wrapped at 72>
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `perf`, `build`, `ci`, `chore`.
Scopes follow the package layout: `encoding`, `data`, `model`, `train`, `eval`,
`compress`, `docs`.

Write the body for someone reading the log in six months with no memory of the
discussion. Say why, not just what — the diff already says what.

## Branches

`<type>/<short-slug>`, for example `feat/relative-position-bias`. Branch off
`main`, open a PR, delete the branch after merge. No long-lived branches: the
previous iteration of this project accumulated eight and they all drifted.

## What never enters git

Datasets, PGN dumps, encoded arrays, and model checkpoints. They live in Drive
(see `docs/data.md`), and every one of them has a documented command that
regenerates it. A generated artifact that cannot be regenerated from a command
in this repo is a bug, not a file.

## Before opening a PR

```bash
ruff check . && ruff format --check . && pytest
```

CI runs the same three.
