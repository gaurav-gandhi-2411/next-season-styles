# next-season-styles

H&M fashion trend forecasting + generation pipeline, built as an agentic workflow exposed via an
MCP server — technical assignment for a Lead/Principal Data Scientist role.

## Package manager

Uses **[uv](https://github.com/astral-sh/uv)** (present on this machine, `uv 0.11.14`). Lockfile
is `uv.lock`, committed for reproducibility per project convention.

## Setup

```
make setup
```

equivalent to `uv sync`, which creates `.venv/` and installs all pinned dependencies (runtime +
dev) from `uv.lock`.

## Shell conventions

Commands in this repo were developed and verified using **Git Bash (POSIX sh)** via Claude Code's
Bash tool, run from PowerShell on Windows 11. All `make` targets and `uv` commands are shell-
agnostic and work identically from PowerShell.

## Status

**Phase 1: data acquisition + verification — in progress.** Kaggle CLI credentials have been
verified present and well-formed; no dataset has been downloaded yet (that is the next step, not
part of this scaffolding commit).

## Project layout

- `src/nss/` — installable package (`data`, `features`, `models`, `viz` subpackages)
- `data/{raw,interim,processed}/` — gitignored data tiers (empty dirs tracked via `.gitkeep`)
- `notebooks/` — exploratory notebooks
- `reports/{figures,tables}/` — generated artifacts
- `tests/` — pytest suite

## Note

Built under a tight assignment deadline; scaffolding favors a disciplined, minimal footprint over
completeness — later phases (data pipeline, panel construction, modeling, MCP server) will extend
this structure incrementally.
