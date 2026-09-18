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

## MCP server

`src/nss/mcp_server.py` exposes 7 read-only tools (transaction queries, style profiles/SHAP
drivers, pre-computed forecasts, reference images, concept generation, concept scoring, and
concept-sheet composition) over the **stdio** transport of the official
[MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) (`mcp` package). Every tool
reads artifacts already on disk — modelling is frozen; nothing here retrains a model or re-runs a
backtest. `generate_concept` is the one tool that performs live GPU (or paid-API) work when a
client actually invokes it.

Run it standalone:

```
uv run python -m nss.mcp_server
```

Point any standard MCP client (Claude Desktop, or any other MCP-compatible client) at it with an
`mcpServers` config block like this (adjust `cwd` to wherever this repo is checked out):

```json
{
  "mcpServers": {
    "next-season-styles": {
      "command": "uv",
      "args": ["run", "--no-sync", "python", "-m", "nss.mcp_server"],
      "cwd": "/absolute/path/to/next-season-styles"
    }
  }
}
```

For Claude Desktop specifically, add that block to `claude_desktop_config.json` (macOS:
`~/Library/Application Support/Claude/claude_desktop_config.json`; Windows:
`%APPDATA%\Claude\claude_desktop_config.json`) and restart the app.
