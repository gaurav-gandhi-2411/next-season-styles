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

## Quickstart

```
uv sync                                   # or: make setup
uv run python scripts/download_data.py    # or: make data (Kaggle CLI creds required)
uv run python scripts/run_pipeline.py     # or: make pipeline -- full end-to-end reproduction
                                           # (panel -> features -> forecast -> top-3 -> briefs ->
                                           # generate -> score -> hero image; GPU required, ~1
                                           # seed/style by default, see script docstring)
```

The hero deliverable image is `reports/figures/FINAL_concepts.png`; the full write-up is
`reports/WRITEUP.md`. Individual pipeline stages (`panel`, `eda`, `backtest`, `train`, `forecast`,
`sweep`, `agent-diagram`, `deliverables`, `test`, `lint`) are each their own `make` target — see the
`Makefile` for the exact command and artifact each one produces.

## Status

**Complete.** All phases (data pipeline, panel construction, modelling, generation, agent layer,
MCP server, write-up) are implemented and committed. See `reports/WRITEUP.md` for the full
narrative, including the honest evaluation-methodology correction (Section 3) and known
limitations (Section 9).

## Project layout

- `src/nss/` — installable package: `data/` (ingestion, exemplar selection), `features/` (panel +
  style_key construction, causal feature set, style-key validation), `models/` (backtest harness,
  baselines, LightGBM, final forecast + selection), `generate/` (SDXL/IP-Adapter + Gemini
  generation, CLIP/DINOv2 margin scoring, VLM judges, QC pipeline), `viz/` (EDA + diagram figures),
  and `mcp_server.py` (the 7-tool MCP server)
- `skills/` — 2 reusable, dataset-agnostic Claude Agent Skills: `style-brief/` (style profile ->
  design brief) and `concept-qc/` (margin-band + blind VLM panel -> pass/fail + retry strategy)
- `agents/` — sub-agent role definitions (`orchestrator.md`, `data-analyst.md`, `forecaster.md`,
  `style-profiler.md`, `concept-designer.md`, `critic.md`) consumed by the agent-layer demo
- `scripts/` — `agent_demo.py` (drives the agent delegation graph end-to-end) and
  `run_pipeline.py` (non-interactive full reproduction, task D3)
- `reports/{figures,tables}/` — every generated artifact this project produces, plus
  `WRITEUP.md` (the full technical write-up) and `agent_run_transcript.md` (the real D4 agent run)
- `data/{raw,interim,processed,generated,images}/` — gitignored data tiers (empty dirs tracked via
  `.gitkeep`)
- `notebooks/` — exploratory notebooks
- `tests/` — pytest suite (unit tests, cross-process determinism regression test)

## Note

Built under a tight assignment deadline; favors a disciplined, minimal footprint over speculative
completeness — every phase (data pipeline, panel construction, modeling, generation, agent layer,
MCP server) reports honest, sometimes negative, results rather than a polished narrative. See
`reports/WRITEUP.md` Section 9 for the full list of known limitations.

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
