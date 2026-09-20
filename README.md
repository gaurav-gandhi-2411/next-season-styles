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

### Running the pipeline

`scripts/run_pipeline.py` has two modes.

**`--dry-run` (reviewers, no GPU).** Exercises every stage (panel → features → forecast → briefs →
generate → score → hero) end to end in about 1–2.5 minutes on CPU:

```
uv run --no-sync python scripts/run_pipeline.py --dry-run
```

It writes only under a scratch directory (`<system temp>/nss_dry_run`, or `--scratch-dir`), never
under `data/` or `reports/`. SDXL generation is skipped and the committed final concepts in
`reports/concepts/` are scored instead. The VLM judges are disabled, so attribute fidelity (Gate 2)
is skipped and the printed verdicts read "did not pass"; the similarity numbers are real. It needs
`data/processed/style_week_panel.parquet` (`make data`, `make panel`) and internet access (the
briefs stage downloads ~16 product photos into the scratch directory).

**Full mode (reproduction, needs a CUDA GPU).**

```
uv run --no-sync python scripts/run_pipeline.py [--n-seeds 2] [--stop-after <stage>]
```

Runs local SDXL generation and the live VLM judges. Outputs go to `reports/pipeline_run/`
(gitignored), never over the shipped deliverables.

## Status

**Complete.** All phases (data pipeline, panel construction, modelling, generation, agent layer,
MCP server, write-up) are implemented and committed. See `reports/WRITEUP.md` for the full
narrative, including the honest evaluation-methodology correction (Section 3) and known
limitations (Section 9).

**Start here:** open `reports/DEMO.html` in any browser (self-contained, no server) for the three
concepts, how each traces back to its forecast, the model evidence and the seasonal view, written
for a non-specialist. The full argument is `reports/WRITEUP.md`; what maps to which requirement is
`reports/SUBMISSION_CHECKLIST.md`.

**Generation, honestly:** an earlier round produced no briefed design change in 12 images. The
cause was measured, not assumed (`reports/tables/n1_levers_summary.md`): an attribute-first prompt
plus single-reference IP-Adapter conditioning. A plain-sentence prompt on both text encoders,
multi-reference conditioning, compel weighting and a per-style scale made the changes visible. Two
of four final concepts (sweater, dress) pass every automatic gate (1, 1b, integrity floor, 2, 3) and
a human check; the white top fails the integrity floor (closest reference 0.733, below the global floor 0.779 that
gates and the per-style floor 0.922 shown as advisory) and the summer bikini top
fails Gate 2 on one reader's answer. The local readers are small (the design-change reader says yes
too easily), so a human check decides. Details: `reports/tables/final_selection_h4.csv`,
`reports/figures/evidence_chain.png`.

**Evaluation correction:** the shipped walk-forward headline (Hit@3-in-top20 0.722) trained on
labels that overlap the test window. With a 13-week gap it is **0.528** (paired drop 0.194, CI
[0.111, 0.250]); see `reports/tables/backtest_embargo_check.csv`. Re-done as a paired comparison
(`backtest_embargo_paired_diff.csv`), the embargoed model is comparable to seasonal naive on top-k
(top-20 +0.233, CI [0.000, 0.567]), better on NDCG@10, Spearman and WMAPE, and better than
persistence and the naive means on every metric. The label-shuffle control (0 hits in 108 picks)
does not detect this leak. COVID: a second model without COVID-overlapping training rows was no
better (`covid_two_model_comparison.csv`).

## Project layout

- `src/nss/` — installable package: `data/` (ingestion, exemplar selection), `features/` (panel +
  style_key construction, causal feature set, style-key validation), `models/` (backtest harness,
  baselines, LightGBM, final forecast + selection), `generate/` (SDXL/IP-Adapter + Gemini
  generation, CLIP/DINOv2 margin scoring, VLM judges, QC pipeline), `viz/` (EDA + diagram figures),
  and `mcp_server.py` (the 8-tool MCP server)
- `skills/` — 2 reusable, dataset-agnostic Claude Agent Skills: `style-brief/` (style profile ->
  design brief) and `concept-qc/` (Gate 1 within-style range + Gate 1b nearest-reference + blind
  VLM fidelity + mandatory human check -> verdict + retry strategy)
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
backtest. `generate_concept` performs live GPU (or paid-API) work when invoked, and `score_concept`
runs the shipped quality gates live (CLIP/DINOv2 on CPU; an optional Groq call for Gate 2).

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

### Verify it works (repeatable smoke test)

`scripts/mcp_smoke_test.py` reads the `mcpServers` block above **as written** (only the documented
`cwd` placeholder is replaced with your checkout path), launches the server with that command over
stdio, connects a real MCP client, and calls `forecast_styles`, `get_style_profile` and `score_concept`. It needs the
`data/` artifacts the tools read (see Quickstart) and exits 0 on success:

```
uv run --no-sync python scripts/mcp_smoke_test.py
```

Expected output (captured from a real run; long lists are elided by the script and say so; paths
under `<your checkout>` are yours). `score_concept` runs the shipped gates on the committed
sweater concept: Gate 1 and Gate 1b pass (the exact-clone control fails as required), Gate 2 is
not run by default, and the human visual check is always required:

```
launching: uv run --no-sync python -m nss.mcp_server  (cwd=<your checkout>)
tools: ['compose_final_sheet', 'forecast_concept', 'forecast_styles', 'generate_concept', 'get_reference_images', 'get_style_profile', 'query_transactions', 'score_concept']
forecast_styles -> [
  {
    "rank": 1,
    "style_key": "Ladieswear || T-shirt || Jersey Basic || Black || Solid",
    "index_group_name": "Ladieswear",
    "product_type_name": "T-shirt",
    "garment_group_name": "Jersey Basic",
    "perceived_colour_master_name": "Black",
    "graphical_appearance_name": "Solid",
    "predicted_intensity": 33.773553456977744,
    "guard1_pass": true,
    "guard1_n_active_articles_trailing_mean": 23.0,
    "guard2_pass": true,
    "guard2_price_index": 1.0589412126425564,
    "guard3_pass": true,
    "guard3_n_weeks_active_trailing": 52,
    "shap_driver_1_feature": "lag_1",
    "shap_driver_1_value": 0.7287921107138857,
    "shap_driver_2_feature": "n_active_articles_level",
    "shap_driver_2_value": 0.2011520713462187,
    "shap_driver_3_feature": "fourier_sin_1",
    "shap_driver_3_value": -0.12149169600317132,
    "shap_driver_4_feature": "garment_group_name",
    "shap_driver_4_value": 0.11608373485876344,
    "shap_driver_5_feature": "perceived_colour_master_name",
    "shap_driver_5_value": 0.11324089707198806,
    "_forecast_origin_date": "2020-09-21",
    "_forecast_horizon_weeks": 13,
    "_requested_origin_date": "2020-09-21",
    "_requested_horizon_weeks": 13
  }
]
get_style_profile -> {
  "style_key": "Ladieswear || T-shirt || Jersey Basic || Black || Solid",
  "attributes": {
    "index_group_name": "Ladieswear",
    "product_type_name": "T-shirt",
    "garment_group_name": "Jersey Basic",
    "perceived_colour_master_name": "Black",
    "graphical_appearance_name": "Solid"
  },
  "trajectory": {
    "available": true,
    "first_week_seen": "2018-09-17",
    "last_week_seen": "2020-09-21",
    "n_weeks_total": 106,
    "recent_weeks": [
      {
        "week_start": "2020-06-29",
        "units": 1857,
        "revenue": 21.284440677966103,
        "price_index": 0.9931281941617366,
        "intensity_shrunk": 66.75452732393227
      },
      {
        "week_start": "2020-07-06",
        "units": 1591,
        "revenue": 17.791305084745762,
        "price_index": 0.9689304409646231,
        "intensity_shrunk": 59.63597755238108
      },
      "... 11 more entries elided"
    ],
    "recent_mean_units": 1564.6923076923076,
    "recent_mean_revenue": 19.329864406779663
  },
  "shap_drivers": {
    "source": "reports\\tables\\top_styles_t1_incumbent.csv",
    "style_specific": true,
    "drivers": [
      {
        "feature": "lag_1",
        "value": 0.7287921107138857
      },
      {
        "feature": "n_active_articles_level",
        "value": 0.2011520713462187
      },
      "... 3 more entries elided"
    ],
    "note": null
  }
}
score_concept -> {
  "concept_path": "<your checkout>\reports\\concepts\\beige-knit-sweater_s0.35_seed45.png",
  "style_key": "Ladieswear || Sweater || Knitwear || Beige || Melange",
  "n_references": 17,
  "gate1": {
    "pass": true,
    "clip": {
      "similarity": 0.925444294424618,
      "limit_p90_of_real_pairs": 0.9664748013019562,
      "pass": true
    },
    "dinov2": {
      "similarity": 0.829064362189349,
      "limit_p90_of_real_pairs": 0.9041306674480438,
      "pass": true
    }
  },
  "gate1b": {
    "pass": true,
    "clone_control_failed_as_required": true,
    "clip": {
      "closest_reference_similarity": 0.949120283126831,
      "limit_p90_of_real_nearest_sibling": 0.9825984597206116,
      "pass": true
    },
    "dinov2": {
      "closest_reference_similarity": 0.90105801820755,
      "limit_p90_of_real_nearest_sibling": 0.9469841361045838,
      "pass": true
    }
  },
  "gate2": {
    "status": "not_run",
    "note": "pass include_fidelity=True (one Groq call) or run `nss.generate.judge_repeat`"
  },
  "human_visual_check": {
    "required": true,
    "status": "not automated",
    "note": "REQUIRED and never automated: look at the image. The automatic gates have passed visibly malformed garments (cut-out defects, pattern drift, straps on a bottom)."
  },
  "automated_gates_pass": null,
  "verdict": "Gate 1 and Gate 1b passed; Gate 2 not run; human visual check REQUIRED"
}
forecast_styles returned 3 rows (only the first is shown above)
MCP smoke test: OK
```

The documented config worked verbatim (`uv` must be on `PATH`; the server itself runs from the
project's own `.venv`).
