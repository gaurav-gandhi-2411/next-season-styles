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

**Generation QC, honestly:** the concept-QC gate rejected every concept until it was re-anchored
on how similar two *real, distinct* H&M articles of the same style are (Section 6). Under the
final gates the dress passes every automatic check, while the T-shirt and the sweater fail
Gate 2 (the second judge scores them 0.333 and 0.617 vs a 0.667 threshold). **None of the three
final concepts shows the design change its brief asked for** (the generator's reference structure
dominated; two rounds of four tries per style), and the human check says so per concept. Fidelity
scores are noisy (about +/-0.21 across sessions). The automatic checks also passed visibly
malformed candidates in earlier runs, so a human check is required. Details:
`reports/tables/final_selection_h4.csv`, `reports/figures/evidence_chain.png`.

**Leakage control:** retraining the same model on shuffled labels collapses to chance (0 hits in
108 picks; `reports/tables/label_shuffle_control.csv`), while the unshuffled control reproduces the
headline Hit@3-in-top20 of 0.722.

## Project layout

- `src/nss/` — installable package: `data/` (ingestion, exemplar selection), `features/` (panel +
  style_key construction, causal feature set, style-key validation), `models/` (backtest harness,
  baselines, LightGBM, final forecast + selection), `generate/` (SDXL/IP-Adapter + Gemini
  generation, CLIP/DINOv2 margin scoring, VLM judges, QC pipeline), `viz/` (EDA + diagram figures),
  and `mcp_server.py` (the 7-tool MCP server)
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
T-shirt concept: Gate 1 passes, Gate 1b fails by 0.0018 on DINOv2 (the exact-clone control fails
as required), Gate 2 is not run by default, and the human visual check is always required:

```
[09/20/26 02:41:43] INFO     HTTP Request: HEAD                 _client.py:1025
                             https://huggingface.co/openai/clip                
                             -vit-large-patch14/resolve/main/co                
                             nfig.json "HTTP/1.1 307 Temporary                 
                             Redirect"                                         
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
                    WARNING  Warning: You are sending              _http.py:993
                             unauthenticated requests to the HF                
                             Hub. Please set a HF_TOKEN to enable              
                             higher rate limits and faster                     
                             downloads.                                        
                    INFO     HTTP Request: HEAD                 _client.py:1025
                             https://huggingface.co/api/resolve                
                             -cache/models/openai/clip-vit-larg                
                             e-patch14/32bd64288804d66eefd0ccbe                
                             215aa642df71cc41/config.json?%2Fop                
                             enai%2Fclip-vit-large-patch14%2Fre                
                             solve%2Fmain%2Fconfig.json=&etag=%                
                             222c19f6666e0e163c7954df66cb901353                
                             fcad088e%22 "HTTP/1.1 200 OK"                     
                    INFO     HTTP Request: HEAD                 _client.py:1025
                             https://huggingface.co/openai/clip                
                             -vit-large-patch14/resolve/main/co                
                             nfig.json "HTTP/1.1 307 Temporary                 
                             Redirect"                                         
                    INFO     HTTP Request: HEAD                 _client.py:1025
                             https://huggingface.co/api/resolve                
                             -cache/models/openai/clip-vit-larg                
                             e-patch14/32bd64288804d66eefd0ccbe                
                             215aa642df71cc41/config.json?%2Fop                
                             enai%2Fclip-vit-large-patch14%2Fre                
                             solve%2Fmain%2Fconfig.json=&etag=%                
                             222c19f6666e0e163c7954df66cb901353                
                             fcad088e%22 "HTTP/1.1 200 OK"                     
                    INFO     HTTP Request: HEAD                 _client.py:1025
                             https://huggingface.co/openai/clip                
                             -vit-large-patch14/resolve/main/mo                
                             del.safetensors "HTTP/1.1 302                     
                             Found"                                            

Loading weights:   0%|          | 0/590 [00:00<?, ?it/s]
Loading weights:   5%|5         | 30/590 [00:00<00:01, 294.00it/s]
Loading weights:  10%|#         | 60/590 [00:00<00:02, 230.43it/s]
Loading weights:  14%|#4        | 84/590 [00:00<00:02, 184.89it/s]
Loading weights:  18%|#7        | 104/590 [00:00<00:03, 155.73it/s]
Loading weights:  21%|##        | 121/590 [00:00<00:03, 150.90it/s]
Loading weights:  23%|##3       | 137/590 [00:00<00:03, 135.11it/s]
Loading weights:  26%|##5       | 151/590 [00:00<00:03, 134.35it/s]
Loading weights:  28%|##7       | 165/590 [00:01<00:03, 135.21it/s]
Loading weights:  30%|###       | 179/590 [00:01<00:04, 102.55it/s]
Loading weights:  34%|###3      | 199/590 [00:01<00:03, 124.35it/s]
Loading weights: 100%|##########| 590/590 [00:01<00:00, 407.74it/s]
[09/20/26 02:41:46] INFO     HTTP Request: GET                  _client.py:1025
                             https://huggingface.co/api/models/                
                             openai/clip-vit-large-patch14/tree                
                             /main/additional_chat_templates?re                
                             cursive=false&expand=false                        
                             "HTTP/1.1 404 Not Found"                          
                    INFO     HTTP Request: HEAD                 _client.py:1025
                             https://huggingface.co/openai/clip                
                             -vit-large-patch14/resolve/main/pr                
                             ocessor_config.json "HTTP/1.1 404                 
                             Not Found"                                        
                    INFO     HTTP Request: HEAD                 _client.py:1025
                             https://huggingface.co/openai/clip                
                             -vit-large-patch14/resolve/main/ch                
                             at_template.json "HTTP/1.1 404 Not                
                             Found"                                            
[09/20/26 02:41:47] INFO     HTTP Request: HEAD                 _client.py:1025
                             https://huggingface.co/openai/clip                
                             -vit-large-patch14/resolve/main/ch                
                             at_template.jinja "HTTP/1.1 404                   
                             Not Found"                                        
                    INFO     HTTP Request: HEAD                 _client.py:1025
                             https://huggingface.co/openai/clip                
                             -vit-large-patch14/resolve/main/au                
                             dio_tokenizer_config.json                         
                             "HTTP/1.1 404 Not Found"                          
                    INFO     HTTP Request: HEAD                 _client.py:1025
                             https://huggingface.co/openai/clip                
                             -vit-large-patch14/resolve/main/pr                
                             ocessor_config.json "HTTP/1.1 404                 
                             Not Found"                                        
                    INFO     HTTP Request: HEAD                 _client.py:1025
                             https://huggingface.co/openai/clip                
                             -vit-large-patch14/resolve/main/pr                
                             eprocessor_config.json "HTTP/1.1                  
                             307 Temporary Redirect"                           
                    INFO     HTTP Request: HEAD                 _client.py:1025
                             https://huggingface.co/api/resolve                
                             -cache/models/openai/clip-vit-larg                
                             e-patch14/32bd64288804d66eefd0ccbe                
                             215aa642df71cc41/preprocessor_conf                
                             ig.json?%2Fopenai%2Fclip-vit-large                
                             -patch14%2Fresolve%2Fmain%2Fprepro                
                             cessor_config.json=&etag=%225a12a1                
                             eb250987a4eee0e3e7d7338c4b22724be1                
                             %22 "HTTP/1.1 200 OK"                             
[09/20/26 02:41:48] INFO     HTTP Request: HEAD                 _client.py:1025
                             https://huggingface.co/openai/clip                
                             -vit-large-patch14/resolve/main/pr                
                             ocessor_config.json "HTTP/1.1 404                 
                             Not Found"                                        
                    INFO     HTTP Request: HEAD                 _client.py:1025
                             https://huggingface.co/openai/clip                
                             -vit-large-patch14/resolve/main/pr                
                             eprocessor_config.json "HTTP/1.1                  
                             307 Temporary Redirect"                           
                    INFO     HTTP Request: HEAD                 _client.py:1025
                             https://huggingface.co/api/resolve                
                             -cache/models/openai/clip-vit-larg                
                             e-patch14/32bd64288804d66eefd0ccbe                
                             215aa642df71cc41/preprocessor_conf                
                             ig.json?%2Fopenai%2Fclip-vit-large                
                             -patch14%2Fresolve%2Fmain%2Fprepro                
                             cessor_config.json=&etag=%225a12a1                
                             eb250987a4eee0e3e7d7338c4b22724be1                
                             %22 "HTTP/1.1 200 OK"                             
                    INFO     HTTP Request: HEAD                 _client.py:1025
                             https://huggingface.co/openai/clip                
                             -vit-large-patch14/resolve/main/co                
                             nfig.json "HTTP/1.1 307 Temporary                 
                             Redirect"                                         
                    INFO     HTTP Request: HEAD                 _client.py:1025
                             https://huggingface.co/api/resolve                
                             -cache/models/openai/clip-vit-larg                
                             e-patch14/32bd64288804d66eefd0ccbe                
                             215aa642df71cc41/config.json?%2Fop                
                             enai%2Fclip-vit-large-patch14%2Fre                
                             solve%2Fmain%2Fconfig.json=&etag=%                
                             222c19f6666e0e163c7954df66cb901353                
                             fcad088e%22 "HTTP/1.1 200 OK"                     
[09/20/26 02:41:49] INFO     HTTP Request: HEAD                 _client.py:1025
                             https://huggingface.co/openai/clip                
                             -vit-large-patch14/resolve/main/to                
                             kenizer_config.json "HTTP/1.1 307                 
                             Temporary Redirect"                               
                    INFO     HTTP Request: HEAD                 _client.py:1025
                             https://huggingface.co/api/resolve                
                             -cache/models/openai/clip-vit-larg                
                             e-patch14/32bd64288804d66eefd0ccbe                
                             215aa642df71cc41/tokenizer_config.                
                             json?%2Fopenai%2Fclip-vit-large-pa                
                             tch14%2Fresolve%2Fmain%2Ftokenizer                
                             _config.json=&etag=%22702bb12920b2                
                             91cade3706cf215c1604d2255d93%22                   
                             "HTTP/1.1 200 OK"                                 
                    INFO     HTTP Request: GET                  _client.py:1025
                             https://huggingface.co/api/models/                
                             openai/clip-vit-large-patch14/tree                
                             /main/additional_chat_templates?re                
                             cursive=false&expand=false                        
                             "HTTP/1.1 404 Not Found"                          
                    INFO     HTTP Request: GET                  _client.py:1025
                             https://huggingface.co/api/models/                
                             openai/clip-vit-large-patch14/tree                
                             /main?recursive=true&expand=false                 
                             "HTTP/1.1 200 OK"                                 
[09/20/26 02:41:56] INFO     HTTP Request: HEAD                 _client.py:1025
                             https://huggingface.co/facebook/di                
                             nov2-base/resolve/main/config.json                
                              "HTTP/1.1 307 Temporary Redirect"                
                    INFO     HTTP Request: HEAD                 _client.py:1025
                             https://huggingface.co/api/resolve                
                             -cache/models/facebook/dinov2-base                
                             /f9e44c814b77203eaa57a6bdbbd535f21                
                             ede1415/config.json?%2Ffacebook%2F                
                             dinov2-base%2Fresolve%2Fmain%2Fcon                
                             fig.json=&etag=%225df0129f878aa42d                
                             71fa74d27a50f382d13ed71e%22                       
                             "HTTP/1.1 200 OK"                                 
                    INFO     HTTP Request: HEAD                 _client.py:1025
                             https://huggingface.co/facebook/di                
                             nov2-base/resolve/main/config.json                
                              "HTTP/1.1 307 Temporary Redirect"                
                    INFO     HTTP Request: HEAD                 _client.py:1025
                             https://huggingface.co/api/resolve                
                             -cache/models/facebook/dinov2-base                
                             /f9e44c814b77203eaa57a6bdbbd535f21                
                             ede1415/config.json?%2Ffacebook%2F                
                             dinov2-base%2Fresolve%2Fmain%2Fcon                
                             fig.json=&etag=%225df0129f878aa42d                
                             71fa74d27a50f382d13ed71e%22                       
                             "HTTP/1.1 200 OK"                                 

Loading weights:   0%|          | 0/223 [00:00<?, ?it/s]
Loading weights:  12%|#2        | 27/223 [00:00<00:00, 265.86it/s]
Loading weights:  29%|##9       | 65/223 [00:00<00:00, 316.65it/s]
Loading weights:  43%|####3     | 97/223 [00:00<00:00, 299.15it/s]
Loading weights: 100%|##########| 223/223 [00:00<00:00, 665.25it/s]
[09/20/26 02:41:57] INFO     HTTP Request: HEAD                 _client.py:1025
                             https://huggingface.co/facebook/di                
                             nov2-base/resolve/main/processor_c                
                             onfig.json "HTTP/1.1 404 Not                      
                             Found"                                            
                    INFO     HTTP Request: HEAD                 _client.py:1025
                             https://huggingface.co/facebook/di                
                             nov2-base/resolve/main/preprocesso                
                             r_config.json "HTTP/1.1 307                       
                             Temporary Redirect"                               
                    INFO     HTTP Request: HEAD                 _client.py:1025
                             https://huggingface.co/api/resolve                
                             -cache/models/facebook/dinov2-base                
                             /f9e44c814b77203eaa57a6bdbbd535f21                
                             ede1415/preprocessor_config.json?%                
                             2Ffacebook%2Fdinov2-base%2Fresolve                
                             %2Fmain%2Fpreprocessor_config.json                
                             =&etag=%22ff5b47c2edcd1d3556d63c01                
                             a65d93b58b9efce1%22 "HTTP/1.1 200                 
                             OK"                                               
                    INFO     HTTP Request: HEAD                 _client.py:1025
                             https://huggingface.co/facebook/di                
                             nov2-base/resolve/main/processor_c                
                             onfig.json "HTTP/1.1 404 Not                      
                             Found"                                            
[09/20/26 02:41:58] INFO     HTTP Request: HEAD                 _client.py:1025
                             https://huggingface.co/facebook/di                
                             nov2-base/resolve/main/preprocesso                
                             r_config.json "HTTP/1.1 307                       
                             Temporary Redirect"                               
                    INFO     HTTP Request: HEAD                 _client.py:1025
                             https://huggingface.co/api/resolve                
                             -cache/models/facebook/dinov2-base                
                             /f9e44c814b77203eaa57a6bdbbd535f21                
                             ede1415/preprocessor_config.json?%                
                             2Ffacebook%2Fdinov2-base%2Fresolve                
                             %2Fmain%2Fpreprocessor_config.json                
                             =&etag=%22ff5b47c2edcd1d3556d63c01                
                             a65d93b58b9efce1%22 "HTTP/1.1 200                 
                             OK"                                               
launching: uv run --no-sync python -m nss.mcp_server  (cwd=<your checkout>)
tools: ['compose_final_sheet', 'forecast_styles', 'generate_concept', 'get_reference_images', 'get_style_profile', 'query_transactions', 'score_concept']
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
    "source": "reports\\tables\\final_three_shap_verdict.csv",
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
  "concept_path": "<your checkout>\\reports\\concepts\\black-jersey-basic-tshirt_seed43.png",
  "style_key": "Ladieswear || T-shirt || Jersey Basic || Black || Solid",
  "n_references": 6,
  "gate1": {
    "pass": true,
    "clip": {
      "similarity": 0.9450633327166239,
      "limit_p90_of_real_pairs": 0.969795036315918,
      "pass": true
    },
    "dinov2": {
      "similarity": 0.8819671074549357,
      "limit_p90_of_real_pairs": 0.896066951751709,
      "pass": true
    }
  },
  "gate1b": {
    "pass": false,
    "clone_control_failed_as_required": true,
    "clip": {
      "closest_reference_similarity": 0.9710571765899658,
      "limit_p90_of_real_nearest_sibling": 0.9781906008720398,
      "pass": true
    },
    "dinov2": {
      "closest_reference_similarity": 0.9130043387413025,
      "limit_p90_of_real_nearest_sibling": 0.9112358093261719,
      "pass": false
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
  "automated_gates_pass": false,
  "verdict": "REJECT: failed gate1b"
}
forecast_styles returned 3 rows (only the first is shown above)
MCP smoke test: OK
```

The documented config worked verbatim (`uv` must be on `PATH`; the server itself runs from the
project's own `.venv`).
