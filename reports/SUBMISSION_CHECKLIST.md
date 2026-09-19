# Submission checklist — what the reviewer receives

Everything below is in the repository (`main`). "Demonstrates" states the claim the file supports;
provenance for every number is the CSV named in the same row. Nothing under `data/` is committed
(raw H&M data, panel, images and generated candidates are git-ignored and regenerable).

## Read first

| File | Demonstrates |
|---|---|
| `reports/WRITEUP.md` | The whole argument in ~1,800 words: style definition, target, backtest, Track G's negative result, the five-stage generation-QC validation arc, limits. |
| `reports/figures/FINAL_concepts.png` | The deliverable: one clean concept per forecast style, captioned with what is *visible* in each image (not what the brief requested). |
| `reports/figures/evidence_chain.png` | Per style: references → brief → concept → Gate 1 vs the real within-style p90 → per-attribute fidelity with its noise bound → verdict, plus the human visual check. |

## Forecasting (frozen model)

| File | Demonstrates |
|---|---|
| `src/nss/features/`, `src/nss/models/` | Style-week panel (313,887 rows), leakage-safe features, LightGBM forecaster, rolling-origin backtest harness. |
| `reports/tables/backtest_summary_v2.csv`, `backtest_paired_diff.csv` | Hit@3-in-top20 0.722 vs random 0.004 and four baselines, with paired-difference CIs. |
| `reports/tables/g1_diagnostics_summary.csv`, `g2_lambdarank_vs_l2.csv`, `g3_variants.csv` | Track G: the "persistence is under-used" premise was tested and rejected; four alternative objectives did not beat L2. |
| `reports/tables/top_styles_final_three.csv`, `final_three_shap_verdict.csv` | The T1/T2 picks and SHAP evidence they are not a Christmas artefact. |

## Agent layer, MCP server, skills

| File | Demonstrates |
|---|---|
| `agents/*.md` | Orchestrator + 5 sub-agents (data-analyst, forecaster, style-profiler, concept-designer, critic); the orchestrator calls no tool directly. |
| `src/nss/mcp_server.py` | 7 read-only MCP tools over stdio; only `generate_concept` does live work. |
| `reports/agent_run_transcript.md` | A real run where the critic's retry loop was exercised (1 attempt + 2 retries per concept). |
| `skills/style-brief/`, `skills/concept-qc/` | Two reusable, dataset-agnostic skills; `concept-qc/SKILL.md` records every gate revision with its evidence. |

## Generation QC — the five-stage instrument validation

| File | Demonstrates |
|---|---|
| `reports/tables/leave_one_out_control.csv` | **Stage 5 evidence.** Each real reference article scored against its siblings through the same gate code: 6/16 pass under the median rule; 16/16 (full benchmark) and 14/16 (held-out benchmark) under p90. |
| `reports/tables/clone_positive_control.csv` | Positive control: an exact copy of a reference passes Gate 1 for 2 of 3 styles — Gate 1 is a range check, not a copy detector; the nearest-neighbour column flags every clone. |
| `reports/tables/within_style_benchmark.csv` | Per-style pairwise-similarity median / p90 / max (screened reference sets). |
| `reports/tables/gate1_rescored_within_style.csv`, `h3_underwear_scored.csv` | Gate 1 for all 16 scored candidates: median rule vs p90 rule side by side, plus the un-gated nearest-neighbour diagnostic. |
| `reports/tables/j4_judge_repeats.csv` | Gate 2: each final concept judged 3× (Groq), median and observed range vs the calibrated 0.513 threshold; H3 seed 45 (folded object) judged once. |
| `reports/tables/final_selection_h4.csv` | The three final verdicts with every input to them. |
| `reports/tables/h2_calibration_recomputed.csv`, `h2_judge_rescore.csv` | Stage-4 judge checklist fix (3 visual attributes) and per-judge thresholds. |
| `reports/tables/margin_anchor_realspace_vs_genspace_gap.csv` | Stages 1–3: why absolute CLIP cosine and real-space anchors were rejected. |

## Reproducibility

| File | Demonstrates |
|---|---|
| `scripts/run_pipeline.py`, `Makefile` | One-command pipeline (panel → forecast → briefs → generate → score → hero); GPU needed for the generate stage. |
| `tests/` | 593 tests (all pass), incl. `test_run_pipeline_wiring.py` (regression test for the hero-stage `TypeError`) and `test_leave_one_out_control.py` (the J1/J2 control on hand-computable vectors). |
| `uv.lock`, `pyproject.toml` | Pinned environment. |

## Known limits the reviewer should weigh

- Gate 1 (p90 range check) does not detect a copy of a single reference (clone control above).
- Automated gates catch drift, not incoherence: three underwear candidates that were visually
  malformed were not stopped by them (see WRITEUP §9).
- Judge scores carry a measured single-call noise of up to ±0.21; each final fidelity is a median
  of 3 calls from one judge model (Groq `qwen3.8-27b`); Gemini was quota-blocked.
- `run_pipeline.py` was not re-run end-to-end in the final session (no new image generation was
  permitted); its wiring is covered by the regression tests.
