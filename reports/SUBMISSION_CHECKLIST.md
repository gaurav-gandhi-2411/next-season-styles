# Submission checklist

The four required items plus the bonus, each mapped to the file that satisfies it. Paths
are relative to the repository root; the reviewer's bundle is `reports/SUBMISSION/`.

## Requirements → files

| Requirement | Satisfied by |
|---|---|
| 1. Forecast next season's top styles, evaluated honestly | `reports/WRITEUP.md` §1–3 and §8; evidence: `reports/tables/backtest_summary_v2.csv`, `backtest_paired_diff.csv`, `label_shuffle_control.csv` (leakage control); shown in `DEMO.html` §3 |
| 2. Generate design concepts for the top styles | `FINAL_concepts.png` (the three concepts); `evidence_chain.png` (references → brief → concept → checks → verdict); `DEMO.html` §1–2 |
| 3. Agent architecture with sub-agents and reusable skills | `agents/*.md` (orchestrator + 5 sub-agents), `skills/style-brief/`, `skills/concept-qc/`, `reports/agent_run_transcript.md` (a recorded run: live tool calls over stdio, generation replayed from recorded results); `WRITEUP.md` §7 |
| 4. An MCP server exposing the tools | `src/nss/mcp_server.py` (8 tools, stdio); README "MCP server" with a repeatable smoke test (`scripts/mcp_smoke_test.py`) and its real output |
| Write-up | `WRITEUP.md` (about 2,200 words by `wc -w`, tables included) |
| Bonus: split by season, show how styles or concepts change | `seasonal_comparison.png` (autumn/winter vs summer concept); `DEMO.html` §4 (four seasonal top-3 tables); `WRITEUP.md` §8 |

## Reviewer bundle (`reports/SUBMISSION/`)

| File | Demonstrates |
|---|---|
| `DEMO.html` | Start here: self-contained page (no server, no internet) for a non-specialist: the concepts, their trace-back to the forecast, model evidence, seasonal view, and what could not be verified |
| `FINAL_concepts.png` | One clean concept per forecast style, captioned with what is visible in each image |
| `evidence_chain.png` | Per concept (three autumn/winter plus summer): references, briefed changes, Gates 1, 1b, integrity floor, Gate 2 per judge, Gate 3, closed-loop forecast, human check, verdict |
| `seasonal_comparison.png` | The same pipeline and selection rules for two seasons: beige sweater (autumn/winter) beside an orange patterned bikini top (summer) |
| `WRITEUP.md` | The argument, the corrected evaluation, the generation root cause and its fix, the quality checks, and the limits |
| `SUBMISSION_CHECKLIST.md` | This file |
| `README.md` | Points at the repository and states the one-command ways to run it |

## Evidence behind the claims (in the repository)

| File | Shows |
|---|---|
| `reports/tables/label_shuffle_control.csv` | Retraining on shuffled labels: 0 hits in 108 picks (does not detect the missing 13-week gap; see `backtest_embargo_check.csv`: 0.722 -> 0.528) |
| `reports/tables/leave_one_out_control.csv`, `reference_threshold_comparison.csv` | Median rule passes 21/61 real articles, p90 passes 61/61; the wider reference base cuts the DINOv2 limits' CI width 2.5-4.4x |
| `reports/tables/clone_positive_control.csv`, `gate1b_nearest_reference.csv` | An exact copy can pass the range check; the nearest-reference check fails it in every style |
| `reports/tables/colour_base_rate.csv`, `final_three_selection_log.csv` | Why the leaderboards are black-heavy (calibrated to the market) and the editorial selection rule with its skip log |
| `reports/tables/final_selection.csv`, `candidates_scored.csv` | The four verdicts and every input; every scored candidate |
| `reports/tables/gate3_validation_local-smolvlm.csv`, `integrity_validation_local-smolvlm.csv`, `local_judge_calibration_*.csv`, `judge_panel_kappa.csv` | Gate 3 and integrity validated on known cases; judge calibration and pairwise kappa |
| `reports/tables/prompt_lever_summary.md` | The prompt-lever experiments: what made the briefed changes appear |
| `reports/tables/covid_two_model_comparison.csv`, `backtest_embargo_check.csv` | COVID two-model comparison; the 13-week-gap re-run of the backtest |
| `reports/tables/concept_forecast_retrieval.csv`, `retrieval_validation.csv` (1,980 candidates), `retrieval_validation_partial_index.csv` (415 candidates), `concept_forecast_validation.csv` | Closed-loop forecasts of the final concepts (retrieval); full-catalogue and partial-index validation; the earlier free-text accuracy on 40 photos |
| `reports/tables/summer_selection_log.csv`, `forecast_spotcheck.csv` | Summer selection under the same rules; the #1 vs #1500 forecast spot-check |
| `tests/` | All tests pass (`uv run pytest tests`); includes the hero-stage regression, dry-run and control tests |
| `scripts/run_pipeline.py --dry-run` | All seven pipeline stages on CPU in about 1–2.5 minutes, writing only to a scratch directory |

## Known limits a reviewer should weigh

- Two of four final concepts pass every automatic gate. The white top fails the integrity floor (closest reference 0.733, below both the global floor 0.779 that gates and the per-style floor 0.922 reported as advisory) and Gate 3 (SmolVLM answers YN), and the summer bikini top fails Gate 2 on one reader's pattern answer; a human confirmed every requested change is visible except that the top's cuffs are narrower than briefed.
- The walk-forward headline Hit@3-in-top20 is 0.528 with a 13-week gap, not 0.722 (`reports/tables/backtest_embargo_check.csv`). Post-embargo, paired: comparable to seasonal naive on top-k (top-20 +0.233, CI [0.000, 0.567]), better on NDCG@10, Spearman and WMAPE, and better than persistence and the naive means on every metric (`backtest_embargo_paired_diff.csv`).
- The image readers are two small local models (SmolVLM gates, Florence-2 is advisory: kappa 0.36-0.48 vs API judges); Groq's daily budget is spent and Gemini's refreshed key authenticates and calibrates (gap 0.522) but its free quota ran out before any concept was re-read, so the bikini's pattern call has no second reader. The design-change reader over-says yes; the human check decides.
- The global integrity floor gates and the per-style floor is advisory; the global floor's one miss is seed 44, sheer mesh, 0.811 (`integrity_global_validation.csv`). Neighbourhood features and buyer-mix and price features were tested and rejected (`neighbourhood_decision.csv`, `buyer_price_decision.csv`). The summer prediction (37.88) matches the realised value (37.94) by luck: typical error is 6 units (`summer_check.csv`).
- The closed loop is an image-retrieval prototype over all 1,980 forecast styles (the local image tree, 8 photos per style). On the original 40 photos (each left out; 1,980 candidates) exact style is 27.5% against the free-text 12.5% (11 vs 5 photos, McNemar p=0.21: not established), top-5 62.5%, product type 82.5%, colour 52.5% (free-text: 70%). Leave-one-out on 159 photos falls from 72.3% (415 candidates, optimistic) to 42.1% (1,980). Two of four concepts map to a neighbouring style under the full index.
- COVID data did not hurt and mildly helped (13 non-COVID origins, second model has ~40% fewer rows).
- Full-mode generation is reproducible with `python -m nss.generate.concept_generation`; the dry run covers wiring.
