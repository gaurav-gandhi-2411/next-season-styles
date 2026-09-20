# Submission checklist

The brief's four required items plus the bonus, each mapped to the file that satisfies it. Paths
are relative to the repository root; the reviewer's bundle is `reports/SUBMISSION/`.

## Requirements → files

| Requirement | Satisfied by |
|---|---|
| 1. Forecast next season's top styles, evaluated honestly | `reports/WRITEUP.md` §1–3 and §8; evidence: `reports/tables/backtest_summary_v2.csv`, `backtest_paired_diff.csv`, `label_shuffle_control.csv` (leakage control); shown in `DEMO.html` §3 |
| 2. Generate design concepts for the top styles | `FINAL_concepts.png` (the three concepts); `evidence_chain.png` (references → brief → concept → checks → verdict); `DEMO.html` §1–2 |
| 3. Agent architecture with sub-agents and reusable skills | `agents/*.md` (orchestrator + 5 sub-agents), `skills/style-brief/`, `skills/concept-qc/`, `reports/agent_run_transcript.md`; `WRITEUP.md` §7 |
| 4. An MCP server exposing the tools | `src/nss/mcp_server.py` (7 tools, stdio); README "MCP server" with a repeatable smoke test (`scripts/mcp_smoke_test.py`) and its real output |
| Write-up | `WRITEUP.md` (about 1,900 words) |
| Bonus: split by season, show how styles or concepts change | `seasonal_comparison.png` (autumn/winter vs summer concept); `DEMO.html` §4 (four seasonal top-3 tables); `WRITEUP.md` §8 |

## Reviewer bundle (`reports/SUBMISSION/`)

| File | Demonstrates |
|---|---|
| `DEMO.html` | Start here: self-contained page (no server, no internet) for a non-specialist: the concepts, their trace-back to the forecast, model evidence, seasonal view, and what could not be verified |
| `FINAL_concepts.png` | One clean concept per forecast style, captioned with what is visible in each image |
| `evidence_chain.png` | Per concept (three autumn/winter plus summer): references, briefed changes, Gates 1, 1b, integrity floor, Gate 2 per judge, Gate 3, closed-loop forecast, human check, verdict |
| `seasonal_comparison.png` | The same pipeline and selection rules for two seasons: beige sweater (autumn/winter) beside an orange patterned bikini top (summer) |
| `WRITEUP.md` | The argument, the corrected evaluation, the generation root cause and its fix, the gates, and the limits |
| `SUBMISSION_CHECKLIST.md` | This file |
| `README.md` | Points at the repository and states the one-command ways to run it |

## Evidence behind the claims (in the repository)

| File | Shows |
|---|---|
| `reports/tables/label_shuffle_control.csv` | Retraining on shuffled labels: 0 hits in 108 picks (does not detect the missing 13-week gap; see `backtest_embargo_check.csv`: 0.722 -> 0.528) |
| `reports/tables/leave_one_out_control.csv`, `reference_threshold_comparison.csv` | Median rule passes 21/61 real articles, p90 passes 61/61; the wider reference base cuts the DINOv2 limits' CI width 2.5-4.4x |
| `reports/tables/clone_positive_control.csv`, `gate1b_nearest_reference.csv` | An exact copy can pass the range check; the nearest-reference check fails it in every style |
| `reports/tables/colour_base_rate.csv`, `final_three_selection_log.csv` | Why the leaderboards are black-heavy (calibrated to the market) and the editorial selection rule with its skip log |
| `reports/tables/final_selection_h4.csv`, `n9_candidates_scored.csv` | The four verdicts and every input; every scored candidate |
| `reports/tables/gate3_validation_local-smolvlm.csv`, `integrity_validation_local-smolvlm.csv`, `local_judge_calibration_*.csv`, `judge_panel_kappa.csv` | Gate 3 and integrity validated on known cases; judge calibration and pairwise kappa |
| `reports/tables/n1_levers_summary.md` | The N1 lever experiments: what made the briefed changes appear |
| `reports/tables/covid_two_model_comparison.csv`, `backtest_embargo_check.csv` | COVID two-model comparison; the 13-week-gap re-run of the backtest |
| `reports/tables/concept_forecast_final.csv`, `concept_forecast_validation.csv` | Closed-loop forecasts of the final concepts; extraction accuracy on 40 real photos |
| `reports/tables/summer_selection_n6.csv`, `forecast_spotcheck.csv` | Summer selection under the N6 rules; the #1 vs #1500 forecast spot-check |
| `tests/` | All tests pass (`uv run pytest tests`); includes the hero-stage regression, dry-run and control tests |
| `scripts/run_pipeline.py --dry-run` | All seven pipeline stages on CPU in about 1–2.5 minutes, writing only to a scratch directory |

## Known limits a reviewer should weigh

- Two of four final concepts pass every automatic gate. The white top fails the integrity floor by mechanism (19 near-identical real articles) and the summer bikini top fails Gate 2 on one reader's pattern answer; a human confirmed every requested change is visible except that the top's cuffs are narrower than briefed.
- The walk-forward headline Hit@3-in-top20 is 0.528 with a 13-week gap, not 0.722 (`reports/tables/backtest_embargo_check.csv`).
- The image readers are two small local models; Groq and Gemini were unavailable this session. The design-change reader over-says yes; the human check decides.
- The closed-loop style extraction is poor on real photos (exact style 12.5%); outputs are labelled low-confidence.
- COVID data did not hurt and mildly helped (13 non-COVID origins, second model has ~40% fewer rows).
- Full-mode generation is reproducible with `python -m nss.generate.n9_generate`; the dry run covers wiring.
