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
| `evidence_chain.png` | Per concept: references, brief, Gate 1 (range), Gate 1b (nearest reference), attribute fidelity with its noise bound, verdict |
| `seasonal_comparison.png` | The same pipeline for two seasons: black jersey T-shirt (autumn/winter) beside a black swim bottom (summer) |
| `WRITEUP.md` | The argument, the five-stage validation of the quality gate, and the limits |
| `SUBMISSION_CHECKLIST.md` | This file |
| `README.md` | Points at the repository and states the one-command ways to run it |

## Evidence behind the claims (in the repository)

| File | Shows |
|---|---|
| `reports/tables/label_shuffle_control.csv` | Retraining on shuffled labels: 0 hits in 108 picks vs 0.722 unshuffled |
| `reports/tables/leave_one_out_control.csv` | Median rule passed 8/18 real articles; p90 passes 18/18 (15/18 held-out) |
| `reports/tables/clone_positive_control.csv`, `gate1b_nearest_reference.csv` | An exact copy passes the range check; the nearest-reference check fails it in all three styles |
| `reports/tables/colour_base_rate.csv`, `final_three_selection_log.csv` | Why the leaderboards are black-heavy (calibrated to the market) and the editorial selection rule with its skip log |
| `reports/tables/final_selection_h4.csv`, `j4_judge_repeats.csv` | The three verdicts and every input; judge repeats with noise bound |
| `reports/tables/seasonal_summer_*.csv`, `forecast_spotcheck.csv` | Summer concept inputs and checks; the #1 vs #1500 forecast spot-check |
| `tests/` | All tests pass (`uv run pytest tests`); includes the hero-stage regression, dry-run and control tests |
| `scripts/run_pipeline.py --dry-run` | All seven pipeline stages on CPU in about 1–2.5 minutes, writing only to a scratch directory |

## Known limits a reviewer should weigh

- **None of the three final concepts shows its briefed design change** (T-shirt: a grey-bodied colour-block tee the brief never asked for; sweater: generic beige V-neck; dress: coral-pink, not red). The T-shirt and sweater fail Gate 2 on the second judge; the dress passes every automatic gate at exactly Gemini's threshold and still fails the human brief check.
- Automatic checks passed visibly malformed candidates in earlier runs (three red-underwear, two summer); a human check is required.
- Fidelity scores are noisy (about ±0.21 across sessions; the summer concept was read by one judge only); the summer concept passes the attribute check only on visible attributes (0.562 vs 0.513; 0.375 counting the non-visual label "Other structure"; readings spread 0.25).
- The summer forecast ranked the right style first but overshot its level (93.1 predicted vs 50.0 realised).
- Full-mode generation was not re-run in the final session; the dry run covers wiring.
