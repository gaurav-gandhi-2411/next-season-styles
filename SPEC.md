# next-season-styles — Engineering Spec

**Owner:** GG · **Working branch:** `feat/v3-improvements` · **Last updated:** 2026-09-23

Read this file, then `CLAUDE.md`, before starting any work. Section 9 records where the work stands;
update it at the end of every session.

---

## 1. Purpose

Two goals, in this order:

1. **A forecasting model demonstrably better than the submitted one**, proven under a metric with
   enough statistical power to detect the difference. "Better" that cannot be measured does not count.
2. **Production grade**: the forecast and concept pipeline runs reproducibly, observably and safely
   outside a research workflow, on GCP.

## 2. Where things stand

The forecasting model is the one submitted: L2 LightGBM on the style-week panel. Every improvement
tried since was rejected under its pre-stated rule. What changed is the rigour of measurement.

| Measure | Value |
|---|---|
| Hit@3-in-top20, embargoed | 0.528 |
| vs seasonal-naive, 4-week grid (10 origins) | +0.233 [0.000, 0.567] |
| vs seasonal-naive, 48 weekly origins | +0.095 [−0.040, +0.278], effective N ≈ 15 |
| NDCG@10 vs seasonal-naive | +0.156 [0.099, 0.254] |
| Spearman vs seasonal-naive | +0.218 [0.203, 0.253] |
| WMAPE vs seasonal-naive | −0.045 [−0.054, −0.033] |
| Random floor, Hit@3-in-top20 | 0.0069 |
| Label-shuffle leakage control | 0 hits in 108 picks |
| Mean gap between true #3 and #4, as % of #3 | 0.61% |

**The binding limit is measurement, not capability.** With an effective sample of about 15 origins, the
CI half-width on Hit@3-in-top20 is about 0.16. Any real gain smaller than that is undetectable here.
The model is already significantly better than every baseline on NDCG, Spearman and WMAPE, which
average over all styles; only the extreme top-k is underpowered.

**Already tried and rejected** (do not repeat without a new mechanism): neighbourhood cohort features;
buyer-mix and price-elasticity features; external search signal (coincident with sales, not leading);
LambdaRank, top-heavy weighting, two-stage re-ranking; an L2 + LambdaRank ensemble (statistically
tied); a growth-ranking redesign; selective prediction by quantile spread; embargoed hyperparameter
re-tuning (0.65% gain, inside grid noise).

**Known finding about the data:** realised growth on this dataset is itself largely mean reversion
(Spearman −0.42 against trailing intensity). No growth target definable here is free of that effect.

## 3. Hard constraints

1. `main` is frozen at `e7910bd`. Never commit to it.
2. `reports/SUBMISSION/` is never modified. Check with `git diff main -- reports/SUBMISSION/`, not raw
   byte hashes — `core.autocrlf=true` makes working-tree hashes misleading.
3. `CLAUDE.md` DATA SAFETY rules are binding. A command's refusal is a stop signal, never an obstacle.
4. Free data and free compute only. Report any cost before incurring it.
5. No `ANTHROPIC_API_KEY` anywhere. LLM orchestration runs through Claude Code on the Max plan.
6. Every generated filename encodes style, scale and seed.

## 4. Method rules

1. **Pre-register.** Any comparison with a pass/fail outcome has its decision rule committed before the
   result exists. The rule's commit must predate the result's commit. Never squash or rebase history —
   it destroys that evidence.
2. **Never change a rule after seeing results.** A negative result is a result. Do not tune toward a win.
3. **Every result carries its floor and its uncertainty**: a random-permutation floor, paired
   block-bootstrap CIs (block length 13), and the effective sample size.
4. **Embargo.** Training labels must not overlap the test window: 13-week horizon, 13-week gap.
5. **Causality.** Every new feature ships with a shuffle test and a negative control.
6. **LLM grading** uses at least two model families, blind, with inter-judge agreement reported, labelled
   LLM-consensus rather than ground truth. Judges score executed actions, not stated reasoning.
7. **Report measured versus assumed** explicitly. Numbers, not adjectives.

## 5. Definitions

- **Style:** the 5-column key — index group, product type, garment group, perceived colour master,
  graphical appearance.
- **Target:** units per active article per week (intensity), over a 13-week horizon.
- **Evaluation:** 48 weekly origins, embargoed; paired against seasonal-naive, EWMA persistence,
  parent-category mean and global mean, plus the random floor.
- **Hit@3-in-top20:** fraction of the model's top-3 picks inside the realised top-20.
- **Demand capture@k** *(introduced in Phase A):* realised intensity of the model's top-k divided by
  that of the true top-k, per origin.
- **Tolerance hit@3** *(introduced in Phase A):* a pick counts if its realised intensity lies within a
  margin of the true #3, the margin derived from the measured near-tie distribution.

## 6. Champion/challenger rule

A challenger replaces the champion only if both hold:

1. Its paired improvement on the **primary metric** has a CI excluding zero in its favour. The primary
   metric is fixed at the end of Phase A, from the power analysis.
2. No **guardrail metric** — Hit@3-in-top20, NDCG@10, Spearman, WMAPE — is significantly worse.

Otherwise the champion stays and the challenger is recorded as a negative result.

## 7. Phases

### Phase 0 — Protect the data
Copy the master dataset to `C:\Users\gaura\hm-data\`, outside every git repository; verify it; apply an
NTFS ACL denying deletion; point this project at it via `NSS_HM_IMAGE_TREE`. Nothing is deleted.

**Done when:** file count, size and CSV hashes match the source; a delete attempt inside the folder
fails while reads succeed; the ACL undo command is documented and tested; retrieval validation
reproduces 27.5 / 62.5 / 75 at 1,980 candidates.

### Phase A — Measure properly
Hit@3 is binary per pick and scores the true #4 as a total miss despite a 0.61% gap to #3. Introduce
demand capture@k and tolerance hit@3, pre-registered, with Hit@3 still reported.

**Done when:** the pre-registration is committed; a power table gives each metric's minimum detectable
effect at 80% power; the current model and all baselines are re-scored on every metric; the primary
metric for Phase B is fixed.

### Phase B — Improve the model
Each lever is pre-registered and judged by the champion/challenger rule.

| Lever | Status before Phase B |
|---|---|
| Visual momentum from the 22,366 CLIP/DINOv2 embeddings | Never used by the forecaster |
| Shrunk intensity as the forecast target | Head stability +0.12 top-3 overlap, paired p = 0.149 (not significant); never tested as the target in a full backtest |
| Hierarchical forecast reconciliation across the attribute hierarchy | Untested |
| Seed / bagging ensemble | Untested (the L2 + LambdaRank ensemble was tested and tied) |
| Quantile output (q10, q50, q90) | Untested; required for production |

**Done when:** every lever has a recorded adopt or reject decision, and the champion is identified.

### Phase C — Production grade (needs a one-time GCP sign-in by GG)
- Data contracts: schema validation on every input.
- Config-driven, tracked, reproducible training.
- Model registry with champion/challenger promotion enforced in CI.
- Forecast API and batch job on Cloud Run.
- Drift monitoring on features and predictions; performance monitoring once actuals arrive.
- CI on every change: tests, lint, secret scan, and the champion/challenger gate.
- GCS with object versioning and soft delete as the data layer.
- Secrets in Secret Manager, never in files.

**Done when:** a clean clone trains the champion reproducibly; the API serves forecasts with intervals;
a challenger that regresses is blocked by CI; a data-restore drill from GCS succeeds.

### Phase D — Second retailer *(after the interview)*
A second corpus and taxonomy transfer. The only real fix for the sample-size ceiling.

## 8. Out of scope for now

Merging to `main`; regenerating the submitted concepts; the `nss-core` package extraction (after the
interview); modifying `agentic-shopping-assistant` or `multimodal-fashion-recommender`.

## 9. Status

Update at the end of every session.

| Phase | Status | Evidence |
|---|---|---|
| 0 | Not started | — |
| A | Not started | — |
| B | Not started | — |
| C | Not started | — |
| D | Deferred until after the interview | — |

## 10. Decision log

| Date | Decision |
|---|---|
| 2026-09-20 | Submission sent. `main` frozen at the submitted state plus a corrections section. |
| 2026-09-23 | Growth ranking found to be largely mean reversion; submitted concepts kept, disclosed in README. |
| 2026-09-23 | Agent layer made real: LLM orchestration via Claude Code; prior human rejection made terminal. |
| 2026-09-23 | Gate 2 measures colour and uses retrieval for product type; Gate 1 made advisory. |
| 2026-09-23 | Data loss via `git worktree remove --force` following a junction; recovered; rules in `CLAUDE.md`. |
| 2026-09-23 | No merge to `main` yet. Next: Phases 0 and A. |
