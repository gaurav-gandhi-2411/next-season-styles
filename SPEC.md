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
| Gap between true #3 and #4 in raw intensity, as % of #3 (48 weekly origins) | median 3.8%, 90th percentile 5.8% (the earlier "0.61%" was a mean on the log1p scale over 12 grid origins) |

**The binding limit on this data is capability, not measurement.** Phase A replaced Hit@3 (CI
half-width about 0.16) with demand capture@20 as the primary. Paired against the champion, Phase B
could detect small gains: MDE at 80% power 0.003 for B.3 (interval [−0.0012, 0.0027]) and 0.019
for B.1 ([−0.016, 0.009]), 0.038 for B.2. None of the four levers produced a gain
(`reports/v3/PHASE_B_results.md`). The extreme head (Hit@3, tolerance hit@3) remains
underpowered, but the primary is not, and on it the model is at its ceiling for the mechanisms
tried here. The champion leads seasonal-naive on capture@20 by +0.185 [0.138, 0.236] and is
significantly better than every baseline on NDCG, Spearman and WMAPE.

**B.4 was a no-op by construction.** The locked config does no row or feature subsampling and
trains deterministically, so seeds cannot change the model: seeds 42-51 gave bit-identical
predictions. A seed ensemble needs stochastic hyperparameters, which the locked-config rule
forbids.

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
Hit@3 is binary per pick and scores the true #4 as a total miss despite a median 3.8% gap to #3. Introduce
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
A second corpus and taxonomy transfer. It tests whether the capability ceiling found in Phase B is
specific to this dataset, and adds independent origins for confirmatory tests.

### Open items for main

Not acted on until GG approves updating `main`; no public text is drafted here.

- **Near-tie gap stated on the log scale.** The "0.61%" gap between the true #3 and #4 is a mean
  on the log1p target; in raw intensity it is median 3.8%, 90th percentile 5.8%. Checked on
  2026-09-23 with `git grep` against `main`: the figure is **not** in the emailed `WRITEUP.md`
  or in `README.md`. On `main` it appears only in the docstring of
  `src/nss/models/lambdarank_model.py` (lines 10–11). A correction goes into `main`'s corrections
  section only if GG approves, and only for where the figure actually appears.

## 8. Out of scope for now

Merging to `main`; regenerating the submitted concepts; the `nss-core` package extraction (after the
interview); modifying `agentic-shopping-assistant` or `multimodal-fashion-recommender`.

## 9. Status

Update at the end of every session.

| Phase | Status | Evidence |
|---|---|---|
| 0 | **Done** 2026-09-23 (`6e7d0d4`) | Copy at `C:\Users\gaura\hm-data`: 105,104 files, 34,558,584,597 bytes; path set and per-file sizes match the source, SHA-256 of all 4 CSVs match, 500 seed-42 random images match. Deny (DE,DC) ACL applied; a delete inside fails and reads succeed; apply/undo tested on a scratch folder first, commands in `CLAUDE.md`. `NSS_HM_IMAGE_TREE` set and canonical path first in `TREE_CANDIDATES`; retrieval validation reproduces 27.5 / 62.5 / 75 at 1,980 candidates (zero diff on `retrieval_validation.csv`). |
| A | **Done** 2026-09-23 (pre-reg `30cc523`, results `7a65786`) | `reports/v3/PHASE_A_measurement.md`. Primary for Phase B: **demand capture@20** (normalised MDE 0.056 vs 0.239 for Hit@3-in-top20). Model − seasonal-naive on it: **+0.185 [0.176, 0.245]**, 42 paired origins, ESS 14.0/23.2, so the lead is demonstrable. No lead at the extreme head: tolerance hit@3 −0.008 [−0.103, 0.103], Hit@3-in-top20 +0.095 [−0.040, 0.278]. All 288 existing (origin, method) metrics reproduced to 1e-9. |
| 0 (read-only) | **Done** 2026-09-23 (`029ad63`) | ACL extended to deny WD, AD, WEA, WA, DE, DC. Tested with its undo on a scratch folder first. On `hm-data` an image reads; opening it for write fails with its hash unchanged; creating, overwriting and deleting files all fail. Commands in `CLAUDE.md`. |
| B | **Done** 2026-09-23. B.0 pre-reg `5d50ca2`, results `f3e92dc`; Phase B pre-reg `ed8d409` + amendment `9105766`; results `c744a70` | `reports/v3/PHASE_B_results.md`. B.0: circular bootstrap; the champion's capture@20 lead over seasonal-naive is +0.185 [0.138, 0.236], so the stop condition is not triggered. Holm (0.0125/0.01667/0.025/0.05): **no lever adopted, the champion stands.** B.1 −0.0016 [−0.016, 0.009], and its random-neighbour control beats the champion, so the controls fail. B.2 −0.013 and fails the Spearman/WMAPE guardrails. B.3 +0.0006 [−0.0012, 0.0027]. B.4 seeds bit-identical (deterministic, no subsampling). B.5 coverage 0.623, outside [0.75, 0.85], so not accepted; q50 non-inferior. No combination. |
| B.5 calibration | **Done** 2026-09-23. Pre-reg `a76adae`; results `5bdcf58`, off-by-one fix `1e73623` | `reports/v3/PHASE_S_T_calibration_and_exploratory.md`. Rolling asymmetric CQR (4 origins, embargoed 13 weeks) on the unchanged quantiles: **pooled coverage 0.803**, target [0.75, 0.85] met, chosen over symmetric (0.805) as narrower (log1p width 1.808 vs 1.825). COVID 0.837, non-COVID 0.748. Only 8 of 48 origins in band (18 below, 22 above, range 0.58-0.98): the 13-16 week calibration lag over-corrects with delay. Median raw width 11.6 → 16.0. q50 unchanged. |
| Market momentum (EXPLORATORY) | Done 2026-09-23 (`812188f`), not adoptable | Explicit market momentum −0.0023 [−0.019, 0.014]; noise column −0.0007; both (a) controls null; the random-neighbour re-run reproduces bit for bit (+0.0088). Reading: the B.1 result does not reproduce as a market effect. Not added to Phase D. |
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
| 2026-09-23 | Canonical H&M data at `C:\Users\gaura\hm-data` under a deny-delete ACL; the `multimodal-fashion-recommender` copy is kept as a backup. |
| 2026-09-23 | Tolerance margin = p90 of the raw-intensity #3-vs-#4 gap = 0.057911. The "0.61%" in Section 2 is on the log1p target (12 grid origins); in raw intensity the gap is median 3.8%, p90 5.8%. |
| 2026-09-23 | Primary-metric rule compares MDEs normalised by range above the random floor, since raw MDEs are not comparable across metrics. |
| 2026-09-23 | **Phase B primary metric: demand capture@20.** Guardrails unchanged (Hit@3-in-top20, NDCG@10, Spearman, WMAPE). Phase B paired comparisons use block length 13 on the 48 weekly origins. |
| 2026-09-23 | Open for Phase B's pre-registration: the existing block bootstrap is non-circular and under-samples edge origins, so CIs sit off-centre; pre-register a circular block bootstrap before the first challenger. |
| 2026-09-23 | `hm-data` made fully read-only (write, create and delete denied). |
| 2026-09-23 | Near-tie gap restated on the raw scale. On `main` the 0.61% figure appears only in the `lambdarank_model.py` docstring, not in `WRITEUP.md` or `README.md`; recorded under Open items for main. |
| 2026-09-23 | **All intervals now use the circular block bootstrap** (L = 13, 2,000 resamples, seed 42). The primary's MDE vs seasonal-naive is 0.071 (was 0.050 under the old scheme). |
| 2026-09-23 | Phase B pre-registration amended before training: the B.1 neighbour pool is the causal panel at the origin, not the survival-filtered model frame. |
| 2026-09-23 | **Phase B: the champion stands.** B.1-B.4 are recorded as negative results. B.4 cannot work with the locked deterministic config. B.5 is not accepted (interval under-covers: 0.623). |
| 2026-09-23 | Open for the next pre-registration, not acted on: (1) calibrating the B.5 interval (e.g. conformal on embargoed residuals), needed before Phase C serves intervals; (2) market-wide momentum, suggested post hoc by B.1's random-neighbour control. Each needs its own rule before any run. |
| 2026-09-23 | SPEC Section 2 restated: the binding limit on this data is capability, not measurement (paired MDE 0.003-0.038 in Phase B, no lever gained). |
| 2026-09-23 | **Production interval: rolling asymmetric CQR** on the B.5 quantiles, calibrated on the 4 most recent fully embargoed origins. It meets 80% coverage pooled, not per week (0.58-0.98); consumers must be told. Reducing the lag needs its own pre-registered rule. |
| 2026-09-23 | Pre-registered conformal quantile named a numpy call that disagreed with its own formula by one order statistic; fixed to the formula, effect <= 0.0005, disclosed. |
| 2026-09-23 | Market momentum closed as an exploratory negative: it does not explain B.1's random-neighbour gain, which stands as one unadjusted comparison among ~8 (p = 0.011). No Phase D hypothesis added. |
