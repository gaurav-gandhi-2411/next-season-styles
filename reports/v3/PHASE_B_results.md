# Phase B: results

SPEC.md Section 7, Phase B, with the B.0 bootstrap fix. Rules: `PREREGISTRATION.md` Sections Q
(`5d50ca2`), R (`ed8d409`) and R8 (`9105766`), every one committed before the result it governs.
Code: `nss.models.circular_bootstrap`, `nss.models.phase_b0_rebaseline`, `nss.models.phase_b`.
Tables: `reports/tables/phase_b0_*.csv`, `phase_b_*.csv`, `phase_b4_seed_variance.json`,
`phase_b5_*.{csv,json}`.

## Outcome

**No lever is adopted, so the champion (the current L2 LightGBM) stands.** None of B.1-B.4 is
significant under Holm on demand capture@20. The smallest p is 0.352, against a first threshold
of 0.0125. B.5's quantile model fails its coverage band (0.623 pooled against 0.75-0.85). This is
the result, not a failure to fix: four pre-registered mechanisms were tried under a test with
enough power to see a gain of about 0.07 capture points, and none produced one.

## B.0 Circular bootstrap (Section Q)

The old moving-block bootstrap under-sampled the first and last 12 of 48 origins. Under the
circular one (same L = 13, 2,000 resamples, seed 42) every origin has equal expected weight.
Old vs new intervals (`phase_b0_summary.csv`, `phase_b0_paired.csv`):

| quantity | mean | old interval | circular interval |
|---|---|---|---|
| Model Hit@3-in-top20 | 0.431 | [0.410, 0.618] | [0.229, 0.590] |
| Model demand capture@20 | 0.614 | [0.563, 0.699] | [0.519, 0.691] |
| Model NDCG@10 | 0.869 | [0.852, 0.907] | [0.825, 0.903] |
| Model − SN, capture@20 | +0.185 | [0.176, 0.245] | **[0.138, 0.236]** |
| Model − SN, capture@10 | +0.184 | [0.150, 0.266] | [0.131, 0.248] |
| Model − SN, capture@3 | +0.150 | [0.037, 0.319] | [0.023, 0.291] |
| Model − SN, Hit@3-in-top20 | +0.095 | [−0.040, 0.278] | [−0.063, 0.246] |
| Model − SN, tolerance hit@3 | −0.008 | [−0.103, 0.103] | [−0.095, 0.095] |
| Model − SN, NDCG@10 | +0.130 | [0.080, 0.226] | [0.062, 0.208] |
| Model − SN, Spearman | +0.200 | [0.192, 0.241] | [0.163, 0.235] |
| Model − SN, WMAPE | −0.041 | [−0.055, −0.034] | [−0.054, −0.024] |

The circular intervals are centred on the mean where the old ones were not. The MDE at 80% power
of the primary against seasonal-naive rises from 0.050 to 0.071. **Stop condition (Q3) not
triggered:** the champion's capture@20 lead over seasonal-naive keeps a lower bound above zero
(0.138).

## Reproduction

This module's own training path reproduces the champion's 141,916 weekly-origin predictions with
max |diff| 0.0 (`repro`), and its scoring reproduces every Phase A per-origin metric to 1e-12. Every
lever differs from the champion only in what the lever names.

## Holm decision (primary: demand capture@20, challenger − champion, 48 origins)

`phase_b_decisions.csv`. Ordered Holm thresholds for m = 4 at family-wise α 0.05:
**0.0125, 0.01667, 0.025, 0.05**.

| Holm rank | lever | mean diff | 95% circular CI | p (two-sided) | threshold | significant | guardrail failures | adopted |
|---|---|---|---|---|---|---|---|---|
| 1 | B.2 shrunk target | −0.0128 | [−0.0387, 0.0128] | 0.352 | 0.0125 | no | Spearman, WMAPE | no |
| 2 | B.3 reconciliation | +0.0006 | [−0.0012, 0.0027] | 0.625 | 0.0167 | no (step-down stopped) | none | no |
| 3 | B.1 visual momentum | −0.0016 | [−0.0159, 0.0092] | 0.863 | 0.025 | no (step-down stopped) | none; controls failed | no |
| 4 | B.4 seed ensemble | 0.0000 | [0, 0] | 1.000 | 0.05 | no | none | no |

Guardrails (95% paired circular intervals, `phase_b_paired.csv`):

| lever | Hit@3-in-top20 | NDCG@10 | Spearman | WMAPE (lower better) |
|---|---|---|---|---|
| B.1 | +0.035 [−0.007, 0.076] | −0.004 [−0.013, 0.006] | −0.002 [−0.007, 0.005] | −0.000 [−0.003, 0.002] |
| B.2 | +0.042 [0.007, 0.070] | −0.003 [−0.011, 0.005] | **−0.110 [−0.137, −0.074] FAIL** | **+0.014 [0.005, 0.023] FAIL** |
| B.3 | +0.007 [0.000, 0.021] | 0.000 [−0.001, 0.001] | +0.001 [0.000, 0.002] | −0.001 [−0.001, −0.000] |
| B.4 | 0 | 0 | 0 | 0 |

## Per lever

**B.1 Visual momentum: rejected.** The per-article embeddings existed (22,468 articles, 3,003
styles, CLIP and DINOv2). First-sale dates did not; they were computed from the transactions
(minimum `t_dat`, CPU, seconds). At each origin, only articles first sold before it enter a style's
vector: 4,412 embedded articles were first sold after the first test origin and are excluded
origin by origin. The feature is present on about 97.5% of eval rows and NaN (missing) on the rest.
Neighbour momentum is trailing-only (4-week vs 13-week EWMA of `intensity_shrunk`, as of the
origin), and the neighbour pool is causal (R8). Result on the primary: −0.0016 [−0.0159, 0.0092].
**Controls:**

- *Causality shuffle test* (feature permuted within origin): +0.0067 [−0.0007, 0.0145]. Passes.
- *Negative control* (10 random neighbours, equal weights): **+0.0088 [0.0016, 0.0176]**, lower
  bound above zero, so the control rule fails.

B.1 would not be adopted even had it been significant. The random-neighbour control doing slightly
better than the visual lever says the visual similarity adds nothing here. The small gain from
averaging 10 random styles' momentum is a post-hoc observation, not a claim. Averaging 10 random
styles plausibly approximates market-wide momentum, and that is a candidate mechanism for a future
pre-registered lever, not a finding.

**B.2 Shrunk-intensity target: rejected.** Evaluated on the same raw `y_true` as the champion.
Primary −0.0128 [−0.0387, 0.0128], not significant. It fails two guardrails: Spearman −0.110 and
WMAPE +0.014. Training on a shrunk label compresses the predictions towards group means. That was
the risk R3 named in advance. Hit@3-in-top20 rises by +0.042 [0.007, 0.070], which is a guardrail
reading, not a criterion.

**B.3 OLS reconciliation: rejected.** Parent forecasts from the same locked LightGBM on aggregated
panels (product type within index group, index group), reconciled in raw-intensity space with
article-count weights. The style forecasts barely move: primary +0.0006 [−0.0012, 0.0027]. It fails
no guardrail and improves WMAPE slightly (−0.0009 [−0.0014, −0.0004]), but it is nowhere near
significant on the primary.

**B.4 Seed ensemble: rejected, and degenerate as predicted in R5.** Seeds 42-51 give bit-identical
predictions: max |diff| 0.0 over 141,916 rows, seed-induced variance exactly 0, variance reduction
not applicable (`phase_b4_seed_variance.json`). The locked config has no row or feature subsampling,
and training is deterministic, so the seed has no effect on the model. A seed ensemble here would
need stochastic hyperparameters, which the locked-config rule forbids. This result is about the
champion's configuration, not about ensembling.

## B.5 Quantile output: not accepted

`phase_b5_summary.json`, `phase_b5_coverage_per_origin.csv`. q10 / q50 / q90, LightGBM `quantile`
objective, locked config, 48 embargoed weekly origins.

| check | result | required | pass |
|---|---|---|---|
| Pooled q10-q90 coverage (141,916 rows) | **0.623** | [0.75, 0.85] | **no** |
| Per-origin coverage, mean [95% circular CI] | 0.626 [0.545, 0.692] | reported | |
| Per-origin coverage, min / max | 0.395 / 0.784 | reported | |
| Crossed intervals (q10 > q90) | 0.006% | reported | |
| q50 − champion, capture@20 | −0.0003 [−0.0157, 0.0154] | CI not entirely below 0 | yes |

The 10-90 interval is too narrow: it covers 62% where it should cover 80%. The median model is
non-inferior on the primary, and it improves WMAPE (−0.0037 [−0.0052, −0.0024]). Per the
pre-registration nothing is recalibrated here. A calibrated interval, such as conformal adjustment
on the embargoed residuals, would need its own pre-registered rule.

## Step 7

No lever was adopted, so no combination is built or tested (R7). **The champion stands.**

## Compute

All CPU; no GPU step and no paid API, so $0. 18 runs on the embargoed 48-origin path (`repro` plus
17 lever and control runs), 38-160 s each, 1,390 s of single-threaded compute in total, run at most
4 in parallel. `data/generated/phase_b_*_timing.json` has each run's time.

## Caveats

1. B.3's parent-level intensity treats `intensity_shrunk` as raw intensity at the parent, and its
   aggregation weights use origin-time article counts for a forward window (both stated in R4).
2. B.1's embedded articles per style were chosen during the concept work (at most 8 per style,
   plus screened reference sets), partly with later data. The first-sale filter removes future
   articles but not that selection (stated in R2).
3. The three first B.1 runs failed with out-of-memory errors in a 4-way parallel batch before
   training started. They were re-run after a memory fix that does not change the computation, and
   are the only B.1 runs used.
