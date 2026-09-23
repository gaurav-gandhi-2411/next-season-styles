# Phase A: measure properly

SPEC.md Section 7, Phase A. Rules: `PREREGISTRATION.md` Section P, committed at `30cc523` before
any metric below was computed on any model, baseline or floor. Code: `nss.models.phase_a_measure`.
All tables: `reports/tables/phase_a_*.csv`, produced by
`uv run python -m nss.models.phase_a_measure` (and `--matched` for the post-hoc check).

## Verdict

- **Phase B primary metric: demand capture@20.** It was chosen by the pre-registered rule: the
  smallest normalised MDE among the eligible candidates. All five candidates passed the relevance
  check.
- **The model's lead over seasonal-naive is demonstrable under it:** +0.185, 95% CI [0.176, 0.245],
  paired over 42 origins, block length 13, ESS 14.0 (autocorrelation) / 23.2 (bootstrap). The model
  captures 0.614 of the true top-20's demand, seasonal-naive 0.473, the random floor 0.114.
- **At the extreme top the lead is not demonstrable, and there is no sign of one.** Model minus
  seasonal-naive: tolerance hit@3 −0.008 [−0.103, 0.103], Precision@3 −0.024 [−0.103, 0.048],
  Hit@3-in-top20 +0.095 [−0.040, 0.278]. The model beats seasonal-naive on how much demand its top
  picks capture, not on exactly which three styles come out on top.

## Reproduction

No prediction frame survived the 2026-09-23 data loss. The model's 48 weekly-origin predictions
were regenerated with the locked `FINAL_MODEL_CONFIG`. Before any new metric was computed, every
existing metric, for every method at every origin (288 rows), was checked against the committed
`v3_power_per_origin_weekly.csv` and matched to 1e-9. These are the predictions already reported,
not a new model. The frame is saved at
`data/generated/phase_a_model_predictions_w48_2019-07-29_2020-06-22.parquet` for Phase B.

## The tolerance margin (pre-registration input, labels only)

`phase_a_near_tie.csv`: over the 48 weekly origins, the raw-intensity gap between the true #3 and
#4 styles is median 3.78%, p90 5.79%, max 17.9%. **Margin m = 0.057911** (p90). The SPEC's "0.61%"
was measured on the log1p target over the 12 grid origins (on the 48 weekly origins the log1p
gap is 0.82%), so it understates the gap in demand units by roughly a factor of 5.

## A.2 Power table: model minus seasonal-naive

`phase_a_power.csv`. Paired over the 42 origins where seasonal-naive has a 52-week lag; moving-block
bootstrap, block 13, 2,000 resamples, seed 42. MDE = 2.8016 × bootstrap SE (two-sided α 0.05, 80%
power).

| metric | mean diff | 95% CI | CI half-width | ESS (ac / boot) | MDE@80% |
|---|---|---|---|---|---|
| hit_at_3_in_top20 | +0.095 | [−0.040, 0.278] | 0.159 | 15.3 / 15.3 | 0.237 |
| hit_at_3_in_top10 | 0.000 | [−0.087, 0.159] | 0.123 | 18.8 / 16.7 | 0.175 |
| precision_at_3 | −0.024 | [−0.103, 0.048] | 0.075 | 33.0 / 23.3 | 0.108 |
| precision_at_10 | +0.012 | [−0.024, 0.045] | 0.035 | 41.7 / 35.6 | 0.049 |
| ndcg_at_10 | +0.130 | [0.080, 0.226] | 0.073 | 6.1 / 6.4 | 0.105 |
| spearman_rho | +0.200 | [0.192, 0.241] | 0.024 | 9.2 / 22.9 | 0.035 |
| wmape (lower is better) | −0.041 | [−0.055, −0.034] | 0.010 | 8.5 / 23.9 | 0.015 |
| demand_capture_at_3 | +0.150 | [0.037, 0.319] | 0.141 | 9.1 / 8.4 | 0.205 |
| demand_capture_at_10 | +0.184 | [0.150, 0.266] | 0.058 | 11.6 / 12.6 | 0.084 |
| **demand_capture_at_20** | **+0.185** | **[0.176, 0.245]** | **0.035** | **14.0 / 23.2** | **0.050** |
| tolerance_hit_at_3 | −0.008 | [−0.103, 0.103] | 0.103 | 30.4 / 22.1 | 0.149 |

## A.4 Primary-metric selection

`phase_a_primary.csv`. Normalised MDE = MDE / (1 − random-floor mean).

| candidate | floor | R2 floor ≤ 0.5 | R3 ≥ 8 paired | MDE@80% | normalised |
|---|---|---|---|---|---|
| tolerance_hit_at_3 | 0.002 | pass | pass | 0.149 | 0.149 |
| demand_capture_at_3 | 0.084 | pass | pass | 0.205 | 0.224 |
| hit_at_3_in_top20 | 0.006 | pass | pass | 0.237 | 0.239 |
| demand_capture_at_10 | 0.100 | pass | pass | 0.084 | 0.093 |
| **demand_capture_at_20** | 0.114 | pass | pass | 0.050 | **0.056** |

Demand capture@20's MDE is about a quarter of Hit@3-in-top20's in normalised terms (0.056 vs
0.239). Under it, Phase B can detect a paired gain of about 0.05 capture points, where Hit@3 needed
about 0.24.

## A.3 Full metric table (48 weekly origins; seasonal-naive 42)

`phase_a_summary.csv`, mean [95% block CI].

| metric | model | seasonal-naive | EWMA | parent-cat. mean | global mean | random floor |
|---|---|---|---|---|---|---|
| hit_at_3_in_top20 | 0.431 [0.410, 0.618] | 0.397 [0.230, 0.619] | 0.271 [0.208, 0.424] | 0.035 [0.000, 0.104] | 0.000 | 0.006 [0.001, 0.010] |
| hit_at_3_in_top10 | 0.278 [0.208, 0.493] | 0.317 [0.183, 0.492] | 0.181 [0.153, 0.292] | 0.021 [0.000, 0.062] | 0.000 | 0.004 [0.001, 0.008] |
| precision_at_3 | 0.049 [0.014, 0.104] | 0.079 [0.000, 0.206] | 0.007 [0.000, 0.021] | 0.000 | 0.000 | 0.002 [0.001, 0.005] |
| precision_at_10 | 0.169 [0.140, 0.260] | 0.181 [0.114, 0.276] | 0.133 [0.115, 0.194] | 0.017 [0.000, 0.050] | 0.000 | 0.004 [0.003, 0.004] |
| ndcg_at_10 | 0.869 [0.852, 0.907] | 0.760 [0.659, 0.830] | 0.807 [0.760, 0.866] | 0.272 [0.190, 0.410] | 0.204 [0.182, 0.219] | 0.403 [0.388, 0.410] |
| spearman_rho | 0.710 [0.640, 0.779] | 0.540 [0.492, 0.579] | 0.594 [0.543, 0.623] | 0.247 [0.216, 0.275] | undefined | 0.001 [−0.001, 0.002] |
| wmape | 0.166 [0.134, 0.195] | 0.190 [0.182, 0.204] | 0.195 [0.172, 0.224] | 0.246 [0.232, 0.264] | 0.306 [0.294, 0.324] | 0.484 [0.479, 0.502] |
| demand_capture_at_3 | 0.569 [0.514, 0.684] | 0.461 [0.292, 0.652] | 0.459 [0.385, 0.585] | 0.062 [0.014, 0.153] | 0.011 [0.008, 0.012] | 0.084 [0.077, 0.092] |
| demand_capture_at_10 | 0.598 [0.562, 0.683] | 0.457 [0.358, 0.534] | 0.492 [0.411, 0.586] | 0.088 [0.044, 0.154] | 0.019 [0.015, 0.022] | 0.100 [0.093, 0.104] |
| demand_capture_at_20 | 0.614 [0.563, 0.699] | 0.473 [0.403, 0.523] | 0.500 [0.408, 0.595] | 0.091 [0.059, 0.121] | 0.019 [0.017, 0.021] | 0.114 [0.106, 0.118] |
| tolerance_hit_at_3 | 0.132 [0.021, 0.299] | 0.159 [0.000, 0.397] | 0.062 [0.021, 0.139] | 0.000 | 0.000 | 0.002 [0.001, 0.005] |

Paired model-minus-comparator differences for every comparator are in `phase_a_paired.csv`. On
demand capture@20 the model's lead excludes zero against every comparator: EWMA +0.114 [0.098,
0.169], parent-category mean +0.523, global mean +0.594, random floor +0.500.

## Post-hoc sensitivity (not pre-registered, not part of the verdict)

The pre-registered convention scores seasonal-naive only on styles with 52 weeks of history, but
scores the model on its full eval set. Re-scoring the model on seasonal-naive's own population
(`phase_a_sensitivity_matched.csv`, from the saved predictions, no retraining):

| metric | model − seasonal-naive, matched population |
|---|---|
| demand_capture_at_20 | +0.204 [0.194, 0.271] |
| demand_capture_at_10 | +0.201 [0.163, 0.291] |
| demand_capture_at_3 | +0.174 [0.041, 0.349] |
| hit_at_3_in_top20 | +0.151 [0.024, 0.302] |
| tolerance_hit_at_3 | 0.000 [−0.103, 0.103] |

The verdict does not depend on the population convention; matching populations enlarges the lead.
Hit@3-in-top20 also excludes zero on the matched population. That is a post-hoc observation and is
not claimed as a result.

## Caveats

1. **The bootstrap is off-centre at the edges.** The project's moving-block bootstrap (unchanged
   here, the same one behind every SPEC number) has no wrap-around. With n = 48 and block 13,
   block starts are drawn from 0–35, so the first and last origins are under-sampled. The CIs sit
   visibly off-centre: the model's Hit@3-in-top20 is 0.431 [0.410, 0.618]. Phase B should
   pre-register a circular block bootstrap. It cannot change this verdict: the primary lead's lower
   bound is 0.176, against an MDE of 0.050.
2. **The primary metric is broad.** The minimum-MDE rule chose k = 20 over k = 3, as pre-registered.
   Demand capture@20 measures how much demand a 20-style range captures, not whether the three
   headline bets are right. Hit@3-in-top20 stays a guardrail, so a challenger cannot win on
   capture@20 while losing significantly at the head.
3. **The MDE is for model vs seasonal-naive.** Phase B compares challenger vs champion, two
   correlated models, whose paired variance should be smaller. The table is a conservative guide.
4. **Overlapping windows.** Adjacent weekly origins share 12 of 13 forecast weeks. The effective
   sample for the primary difference is 14–23, not 42.
