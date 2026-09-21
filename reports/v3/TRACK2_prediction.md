# Track 2: predicting the winners

Branch `feat/v3-improvements`. Every rule below was committed in `reports/v3/PREREGISTRATION.md` before the run it governs. Every comparison carries a random floor and paired block-bootstrap intervals.

## 2a. Growth backtest: PARTIAL

**Headline, stated plainly.** The growth ranking that picks the two emerging concepts of the final three beats *chance* by a wide margin, but it does **not** beat any of the four simple baselines on top-k. On Hit@3-in-top20 it is statistically indistinguishable from seasonal naive, EWMA persistence, the parent-category mean and the global mean. So the emerging half of the selection rests on a ranking that is validated against random and not shown to be better than simple baselines. It is not "unvalidated" in the sense of being no better than chance, and it is not "validated" in the sense the rule required.

**What was tested.** At each embargoed test origin (same model, locked shipped config, training origins at least 16 weeks earlier), the deployed selection quantity was rebuilt: `growth = expm1(prediction) / trailing 13-week mean intensity`, for the model and for each baseline's own forecast, and ranked against the realised `expm1(y_true) / trailing mean` over the same population. Primary population (P2) is the pool the emerging list is actually drawn from: passes guards 1-3, trailing mean above zero, and predicted intensity at or above the median of the guard-passing styles (median 130 styles per origin). P1 drops the model-derived floor (median 261 styles).

**Only 10 of the 12 origins can be evaluated.** At the first two test origins (2019-07-29 and 2019-08-26) no style has a `price_index` yet (it needs 52 weeks of history), so guard 2 fails every style. The deployed pipeline would return an empty list there too. Everything below is on 10 origins; seasonal naive is also defined on those 10.

**Hit@3-in-top20, pooled over 10 origins (P2), mean and 95% block-bootstrap interval:**

| Method | Hit@3-in-top20 |
|---|---|
| seasonal naive | 0.800 [0.733, 0.900] |
| **model (LightGBM)** | **0.767 [0.733, 0.867]** |
| parent-category mean | 0.767 [0.700, 0.933] |
| global mean | 0.733 [0.633, 0.967] |
| EWMA persistence | 0.633 [0.467, 0.967] |
| **random floor** | **0.177 [0.170, 0.205]** |

**Paired, model minus comparator (P2).** Rule: beat a comparator only if the Hit@3-in-top20 interval excludes zero (A), or both NDCG@10 and Spearman intervals exclude zero with Hit's mean not worse (B).

| Comparator | Hit@3 top 20 | NDCG@10 | Spearman | WMAPE (lower is better) | Beats? |
|---|---|---|---|---|---|
| seasonal naive | -0.033 [-0.100, +0.100] | -0.005 [-0.053, +0.052] | -0.025 [-0.115, +0.050] | -0.045 [-0.108, +0.028] | no |
| EWMA persistence | +0.133 [-0.133, +0.267] | +0.074 [-0.059, +0.128] | +0.176 [-0.007, +0.255] | +0.002 [-0.026, +0.038] | no |
| parent-category mean | -0.000 [-0.100, +0.033] | +0.066 [+0.009, +0.088] | +0.205 [+0.183, +0.225] | -0.134 [-0.180, -0.071] | no, by a float artefact (see below) |
| global mean | +0.033 [-0.133, +0.100] | +0.069 [+0.012, +0.091] | +0.194 [+0.174, +0.213] | -0.224 [-0.285, -0.146] | yes, by rule B only |
| random floor | +0.590 [+0.557, +0.663] | +0.378 [+0.355, +0.451] | +0.614 [+0.514, +0.744] | -0.219 [-0.337, -0.168] | yes |

The machine-computed verdicts are in `reports/tables/v3_growth_decision_P2.csv`: the model beats the random floor and the global mean (rule B), and does **not** beat seasonal naive or EWMA persistence.

**The parent-category row is decided by a float artefact, disclosed.** Model and parent-category mean both have Hit@3-in-top20 = 23/30 = 0.7667, so their paired mean difference is mathematically zero; in floating point it is -5.6e-18. Rule B requires "Hit@3 mean not worse (>= 0)", which that value fails, so the rule as pre-registered and coded says "does not beat". Read as zero, rule B would say "beats" (NDCG +0.066 [+0.009, +0.088], Spearman +0.205 [+0.183, +0.225] both exclude zero). I did not change the rule. The category does not depend on it: seasonal naive and EWMA persistence are not beaten either way. This is a defect in how the rule was written; later rules in this pass carry a 1e-9 tolerance on equality.

**Category: PARTIAL.** The model beats the floor but not all four baselines. P1, the sensitivity population, gives the same category.

**Why the ranking beats chance, and why that is weak evidence for the model.** The deployed growth ratio divides every method's forecast by the same trailing mean. So any forecast, even a constant, ranks styles by *low recent intensity first*: the global-mean baseline's growth ranking is just "lowest trailing intensity", and it scores 0.733 against the floor's 0.177. Most of the growth ranking's skill is therefore a base effect (styles with a low recent level tend to grow), which the ratio builds in. The model adds a little on rank correlation and error (better Spearman and WMAPE than the global and parent-category means) but nothing demonstrable on top-k over the base effect.

**Lift of the top-3 picks** (mean realised growth of the method's top 3 over the population mean; 1.0 is chance, P2): model 2.24 [1.92, 2.89], seasonal naive 2.30, parent-category mean 2.25, global mean 2.25, EWMA 2.01. Every method's picks grow about 2.2 times faster than the population; the model is not distinguishable from them.

**Limits.** Ten origins overlapping by 9 of 13 weeks is a small effective sample (2b addresses power). Only styles with an observed 13-week outcome can be scored. The two most recent-history baselines (seasonal naive, EWMA) are strong on this target.

Tables: `v3_growth_{per_origin,summary,paired,decision,lift,lift_ci,population}_{P2,P1}.csv`. Code: `src/nss/models/growth_backtest.py`; tests: `tests/test_growth_backtest.py`.

## 2b. Evaluation power: the interval still includes zero

Rule committed in `a0a13ad` (erratum `b0c475b`: 48 weekly origins, not 47). **This is the same model measured with more origins, not a stronger model.** The 12 models are unchanged; each weekly origin uses its 4-week block's model (identical training set under the 16-week embargo). The reproduction gate passed: at the 12 grid origins the per-origin model metrics equal the committed table to 1e-9.

**The pre-registered question: at weekly spacing, does the paired interval for model minus seasonal naive on Hit@3-in-top20 exclude zero? No.**

| Spacing | Origins with seasonal naive defined | Block | Effective sample size (ESS_ac / ESS_boot) | Mean difference | 95% interval |
|---|---|---|---|---|---|
| 4-week (reported) | 10 of 12 | 4 | 9.4 / 6.0 | +0.233 | [0.000, +0.567] |
| **1-week** | **42 of 48** | **13** | **15.3 / 15.3** | **+0.095** | **[-0.040, +0.278]** |
| 1-week, block 4 (sensitivity) | 42 of 48 | 4 | 15.3 / 24.4 | +0.095 | [-0.024, +0.246] |

(Seasonal naive needs a 52-week lag, so it is defined at 10 grid and 42 weekly origins. The `n_origins` column in `v3_power_paired.csv` counts joined rows including the undefined ones; the effective sample sizes above are computed on the defined ones.)

**What denser origins bought, in numbers.** Going from 12 to 48 origins (4x) raised the effective sample size for this comparison from about 6-9 to about 15 (roughly 1.6-2.5x), because adjacent weekly origins share 12 of their 13 forecast weeks. The interval narrowed from 0.567 wide to 0.318 wide.

**The reported +0.233 was a favourable draw.** Where seasonal naive is defined, the mean difference is +0.233 at the 10 grid origins and only +0.052 at the 32 non-grid weekly origins; the weekly average is +0.095. The denser estimate is smaller and its interval contains zero. The top-3 comparison with seasonal naive therefore stays undecided, now with a smaller point estimate than the write-up carried. That figure is in the submitted write-up (Section 3, "+0.233, CI [0.000, 0.567]"); it is unchanged there, and this is the disclosure of its fragility.

**Against the other comparators the conclusion does not change** (Hit@3-in-top20, weekly, block 13): EWMA persistence +0.160 [+0.111, +0.271], parent-category mean +0.396 [+0.361, +0.590], global mean +0.431 [+0.410, +0.618], random floor +0.425 [+0.409, +0.607]; all exclude zero, as at 4-week spacing.

**On ranking quality and error the model's edge over seasonal naive is stable across densities** (weekly, block 13): NDCG@10 +0.130 [+0.080, +0.226] (ESS about 6), Spearman +0.200 [+0.192, +0.241], WMAPE -0.041 [-0.055, -0.034]. The intervals exclude zero at both spacings. What is not established is a top-3 advantage.

Tables: `v3_power_per_origin_weekly.csv`, `v3_power_paired.csv`. Code: `src/nss/models/eval_power.py`; tests: `tests/test_eval_power.py`.
