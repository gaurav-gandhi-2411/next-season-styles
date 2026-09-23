# Model card: next-season style demand forecaster

**For:** a retailer's data team deciding whether to put this forecast in front of buyers.
**Version:** the champion as of 2026-09-23: an L2 LightGBM point forecast plus a rolling
asymmetric conformal interval. **Status:** research-validated on public H&M data; not yet run on
live data.

## What it does

For every style (a combination of five catalogue attributes: index group, product type, garment
group, perceived colour and graphical appearance; 3,076 styles), it forecasts **demand intensity
over the next 13 weeks**: average units sold per active article per week. Each forecast comes with
an 80% interval and the style's rank among all styles forecast that week.

## Intended use

**Use it to shortlist about 20 styles** worth a closer look for next season's range, and to see
roughly how much demand the shortlist carries.

**Do not use it to pick the exact top 3.** It has **no demonstrated edge** over a simple
seasonal-naive rule (last year's same 13 weeks) at that. On tolerance hit@3, which asks whether
each top-3 pick is within 5.8% of the true #3 style's demand, it scores −0.008 [−0.095, +0.095]
against seasonal-naive. On exact top-3 precision it is no better either. The shortlist, not the
podium, is where the model earns its keep.

Also not intended for: forecasting a single article or SKU, new styles with no sales history (the
model needs trailing weeks), pricing decisions, or horizons other than 13 weeks.

## Performance

48 weekly forecast dates, 2019-07-29 to 2020-06-22, each scored only on data the model could have
seen then (see *Protocol*). Intervals are 95% circular block-bootstrap intervals (block length 13
weeks), which respect the overlap between consecutive forecast windows. Seasonal-naive can only
forecast styles with a year of history, so its comparisons use the 42 dates where it applies.

**Primary metric: demand capture@20.** The realised demand of the model's top-20 styles divided by
that of the true top-20. 1.0 means the shortlist carries as much demand as the best possible one.

| method | demand capture@20 | model's advantage |
|---|---|---|
| **this model** | **0.614** [0.519, 0.691] | — |
| EWMA persistence (recent trend carried forward) | 0.500 [0.391, 0.594] | +0.114 [+0.070, +0.159] |
| seasonal-naive (same weeks last year) | 0.473 [0.411, 0.538] | +0.185 [+0.138, +0.235] |
| random shortlist | 0.114 [0.107, 0.124] | +0.500 [+0.406, +0.574] |
| parent-category mean | 0.091 [0.061, 0.119] | +0.523 [+0.412, +0.625] |
| global mean | 0.019 [0.017, 0.021] | +0.594 [+0.500, +0.672] |

**Other metrics, model's advantage over seasonal-naive and over EWMA:**

| metric | what it measures | model | vs seasonal-naive | vs EWMA |
|---|---|---|---|---|
| Hit@3-in-top20 | share of the top-3 picks that land in the true top 20 | 0.431 | +0.095 [−0.063, +0.246] | +0.160 [+0.063, +0.257] |
| Tolerance hit@3 | share of top-3 picks within 5.8% of the true #3's demand | 0.132 | −0.008 [−0.095, +0.095] | +0.069 [−0.007, +0.153] |
| NDCG@10 | quality of the top-10 ordering | 0.869 | +0.130 [+0.062, +0.208] | +0.062 [+0.033, +0.100] |
| Spearman | rank agreement across all styles | 0.710 | +0.200 [+0.163, +0.235] | +0.116 [+0.066, +0.165] |
| WMAPE (lower is better) | weighted absolute error | 0.166 | −0.041 [−0.054, −0.024] | −0.030 [−0.043, −0.014] |

Reading it plainly: the model is reliably better than every simple rule at ranking the whole
catalogue and at building a 20-style shortlist. At the very top it is not reliably better than
last year's numbers.

## The 80% interval: what it means

- **It is a long-run 80% interval, not a weekly one.** Across all 48 forecast dates, 80.3% of
  outcomes fell inside it (target 75-85%).
- **By regime:** 83.7% during the COVID-affected period, **74.8% outside it**.
- **By week it swings from 58% to 98%.** Only 8 of 48 weeks landed within 75-85%; 18 fell below
  and 22 above. Any single week's interval may be too narrow or too wide.
- **Why:** the interval is corrected using how wrong past forecasts were. A 13-week forecast is
  only known to be right or wrong 13 weeks later, so each correction uses errors 13-16 weeks old.
  When demand shifts (as in March 2020), the correction arrives a season late.
- **How wide:** the median interval is **2.7x the median forecast** across all styles. On the
  top-20 shortlist it is **1.3x** (e.g. a forecast of 61 units per article per week with an interval
  about 79 wide), and 85% of shortlisted outcomes fell inside it.
- Use the interval to judge how uncertain a style is relative to others, and to plan for a
  plausible range across many styles. Do not treat one week's interval as a guarantee.

## Training data

- H&M Personalized Fashion Recommendations (public Kaggle data): 31,788,324 transactions from
  2018-09-20 to 2020-09-22, 105,542 articles, 1,371,980 customers.
- Aggregated to a style-week panel: 313,887 rows, 3,076 styles, weeks 2018-09-17 to 2020-09-21.
- Features, all computed only from weeks up to the forecast date: recent demand levels and lags
  (1-52 weeks), short and long moving averages, 4- and 13-week trends, number of active articles,
  relative price, the style's share of its category, time since launch, and seasonality terms.
- **Not used:** customer attributes, images, external signals. Images and a search-trend signal
  were tested and rejected (see below).

## Protocol

- **Retraining:** a new model every 4 weeks. Each week's forecast uses the most recent model.
- **Embargo:** a model only learns from forecast dates whose 13-week outcomes were fully known by
  the date it forecasts. No training label overlaps the period being forecast.
- **Fixed settings:** 63 leaves, learning rate 0.05, 200 trees, minimum 50 rows per leaf, chosen
  once and never tuned on the evaluation dates. Training is deterministic: the same data gives
  bit-identical forecasts.
- **Interval:** q10 and q90 LightGBM quantile models, widened or narrowed per forecast date by a
  conformal correction learned from the 4 most recent forecast dates whose outcomes were known
  (asymmetric: the low and high ends are corrected separately).
- **Every comparison above was pre-registered:** the decision rule was committed to version
  control before the result was computed.

## Limitations

1. **One retailer, two years, one pandemic.** Of the 48 forecast dates, 30 have horizons that
   overlap March-June 2020. Performance on a normal year and on another retailer is unmeasured.
2. **Effective sample is small.** Consecutive weekly forecasts share 12 of 13 weeks, so the 48
   dates carry the information of roughly 12-23 independent ones. This is why the top-3 comparison
   is inconclusive.
3. **Intervals lag regime changes** by a season (see above).
4. **Established styles only.** A style needs trailing sales to be forecast; genuinely new designs
   are out of scope.
5. **Style, not article.** The forecast is for a five-attribute style; how demand splits across
   the articles inside it is not modelled.
6. **Rankings drive most of the value.** Absolute accuracy (WMAPE 0.166) is better than the
   baselines, but the model was optimised and judged on ranking.

## Tried and rejected

Each was judged against this model under a pre-stated rule; none beat it.

- **Neighbourhood cohort features** (demand of similar styles): no gain.
- **Buyer-mix and price-elasticity features:** no gain.
- **External search-trend signal:** it moves with sales rather than ahead of them; no gain.
- **LambdaRank, top-heavy weighting, two-stage re-ranking:** no gain at the top.
- **Ensemble with a LambdaRank model:** statistically tied.
- **A growth-ranking redesign:** realised growth here is mostly mean reversion; not adopted.
- **Selective prediction by quantile spread:** did not identify reliable picks.
- **Hyperparameter re-tuning under the embargo:** 0.65% gain, within noise.
- **Visual momentum from product-image embeddings:** −0.002 capture@20; controls failed.
- **Training on a noise-shrunk demand target:** −0.013, and worse rank agreement and error.
- **Hierarchical reconciliation across category levels:** +0.0006, not significant.
- **Seed ensemble:** a no-op; training is deterministic, so all seeds give the same model.
- **Market-wide momentum (exploratory):** −0.002; no market effect.

## Sources

All figures are reproducible from the repository: `reports/tables/phase_b0_summary.csv` and
`phase_b0_paired.csv` (performance), `phase_s_summary.csv` and `phase_s_coverage_per_origin.csv`
(interval coverage), `c1_interval_width.csv` (width), `reports/v3/PHASE_A_measurement.md`,
`PHASE_B_results.md`, `PHASE_S_T_calibration_and_exploratory.md`, and the pre-registered rules in
`reports/v3/PREREGISTRATION.md`.
