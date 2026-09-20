# Experiment: does external search interest lead sales? REJECTED

Branch `feat/external-signals`. Not merged; `main` is untouched.

**Result.** Adding nine Google Trends features to the shipped model did not clear the decision rule I fixed beforehand. Hit@3-in-top20 rose from 0.528 to 0.583 (+0.056), but its 95% interval is [0.000, 0.139], so the lower bound does not clear zero. Spearman's interval excludes zero, but its gain is +0.004, and NDCG@10's interval straddles zero. Decision: **REJECT**.

**Why, in one line.** The signal moves *with* sales, not ahead of them: its link to sales movement is strongest at lag 0 and shrinks at every longer lag I tested (2, 4 and 8 weeks). That is what a leading indicator would not look like. Coverage and query resolution may also have limited it; this experiment cannot separate those (section 8).

## 1. Hypothesis and decision rule

Every feature in the model, and the seasonal-naive baseline, comes from the retailer's own sales history. That symmetry is why the model only matches seasonal-naive on top-k. Public search interest is information seasonal-naive structurally cannot have: a style could rise in search before it rises in sales.

One experiment, one shot, one locked configuration (`final_forecast.FINAL_MODEL_CONFIG`, the shipped model) for both arms; nothing tuned. Same embargoed protocol as the reported headline (12 paired test origins, training origins at least 16 weeks before each test origin, seed 42). The control arm had to reproduce the recorded 0.528 before the treatment arm trained; it did (0.527778).

The rule, in the module docstring and `RULE_TEXT` of `src/nss/models/external_signals_experiment.py`, committed in `f6b6960` before the treatment arm was ever run:

> ADOPT iff the pooled paired treatment-minus-control difference has (A) a 95% block-bootstrap CI excluding zero on Hit@3-in-top20, OR (B) a CI excluding zero on both NDCG@10 and Spearman with Hit@3-in-top20's point estimate not worse. Otherwise REJECT.

One run happened before the reported one: the first attempt crashed inside the *control* arm's LightGBM fit (a Windows access violation caused by importing `shap` before `lightgbm`). No result existed. I fixed the import order and ran again. A further run, from the committed weekly table instead of the API cache, reproduced all five result files byte for byte. Those two edits (the import order, and a fallback that reads the committed table when the API cache is absent) are the only changes to the experiment module after the rule commit; the rule, protocol, config and features were not touched.

## 2. The two signal sources

Both were tested before choosing. Google Trends was the model's source, chosen in advance as the more direct signal.

| | Google Trends (pytrends) | Wikipedia pageviews (REST API) |
|---|---|---|
| Works today? | Yes. 25/25 test queries, then 501/501 terms, no key needed | Yes, but see rate limit |
| Time cost | 2,272 s (37.9 min) for 501 terms, one query each | 247 s for 60 articles |
| Rate limiting | No 429 seen; three transient connect timeouts, each recovered on the first retry | **HTTP 429 at the 11th request** at 0.2 s spacing, despite "no rate limits"; needed `Retry-After` backoff and 1 s spacing |
| Granularity | Colour + product query ("beige sweater"), weekly | One article per product type or colour, daily (summed to weeks) |
| Styles served | 2,595 of 3,076 (84.4%) with a usable series | 2,565 (83.4%), but only **46 distinct series** |
| Styles per series | about 6.4 (2,595 styles over 406 series) | about 56 |
| Series-level caveat | Each series is rescaled 0-100 by its own maximum over the window (all 488 non-empty series peak at exactly 100; section 4) | Raw counts; no normalisation |

Trends' answers drift over time and the raw responses are not committed (`data/external/` is gitignored), so the compact weekly table is: `reports/tables/external_signal_trends_weekly.csv` (501 terms, 488 non-empty). Values for 2018-2020 were retrieved in September 2026 and are the API's sampled figures, worldwide, all retailers, not H&M-specific.

Wikipedia was evaluated for comparison only; using both would have been a second variant. Its link to sales was weaker at every lag (section 6). It served 46 series to about 2,565 styles, so it could not distinguish styles within a product type.

## 3. Term mapping and coverage

A style's query is "<colour word> <product word>". Only attributes with an unambiguous everyday search word are mapped (`src/nss/external/term_mapping.py`). No term was invented to raise coverage.

| Tier | Styles | Share of 3,076 |
|---|---|---|
| Mapped to a query term (501 unique terms) | 2,869 | 93.3% |
| Unmapped: product type has no clear search word (e.g. "Garment Set", "Other accessories", "Underwear body") | 110 | 3.6% |
| Unmapped: colour has no clear search word ("Mole", "Metal", "Unknown") | 92 | 3.0% |
| Unmapped: both | 5 | 0.2% |
| Term returned any data (13 terms returned nothing) | 2,846 | 92.5% |
| **Usable series** (non-zero in at least 90% of weeks; 406 of 488 non-empty terms) | **2,595** | **84.4%** |

At the 12 test origins, 84.5-84.7% of the styles being scored had a signal (`external_signals_origin_coverage.csv`); the rest had all nine features null, nothing imputed. Pattern (Solid, Stripe, Melange...) is deliberately not in the query: it makes queries too sparse, and it means styles differing only in pattern share one series. That is a real resolution limit.

## 4. Causality

Nine features (`src/nss/features/signal_features.py`): recent level vs the trailing quarter, 4- and 13-week slopes, level vs the trailing 52-week baseline, the 4-week signal slope as of 2, 4 and 8 weeks earlier (the lead features), and signal-minus-sales slope over 4 and 13 weeks (the divergence).

**A Trends-specific leak, and how it was closed.** Trends rescales every series to 0-100 by the series' own maximum over the requested window. A raw value in 2018 therefore depends on how high the series climbs in 2020. Any raw level or raw difference would leak the future. Every feature here is a log-ratio of two values of the same series, which a positive constant cancels; a test multiplies a series by 3.7 and requires identical features. The consequence: the brief's "signal level" is implemented as *recent level relative to the trailing quarter*, not the raw 0-100 number.

Alignment: a Trends week labelled by its starting Sunday is joined to the Monday after, so the signal week ends (Saturday) inside the panel week whose sales the model already treats as observed at the origin.

`tests/test_signal_features.py` (13 tests): hand-computed values for every feature; the causality check (shuffle every signal and sales value after the origin; features must not change); its **negative control** (a signal deliberately slid 3 weeks into the future must fail that same check); pre-origin sensitivity; scale invariance; independence from which other origins are requested. The existing `model_features` and `price_features` causality tests pass unchanged. Full suite on the branch: 785 passed, 3 skipped, 1 xfailed.

## 5. Paired results (12 origins, block bootstrap 4 / 2,000 resamples / seed 42)

Treatment minus control (same locked config, same origins):

| Metric | Control | Treatment | Difference | 95% CI |
|---|---|---|---|---|
| Hit@3 in top 20 (rule A) | 0.5278 | 0.5833 | +0.0556 | [+0.0000, +0.1389] |
| Hit@3 in top 10 | 0.3056 | 0.3611 | +0.0556 | [+0.0000, +0.1667] |
| NDCG@10 (rule B) | 0.8723 | 0.8732 | +0.0009 | [-0.0049, +0.0089] |
| Spearman (rule B) | 0.7116 | 0.7157 | +0.0041 | [+0.0010, +0.0086] |
| WMAPE (lower is better) | 0.1696 | 0.1672 | -0.0024 | [-0.0049, -0.0011] |
| Precision@10 | 0.1917 | 0.2083 | +0.0167 | [-0.0083, +0.0417] |

Treatment minus each comparator, paired per origin (seasonal-naive is defined on 10 of the 12 origins):

| Comparator | Hit@3 top 20 | Hit@3 top 10 | NDCG@10 | Spearman | WMAPE | Precision@10 |
|---|---|---|---|---|---|---|
| seasonal_naive | +0.300 [+0.033, +0.633] | +0.100 [+0.000, +0.300] | +0.157 [+0.098, +0.265] | +0.222 [+0.208, +0.256] | -0.047 [-0.059, -0.035] | +0.060 [+0.020, +0.100] |
| ewma_persistence | +0.306 [+0.250, +0.445] | +0.222 [+0.167, +0.417] | +0.069 [+0.051, +0.094] | +0.127 [+0.098, +0.187] | -0.031 [-0.049, -0.025] | +0.075 [+0.042, +0.133] |
| global_mean | +0.583 [+0.556, +0.833] | +0.361 [+0.306, +0.583] | +0.670 [+0.621, +0.739] | undefined | -0.141 [-0.169, -0.120] | +0.208 [+0.158, +0.333] |
| parent_category_mean | +0.556 [+0.556, +0.833] | +0.361 [+0.306, +0.583] | +0.614 [+0.520, +0.712] | +0.470 [+0.420, +0.524] | -0.080 [-0.105, -0.060] | +0.200 [+0.150, +0.317] |

(`global_mean`'s ranking metrics depend only on tie ordering; see the README's "Corrections since submission".)

**What the top-20 gain is made of.** Hit@3 has a granularity of one pick in three per origin. The +0.056 is one more hit at each of three origins and one fewer at one origin, eight origins unchanged: a net of two picks out of 36. The interval's lower edge sits at exactly 0.0000 because the per-origin differences are multiples of one third; it is not a near miss of a continuous quantity, it is a small, discrete gain.

**Do not read the seasonal-naive row as adoption.** The treatment's interval against seasonal-naive on top-20 now excludes zero ([+0.033, +0.633]) where the control's did not ([0.000, 0.567] in the write-up). The rule I fixed asks whether the signal improves on the *current model*, and it does not clear that. I also looked at seven metrics without correcting for it. The two intervals that exclude zero (Spearman +0.004, WMAPE -0.002) are small in absolute terms: about 0.6% and 1.4% of the control's values.

## 6. Does the signal lead sales? (the actual hypothesis)

Pooled cross-sectional Spearman, weekly, moving-block bootstrap. X is the style's 4-week signal slope as of L weeks earlier; Y is the style's sales slope now (concurrent) or the realised 13-week-ahead growth (forward). Lag 0 is the control. Table: `reports/tables/external_signal_lead_lag.csv`. The `pre_test` window uses only information available before the first test origin, the only place a lead length could legitimately be chosen; no feature was selected from any of this, all three lags entered the model.

Google Trends, concurrent (signal slope at t-L vs sales slope at t), pre-test, 38 weeks:

| Lag L | rho | 95% CI |
|---|---|---|
| 0 (control) | +0.111 | [+0.076, +0.161] |
| 2 | +0.086 | [+0.039, +0.145] |
| 4 | +0.068 | [+0.008, +0.138] |
| 8 | +0.035 | [-0.022, +0.109] |

The association is strongest at lag 0 and decays monotonically. A leading indicator would peak at a positive lag. Forward growth shows the same shape (all 20 origins, descriptive: +0.250, +0.223, +0.200, +0.113 at lags 0, 2, 4, 8; the pre-test window has only 5 origins, so its intervals are nearly degenerate). At lag 0 the signal is as predictive of forward growth as the style's own sales slope (+0.250 vs +0.203, intervals overlap) and so probably carries much of the same seasonal information the model already has (Fourier terms, sales slopes).

Wikipedia was weaker at every lag (concurrent pre-test +0.067, +0.062, +0.050, +0.016).

**Which lead length carries the most information?** None meaningfully. By correlation, among positive leads, 2 weeks. By in-sample SHAP, 4 weeks (0.33% of total attribution vs 0.20% for 8 and 0.19% for 2). The gaps are far inside the noise.

## 7. SHAP

In-sample mean |SHAP|, treatment model on the last embargoed fold (test origin 2020-06-01; 16 training origins, 48,003 rows, 35 features). Full table: `external_signals_shap.csv`.

| Group | Share of total attribution |
|---|---|
| Existing sales-history and attribute features | 92.4% |
| Signal features (level, slopes, 52-week, divergence) | 6.9% |
| Signal **lead** features (lags 2, 4, 8) | **0.7%** |

Signal features do get used: `sg_level_rel13` ranks 10th of 35 and `sg_rel_52w` 13th. Those are *level* features (recent level vs the trailing quarter and year), not leads. The three lead features rank 30th, 34th and 35th, the bottom of the table. In-sample SHAP shows what the model used, not that the use generalises.

## 8. Most likely reasons for the negative, in order of evidence

1. **The signal is coincident, not leading at these horizons.** Direct evidence: section 6, plus the lead features carrying 0.7% of attribution. Public search and retail sales in the same weeks are both driven by season and by the same trend, so search mostly re-expresses information the model already gets from sales and its seasonal terms.
2. **Resolution.** 406 series serve 2,595 styles (about 6.4 each), and a global colour-plus-product query is dominated by non-H&M searches. A style-specific rise is invisible in a term shared with five other styles. I cannot separate this from (1) here.
3. **Coverage.** 15.4% of styles have no usable series and get null features; the achievable gain shrinks accordingly.
4. **Power.** Twelve test origins, 16 training origins, two years, and a discrete top-3 metric: the interval on Hit@3-in-top20 reaches about +0.14 on the upside, so a real gain smaller than that cannot be told apart from zero here. A real but small lead would look exactly like this. This experiment does not show that no signal exists; it shows this signal, at this resolution, on this data, does not clear the rule.

## 9. What was not done

- **No placebo arm** (signal shuffled across styles). It would only matter if the treatment had won; the brief allows one shot.
- **No tuning, no second variant, no second source in the model.** Wikipedia was diagnostic only.
- **No geography or category filtering** of Trends; worldwide, all-category.
- **Not concluded:** that external signal cannot help. Only that this one, built this way, did not clear a pre-stated bar.

## 10. Files and reproduction

| | |
|---|---|
| Fetch | `nss.external.trends_fetch` (run with `uv run --no-sync --with pytrends`, so `pyproject.toml` and `uv.lock` are untouched), `nss.external.wikipedia_fetch` |
| Features and tests | `src/nss/features/signal_features.py`, `tests/test_signal_features.py`, `tests/test_external_signal_loading.py` |
| Diagnostics | `python -m nss.external.signal_diagnostics` -> `external_signal_{coverage,lead_lag,style_terms,trends_weekly,wikipedia_weekly}.csv` |
| Experiment | `python -m nss.models.external_signals_experiment` -> `external_signals_{per_origin,paired_diff,decision,shap,origin_coverage}.csv` (needs the H&M panel; reads the committed weekly table if the API cache is absent) |

Numbers here come from the tables above, produced on this branch (see `git log`); the experiment output is deterministic (a second run reproduced all five files byte for byte).
