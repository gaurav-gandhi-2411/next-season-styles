# Track 3a: per-season performance of the intensity backtest (evaluation only)

Rule committed in `0394197` before the run. Source: the 48 weekly origins of 2b (same 12 models). Season = the month of the midpoint of an origin's 13-week target window. Paired differences are model minus comparator, with a moving-block bootstrap (block 4). Every number below is in `reports/tables/v3_season_split.csv` (140 cells: 4 seasons x 5 comparators x 7 metrics).

## Read this first: what a season cell can and cannot say

- **Each season occurs once.** The test period is about one year, so a season's origins are one contiguous run of 11-13 weekly origins whose 13-week outcome windows overlap by 12 of 13 weeks. **Every cell is, in effect, one outcome window.** An interval here describes that stretch of time, not "that season in general". No claim about seasonal patterns in model quality can be made from it.
- **The pre-registered thin-cell rule is more lenient than that structure.** It flags a cell only if fewer than 8 origins have both methods defined or `ess_ac` is under 5. That flagged **22 of 140 cells**. The others are *not* strong evidence: the `ess_ac` values close to the origin count (many read 13.0) come from an autocorrelation estimator on at most 13 points and are not credible for heavily overlapping windows. I did not change the rule after seeing this; treat "not THIN" as "not flagged by the rule", never as "adequate".
- **Confounds.** Each season's models were trained on a different, growing training set (autumn's are the smallest). Spring and summer windows overlap the March-June 2020 COVID period: all 13 spring and all 11 summer origins are COVID-flagged, and 6 of the 13 winter ones.
- **Undefined comparators.** The global-mean baseline predicts a constant, so Spearman is undefined for it (empty cells). Seasonal naive needs a 52-week lag, so only 5 of the 11 autumn origins have it.

## The model, by season

| Season | Origins (first to last) | Of which COVID-flagged | Hit@3-in-top20 | NDCG@10 | Spearman | WMAPE (lower is better) |
|---|---|---|---|---|---|---|
| Autumn (windows in Sep-Nov 2019) | 11 (2019-07-29 to 2019-10-07) | 0 | 0.212 | 0.787 | 0.541 | 0.250 |
| Winter | 13 (2019-10-14 to 2020-01-06) | 6 | 0.487 | 0.887 | 0.719 | 0.153 |
| Spring | 13 (2020-01-13 to 2020-04-06) | 13 | 0.641 | 0.893 | 0.762 | 0.146 |
| Summer | 11 (2020-04-13 to 2020-06-22) | 11 | 0.333 | 0.902 | 0.807 | 0.119 |

The autumn cell is the weakest on every metric; it is also the one whose models were trained on the fewest origins.

## Hit@3-in-top20, model minus each comparator (mean [95% interval]; origins with both defined; flag)

| Season | Seasonal naive | EWMA persistence | Parent-category mean | Global mean | Random floor |
|---|---|---|---|---|---|
| Autumn | +0.133 [+0.067, +0.133] (5, **THIN**) | +0.182 [0.000, +0.394] (11) | +0.212 [0.000, +0.424] (11) | +0.212 [0.000, +0.424] (11) | +0.211 [0.000, +0.420] (11) |
| Winter | +0.179 [+0.154, +0.359] (13) | +0.077 [-0.154, +0.308] (13) | +0.462 [+0.333, +0.718] (13) | +0.487 [+0.359, +0.718] (13) | +0.486 [+0.359, +0.715] (13) |
| Spring | +0.179 [-0.154, +0.462] (13) | +0.308 [+0.128, +0.410] (13) | +0.538 [+0.410, +0.641] (13) | +0.641 [+0.513, +0.667] (13) | +0.633 [+0.497, +0.667] (13) |
| Summer | **-0.121 [-0.242, -0.030]** (11) | +0.061 [0.000, +0.121] (11) | +0.333 [+0.030, +0.576] (11) | +0.333 [+0.030, +0.576] (11) | +0.321 [+0.021, +0.562] (11) |

- **Against seasonal naive on top-k, the sign is not stable across seasons:** positive in autumn (5 origins, THIN), winter and spring (spring's interval includes zero), **negative in summer** (the interval excludes zero, for one 11-origin stretch that lies entirely in the COVID months). This is the same "no demonstrated top-3 edge" as the pooled result, now seen to hide a season where seasonal naive did better on top-k.
- **Against the random floor and the two means the model's point estimate is ahead in every season**, by about 0.2 to 0.6 in Hit@3 (the autumn intervals reach down to 0.000).
- **EWMA persistence is the closest rival on top-k**, and the model's lead over it includes zero in autumn and winter.

## Other metrics against seasonal naive (mean difference [95% interval]; flag)

| Season | NDCG@10 | Spearman | WMAPE (negative = model better) |
|---|---|---|---|
| Autumn | +0.064 [+0.051, +0.064] (**THIN**) | +0.072 [+0.051, +0.075] (**THIN**) | +0.017 [+0.016, +0.025] (**THIN**) |
| Winter | +0.224 [+0.189, +0.283] | +0.230 [+0.198, +0.267] | -0.046 [-0.059, -0.033] |
| Spring | +0.128 [+0.052, +0.210] | +0.217 [+0.205, +0.230] | -0.049 [-0.052, -0.043] |
| Summer | +0.050 [+0.035, +0.075] | +0.202 [+0.196, +0.220] | -0.053 [-0.060, -0.051] |

On ranking quality and error the model is ahead of seasonal naive in winter, spring and summer, including summer, where its top-3 picks are worse. Autumn's five defined origins are too few for any statement (and WMAPE there points the other way, +0.017).

Full 140-cell table, with the comparator means, both effective-sample-size estimators and the flag for every cell: `v3_season_split.csv`; per-season model means: `v3_season_summary.csv`. Code: `src/nss/models/season_split.py`; tests: `tests/test_season_split.py`.
