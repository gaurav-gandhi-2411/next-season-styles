# Interval calibration (S) and the EXPLORATORY market-momentum check (T)

Rules: `PREREGISTRATION.md` Sections S and T, committed at `a76adae` before anything below was
computed. Code: `nss.models.phase_s_calibration`, `nss.models.phase_t_exploratory`.

## S. Rolling conformalised quantile regression

The q10 and q90 models are B.5's, unchanged (their 48 test-origin predictions are checked to
reproduce B.5 exactly). q50 is untouched. Each test origin is calibrated on the 4 most recent
weekly origins whose 13-week outcomes closed before it (`t − 16w .. t − 13w`, about 12,000
scores). The first test origins draw on 16 calibration-only origins (2019-04-08 .. 2019-07-22)
from the same embargoed serving rule.

### Coverage (`phase_s_summary.csv`, `phase_s_coverage_per_origin.csv`)

| interval | pooled (48 origins) | COVID (30) | non-COVID (18) | per-origin mean [95% circular CI] | per-origin min / max | origins in [0.75, 0.85] | mean width (log1p) | median width (raw units/article/wk) |
|---|---|---|---|---|---|---|---|---|
| raw q10-q90 | 0.623 | 0.685 | 0.525 | 0.626 [0.545, 0.692] | 0.395 / 0.784 | 6 (42 below) | 1.293 | 11.6 |
| CQR symmetric | 0.805 | 0.841 | 0.749 | 0.807 [0.727, 0.890] | 0.583 / 0.988 | 8 (18 below, 22 above) | 1.825 | 16.9 |
| **CQR asymmetric (chosen)** | **0.803** | 0.837 | 0.748 | 0.804 [0.726, 0.886] | 0.583 / 0.984 | 8 (18 below, 22 above) | **1.808** | 16.0 |

**Correction disclosed.** The first run (`5bdcf58`) computed the conformal quantile as
`np.quantile(scores, k/n, method="higher")`, which the pre-registration also named. Numpy indexes
over `n - 1`, so that returns the (k+1)-th smallest score, one order statistic above the S1 formula
`k = ceil(0.8 (n + 1))`. A unit test caught it. With about 12,000 scores per window the effect is
at most 0.0005 on any figure above (asymmetric pooled 0.80274 → 0.80261, width 1.80807 →
1.80763), and the choice and band counts are unchanged. The table shows the corrected run.

**Verdict: the pre-registered target is met.** Both CQR variants land pooled coverage inside
[0.75, 0.85], and the asymmetric one is narrower (log1p width 1.808 vs 1.825), so it is chosen, as
the rule said in advance. The cost is width: the median raw interval goes from 11.6 to 16.0 units
per active article per week (+38%).

**What the pooled number hides.** Only 8 of 48 origins individually fall in the band; coverage
swings from 0.58 to 0.98 (`reports/figures/phase_s_coverage_over_time.png`). The swings follow the
raw interval's own cycle, delayed. Each correction is estimated from outcomes 13-16 weeks old, so
it corrects for an error that has since moved. Under-coverage in Aug-Oct 2019 was followed by
over-coverage in Nov 2019-Jan 2020, and the March 2020 COVID shift drove coverage down to 0.64
before the window caught up. Non-COVID origins sit just below the band (0.748), COVID origins just
above it (0.837). The earliest origins also calibrate on models trained on fewer origins, which
the pre-registration flagged as a source of error. Here those origins nonetheless under-cover.

**Exchangeability caveat.** Conformal guarantees assume exchangeable calibration and test scores.
This series is not exchangeable: demand drifts, COVID is a regime shift, and the rows of one
origin share a shock. The rolling, embargoed window is a practical response that meets the pooled
target on these 48 origins; it does not guarantee coverage at any single origin, and the figure
shows it does not deliver it.

**For production (Phase C):** the calibrated interval is usable as a long-run 80% band. A consumer
who needs per-week reliability should know it can run from about 60% to 98%. Reducing the lag
(e.g. adaptive conformal updating on partially observed outcomes) would need its own
pre-registered rule.

## T. EXPLORATORY: market momentum

**Everything in this section is EXPLORATORY.** The hypothesis came from B.1's random-neighbour
control on these same 48 origins. Results are unadjusted for multiplicity, and nothing is adopted.
Files: `phase_t_exploratory_{per_origin,paired}.csv`, `phase_t_exploratory_verdict.json`.

| arm | what it adds | capture@20 vs champion [95% circular CI] | p | helps? |
|---|---|---|---|---|
| (a) market momentum | 4w and 13w trailing slopes of mean intensity across active styles | −0.0023 [−0.0192, 0.0136] | 0.80 | no |
| (a) shuffle test | (a)'s origin-level values moved to other origins | +0.0001 [−0.0098, 0.0112] | 1.00 | no |
| (a) negative control | origin-level Gaussian noise, (a)'s mean and SD | +0.0011 [−0.0069, 0.0102] | 0.79 | no |
| (b) pure noise | random-neighbour values permuted globally | −0.0007 [−0.0121, 0.0114] | 0.87 | no |
| (c) random-neighbour re-run | B.1's control, fresh process | +0.0088 [0.0016, 0.0176] | 0.011 | yes |

Arm (c) reproduces bit for bit (max |diff| 0.0). That is computational reproducibility of
deterministic code, not independent evidence. No guardrail moves significantly in any arm
(`phase_t_exploratory_paired.csv`).

**Interpretation, by the rule fixed in Section T:** (a) does not help and (b) does not help →
**the B.1 result does not reproduce as a market effect.** An explicit market-momentum signal adds
nothing (−0.0023), and neither does an arbitrary extra column, so the random-neighbour gain is
explained neither by market momentum nor by generic regularisation. It stands as one comparison
among about eight exploratory and control comparisons run on these origins (B.1's two controls
plus the five arms here), with p = 0.011 and no multiplicity adjustment. That is consistent with
chance. **Market momentum is therefore not added to Phase D.**
