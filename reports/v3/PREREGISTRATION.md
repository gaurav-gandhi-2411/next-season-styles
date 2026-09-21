# v3 improvements: pre-registered rules

Branch `feat/v3-improvements`, cut from `main` at `119d187`. `main` and `reports/SUBMISSION/` are not touched.

**Rule for the whole pass.** Before any experiment that produces a comparison runs, its decision rule is committed here. The commit that adds a section predates every result that section governs (check `git log -- reports/v3/PREREGISTRATION.md` against the result commits). A rule is not edited after results are seen; if a rule turns out to be badly specified, that is reported as a defect in the next section, not fixed silently. Every comparison carries a floor and a paired block-bootstrap CI or, for proportions, a Wilson interval. Measured versus assumed is stated explicitly.

---

## Track 1: concepts

### 1a. White-top selection (rule is the repo's existing one, restated as code)

The rule is implemented in `src/nss/generate/candidate_selection.py` and unit-tested. It restates what `final_selection_figures` already documented ("passes every automatic gate -> most briefed changes visible -> highest fidelity, then confirmed by a human"); it does not change it.

- **Pass:** every automatic gate passes: Gate 1, Gate 1b, the global integrity floor, Gate 2 (SmolVLM gates; Florence-2 is advisory) and Gate 3. A missing or null gate is a fail. The per-style integrity floor is advisory.
- **Rank among passers:** (1) number of briefed changes Gate 3 judged visible, (2) highest SmolVLM fidelity, (3) highest Florence-2 fidelity (tie-break), (4) lowest seed.
- **Human visual check:** a veto applied afterwards by a person looking at the image. It is never a rank input and never inferred from a score.
- **No passer:** the rule returns nothing; it never promotes a failure.

*Disclosure:* the 8-seed scores for the white top existed and had been read before this rule was committed, so 1a is not a blind prediction. It is a check that the rule, applied as written, reproduces the prior pass counts (sweater 1/8, white top 2/8, dress 4/8, bikini 0/8; it does) and reports what it picks. Nothing is adopted: the user does the human check.

### 1b. Second-opinion re-read with Gemini

- **Images:** the four submitted concepts, plus the white-top rule pick and runner-up from 1a.
- **Judges:** Gemini (3 readings per image, median per attribute dimension), SmolVLM (gating, one greedy reading), Florence-2 (advisory, one reading).
- **Scores:** each judge's mean attribute score per image, and each judge's Gate 2 pass at its own calibrated threshold (unchanged, `judge_repeat.judge_thresholds`).
- **Agreement:** pairwise Cohen's kappa on binarised per-(image, attribute) items (score >= 0.5), with n, computed as in `judge_panel_kappa`. Kappa says whether two judges make the same calls, not whether either is right; with a handful of images it is a rough figure and will be reported with n.
- **No verdict changes** are adopted from this. If the Gemini reading contradicts a gate, that is reported, not acted on.

### 1c. Yield at 24 seeds

- **Config unchanged:** IP-Adapter scale 0.35, compel weight 1.5, multi-reference concatenation of the 8 best screened references, same prompts, negatives and references as the submitted run. Seeds 42-49 already exist and are reused; seeds 50-65 are new (24 in total per style, four styles).
- **Pass:** exactly the 1a definition.
- **Reported per style:** passes/24 pooled (the headline yield estimate), and the prior 8 seeds versus the new 16 seeds, each with a Wilson 95% interval.
- **"Yield changed" criterion:** claimed only if the prior-8 and new-16 Wilson intervals are disjoint. Otherwise the result is "consistent with the prior rate", never "improved".
- **Floor:** cross-style mismatch. Each style's 24 images are scored as if they belonged to the next style in registry order (wrong references, wrong attributes, wrong briefed changes) through the same gate stack. The false-pass rate is reported next to the yield. Expected near zero; any pass is a defect of the gate stack and is reported as one.
- **Selection:** the 1a rule applied to each style's pool of 24. The pick is a candidate only. The human check is the user's. No figure under `reports/figures/` and nothing in `reports/SUBMISSION/` is changed.

### 1d. FLUX comparison (optional)

Not started. Requires a GCP GPU, and I cannot verify from the CLI that free credits cover it. The cost estimate and a request for confirmation go to the user first; nothing is provisioned before that. If approved, its comparison rule will be committed in this file before it runs.

**Decision (user): 1d is SKIPPED.** The binding constraints are the judges and the pass rates, not the generator.

---

## 1e. "All over pattern" is satisfied by any named visible pattern

*Motivation, measured:* SmolVLM's fidelity for the bikini is exactly 0.283 on all 24 seeds, so Gate 2 carries no seed-level information for it. The judge always names a pattern (Melange 8, Floral 5, Checkered 5, Polka dot 3, other 3) and never says "solid", but it never says the bucket label "All over pattern". H&M's `graphical_appearance_name` values "Jersey Basic" and "Other structure" were already excluded for the same reason (no describable visual referent).

*Correction to my Track 1 note:* the 0.283 is the colour dimension alone (0.85 / 3). The product-type dimension also scores 0 for the bikini, because SmolVLM answers "Bikini." and the label is "Bikini top" (the scorer keeps the answer's trailing period, so neither string contains the other). That is a second, separate defect. It is **not** changed here; it is out of the 1e scope and is reported only.

**Rule (implemented in `src/nss/generate/pattern_label.py`, committed before any control was run):**

- Active only when the ground-truth label is exactly "All over pattern" and the judge's answer is at most 6 words (long free-text captions, i.e. Florence-2, keep the legacy score: they say "plain background").
- Satisfied (score 1.0) iff the answer contains a word from an explicit pattern list and no word from the plain list (`solid`, `plain`, `none`, ...). Everything else (solid, plain, nonsense, empty, or "solid, striped") scores 0.0.
- The pattern list is explicit and includes `melange` (the rule is "any named pattern"). Disclosure: I had already seen the 24 bikini answers (8 were "Melange").
- No threshold changes: Gate 2's per-judge threshold stays the calibrated 0.75 x positive mean (SmolVLM 0.384).

**Mandatory negative control:** 6 real catalogue photos of SOLID-colour bikini tops (the lowest article id per colour among Orange, Black, White, Red, Blue, Pink, fetched with the existing fetcher). SmolVLM's short answer for each is run through the rule with the label "All over pattern". **Every one must FAIL.** If any solid bikini passes, the mapping is broken: it is reverted (the `concept_scoring` hook removed) and the failure is reported.

**Positive control (reported, not gating):** all real "All over pattern" bikini tops with a local photo (22 found). The pass rate says whether the rule is usable on real prints.

**No-side-effect check:** re-score all 96 candidates (24 seeds x 4 styles) with the mapping on, from the stored extractions (deterministic judges), and compare every gate column with the pre-mapping table. For the three non-bikini styles the result must be identical. (The rule is inactive for any label except "All over pattern", so any change there would be a bug.)

**Reported:** the bikini's Gate 2 pass count and its all-gate pass count across 24 seeds under the mapped rule, beside the unmapped 0/24.

---

## Track 2

### 2a. Growth backtest: is the emerging (growth) ranking validated?

The final three take one incumbent (ranked by predicted intensity) and two **emerging** styles ranked by `growth_ratio = predicted_intensity / trailing_13w_mean_intensity` (`diversity_forecast`). The existing backtest only ever scored the intensity ranking. This evaluates the growth ranking itself.

**What is ranked and scored.** At each of the 12 shared embargoed test origins (same model, same training origins at least 16 weeks before each test origin, same locked `FINAL_MODEL_CONFIG`, seed 42):

- **Realised growth** of a style = `expm1(y_true) / trailing_13w_mean_intensity`, where `y_true` is the realised `log1p(mean units per active article)` over the next 13 weeks and the denominator is the deployed one (raw `units_per_active_article`, trailing 13 weeks inclusive of the origin week).
- **Predicted growth** of a method = `expm1(its prediction) / trailing_13w_mean_intensity`. The model uses its own prediction; each baseline (seasonal naive, EWMA persistence, global mean, parent-category mean, from `backtest.build_predictions_frame`) uses its own. A baseline's growth is therefore a level forecast divided by the same denominator, which is exactly how the deployed ranking is built.
- **Random floor:** the realised growth ratios permuted across the eligible styles (20 seeds, the same construction as the intensity floor).
- **Metrics:** the same seven as the intensity backtest (Hit@3-in-top20 is the headline), per origin, then pooled with the moving-block bootstrap (block 4, 2,000 resamples, seed 42) and paired per origin against each comparator.

**Populations (both reported; the first is primary).**

- **P2, deployed (primary):** the population the emerging list is actually drawn from: passes guards 1-3, trailing mean > 0, and predicted intensity at or above the median predicted intensity of the guard-passing styles at that origin (the model's own floor, `t2_absolute_intensity_floor`). The same set of styles is used for every method, so comparisons are paired on identical populations.
- **P1, guard-passing (sensitivity):** guards 1-3 and trailing mean > 0, no model-derived floor.
- **Restriction, stated:** only styles with an observed full 13-week outcome can be scored, so a style that stops selling within 13 weeks is absent from the population and from the floor's median. This differs from the live forecast, which cannot see it.

**Decision categories, fixed now, on P2, pooled over the 12 origins, using the same rule form as the earlier experiments.** A method X "beats" a comparator iff the paired X-minus-comparator difference has (A) a 95% CI excluding zero on the improving side for Hit@3-in-top20 (`ci_lo > 0`), OR (B) `ci_lo > 0` on both NDCG@10 and Spearman with a Hit@3-in-top20 mean difference of at least 0.

- **VALIDATED:** the model beats every one of the four baselines *and* the random floor.
- **PARTIAL:** the model beats the random floor but not all four baselines. The result then says which baselines it does not beat.
- **NOT VALIDATED:** the model does not beat the random floor.

If the result is PARTIAL or NOT VALIDATED it is reported as such: the emerging half of the final-three selection then rests on a ranking that has not been validated, and that is stated at the top of the report. P1 is reported as a sensitivity check and cannot change the category. No tuning, no second variant.

**Secondary (reported, not decision-bearing):** the mean realised growth ratio of each method's top 3 by predicted growth, relative to the population mean (a lift), per origin with a block-bootstrap CI.

*Defect disclosed after the 2a run, for later rules:* an equality between two hit rates evaluated to -5.6e-18 and decided a "not worse" clause. Every "mean difference >= 0" clause from here on is evaluated as `mean_diff >= -1e-9`. The 2a rule itself was not edited.

### 2b. Evaluation power: the same model at a 1-week origin step

**Question (the only decision-bearing one):** at 1-week origin spacing, does the paired 95% interval for the embargoed model minus seasonal naive on **Hit@3-in-top20** (the reported intensity ranking, all styles) exclude zero? At the 4-week spacing it did not: +0.233 [0.000, 0.567].

**This is the same model measured with more origins, not a stronger model.** The model is the shipped locked config under the 16-week embargo. For a weekly test origin `t`, the training origins are the 4-week-grid origins at least 16 weeks before `t`. That set is identical to the training set of the greatest grid test origin at or before `t`, so 12 models are trained (as in the reported run) and each weekly origin uses its block's model with its own features. No retraining, no tuning.

- **Weekly origins:** every Monday from the first walk-forward test origin (2019-07-29) to the last origin whose 13-week outcome fits in the panel (2020-06-22): 48 origins (the first draft of this line said 47; a counting error, corrected before any weekly result existed). The 12 grid origins are a subset.
- **Reproduction gate:** at the 12 grid origins the per-origin model metrics must equal the committed embargoed per-origin table (`backtest_embargo_per_origin.csv`, lightgbm) to 1e-9, or the run stops.
- **Uncertainty:** moving-block bootstrap on the per-origin paired differences, **block length 13 origins (the 13-week horizon)**, 2,000 resamples, seed 42, at weekly spacing. The 4-week run uses the reported block length 4 (which is also about the 13-week horizon expressed in origins). Both are shown side by side. Seasonal naive is undefined without a 52-week lag, so its paired rows use fewer origins; n is reported.
- **Effective sample size, two estimators, both reported.** `ESS_ac = n / (1 + 2 * sum_{k=1..L} (1 - k/(L+1)) * rho_k)` with `L = 12` (Bartlett-weighted autocorrelation of the paired difference series; floored at 1, capped at n) and `ESS_boot = n * Var_iid(mean) / Var_block(mean)`, the iid variance of the mean over the block-bootstrap variance of the mean. The raw origin count is never presented as the sample size.
- **Floor and comparators:** the random-permutation floor and all four baselines are in the same table; the question concerns seasonal naive only.
- **Reporting rule:** the result is stated as "the interval does / does not exclude zero at weekly spacing". If it does, it is presented as "the same model with more power", never as an improved model.

### 2c. Selective prediction: does the model's own uncertainty identify its reliable picks?

**Uncertainty measure (fixed).** Two extra LightGBM models per training set, identical to the point model (same features, locked `FINAL_MODEL_CONFIG`, deterministic settings, seed 42, same embargoed training origins) except for the objective: `quantile` with alpha 0.1 and 0.9. A style's uncertainty is the **spread** `w = pred_q90 - pred_q10` on the log1p scale. The point model is unchanged, so the picks are exactly the reported ones.

**Picks and hits.** At each origin the picks are the point model's top 3 among the evaluated styles (the same 3 that Hit@3-in-top20 counts). A pick is a hit iff it is in the realised top 20 of that origin's full evaluation set.

**Confident subset (threshold fixed before any spread was computed).** A pick is *confident* iff its spread `w` is at or below the **median spread of all evaluated styles at that origin**. The model abstains on the other picks. No other threshold enters the decision.

**Reported.** Hit rate of confident picks versus all picks (pooled over origins), and the coverage fraction (confident picks / all picks).

**Evaluation sets.** Primary: the 12 shared embargoed test origins, block bootstrap block length 4. Secondary: the 48 weekly origins of 2b (same 12 models), block length 13, with the effective sample size reported. The primary is the decision.

**Interval.** Moving-block bootstrap over origins (2,000 resamples, seed 42) of the statistic `hit_rate(confident) - hit_rate(all)`, resampling whole origins so that all 3 picks of an origin stay together.

**Random floor (selective).** Within each origin, the confidence labels are permuted across its 3 picks, keeping the number of confident picks per origin fixed (2,000 permutations, seed 42), and the same statistic is computed. This is what abstaining at random would give. The observed statistic is reported against this null (its mean and 95% range, and a one-sided permutation p).

**Decision (fixed).** Selective prediction is reported as **HELPS** iff the block-bootstrap 95% interval of `hit_rate(confident) - hit_rate(all)` has a lower bound above 0 (with the 1e-9 tolerance) **and** coverage is at least 0.30. Otherwise it is reported as **NOT DEMONSTRATED**, with the reason (interval, coverage, or both). No threshold sweep can change that verdict.

**Descriptive, labelled exploratory and not decision-bearing:** (i) the same statistic at spread quantiles 0.25 and 0.75; (ii) the Spearman correlation, per origin then averaged, between spread and the absolute error of the point prediction, which says whether the spread carries any information about error.

---

## G1/G2. Redesigning the emerging ranking (Tracks 3 and 4 and the white-top check are on hold)

**Finding that motivates this (2a):** the deployed emerging score, `predicted intensity / trailing 13-week mean`, mostly ranks styles by low recent intensity (mean reversion). A constant global-mean forecast scores 0.733 Hit@3-in-top20 on it, against the model's 0.767 and seasonal naive's 0.800 (random floor 0.177).

**Three scores, all computed for the model and for every baseline's own forecast (log1p-scale forecasts `f`, `sn` = seasonal-naive forecast for the same style and origin, `trailing` = trailing 13-week mean intensity):**

1. **Current ratio:** `expm1(f) / trailing`.
2. **Redesigned (the candidate): excess over the naive forecaster:** `f - sn` on the log1p scale, i.e. the model's forecast minus the seasonal-naive forecast for the same style and horizon. No trailing-mean denominator. For the seasonal-naive baseline this score is identically 0 (a constant, no ranking), so it is reported as degenerate rather than ranked.
3. **Residualised ratio:** at each origin, ordinary least squares of the current ratio on `trailing` over the scored population, then the residuals (literal reading of the request; comparison only).

**Population (identical for every score, method and floor, so all comparisons are paired).** P3 = the deployed emerging pool (guards 1-3, trailing mean above zero, predicted intensity at or above the median of the guard-passing styles) **intersected with styles for which a seasonal-naive forecast exists** (a full 13-week window one year earlier). Limitation, stated in advance: the redesigned score cannot rank a style younger than about 65 weeks, so it can never nominate the newest styles; the number of styles lost to this is reported per origin.

**Realised target (all scores):** the 2a realised growth, `expm1(y_true) / trailing`, so that every score is judged against the same outcome.

**Origins.** The 10 evaluable grid origins from 2a (the first two have no `price_index`), and the weekly grid of 2b (48 origins, of which those with a non-empty P3 are evaluable, count reported, same 12 models), block length 4 and 13 respectively, 2,000 resamples, seed 42.

**PRIMARY (decision-bearing).** The redesigned score's Hit@3-in-top20 minus **seasonal naive's current-ratio Hit@3-in-top20** (the strongest baseline on the deployed formula, 0.800 in 2a), paired per origin, must have a 95% block-bootstrap interval whose lower bound is above 0 (with the 1e-9 tolerance) **on both** the 10-origin grid **and** the weekly grid. One of the two is a failure.

**BASE-EFFECT DIAGNOSTIC (mandatory, stated first in the report).** Under the redesigned score, the constant global-mean baseline (`global_mean forecast - sn`) must fall to within noise of the random floor. Fixed criteria, both required, on both origin sets: (i) its pooled Hit@3-in-top20 is within 0.10 of the random floor's, and (ii) its paired difference from the random floor has a 95% interval whose lower bound is at or below 0. If it still scores near 0.733 the base effect was not removed, and that is reported as a failure regardless of the primary.

**ADOPTION RULE.** The redesigned score is adopted for selection **only if the primary passes AND the diagnostic passes.** Otherwise it is not adopted.

**FALLBACK (stated now, no further variants).** If the redesigned score is not adopted, the final three are chosen from the validated INTENSITY table (the incumbent list, guard-passing styles ranked by predicted intensity with the existing diversity constraint on (product type, colour)), then through the unchanged `reselect_final_three` rules: intimates and visual-ambiguity exclusions, no two styles sharing a colour or a product type, first three that survive, shortfall reported and not backfilled.

**G2 selection procedure.** At the live forecast origin 2020-09-21 with the frozen final model: under adoption, the emerging list is the P3-style pool ranked by the redesigned score (diversity-constrained on (product type, colour), top 10) and then the same `reselect` rules; under the fallback, as above. A control gate first reproduces the committed `top_styles_emerging.csv` and the current final three with the existing code. Reported: the new final three, the full skip log, and which of the current three (sweater, dress, white top) survive. **Nothing is regenerated**; the user decides whether regeneration is worth it.

**Outcome and decisions after G1/G2 (user):** the redesigned score was not adopted; the fallback three were reported; concepts are NOT regenerated and the submitted three remain the deliverable; the four-season concept generation (Track 3b) is dropped for the same selection problem; the per-season performance split (Track 3a) is kept because it is evaluation only.

---

## H2. Per-season performance split of the intensity backtest (evaluation only)

**Source.** The 48 weekly origins of 2b (same 12 models, intensity ranking, all styles), from `v3_power_per_origin_weekly.csv`: model, the four baselines and the random floor, all seven metrics.

**Season of an origin (fixed).** The calendar month of the **midpoint of its 13-week target window** (origin + 7 weeks), mapped with `final_forecast.SEASON_MONTHS`: winter Dec-Feb, spring Mar-May, summer Jun-Aug, autumn Sep-Nov.

**Reported per season, per comparator (seasonal naive, EWMA persistence, parent-category mean, global mean, random floor), per metric:** the model's mean, the comparator's mean, the paired mean difference (model minus comparator) and a 95% moving-block bootstrap interval (2,000 resamples, seed 42) with **block length 4** (the project standard; the 13-origin block of 2b cannot be used inside a cell of about a dozen origins). Also the number of origins, the number where the comparator is defined, and both effective-sample-size estimators of 2b.

**Thin-cell rule (fixed).** A cell is flagged **THIN** if fewer than 8 origins have both methods defined, or its `ess_ac` is below 5. A THIN cell is reported with its point estimate and its interval, marked "direction only, not a claim". A cell that is not THIN is still one season observed once.

**Stated in advance, not to be softened later:**
- Each season occurs **once** in the test period (about one year of test origins), so a season's origins are one contiguous stretch whose outcome windows overlap heavily. A per-season interval describes one 13-week stretch, not "that season in general". No claim of a seasonal pattern in model quality can be made from this.
- Season is confounded with calendar time (each season's model was trained on a different training set) and with COVID (windows overlapping March-June 2020 fall in spring and summer cells).
- No between-season comparison is tested; the table lets the reader see the numbers, nothing more.
- Seasonal naive is undefined before 2019-09-16 (52-week lag), which thins the autumn cell.
