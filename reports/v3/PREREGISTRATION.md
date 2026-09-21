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

- **Weekly origins:** every Monday from the first walk-forward test origin (2019-07-29) to the last origin whose 13-week outcome fits in the panel (2020-06-22): 47 origins. The 12 grid origins are a subset.
- **Reproduction gate:** at the 12 grid origins the per-origin model metrics must equal the committed embargoed per-origin table (`backtest_embargo_per_origin.csv`, lightgbm) to 1e-9, or the run stops.
- **Uncertainty:** moving-block bootstrap on the per-origin paired differences, **block length 13 origins (the 13-week horizon)**, 2,000 resamples, seed 42, at weekly spacing. The 4-week run uses the reported block length 4 (which is also about the 13-week horizon expressed in origins). Both are shown side by side. Seasonal naive is undefined without a 52-week lag, so its paired rows use fewer origins; n is reported.
- **Effective sample size, two estimators, both reported.** `ESS_ac = n / (1 + 2 * sum_{k=1..L} (1 - k/(L+1)) * rho_k)` with `L = 12` (Bartlett-weighted autocorrelation of the paired difference series; floored at 1, capped at n) and `ESS_boot = n * Var_iid(mean) / Var_block(mean)`, the iid variance of the mean over the block-bootstrap variance of the mean. The raw origin count is never presented as the sample size.
- **Floor and comparators:** the random-permutation floor and all four baselines are in the same table; the question concerns seasonal naive only.
- **Reporting rule:** the result is stated as "the interval does / does not exclude zero at weekly spacing". If it does, it is presented as "the same model with more power", never as an improved model.
