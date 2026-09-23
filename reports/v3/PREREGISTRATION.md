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

---

## H3. The agent layer: live run (4a), critic decisions and routing (4b), gate ablation (4c)

Committed before any H3 harness or result. **What the "agent layer" is, stated plainly:** `agents/*.md` are role definitions and `scripts/agent_demo.py` is a scripted driver; no LLM makes any agent decision. So "the critic's accept/reject decision" is the deterministic rule in `agents/critic.md` (REJECT if any gating gate fails; PASS_PENDING_HUMAN if all five gating gates pass; INCONCLUSIVE on a tool error or a `not_run`/null/malformed result after one scoring retry), and "the orchestrator routes" is `agents/orchestrator.md`'s delegation flow, implemented for this evaluation as an executable `route()` function and checked against an expectation table written below, before it is coded. What is evaluated is that rule plus the gates behind it, against human-derived labels. A finding about it says nothing about an LLM agent.

**Disclosure of what was already known.** The stored gate columns of the 96 wrong-style rows (`v3_yield_floor_scored.csv`) are Track 1 results and were seen before this rule was written: integrity fails 96, Gate 2 fails 95, Gate 3 fails 94, Gate 1 and Gate 1b fail 0, and 0 of 96 pass every gate. Nothing else in this section has been run.

### 4b. Decision eval (critic) and routing

**Unit and scoring.** One case = (image, target style, briefed changes), scored once through the MCP `score_concept` tool with `include_fidelity=true` and the brief's changes (`concept_generation.CHANGES[style]`, or the image's sidecar for H3), over the same stdio protocol the demo uses, CPU only. The critic decision is taken on the first attempt of each case.

**Cases and labels (fixed).**

| Stratum | Cases | Truth | Label source |
|---|---|---|---|
| N1 malformed | the three malformed underwear images: `final_concepts_h3/..._seed42, _seed44, _seed45` | REJECT | `integrity_labels.json` (`coherent: false`), `underwear_scored.csv` `visual_qc` |
| N2 swatch | `n1_levers/..beige_melange/concat-0.15_s43.png`, `concat-style-only-1.0_s42.png` (sweater style) | REJECT | `integrity_labels.json` ("fabric swatch, no garment") |
| N3 exact clone | for each of the four styles in `ALL_SELECTED`, a byte copy of that style's first screened reference photo (`load_refs()[style][0]`), placed under `data/generated/agent_eval/` | REJECT (a copy of a real product is not a new concept, by construction) | positive-control design, `clone_positive_control.csv` |
| N4 wrong style | the 96 rows of `v3_yield_floor_scored.csv` (24 images of each source style scored against each of the other target styles) | REJECT | the image depicts a different style; Track 1 |
| P1 approved (primary) | the four submitted concepts, `final_selection_figures.ALL_SELECTED` | ACCEPT | recorded human check `HUMAN_CHECK` with `HUMAN_BRIEF_MET` all `True` |
| P2 coherent (secondary) | generated images labelled `coherent: true` in `integrity_labels.json`, not in P1: H3 seed 43, the three `n1_levers` clean sweaters, the M3 sweater and the M3 dress | ACCEPT | `integrity_labels.json`. **Coherent is not "brief met":** the human label does not cover Gate 3's question, so P2 is reported separately and never pooled into the primary recall |

Real catalogue photos labelled coherent in `integrity_labels.json` are excluded: they are members of the reference set, so a Gate 1b reject is correct by construction and they are not concepts.

N4 uses the stored gate columns; a **live parity sample** of 6 of the 96 rows (`random.Random(42).sample(range(96), 6)`, indices into the CSV order) is re-scored live, and any disagreement between live and stored gate values is reported and the live value used for that case.

**Metrics (fixed).** Positive class = ACCEPT (the critic returns PASS_PENDING_HUMAN). INCONCLUSIVE counts as not-accept and is listed separately. Recall = accepted / all truth-ACCEPT cases; precision = truth-ACCEPT among all accepted. **Primary:** positives = P1 (n=4). **Two negative sets, both reported, neither chosen after seeing results:** *hard* negatives = N1+N2+N3 (n=9); *all* negatives = N1..N4 (n=105). Wilson 95% intervals on every proportion. Secondary: the same with P1 + P2 as positives (n=10). **Floors:** the accept-everything rule (precision = prevalence, recall = 1) and a random accept rule at that prevalence (precision = prevalence, recall = prevalence); the prevalence is stated for each negative set. The false-accept and false-reject cases are listed by name. Known limits stated now: n=4 positives cannot support a recall interval narrower than about [0.51, 1.00] even at 4/4; most negatives are the easy wrong-style set, so *all-negative* precision is inflated and *hard-negative* precision is the informative one; and the human check is outside this measurement (a PASS_PENDING_HUMAN on a known-bad case is a case the human must catch, not a shipped one).

**Routing expectations (written from `orchestrator.md` and `critic.md` now; `route()` is coded after).**

| Situation | Expected next hop |
|---|---|
| REJECT at attempt 1 or 2 | `concept-designer` (via the orchestrator) with one adjusted parameter: `seed` if integrity is the only failing gate, else `ip_adapter_scale` (`critic.md`); prompt, references and backend unchanged |
| REJECT at attempt 3 (cap: 1 original + 2 retries) | none; the orchestrator reports `FAILED` with the full history; no `concept-designer` call |
| PASS_PENDING_HUMAN | `forecaster` (`forecast_concept`), then the human visual check is recorded; never final without it |
| INCONCLUSIVE after one scoring retry | escalate to the orchestrator as "QC inconclusive"; no `forecaster`, no `concept-designer` |

Checks, all fixed: (a) for every case `route(verdict, attempt=1)` equals the table's hop for its actual verdict (this tests the function against the spec, not the verdict against the truth); (b) the retry cap on a three-attempt scenario built from the three N1 images in seed order 42, 44, 45: attempts 1 and 2 that REJECT route to `concept-designer`, a REJECT at attempt 3 routes to FAILED, and there is no fourth call; (c) five fail-closed unit cases with a stubbed `score_concept`: it raises twice (expect INCONCLUSIVE after exactly one retry), it raises once then succeeds (expect the normal verdict after exactly one retry), a result with Gate 2 `not_run` (INCONCLUSIVE), a result with Gate 3 `pass: null` (INCONCLUSIVE), and a result with `clone_control_failed_as_required: false` and Gate 1b `pass: false` (INCONCLUSIVE, per `critic.md`'s malformed-data rule); (d) tool allowlists parsed from `agents/*.md`: the hop's target agent must list the tool that hop needs (`concept-designer`: `generate_concept`; `forecaster`: `forecast_concept`), and the critic must not list `generate_concept`, `forecast_concept` or `forecast_styles`. A failed check is reported as a defect in the spec or the router; I do not edit an agent definition to make a check pass.

### 4c. Gate ablation

**Design.** Over every case scored in 4b (N1..N4, P1, P2), one score per image (the N4 rows use the stored columns). Remove each of the five gating gates in turn from the decision rule and recompute every decision. This is valid at the decision layer because each gate's `pass` is computed independently of the others in `qc_gates.score_gates` and `concept_scoring.apply_panel_rule` (read before writing this rule: no gate's pass reads another's result; the per-judge rows carry the integrity flag only as a column). Per gate, per stratum, report: **catches** (the gate is among the failing gates), **unique catches** (the gate is the only failing gate, i.e. the case newly becomes PASS_PENDING_HUMAN if the gate is removed), and **unique false rejects** (truth-ACCEPT cases for which the gate is the only failing gate). Also report the recomputed precision and recall with each gate removed, and the set of known-bad cases that pass every automatic gate (only the human check stops them).

**Interpretation rule (fixed).** A gate with zero unique catches over N1..N4 is reported as a finding, **not removed** and not proposed for removal here: unique-catch is defined against this case set only (a control's surface is the cases it was tried on), overlap between gates is what defence in depth looks like, and the removal decision is the user's. The report states each such gate's other roles (for example Gate 1b's clone validation) and the size of the case set, without a claim of "redundant".

### 4a. Fully live run

**Scope.** The three autumn/winter styles at forecast origin 2020-09-21 (`SWEATER`, `DRESS`, `TOP`). The summer concept is excluded: `forecast_styles` serves only the 2020-09-21 tables, so a summer step would have to be replayed, and nothing may be.

**Labels allowed in the transcript.** `LIVE` = a real MCP tool call over stdio through the allowlist-checked driver; `INPUT` = a fixed human-authored input (the brief `CHANGES`, the seeds, the starting scale); `LOCAL` = an in-process pure function (prompt builder, critic decision, router); `NOT PERFORMED` = the human visual check. **The word REPLAY must not appear as a step label anywhere**; the transcript ends with a count of steps by label, and the driver fails its own lint if a step is unlabelled. The existing `reports/agent_run_transcript.md` is not modified; the new file is `reports/agent_run_transcript_live.md`.

**Chain per style:** forecaster `forecast_styles` (live) -> style-profiler `get_style_profile` (live) -> concept-designer `generate_concept` (live, GPU; a separate server subprocess from the CPU scoring server) -> critic `score_concept` (live, `include_fidelity=true`, changes from the brief) -> on REJECT, the critic's adjusted parameter to concept-designer, cap 1 + 2 retries -> on PASS_PENDING_HUMAN forecaster `forecast_concept` (live, default origin) -> the human check recorded as NOT PERFORMED with the image path for a person to open.

**Generation call, fixed to what the MCP tool exposes:** `backend=local_sdxl`, prompt = `natural_prompt(style, changes, None)`, `reference_images` = the first 8 screened references (`N_REFS`), `ip_adapter_scale=0.35` on attempt 1, `seed=42`, `n=1`. **Retry rule (from `critic.md`, fixed now):** if integrity is the only failing gate, the next attempt changes only the seed (+1); otherwise it changes only `ip_adapter_scale` (-0.10, floor 0.15). No other parameter is changed.

**Reported regardless of outcome:** which gates each live attempt failed; the finding that the MCP `generate_concept` path is **not** the configuration that made the deliverables (the deliverables used `levers.generate_variant`, `mode="concat"`, a weighted compel prompt and a negative prompt; the MCP tool exposes neither a negative prompt nor concat mode), so its gate outcomes are not evidence about the deliverables; the wall time per step; $0 cost (local). No image from the live run is a candidate for the submission, and I do not look at the images to fill the human-check step: that step is the user's.

---

## J. LLM orchestration, hard negatives, critic strictness

Committed before the J2 runs, the J6 generation and scoring, and the J7 diagnosis. Already done and NOT covered by a pre-registered comparison: J1 (feasibility: headless Claude Code runs the sub-agent pattern with MCP tools, probe run before this commit), J3 (the unified rule below in "J3 reference"), J4 (byte-level reproduction check), J5 (filename fix and test).

### J3 reference (the deterministic rule J2 is scored against)

`nss.generate.critic_rule.decide` plus `agent_eval.route`, as committed with J3: a gate is unmeasured if its `pass` is null/missing or it is Gate 1b with a clone control that did not fail as required; any definite failure -> REJECT (even with an unmeasured gate); else any unmeasured gate -> INCONCLUSIVE (escalate, no regeneration); else PASS_PENDING_HUMAN. Routing: REJECT at attempt 1-2 -> `concept-designer` with `seed` (integrity is the only failing gate) or `ip_adapter_scale`; REJECT at attempt 3 -> FAILED, no further call; PASS_PENDING_HUMAN -> `forecaster` (`forecast_concept`), then the human check; INCONCLUSIVE -> escalate to the orchestrator, no `forecaster`, no `concept-designer`.

### J2. LLM decision agreement and consistency

**Set.** The 115 routing cases of H3 (19 scored live + 96 wrong-style stored rows; strata N1 malformed 3, N2 swatch 2, N3 clone 4, N4 wrong style 96, P1 approved 4, P2 coherent 6) at attempt 1, plus a **cap set**: the 19 live cases at attempt 3 (history states attempts 1 and 2 were REJECTs). Each case is shown to the model as the `score_concept` result of its critic sub-agent, projected to one schema for live and stored cases alike: per gate `pass` (true / false / null), `gate1b.clone_control_failed_as_required`, and Gate 3 `status` (null pass = not run). Measured values are not shown.

**Model call.** `claude -p` headless (Claude Max login, `ANTHROPIC_API_KEY` removed from the environment), `--model sonnet`, no tools, no MCP, no settings/hooks (`--tools "" --setting-sources ""`), no session persistence, system prompt = the bodies of `agents/critic.md` and `agents/orchestrator.md` with the instruction to play both roles for this one decision. The reply must be one JSON object: `verdict`, `outcome` (`RETRY` | `FORWARD` | `FAILED` | `ESCALATE_INCONCLUSIVE`), `next_agent`, `adjust` (`seed` | `ip_adapter_scale` | null), `reason`. Each case is run **3 times** independently (default sampling). An unparseable reply counts as a disagreement and is reported as such. Marginal cost is $0 on the Max plan; about 500 input tokens per call.

**Reported (per stratum and overall).** (1) Run-level agreement with the reference on `verdict`, on `outcome`, and on `adjust` for RETRY cases, with Wilson 95% intervals (runs of one case are correlated, so the intervals are optimistic; stated). (2) Case-level: **consistency** = share of cases whose 3 runs give identical (verdict, outcome, adjust); **agreement of the majority run**; cases where consistent-and-wrong. (3) Every disagreeing case and run listed, with the model's reason and an adjudication: whichever of the model and the rule is right **per the text of `critic.md` / `orchestrator.md`**; where the text is silent or ambiguous it is labelled a spec gap and scored as neither. `adjust` is expected to be spec-underdetermined (`critic.md` says "a different `ip_adapter_scale` if a Gate 1/1b/2/3 miss, or a different `seed` if the failure looked like a one-off sampling artifact"; the reference makes it a rule for integrity-only failures), so it is reported apart from verdict and outcome.

### J6. Hard negatives for the ablation

Generation config: the deliverables' (`levers.generate_variant`, concat mode over the style's 8 best references, scale 0.35, seed 42, compel-weighted prompt, brief negative prompt) for the four styles in `ALL_SELECTED`, GPU, $0. Every case is scored once through the MCP `score_concept` (CPU, `include_fidelity=true`, briefed changes = `CHANGES[style]`) and its truth is REJECT.

| Class | Cases | How built | Truth by |
|---|---|---|---|
| H1 wrong attribute (right style, wrong colour/pattern) | the M3 coral-pink "red" dress in both M3 attempts (`final_concepts_m3/attempt1|2/..._dress_..._seed45.png`, identified by looking at all eight M3 dress images against the description; **the same file as P2 `m3_dress`**) + one generation per style with the colour or pattern replaced in the prompt (sweater "navy blue", dress "emerald green", top "black", bikini top "plain blue"), the changes kept | 2 + 4 = 6 | the attribute is wrong by construction |
| H2 briefed change absent | one generation per style with the briefed changes left out of the prompt and added to the negative prompt | 4 | the brief is absent by construction |
| H3 averaged garment | one image per style: the pixel mean of the style's 8 best screened references (resized to 1024 square) | 4 | no single coherent garment by construction |

Label QA: I look at each generated image once and **exclude and report** any H1/H2/H3 case that visibly does not have the intended defect (colour still the original, a briefed change visibly present, a coherent garment). That check is mine, not the human visual check, and is disclosed as such. The M3 coral-pink dress was identified by me from the images; I ask you to confirm.

**Reported.** The per-case gate results; per class and gate: catches, **unique catches** (removal would newly let the case reach PASS_PENDING_HUMAN; also the sole-failing count), and any hard negative that passes every automatic gate (a real false accept, listed). The H3 correction: the M3 dress `m3_dress` was scored as an ACCEPT (coherent) in H3 P2 but is coral-pink, i.e. also a wrong-attribute case; P2 is recounted without it and both recalls are stated. No gate is removed and no threshold changed; the report says which gate uniquely catches each class, or that none does.

### J7. Critic strictness (report only)

For the two submitted concepts chosen without a gate pass (white jersey top, bikini top), from the cached live scores (`v3_agent_eval_scores.jsonl`): per failing gate, the measured value against its threshold and the margin (absolute and relative); for Gate 3 the per-change answers; for Gate 2 the judge's per-attribute reading against the style's attributes and against the recorded human observation (`final_selection_figures.HUMAN_CHECK`). Classification, fixed now: **marginal** if the measured value is within 10% of its threshold (relative), otherwise **not marginal**; **judge misread** if the judge's answer contradicts the recorded human observation, otherwise **judge consistent with the human record**. The report says which gate drove each rejection and whether the threshold or the judge is implicated, with the evidence; **no threshold, judge or gate is changed.**

---

## K. Where the rule is silent (K2), Gate 2 identity constraints (K4), Gate 1 advisory (K5)

Committed before K2 is run and before K4 is scored. K3 (the scale-direction rule) and K5 are decisions plus a replay of recorded cases, described here so their counting is fixed.

### K2. Judgment where the written rule is silent

**Cases.** Ten scenarios, `evals/fixtures/silent_rule/S01..S10.json` (generator `evals/build_silent_rule_fixtures.py`, committed with this section): scale direction for a Gate 3 miss at 0.35 (S01) and at 0.55 (S02), an integrity-only miss (S03), integrity and Gate 3 failing together (S04); a Gate 2 judge that reads the same thing on every seed while the advisory judge disagrees, on the last attempt (S05); a gate returning an error, transient (S06) and deterministic (S07); conflicting judges in both directions (S08, S09); every gate passing after a person has said the brief is not met (S10). Each fixture carries the scenario, the `score_concept` result with measured values (not only flags), and a **rubric written now**: the good decision, what is equivalent, what is worse, and what would make a decision better.

**Model.** Headless Claude (Sonnet), same call shape as J2 (no tools, no MCP, no settings, `ANTHROPIC_API_KEY` removed), system prompt = `agents/critic.md` and `agents/orchestrator.md` **as they stand at commit 896b10d, i.e. before the K3/K5 edits, so the rule is genuinely silent**, plus an instruction to answer with one JSON object (`verdict`, `next_agent`, `outcome`, `adjust` as {param, direction, value}, `escalate_to_human`, `flags_for_human`, `reason`). **Two arms**, 3 runs each: **A** spec only; **B** spec plus a fixed "project evidence" note (the lever and calibration facts below). 10 cases x 2 arms x 3 runs = 60 calls.

*Project evidence note (arm B), fixed now:* with the production prompt and 8 concatenated references, briefed changes appear at ip_adapter_scale 0.25 to 0.35 and are weaker at 0.45; at 0.6 to 0.7 the references dominate and the changes disappear; with the older prompt, 0.15 and 0.25 collapsed into fabric swatches; the swept window is 0.15 to 0.45 and the per-style choice is 0.35. Gate 3's local reader says yes too easily (specificity 0.58). SmolVLM's Gate 2 reading of the bikini top is identical (fidelity 0.283) on all 24 seeds, and Florence-2 reads its pattern correctly.

**Grading.** Each run is graded against its rubric as **better** (meets a `better_if` beyond the good decision), **equivalent** (the good decision or a listed equivalent), **worse** (a listed worse, or violates the good decision without justification) or **other** (a defensible call the rubric did not anticipate, argued in the report). I grade; the reference for "what a person would choose" is the rubric I wrote from the project evidence, standing in for you, and you can overrule any grade. The report gives per case and arm: the three grades, the model's reasoning for each divergence, and the consistency across the 3 runs. Nothing is tuned after seeing answers; a rubric found wrong is reported as wrong, not edited.

### K4. Gate 2: product type and colour as hard constraints

**Rule.** Gate 2 passes iff **(a)** the gating judge's fidelity is at or above its calibrated threshold (**scorer, averaging and threshold untouched**, so the calibration stands) **and (b)** the gating judge's product-type reading and colour reading each match the style's value under `nss.generate.identity_match` (`PRODUCT_SYNONYMS`, `COLOUR_SYNONYMS`, committed with this section). Pattern is not constrained beyond the average. Florence-2 stays advisory and unconstrained.

**Matching.** Normalise (lower case, punctuation to spaces). Find every word of every family that occurs as a whole word in the reading, keeping the longest overlapping phrase. Match iff the set of families found is exactly {the style's family}. So a reading naming a different family ("sweater dress", "red and white", "bikini top" for a Top) fails; a reading naming no known word ("coral") fails; an unknown style value falls back to token containment in either direction.

**Why this cannot loosen Gate 2.** (b) is an AND on top of the unchanged (a): it can only turn a pass into a fail. The synonym lists can only prevent new failures. Consequence stated in advance: **the bikini cannot flip to pass**, because its fidelity (0.283) is below 0.384 regardless of synonyms. I will still report exactly which reading and entry decides each result. The separate scorer defect (the trailing period makes "Bikini." fail to match "Bikini top" in the averaged score) is NOT fixed here, since it would move the calibration; it is reported.

**Synonym lists and justification (fixed before any scoring).** *I had seen these judge readings before writing the lists: "Sweater.", "Turtleneck.", "T-shirt.", "Dress.", "Bikini.", and the colours "Orange.", "Green.", "Brown.", "White." from earlier reports; I had not seen the readings of any candidate under the new rule.*

| Style value | Accepted words | Justification |
|---|---|---|
| Sweater | sweater, jumper, pullover, turtleneck, polo neck, roll neck, knit, knitwear | Names for one garment (UK/US); turtleneck and roll neck are neck variants of a sweater, and the sweater brief itself asks for a high funnel neck, so "turtleneck" is a *correct* reading of the design; "knit/knitwear" is the catalogue's word. Not accepted: cardigan, hoodie, sweatshirt, top. |
| Dress | dress, sundress, frock | Same garment. |
| Top | top, blouse, shirt, t-shirt, tee, tank top | Upper-body jersey/woven tops a judge reasonably calls a plain top; blouse and shirt as you listed. Loosest entry: t-shirt/tee (a separate H&M type). It can only prevent a new failure. |
| Bikini top | bikini top, bikini, swim top, swimsuit top, bikini bra | "Bikini" is the head noun a judge gives for a single-top image; it also fits a bikini bottom, which is the looseness. |
| T-shirt | t-shirt, tee, tshirt, t shirt | Spelling variants only ("top" is deliberately not accepted). |
| Underwear bottom | underwear bottom, underwear, briefs, panties, knickers, underpants | Names for the same garment. |
| Trousers | trousers, pants, slacks, chinos | Names for the same garment (jeans deliberately not). |
| Blazer / Cardigan | blazer, suit jacket / cardigan, cardi | Same garment. |
| Colour | Accepted words | Justification |
|---|---|---|
| Beige | beige, tan, sand, oatmeal | Light warm neutrals inside H&M's Beige master colour (camel and taupe left out: browner). |
| White | white, off-white, ivory, cream | H&M's White includes off-white; cream is between white and beige and is put with white. |
| Black | black | No neighbours. |
| Red | red, crimson, scarlet, burgundy, maroon, wine | Shades of red. **Deliberately excluded: coral, salmon, pink, orange, rust**, the neighbours the coral dress and a colour-swapped dress need to fail. |
| Orange | orange, tangerine, amber | Shades of orange (rust and coral left out). |
| Blue, Green, Grey, Pink, Brown, Yellow, Purple, Turquoise | blue/navy/cobalt; green/emerald/olive/mint; grey/gray/silver/charcoal; pink/rose/blush/fuchsia; brown/chocolate/mocha/chestnut; yellow/mustard; purple/lavender/violet; turquoise/teal | Shades inside each master colour; also the vocabulary that makes "red and white" or "emerald" a conflict for a red style. |

**Required outcomes (stated before scoring).** The emerald-green dress (`H1_wrong_attribute_Dress`) and both coral dresses (`m3_coral_dress_attempt1`, `_attempt2`) must FAIL Gate 2; the submitted sweater and dress must still PASS it. If any of these is not met, that is reported and the rule is not adjusted to fit. **Reported separately and plainly:** the effect on the submitted white top and bikini top; every Gate 2 verdict that changes across all stored cases (the 129 H3/J6 cases and the 54 recorded candidates, rescored offline from the stored extractions, no judge re-run).

### K5. Gate 1 advisory

Gate 1 keeps being computed and reported, and is removed from the accept/reject decision (`critic_rule`, `critic.md`, `SKILL.md` together). Expected and to be reported: **no verdict changes** across the 129 cases (Gate 1 failed in none of them).

### K3. Scale-direction rule (decision, with a replay)

Written into `critic.md` from the lever evidence above, per failure type. To be reported: which recorded retry decisions it would have changed (the scripted live run, the LLM run's attempts, and the J2 `adjust` reference).

---

## L. Measured colour, retrieval product type, restructured identity check, blind re-grade

Committed before any L1/L2/L3 scoring, before the L4 judges are called and before the L5 re-run. Environment facts already known and stated: `rembg` is not installed (and needs a model download); `scikit-image`, `scikit-learn`, `scipy` and `opencv` are.

### L1. Measured colour check

**Why border sampling and not rembg.** rembg is not installed, would need a network model download, and its failure modes on white-on-white flat-lays are unknown; a border-sampled background needs no dependency, is deterministic and is inspectable. Its known weakness (a white garment on a near-white background yields a tiny mask) is handled by a stated fallback below, and reported.

**Method (fixed).** (1) Load RGB, resize so the longest side is 256 px (LANCZOS). (2) Convert to CIELAB (skimage, D65). (3) Background colour = per-channel median Lab of the outer ring (4% of width and height). (4) Candidate mask = pixels whose CIE76 distance from the background is above 12. (5) Morphological opening (3x3), keep the largest connected component, fill holes, erode by 3 px (drops the anti-aliased edge and drop-shadow fringe). (6) If the mask covers under 3% of the image, use the central 40% box instead (the white-on-white fallback; every use is counted and reported). (7) Dominant garment colour = centre of the largest cluster of k-means (k=3, `random_state=42`, `n_init=10`) over the mask's Lab pixels.

**Distance and statistic (fixed).** CIEDE2000 between dominant colours. For a concept, `d` = the minimum CIEDE2000 from its dominant colour to the dominant colour of each of the style's real reference articles (`qc_gates.reference_paths_for_style`, the same references the other gates use).

**Threshold, calibrated exactly like Gate 1b (fixed).** Per style, the p90 of the real nearest-sibling distances: for each real reference, the minimum CIEDE2000 to the other references; the threshold is the 90th percentile of those values. A concept passes iff `d` is at or below its style's threshold.

**Leave-one-out validation on real articles (fixed).** For each real reference `i` of a style: remove it, recompute the other references' nearest-sibling distances among the remaining ones and their p90, then test whether `i`'s minimum distance to the remaining references is at or below that threshold. Report the real-article pass rate per style and overall. **Stop rule:** if the overall pass rate is below 80%, or any of the four final styles is below 75%, the mask or the distance is wrong: report and stop, without scoring the concepts.

**Required outcomes (stated before scoring; no tuning if unmet).** **FAIL:** the emerald-green dress (`hard_negatives/H1_wrong_attribute_Dress.png`) and both coral dresses (`final_concepts_m3/attempt1|2/..._dress_..._seed45.png`). **PASS:** the submitted dress (`n9/..red_solid/s0.35_seed44.png`), the submitted sweater (`n9/..beige_melange/s0.35_seed45.png`), and the two red dresses K4 rejected (`n9/..red_solid/s0.35_seed47.png` and `s0.45_seed43.png`). Reported alongside, not required: the submitted white top and bikini, the mask-fallback count, and the colour distances of every case.

### L2. Product type from retrieval

The product type of an image is that of the top-1 style of `concept_forecast_index` retrieval (CLIP + DINOv2 averaged cosine to the mean embedding of each style's real photos, at most 8 per style, headline view unchanged), over the union of the autumn table (1,980 styles) and the summer table (3,000 styles), 3,003 distinct styles, all with photos. The check passes iff that style's `product_type_name` equals the style's own product type as an exact H&M string: **no synonym list**. The 82.5% product-type accuracy on the 40 real photos is the submission's figure (`reports/SUBMISSION/WRITEUP.md`) and is not re-measured here. **Required:** the four submitted concepts (sweater, dress, white top, bikini) pass; reported plainly for the white top and the bikini whatever happens.

### L3. Restructured Gate 2

Gate 2 passes iff **(a)** the SmolVLM averaged fidelity clears its calibrated threshold (scorer, averaging, threshold untouched; it still contains the judge's own colour and product readings, which no longer decide anything alone) **and (b)** the measured colour check passes **and (c)** the retrieval product type passes. The K4 VLM-reading constraints (`identity_match`) are removed from the decision. Pattern stays VLM-scored and averaged. Gate 3 stays the only VLM-decided yes/no question. `critic_rule`, `critic.md`, `SKILL.md` and `qc_gates`/`concept_scoring` change together. **Re-run** on the 129 cases (115 H3 + 14 J6, including the 4 label-QA exclusions, flagged) and the 54 recorded candidates, offline from stored judge readings for the fidelity part (no judge re-run) and freshly computed colour and retrieval. **Reported:** the all-gate pass count on the 54 recorded candidates against the K4 figure (9 before K4, 5 after) and the new figure; every verdict that changes against both the pre-K4 and the K4 verdicts, with the reason (which of (a), (b), (c) flipped).

### L4. Blind independent re-grade of K2

The 60 K2 decisions (`v3_silent_rule_runs.jsonl`), the ten scenarios and the rubric text (`evals/fixtures/silent_rule/`) go to two non-Claude text models from different families: **Gemini** and a **Groq-hosted Llama or Qwen** (the first available of `llama-3.3-70b-versatile`, then a Qwen model; recorded). They receive, per case, the scenario, the `score_concept` result, the rubric and the six decisions with **shuffled anonymous ids** (seed 42) and **without the arm, the run number or any Claude grade**; the instruction is to grade each decision better / equivalent / worse / other against the rubric, as JSON, temperature 0. One request per case (10 per judge; Gemini's free tier is 20 requests a day). **Reported:** each judge's better/equivalent/worse/other counts; pairwise Cohen's kappa over the graded decisions among {my grades, Gemini, the Groq model}, on the four grades and on a collapsed "acceptable (better or equivalent) vs not"; every decision where either independent judge disagrees with my grade, with the disagreement. Labelled **LLM consensus, not human ground truth**. If a quota blocks a judge, I report which and how far it got and do not substitute a Claude model.

### L5. The scale habit against the K3 rule

S09 only, 3 runs, Sonnet, same call shape as K2, system prompt = the **current** `agents/critic.md` and `agents/orchestrator.md` (K3 in context) and no evidence note. **Good** (the K3 rule): REJECT, RETRY with a **seed** change and the scale kept (this is attempt 1, so no repeat yet; escalation is for a repeated reading). **The habit persists** iff at least one of the 3 runs changes `ip_adapter_scale`. Utilisation is checked and reported before the run.

---

## M. Shrunk colour thresholds, rembg masks, a borderline band, and the K2 statement

Committed before any M1-M3 scoring. **Facts stated in advance.** Before M2: the L1 thresholds (CIEDE2000, p90 of real nearest-sibling distances, border-sampled mask) were sweater 3.44, dress 4.50, white top 1.06 (all 19 references on the central-box fallback), bikini top 16.44 (8 references), underwear 23.72 (4 references); nested leave-one-out on the four final styles 0.884 (`v3_colour_loo.csv`). Installing rembg needs 5 packages into the project venv with `pillow` held at the project's pinned 10.4 (rembg 2.0.69, onnxruntime 1.30.0, pooch, pymatting, flatbuffers) and downloads the u2net model, **175.997 MB** (`u2net.onnx`, checked with a HEAD request; `U2NET_HOME=data/rembg_models`, not committed).

### M2. rembg mask (changes the mask source only)

The garment mask is the rembg `u2net` alpha mask of the same 256-px image (`only_mask=True`, binarised at 128), then the unchanged steps: largest connected component, holes filled, 3 px erosion, k-means dominant colour, the 3% fallback to the central box (counted and reported). Applied to **every** style and to every reference and concept. Reported: masks still on the fallback, the white top's new threshold, the per-style leave-one-out pass rates, and any style whose raw threshold moves by more than 25% (relative) or more than 2 CIEDE2000 units.

### M1. Empirical-Bayes shrinkage of the threshold

`T_style = w * p90_style + (1 - w) * T_global`, with `w = n / (n + K)`, `n` the style's reference count, `T_global` the **median of the raw per-style p90 thresholds** over the five calibrated styles (sweater, dress, white top, bikini top, underwear), and **`K` = the median reference count over those five styles = 17** (17, 25, 19, 8, 4), the same convention as `K_SHRINKAGE` for intensity (a style with the typical amount of evidence gets `w = 0.5`). Fixed now; not tuned. In the nested leave-one-out, the held-out article is removed from its style (`n - 1`), the style's p90 is recomputed without it, and `T_global` is recomputed with that style's leave-one-out p90. **Required outcomes unchanged** (evaluated on the binary pass/fail): FAIL the emerald-green dress and both coral dresses; PASS the submitted dress, the submitted sweater and the two red dresses K4 rejected. Reported: old (L1) versus new raw and shrunk threshold per style.

### M3. Borderline band -> ESCALATE

**Band half-width `w_band` from measured mask noise, not chosen.** For each real reference article of the four final styles, 8 jittered crops (a window of 90% of the width and of the height at a uniformly random offset, `random.Random(42 + image index)`), run through the full M2 pipeline (rembg mask, k-means); the deviation is the CIEDE2000 between the crop's dominant colour and the full image's. **`w_band` = the 95th percentile of the pooled deviations** (also reported per style). Because a concept's statistic is a minimum distance to references, a dominant-colour error of `delta` moves it by at most `delta`, so a distance within `w_band` of the threshold can be flipped by mask noise alone. **Verdict:** `d <= T - w_band` PASS; `d >= T + w_band` FAIL; otherwise **ESCALATE** (Gate 2 unmeasured, hence INCONCLUSIVE in `critic_rule` unless another gate already failed). If `w_band >= T` for a style there is no outright-pass region for it; that is reported, not adjusted. The noise is measured on real references only, before any concept is scored, and the value is committed before scoring. The required outcomes stay binary (above); the report also lists every case that falls in the band.

### M4. K2 statement and the second Gemini judge

The K2 conclusion is restated on the defensible measure only: acceptable-versus-not, Claude-Qwen kappa 0.84 (n = 59), labelled LLM consensus; four-way agreement was 0.51 and the four-way counts are not stable across graders, so they are not a finding. The post-hoc re-reading of the disagreements after seeing Qwen's grades is withdrawn (it revised my grades toward the more favourable result). When the Gemini free-tier quota allows, `gemini-2.5-flash` (the model that graded the first 12 decisions) grades the remaining 47 under the same blind protocol; pairwise kappa is reported once it is complete, and if it is blocked I report how far it got. No Claude model is substituted.

---

## N. Catalogue-wide colour priors by pattern class, and distribution-based colour for patterned garments

Committed before any N3 or N4 scoring. **Facts already known and stated, decided from evidence already on record before this section is scored (not from any N3/N4 result):** the M3 per-style mask-noise p95 (`v3_colour_mask_noise.csv`) is Sweater 0.179, Dress 0.203, Top 0.231, Bikini top 28.93 -- three styles cluster below 0.25 and one is two orders of magnitude apart. Of the five calibrated styles, `graphical_appearance_name` is Sweater = Melange, Dress = Solid, Top = Solid, Bikini top = All over pattern, Underwear = Solid.

### N3. Catalogue-wide shrinkage prior, stratified by pattern class

**Class definition (fixed before scoring).** Two classes over `graphical_appearance_name`: **solid** = `{Solid, Melange}`; **patterned** = every other value (22 remaining labels in the 1,980-style autumn table, pooled). Melange is grouped with Solid, not left as its own class or pooled with the prints, because the only calibrated Melange style (the sweater) already has a measured mask-noise p95 (0.179) indistinguishable from the three Solid styles (0.203, 0.231) and two orders of magnitude below the one patterned style's (28.93) -- the noise evidence, not the label text, is what the shrinkage prior is meant to describe. This is the same two-way split N4 uses to route styles to a colour statistic.

**Universe.** The 1,980 autumn forecast-eligible styles (`reports/tables/forecast_all_styles.csv`), not the five calibrated styles.

**Method (fixed, the existing M1/M2 dominant-colour pipeline, unchanged).** For each of the 1,980 styles: collect up to 8 real catalogue photos (`concept_forecast_index.collect_index_images`, `PER_STYLE_CAP` = 8, `data/images` plus the local image tree, texture crops excluded per the existing screen); run each photo through the M2 rembg-mask dominant-colour extraction (`colour_check.dominant_colour`, unchanged). A style's raw threshold is the p90 of the nearest-sibling CIEDE2000 distances among its own photos' dominant colours (`colour_check.threshold_from`, unchanged), defined only for styles with >= 2 photos; styles with 0 or 1 photo are excluded from the prior and counted separately. **Per class:** the prior is the **median of the per-style raw thresholds** within that class (the same statistic M1 used globally, now computed within each class instead of pooling both). **Reported before any N4 scoring:** each class's size (styles with a defined threshold) and its median threshold; the count excluded for having under 2 photos; if a class has under 20 styles with a defined threshold (an arbitrary but stated floor -- under 20 the median is unstable to single outliers the way the bikini's n=8 reference threshold already is), the pooled catalogue-wide median is used for that class instead and this is stated plainly, not silently.

**This does not change M1's calibrated thresholds.** The five calibrated styles keep the global (unsplit) shrinkage of M1/M2 unless N4 explicitly re-derives one; N3 only computes the two class priors for N4 to use.

### N4. Distribution-based colour for patterned garments

**Which styles switch (fixed, from the N3/M3 evidence above, before any distance is computed).** Only styles in the **patterned** class (N3's definition) among the five calibrated styles switch to the histogram statistic below. That is the **bikini top only**: Sweater, Dress, Top and Underwear stay on M1/M2 dominant-colour distance and shrinkage, unchanged in every respect (mask, threshold, band). "Solid styles must be unchanged" below means these four, not the literal `Solid` label.

**Statistic (fixed).** Per image: the same rembg garment mask as M2 (unchanged), then the mask's Lab pixels binned into three independent 1-D histograms (32 bins each, fixed edges `L in [0, 100]`, `a, b in [-100, 100]`, matching skimage's Lab output range; counts normalised to sum to 1). Distance between two images = **sum of the three per-channel 1-D Wasserstein (earth-mover's) distances** (`scipy.stats.wasserstein_distance`, bin centres as values, normalised counts as weights) -- a documented simplification of the joint 3-D EMD, tractable and, because each channel distance is itself in native Lab units, on the same rough numeric scale as CIEDE2000 (not identical: this is a modelling choice, not a unit conversion, and is reported as such). A concept's statistic is the minimum of this distance to the style's real reference articles, exactly as the dominant-colour check does.

**Threshold (fixed).** Raw threshold = p90 of the real nearest-sibling histogram distances among the bikini's own 8 references (`threshold_from`-equivalent, same percentile, new statistic). Shrunk toward the **N3 patterned-class prior** with the unchanged M1 formula (`w = n / (n + 17)`, `n` = 8). This mixes a dominant-colour-CIEDE2000-scaled prior with a histogram-EMD-scaled raw threshold; that mismatch is accepted here as the simplest option that follows the brief's instruction literally, and its effect (if any) is exactly what the required outcomes below are designed to catch -- it is not adjusted after the fact if it produces a bad result.

**Noise band (fixed, re-derived, not reused from M3).** The same M3 jitter procedure (8 crops per reference, 90% window, `random.Random(42 + image index)`) run through the histogram statistic instead of CIEDE2000, on the bikini's 8 references only (64 crops). The new band half-width is the p95 of those 64 deviations. Verdict rule unchanged (`verdict_for`, tri-state pass/fail/escalate).

**Required outcomes (stated before scoring; no tuning if unmet).**
1. Every M1 required outcome still holds under the mixed pipeline (four styles on M1/M2 dominant colour + shrinkage, bikini on the new statistic): **FAIL** the emerald-green dress and both coral dresses; **PASS** the submitted dress, the submitted sweater and the two red dresses K4 rejected -- all five of these are solid-class styles and untouched by N4, so this is a regression check, not a new measurement.
2. The bikini's real-article nested leave-one-out pass rate (same nested procedure as M1/M3, new statistic and new shrunk threshold) is **>= 0.80** (was 0.75 unshrunk L1, 0.50 under M1 dominant-colour shrinkage).
3. The bikini's new noise-band p95 is **well below** the old dominant-colour p95 of 28.9 (no numeric floor stated in advance beyond "well below": reported plainly whatever it is).

**If the bikini does not recover to >= 0.80, this is reported plainly and the statistic, threshold or shrinkage weight is not adjusted post hoc to force it.**

---

## O. A3: a prior human REJECT is terminal (spec-gap fix, S10 re-run)

Committed before any S10 re-run. **Facts already known and stated, decided from evidence already
on record (A2's adjudication, not from anything the re-run below will produce):** S10 is the only
one of the ten K2 silent-rule scenarios where an earlier attempt of the SAME concept was rejected
by a HUMAN (the other prior-REJECT scenarios, S02 and S05, were rejected by an automatic gate on
an earlier attempt, which the existing retry rule already handles correctly). Neither
`critic_rule.decide` nor `agents/critic.md` had any rule for this case; the live LLM orchestrator
improvised `PASS_PENDING_HUMAN` + `FORWARD` to the forecaster + a second human escalation, which
A2 adjudicated as `worse` (the executed action re-asks an already-decided question and forwards an
unshippable concept, contradicting the reply's own "stays unshippable" reasoning).

**Rule (fixed, already implemented in this commit's parent, not by this pre-registration): if a
prior human REJECT exists for this concept request (same style_key + brief, an earlier attempt in
the same retry chain), the verdict is REJECT -- no FORWARD, no PASS_PENDING_HUMAN, no
re-escalating the same already-decided question -- checked before any gate.**
`nss.generate.critic_rule.decide` takes `prior_human_reject: bool = False`; no existing caller
passes it, so every previously-scored case (the 129 H3/J6 cases, the 54 recorded candidates, S01
through S09) is provably unaffected by construction (default-argument semantics) and by the full
suite passing unchanged (913 pre-existing + 31 new, all green). `agents/critic.md`'s
Failure/escalation section and `skills/concept-qc/SKILL.md`'s CURRENT (A3) section state the same
rule for the live orchestrator; `evals/fixtures/silent_rule/S10.json`'s rubric now requires REJECT
(no FORWARD) as the good/equivalent behaviour, matching this rule.

**Re-run (fixed).** S10 only, 3 runs, **Sonnet** (not Fable, not Opus -- matching K2/L5's model),
same call shape as K2/L5: system prompt = the CURRENT `agents/critic.md` and
`agents/orchestrator.md` (with the A3 rule in context), no evidence note beyond what the agent
files themselves now say. Claude Max plan utilisation is checked and reported before the run (it
was last reported ~0.90); if utilisation does not allow 3 runs, this is reported and the re-run
waits rather than proceeding.

**Required outcomes (stated before running; no tuning if unmet).** All 3 runs return verdict
`REJECT`, with no `FORWARD` outcome and no `escalate_to_human: true` in the reply. **Also
reported, not required to pass/fail:** the deterministic rule's own verdict on S10 with
`prior_human_reject=True` (already confirmed REJECT, see the commit implementing the rule), and
that no case in the 129/54/S01-S09 set changes verdict (see above -- structural guarantee, checked
via the passing test suite, not re-scored here since that would separately exercise N4's
not-yet-measured histogram band on the bikini-style stored cases, an unrelated pending gap).

---

## P. Phase A: demand capture@k, tolerance hit@3, power, and the Phase B primary metric

SPEC.md Sections 5 and 7. Committed before any of the metrics below has been computed on any
model, baseline or floor. The only numbers computed before this commit are label-only: the realised
#3-vs-#4 gaps in `reports/tables/phase_a_near_tie.csv` (`python -m nss.models.phase_a_measure
--near-tie`), which read realised targets and no predictions. Code: `nss.models.metrics`
(`demand_capture_at_k`, `tolerance_hit_at_k`) and `nss.models.phase_a_measure`, committed together
with this section.

**Why these metrics, and why now.** The near-tie finding (the true #3 and #4 styles differ by
0.61% of the #3 value, `ranking_diagnostics_summary.csv`) predates both metrics; it is the reason
for them, not a result they were tuned to. A buyer commits inventory to a handful of styles and is
paid by the demand those styles realise. Demand capture scores exactly that: how much of the best
achievable demand the picks deliver, with partial credit for a near-miss. Hit@3 is binary per pick:
a pick one place outside the true top-N scores 0, however close it was.

**Correction to the premise, measured before this commit.** The 0.61% was computed on the
`log1p` target (about 4.7 at #3) over the 12 grid origins, not on raw intensity. On the 48 weekly
origins used here, the log1p gap averages 0.82%. In raw intensity (units per active article per
week, `expm1` of the target), the #3-vs-#4 gap has median 3.78%, 90th percentile 5.79%, maximum
17.9%. Both new metrics use raw intensity, so the margin is derived on that scale.

### P1. Metrics (all per origin, then averaged over origins)

- **Demand capture@k, k = 3, 10, 20:** the summed realised intensity of the method's top-k picks
  divided by the summed realised intensity of the true top-k, in raw intensity. In [0, 1]; 1 for a
  perfect pick. Oracle = 1.
- **Tolerance hit@3:** the fraction of the method's top-3 picks whose realised intensity is at
  least `(1 - m)` times the true #3's intensity. Oracle = 1.
- **The margin `m`, derived:** the 90th percentile, over the 48 weekly origins, of
  `(I3 - I4) / I3` in raw intensity (numpy's default linear interpolation): **m = 0.057911**.
  Reasoning: the SPEC treats the true #4 as indistinguishable from #3; the margin is the smallest
  that counts the true #4 as a hit at 90% of origins. It is a quantile of the measured gap, not a
  round number, and the code recomputes it at run time and stops if it differs.
- **Population convention**, unchanged from the existing metrics: each baseline is scored on the
  styles where its prediction is non-null (seasonal-naive drops styles without 52 weeks of
  history), the true top-k taken within that population; the model is scored on its full eval set;
  the random floor permutes realised targets over the full eval set, averaged over the 20 existing
  floor seeds.
- **Nothing is replaced.** Hit@3-in-top20 and every existing metric (Hit@3-in-top10,
  Precision@3/10, NDCG@10, Spearman, WMAPE) are reported alongside, on the same origins.

### P2. Protocol (A.2, A.3)

The model and all four baselines (seasonal-naive, EWMA persistence, parent-category mean, global
mean) and the random floor, at the 48 embargoed weekly origins of 2b (`eval_power`). The model's
predictions are regenerated with the locked `FINAL_MODEL_CONFIG` (no cached frame survived the
data loss); the run stops unless every existing metric, for every method at every origin, equals
`v3_power_per_origin_weekly.csv` to 1e-9. No retraining beyond that reproduction; no tuning.

For each metric, paired model-minus-comparator differences per origin, moving-block bootstrap,
block length 13, 2,000 resamples, seed 42. Reported against seasonal-naive (the power table) and
every other comparator: n paired origins, mean difference, 95% percentile CI, CI half-width,
bootstrap SE, effective sample size (`ess_ac` and `ess_boot`, as in 2b), and the **minimum
detectable effect at 80% power**, `MDE = (z_0.975 + z_0.80) * SE_block = 2.8016 * SE_block`
(two-sided alpha 0.05).

### P3. Rule for choosing the Phase B primary metric (A.4)

**Candidates:** tolerance hit@3, demand capture@3, Hit@3-in-top20, demand capture@10, demand
capture@20. The other existing metrics stay guardrails (SPEC Section 6) and are not candidates.

**Relevance check, stated in advance.** A candidate is eligible only if all three hold:

- **R1 decision alignment:** it depends only on the method's top-k picks, k <= 20, scored by
  their realised demand. True of every candidate by construction; excludes Spearman, WMAPE and
  NDCG@10 (log-scale relevance).
- **R2 separation from chance:** the random floor's mean is at most 0.5, halfway between an
  uninformed pick and the oracle's 1.0.
- **R3 estimable:** at least 8 origins with a paired model-minus-seasonal-naive difference, and a
  nonzero bootstrap SE.

**Selection.** Among eligible candidates, the one with the smallest **normalised MDE**,
`MDE / (1 - floor mean)`: the MDE as a fraction of that metric's achievable range above chance.
Raw MDEs are not comparable across metrics on different scales, which is why the rule normalises;
with oracle = 1 for every candidate, `1 - floor` is that range. Ties at 3 decimals go to the
earlier candidate in the list above (smaller k). **Only dispersion enters the choice.** The sign
and size of the model's lead play no part in it.

**Verdict.** The model's lead over seasonal-naive counts as demonstrable under the chosen primary
metric if and only if its 95% paired block-bootstrap CI (block 13) excludes zero in the model's
favour. Either answer is reported plainly. No metric, margin, candidate, block length or threshold
in this section is changed after results are seen.

**Caveat, stated now.** The MDE is computed for model versus seasonal-naive. In Phase B the
comparison is challenger versus champion, two correlated models, and the variance of that paired
difference will generally be smaller. The Phase A MDE is therefore a conservative guide for Phase
B, not an exact one.

---

## Q. B.0: circular block bootstrap, and re-baselining under it

Committed before any interval has been computed with the circular bootstrap.

**Defect being fixed (reported in Section P's results, `PHASE_A_measurement.md` caveat 1).** The
moving-block bootstrap used so far (`backtest.block_bootstrap_ci`, `phase_a_measure.block_means`)
draws block starts uniformly from `0 .. n - L`, with no wrap-around. With n = 48 and L = 13, the
first and last 12 origins appear in fewer resampled blocks than the middle ones, so the intervals
are skewed (the model's Hit@3-in-top20: mean 0.431, CI [0.410, 0.618]). Phase B challengers are
expected to sit near zero, where that skew could flip a decision.

**Q1. The circular block bootstrap (Politis and Romano 1992), exactly:**

- The per-origin series `x_0 .. x_{n-1}` (chronological, NaN dropped) is treated as a circle.
- Each resample draws `ceil(n / L)` block starts **uniformly from `0 .. n - 1`**. Each block is
  `x_{s mod n}, x_{(s+1) mod n}, ..., x_{(s+L-1) mod n}`. Blocks are concatenated and truncated
  to n; the resample's mean is recorded.
- **L = 13**, **2,000 resamples**, numpy `default_rng(seed=42)`, starts drawn by
  `rng.integers(0, n, size=ceil(n/L))`: the same resample count, seed and block length as before.
  Only the start range and the wrap-around change.
- 95% interval: 2.5th and 97.5th percentiles of the resample means (numpy default interpolation).
  SE = SD of the resample means (ddof 1). MDE at 80% power = 2.8016 × SE. The point estimate is the
  plain mean.
- Every origin appears in exactly L of the n possible blocks, so each origin has the same expected
  weight in a resample.

**Q2. Re-baseline.** Recompute every metric in `phase_a_per_origin.csv` (the 7 existing and the 4
Phase A metrics), for the model, all four baselines and the random floor, with the circular
bootstrap: per-method means with intervals, and paired model-minus-comparator differences with
intervals, SE, ESS and MDE. Report the old (non-circular) and new intervals side by side. The
per-origin values are not recomputed; only the resampling changes.

**Q3. Stop condition.** If the model's demand capture@20 lead over seasonal-naive, under the
circular bootstrap, has a 95% lower bound at or below zero, Phase B stops there and this is
reported. Otherwise Phase B proceeds under the Section R rules, which will use this bootstrap.
The Phase A choice of primary metric is not re-run under the new bootstrap: it was fixed at the end
of Phase A, and this section changes the resampling, not the metric.

---

## R. Phase B: four accuracy levers, quantile output, combination

Committed before any lever model is trained. Governs Phase B (SPEC Section 7) under the SPEC
Section 6 champion/challenger rule. Code: `nss.models.phase_b`, committed after this section.

### R0. Common protocol (every lever, every control)

- **Champion:** the current L2 LightGBM (`FINAL_MODEL_CONFIG`, objective `regression`, seed 42,
  the determinism parameters of `lightgbm_model`). Its per-origin metrics are the Phase A ones
  (`phase_a_per_origin.csv`, reproduced to 1e-9).
- **Origins and embargo:** the 48 weekly origins, each served by the model of its 4-week grid
  block and trained on the embargoed origins (`growth_backtest.predictions_for_origins`). All
  levers go through this path.
- **Locked hyperparameters:** `num_leaves 63, learning_rate 0.05, n_estimators 200,
  min_child_samples 50` everywhere. Only what a lever names changes (a feature, the training label,
  a post-hoc projection, the seed, the objective). No tuning inside Phase B, and nothing below is
  changed after results are seen.
- **Evaluation:** every challenger's style-level predictions are scored on the champion's eval set
  (same styles, same origins, same realised raw target `y_true`) with the same metric code. Paired
  challenger-minus-champion differences per origin. Circular block bootstrap (Section Q),
  L = 13, 2,000 resamples, seed 42.

### R1. Decision rule

- **Primary metric:** demand capture@20.
- **p-value:** two-sided circular-bootstrap p for "mean paired difference = 0",
  `p = min(1, 2 * min(#{m <= 0} + 1, #{m >= 0} + 1) / 2001)` over the 2,000 resample means `m`
  (`circular_bootstrap.bootstrap_p_two_sided`). This is dual to the percentile interval: the
  two-sided `(1 - a)` interval excludes zero when `p <= a`.
- **Multiple comparisons: Holm-Bonferroni across B.1-B.4 (m = 4), family-wise alpha 0.05.** Sort
  the four p-values ascending. The i-th smallest is compared with these ordered thresholds:
  **0.0125, 0.01667, 0.025, 0.05**. Step down: stop at the first p above its threshold; that lever
  and every lever after it are not significant.
- **Guardrails:** Hit@3-in-top20, NDCG@10, Spearman, WMAPE. A guardrail **fails** if its 95%
  paired circular interval lies entirely on the worse side of zero (upper bound < 0 for Hit@3,
  NDCG, Spearman; lower bound > 0 for WMAPE). Guardrails are not multiplicity-adjusted, which makes
  failing easier and adoption harder.
- **Adoption:** a lever is adopted only if (a) it is significant under Holm on the primary, (b)
  its mean paired difference is positive, (c) it fails no guardrail, and, for B.1 only, (d) its
  controls pass (R2). Otherwise it is recorded as a negative result.
- **The family is fixed at m = 4** whatever happens. If a lever turns out degenerate (e.g. B.4's
  seeds give identical predictions), it keeps its place with its p-value (p = 1 for an identically
  zero difference) and the thresholds do not change.

### R2. B.1 Visual momentum (new feature `vis_nbr_momentum`)

- **Embeddings:** the existing per-article CLIP and DINOv2 vectors in
  `data/retrieval_cache/emb_{clip,dino}.npz` (22,468 articles, 3,003 styles). No new embedding is
  computed. Per-article first-sale dates **did not exist**; they are computed from
  `data/interim/transactions_train_parquet` as each article's minimum `t_dat` (about 3 s on CPU).
  Articles never sold (52 embedded ones) are excluded.
- **Leakage guard 1, the images:** at origin `t`, a style's visual vector uses **only articles
  whose first sale is strictly before `t`**. The CLIP and DINOv2 means are taken over those
  articles and L2-normalised per model. A style with no such article has no vector at `t`: it gets
  a null feature and is never a neighbour. Measured coverage: 97.6-100% of eval styles per origin
  (median 99.4%). 4,412 embedded articles were first sold after the first weekly origin; this
  filter excludes them origin by origin.
- **Residual selection caveat, stated now:** which articles of a style were embedded (at most 8
  per style, in scan order, plus the screened reference sets) was decided during the concept work,
  partly with later data. That choice affects which pre-origin photos represent a style, never a
  label or a trend.
- **Similarity:** the mean of the CLIP cosine and the DINOv2 cosine between the two styles'
  vectors (the retrieval index's `avg` view).
- **Neighbours:** the **k = 10** most similar other styles present in the model frame at `t`.
  k is fixed a priori, not tuned; the style itself is excluded.
- **Leakage guard 2, the trend (trailing only):** a neighbour `j`'s momentum at `t` is
  `g_j = log1p(ewma_halflife_4w_j) - log1p(ewma_halflife_13w_j)`. Both are existing causal features
  computed from panel weeks at or before `t` (`model_features`). Feature: `vis_nbr_momentum_s =
  sum_j max(sim_sj, 0) * g_j / sum_j max(sim_sj, 0)` over neighbours with non-null `g_j`; null if
  none.
- **Controls, both run whatever B.1's result:**
  - *Causality shuffle test:* within each origin, `vis_nbr_momentum` is permuted across styles
    (seed 42), in training and test frames alike, and the model retrained. This keeps the feature's
    distribution but breaks its link to each style.
  - *Negative control:* the same feature built from **10 random other styles** (seed 42, equal
    weights) instead of the 10 visually nearest.
  - **Control rule (adoption condition (d)):** both control models' paired capture@20 difference
    against the champion must have a 95% circular lower bound <= 0. If either control also beats
    the champion, the gain is not attributable to visual similarity and B.1 is not adopted.

### R3. B.2 Shrunk-intensity target

Training label: `log1p(mean(intensity_shrunk))` over the same 13-week forward window
(`compute_forward_target(..., target_column="intensity_shrunk")`), the same full-window rule and
the same features. The shrinkage prior is trailing-only with K = 3 fixed a priori
(`style_panel.add_intensity_shrunk`), so the label adds no future information beyond the window
it describes. **Evaluation uses the same realised raw outcome `y_true`** (log1p of mean raw
intensity) as the champion. Predictions are used as they come, with no rescaling. A systematic
level shift from shrinkage would therefore show up in WMAPE, and that guardrail is expected to be
the one at risk.

### R4. B.3 Hierarchical reconciliation: **OLS** (pre-registered choice)

- **Levels:** style (bottom), product type within index group (`index_group_name x
  product_type_name`), index group. Nested by construction.
- **Parent forecasts:** the same LightGBM, locked config, trained on a parent-level panel built
  by aggregating the style panel per (parent, week): `units`, `revenue`, `n_active_articles`
  summed; `units_per_active_article = units / n_active_articles` (0 when no active articles);
  `intensity_shrunk` set to the parent's raw intensity (with parent article counts in the
  hundreds, K = 3 shrinkage is negligible; stated as an approximation); `price_index` averaged
  with `n_active_articles` weights; `first_week_seen` the minimum; attribute columns not in the
  parent key set to `"ALL"`. Densified weekly between each parent's first and last week (missing
  weeks: zero units and articles). Features and targets use the same functions as for styles.
- **Aggregation:** intensity is a per-article mean, so a parent is the article-weighted average of
  its children. At origin `t` the aggregation matrix row for parent `P` has weight
  `n_active_articles_level_i / sum over children in P of the same` on each child style `i` present
  in the style model frame at `t`. Using origin-time weights for the forward window is an
  approximation, stated here.
- **Reconciliation:** in raw-intensity space (`expm1` of each prediction), per origin,
  `y_tilde = S (S'S)^{-1} S' y_hat`, with `S = [I; A_pt; A_ig]` and `y_hat` the stacked base
  forecasts. OLS, identity weights, no covariance estimate. Reconciled style values are clipped at
  0 and mapped back with `log1p`. Only the reconciled style forecasts are evaluated.

### R5. B.4 Seed ensemble

The champion trained with seeds **42, 43, ..., 51**: `random_state` and the three pinned seeds
(`bagging_seed`, `feature_fraction_seed`, `data_random_seed`) all set to the seed. Predictions are
averaged on the log1p scale. **Stated in advance:** the locked config uses no row or feature
subsampling, and training is deterministic, so the ten models may be identical. Seed-induced
variance is reported as the mean over (style, origin) of the across-seed SD of predictions, and the
ensemble's variance reduction as `1 - Var(ensemble) / mean Var(single)` of that seed component. If
the seeds are identical, B.4 is a zero-difference lever with p = 1, and making it stochastic would
need changed hyperparameters, which R0 forbids.

### R6. B.5 Quantile output (production requirement, not in the Holm family)

- Three models, objective `quantile` with alpha **0.10, 0.50, 0.90**, locked config otherwise,
  seed 42, trained on the champion's label. Same 48 embargoed weekly origins.
- **Coverage:** a (style, origin) is covered if `q10 <= y_true <= q90` (log scale; quantiles are
  invariant to the monotone `log1p`). Crossed intervals (`q10 > q90`) count as not covered, and
  their rate is reported. Coverage is reported per origin and pooled over all (style, origin) rows,
  with a 95% circular interval on the per-origin series.
- **Acceptance, both required:** (1) pooled empirical coverage **in [0.75, 0.85]** (nominal
  0.80), judged on the point estimate; (2) **q50 non-inferior to the champion on demand
  capture@20**, meaning the 95% paired circular interval of q50-minus-champion is not entirely
  below zero. Failing either is reported plainly; nothing is recalibrated.

### R7. Combination

- If **two or more** levers are adopted, they are combined (B.1's feature, B.2's label, B.3's
  projection and B.4's averaging, as applicable, in that order) and **one** final test of the
  combination against the champion is run: 95% paired circular interval on capture@20 excluding
  zero in its favour (alpha 0.05, a single pre-planned test), and no guardrail failure. Pass: the
  combination is the new champion. Fail: the single adopted lever with the smallest Holm p-value
  becomes champion.
- If **exactly one** lever is adopted, it is the new champion; no combination test.
- If **none** is adopted, the current champion stands. That is the result.

### R8. Amendments before any training (2026-09-23)

Found while writing the code, before any lever model was trained or any Phase B metric computed.
Both close a gap in R2 or R4; neither was chosen by looking at a result.

- **R2, neighbour pool (a look-ahead fix).** R2 said neighbours are drawn from "styles present in
  the model frame at `t`". The model frame keeps only styles whose 13-week forward target is
  complete, so that pool conditions on a style surviving the next 13 weeks: future information.
  **Amended:** the neighbour pool, and the momentum `g_j`, come from styles with a panel row at `t`
  (`build_features` output, before any target filter), which is causal. The feature is still
  attached only to model-frame rows, as every other feature is.
- **R4, parents without a base forecast.** A parent whose own 13-week target is incomplete at `t`
  has no row in the parent model frame, so it has no base forecast. **Amended:** such a parent,
  and any parent whose children in the style eval set have zero total `n_active_articles_level`,
  is left out of `S` at that origin. Its children are then reconciled against the remaining levels
  only.
