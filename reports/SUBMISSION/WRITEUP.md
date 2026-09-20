# next-season-styles — Write-up

## 1. Style definition and why

The unit of forecasting is not the H&M `article_id` (a single SKU/colourway) but a `style_key`: a
composite of 5 attribute columns from `articles.csv` — `index_group_name`, `product_type_name`,
`garment_group_name`, `perceived_colour_master_name`, `graphical_appearance_name`. The reasoning is
commercial: a buyer briefs a design team on "black jersey basic T-shirts," not on article
0554598001, and one archetype is realised across many near-duplicate SKUs that turn over while the
archetype persists.

Coherence is reported honestly (`reports/tables/style_key_silhouette_comparison.csv`): the 5-column
key's silhouette over `detail_desc` embeddings is a weak -0.453, but against a 50-shuffle
permutation null (mean -0.700) that is z=145.5. A 4-column key keeps more lifetime units (90.33% vs
85.07%) but is less coherent (z≈89.0); the 5-column key was kept.

## 2. Target definition

The target is `units_per_active_article` (intensity), not raw volume: the top-20 styles by lifetime
units and by mean intensity have zero overlap (`reports/tables/intensity_comparison.csv`), because
raw volume rewards assortment breadth, not per-SKU demand. The intensity leaderboard is not
markdown-driven: only 1 of its 18 priced top-20 styles has a price index below 0.9 (mean ≈1.043).

## 3. Evaluation methodology, and a correction

The harness uses 20 rolling origins (4-week step, 13-week horizon); the paired comparison runs on the 12 origins every method shares (`backtest_summary_v2.csv`).

**Correction to the shipped headline.** Each walk-forward model trained on every earlier origin, but the last three have 13-week labels that run into the test window, so training labels contained the outcomes being forecast (`backtest_embargo_check.csv`). With a 13-week gap, Hit@3-in-top20 falls **0.722 → 0.528** (paired drop 0.194, CI [0.111, 0.250], 12 origins), Spearman 0.803 → 0.712, and six of seven measures are lower (Precision@3 is unchanged). The label-shuffle control (0 hits in 108 picks) does not detect this: it tests a different leak. Re-doing the PAIRED comparison with the embargoed model (`backtest_embargo_paired_diff.csv`, same 12 origins, block bootstrap) changes the claim. Against **seasonal naive** (defined on 10 of the 12 origins, the ones with a 52-week lag) the embargoed model is comparable on top-k: Hit@3-in-top20 +0.233, CI [0.000, 0.567]; top-10 +0.033, [-0.100, 0.267]. It is better on the graded measures: NDCG@10 +0.156 [0.099, 0.254], Spearman +0.218 [0.203, 0.253], WMAPE -0.045 [-0.054, -0.033]. It beats EWMA persistence, global mean and parent-category mean on every metric (e.g. top-20 +0.250 [0.222, 0.361] against persistence). So the honest claim is: a clear lead over persistence and the naive means, and a lead over seasonal naive on ranking quality but not demonstrably on picking the top three. Random floor 0.004. The embargo also shrinks training sets, so leakage and lost data are not separable.

## 4. The stock limitation

Every number derives from realised transactions, censored by what was stocked and merchandised, not latent demand. A drop-then-recover detector flags a stockout signature in 3.76% of eligible style-weeks, a lower bound. No inventory data was invented.

## 5. COVID: two models, compared directly

I trained a second model without any training row whose 13-week label window touches 2020-03-01 to 2020-06-30 and scored both on non-COVID origins (`covid_two_model_comparison.csv`). A walk-forward comparison is vacuous there (both train on identical pre-COVID data), so the primary design scores each of 13 non-COVID origins out of sample, both models trained on origins whose labels do not overlap it. Removing COVID data made **rank correlation worse (−0.056, CI [−0.126, −0.015]) and WMAPE worse (+0.023, [0.008, 0.045])**, and top-k measures no different. Verdict: COVID data did not hurt and mildly helped. Caveats: the second model has about 40% fewer rows, so this cannot separate "COVID helped" from "more data"; a 5-origin subset points the other way with degenerate CIs; the deployed post-COVID case has no scored origins.

## 6. Generation: what was wrong, and the fix

The similarity gates were validated by controls: leave-one-out showed the median threshold rejected 66% of 61 genuine reference articles (21/61 pass) while the p90 limit passes 61/61, and an exact copy can pass Gate 1 but fails Gate 1b in every style.

**N1: the root cause was diagnosed wrongly twice, then measured.** Two rounds of four seeds per style gave zero briefed design changes. I first blamed IP-Adapter conditioning on one reference. Testing the levers on the sweater showed that was only half: (a) multi-reference (`ip_adapter_image=[[…]]`, which the installed diffusers concatenates into one attention over N images' tokens) alone changed the garment (crew for V neck) but not the brief; (b) lowering the scale to 0.15–0.25 collapsed into fabric swatches, and text alone (scale 0) did too, because the attribute-first prompt ("Sweater, knitwear construction, beige melange. Novel accent: …") with an attribute-only second-encoder prompt made texture words dominate; (c) per-block scales (supported by the installed version, InstantStyle-style) gave swatches (style block only) or no change (style plus layout); (d) a plain sentence naming the garment and its two changes, given to **both** text encoders, made a funnel neck and brown rib appear even from a single reference at scale 0.25–0.35 (`n1_levers_summary.md`), and compel weighting (1.5) on the change clauses restored brown accents at 0.45 (2/2 against 0/2). The shipped configuration is therefore prompt structure first, multi-reference second: natural sentence on both encoders, 8-reference concat, compel 1.5, per-style scale (0.35 chosen from a 0.15–0.45 sweep; 0.6–0.7 removes the changes).

**N2/N3: two gates that were missing.** *Gate 3* asks, per briefed change, "does this garment have X?" and passes on a strict majority. Validated against my labels on 23 images (46 questions) the small local reader has recall 0.95 but specificity 0.58 (accuracy 0.74): it says yes too easily, so Gate 3 supports but does not replace the human check. *Integrity*: asking a small VLM "is this a coherent garment?" caught **0 of 3** known-malformed images (cut-out, sheer mesh, folded object), i.e. it does not work. A reference-based floor (closest real reference at least as close as the 10th percentile of real nearest-sibling similarity, DINOv2) catches 3/3 and 5/5 of all known-bad images, and passes 8/9 known-good (one real photo is an outlier). It is calibrated within the style, which makes it a similarity gate for near-identical styles, so I tested a **global** floor (10th percentile pooled over 154 real photos in 8 styles, 0.779): it passes 9/9 known-good but misses one of the three malformed images (sheer mesh, 0.811), so it does not work and the per-style floor stays; it does not rescue the white top (0.733) either (`integrity_global_validation.csv`).

**N4: judges.** Groq and Gemini were unavailable (a daily token limit; an invalid key), so the panel is two local models calibrated as before (positive minus negative mean ≥ 0.3 on 26 controls): SmolVLM-500M (gap 0.338, pass mark 0.384) and Florence-2-base (0.373, 0.337; a captioner, so Gate 2 only). Moondream2 failed to load under transformers 5.17. Groq answered once when re-tried, then hit its daily limit again; Gemini's key still returns 401. **Florence-2 is therefore advisory, not gating** (its kappa against the API readers, 0.36–0.48, is too low to carry a verdict; SmolVLM's is 0.68 with Groq): a decision on measured agreement, not to pass concepts, and it changes the white top's Gate 2 from fail to pass (its overall verdict stays FAIL). Pairwise kappa on binarised attribute calls (`judge_panel_kappa.csv`): Groq–Gemini 0.89 (n=18), Gemini–SmolVLM 0.78 (18), Groq–SmolVLM 0.68 (51), Florence–SmolVLM 0.53, Florence–Groq 0.48, Florence–Gemini 0.36.

**N5: a wider reference base.** References rose from 4–6 to 17 (sweater), 25 (dress) and 19 (white top: only 20 such articles exist), screened by the local judge plus, after a review found a fabric close-up in the base (the small judge's framing answer was "yes" for every image), a border-variance check. The sweater was regenerated with the corrected base. Bootstrap CI widths of the DINOv2 limits shrank 2.5–4.4×: Gate 1 0.180 → 0.072 (sweater), 0.186 → 0.064 (dress); Gate 1b 0.180 → 0.048 and 0.150 → 0.034.

## 7. Agent architecture

An orchestrator delegates to 5 sub-agents (`reports/figures/agent_architecture.png`): `data-analyst`, `forecaster`, `style-profiler`, `concept-designer`, `critic`. Two dataset-agnostic skills carry the logic (`skills/style-brief/`, `skills/concept-qc/`, now with Gate 3 and the integrity floor). The MCP server exposes 8 tools over stdio; `generate_concept`, `score_concept` and the new `forecast_concept` do live work.

## 8. Results

**Selection: the model surfaces, a human judges.** All three from the emerging table (guards, and a predicted intensity at or above the guard-passing median by construction), excluding intimates and garments not identifiable in a flat photo (swim bottoms, hosiery), with no shared colour or product type; fixed before inspecting output (`final_three_selection_log.csv`). Skipped: T2 #1 red underwear (category), #4 sweater (product-type collision). Chosen: beige melange sweater (17.8, growth 1.59), red dress (11.0, 1.41), white jersey top (13.9, 1.17); SHAP leads with `n_active_articles_level`.

**Final concepts** (best of 8 seeds; Gate 2 needs both judges; human = every briefed change visible):

| Concept | G1 | G1b | Integrity | G2 | G3 | Human | Closed-loop forecast |
|---|---|---|---|---|---|---|---|
| Sweater | pass | pass | pass | pass | pass | yes | Beige Solid: 12.8, rank 205/1,980, low |
| Dress | pass | pass | pass | pass | pass | yes | Jersey Fancy Red Solid: 7.3, rank 522, low |
| White top | pass | pass | **fail** (0.733 < 0.922) | pass (Florence advisory: fail) | fail (YN) | mostly (narrow cuffs) | Sweater White Solid: 25.9, rank 25, low |
| Bikini top (summer) | pass | pass | pass | fail (SmolVLM 0.28) | pass | yes | Orange Solid: 40.8, rank 98/3,000, medium |

Two of four pass every automatic gate. **The white top fails by mechanism, not luck:** its 19 real articles are near-identical, so the floor sits at 0.922; a design that adds a neckline and sleeves falls to 0.73, and raising the scale to 0.6–0.7 passes the floor (0.937) only by deleting the changes (Gate 3 YN/NN). I did not loosen the gate. The bikini's Gate 2 failure is SmolVLM answering "ORIGINAL" for the pattern.

**N8, closed loop.** Image → blind reading → nearest catalogue style → the frozen model's forecast (`concept_forecast.py`, MCP `forecast_concept`, DEMO). On 40 real catalogue photos with known styles the extraction is **poor**: SmolVLM reads product type 65%, colour 70%, pattern 48% (all three 28%, exact style 12.5%); Florence 63/38/50% (13%, 2.5%). A multiple-choice prompt did not help (exact 18%, type worse). Hence most outputs are low-confidence and the sweater and dress map to neighbouring styles; this is a forecast for the archetype the picture reads as, not for the design.

**Seasonal view.** The same rules, with the summer model trained under a 13-week embargo, pick an orange all-over-pattern bikini top (swim bottoms excluded): predicted 37.88, realised 37.94, computed independently (the model never sees the June to August window). One style is one data point: across 127 emerging candidates the median absolute error is 6.0 units (correlation 0.82), so the 0.06 gap is luck.

## 9. Limitations

The image readers are small: the design-change reader over-says yes and the integrity question fails, so a human check decided every concept, and quota-limited API judges were unavailable. Limits rest on 8–25 real photos per style. Gate 2 penalises a brown trim on a beige garment. The forecast is for styles, not for whether these pictures would sell. The hyperparameter grid was not re-run under an embargo.
