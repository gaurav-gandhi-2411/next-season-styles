# Forecasting next season's winning styles, and designing for them

## 1. What a "style" is

I forecast styles, not articles. A buyer briefs a design team on "black jersey basic T-shirts", not
on article 0554598001, and one archetype sells through many near-duplicate SKUs. A style is a
five-column key: index group, product type, garment group, perceived colour and graphical
appearance. It is only weakly coherent (silhouette over article-description embeddings −0.453,
though z = 145.5 against a shuffle null); a four-column key keeps more lifetime units (90.33%
against 85.07%) but is less coherent (z ≈ 89.0), so I kept five.

Sources: `style_key_silhouette_comparison.csv`.

## 2. What I forecast

The target is units per active article per week (intensity), not raw volume. The top 20 styles by
lifetime units and by mean intensity do not overlap, because volume rewards a wide assortment, not
demand per product. Only 1 of the 18 priced top-20 intensity styles has a price index below 0.9, so
the ranking is not just a markdown effect.

Sources: `intensity_comparison.csv`.

## 3. How I evaluated it, and a correction to my own headline

I backtest with 20 rolling origins (4-week step, 13-week horizon) and compare methods on the 12
they all share.

My first headline was wrong. Each walk-forward model trained on every earlier origin, but for the
last three the 13-week training labels run into the test window, so the model had seen outcomes it
was asked to forecast. A label-shuffle control missed it because it tests a different leak. With a
13-week gap, Hit@3-in-top20 falls from 0.722 to **0.528** (paired drop 0.194, CI [0.111, 0.250])
and Spearman from 0.803 to 0.712; six of seven measures get worse. The embargo also shrinks the
training sets, so leakage cannot be separated from lost data.

Redoing the paired comparison with the embargoed model changes the claim. It beats EWMA
persistence, the global mean and the parent-category mean on every measure. Against seasonal naive
(defined on 10 of the 12 origins, those with a 52-week lag) it is comparable on top-k and better on
ranking quality:

| Embargoed model minus seasonal naive | Difference | 95% CI |
|---|---|---|
| Hit@3 in top 20 | +0.233 | [0.000, 0.567] |
| Hit@3 in top 10 | +0.033 | [−0.100, 0.267] |
| NDCG@10 | +0.156 | [0.099, 0.254] |
| Spearman | +0.218 | [0.203, 0.253] |
| WMAPE (lower is better) | −0.045 | [−0.054, −0.033] |

So the honest claim: a clear lead over persistence and the naive means, and over seasonal naive a lead on ranking quality but not a demonstrated one on the top three.

Sources: `backtest_embargo_check.csv`, `backtest_embargo_paired_diff.csv`.

## 4. The stock limitation

Every number comes from realised transactions, censored by what was stocked and merchandised, so
none of it is latent demand. A drop-then-recover detector flags a stockout signature in 3.76% of
eligible style-weeks, a lower bound.

## 5. COVID: two models, compared directly

I trained a second model without any row whose 13-week label window touches 1 March to 30 June
2020, and scored both out of sample on 13 non-COVID origins. Removing COVID data made rank
correlation worse (−0.056, CI [−0.126, −0.015]) and WMAPE worse (+0.023, [0.008, 0.045]), with no
difference on top-k, so COVID data did not hurt and mildly helped. The second model has about 40%
fewer rows, which cannot be told apart from "more data", and a 5-origin subset points the other way.

Sources: `covid_two_model_comparison.csv`.

## 6. Generation: nothing usable at first, and what was actually wrong

At first the generation step produced nothing usable, and I could not yet trust the checks that
would say so. Gate 1 checks that a concept is not too close to its references on average, Gate 1b
that it is not a near-copy of any single one, Gate 2 that it reads as the right product type,
colour and pattern, and Gate 3 that the changes I briefed are visible. I tested the similarity
checks with controls first. Leave-one-out showed a median threshold rejected 66% of 61 genuine
reference articles (21 passed), while a 90th-percentile limit passed all 61; an exact copy can
pass Gate 1 but fails Gate 1b in every style.

**I diagnosed the root cause wrongly twice before I measured it.** Two rounds of four seeds per
style produced no briefed design change. I first blamed conditioning on a single reference; testing
levers on the sweater showed that was half the story. More references changed the garment (crew for
V-neck) but not the brief, and lower image scales or text alone collapsed into fabric swatches,
because my attribute-first prompt ("Sweater, knitwear construction, beige melange. Novel accent: …")
let texture words dominate. What worked was a plain sentence naming the garment and its two
changes, given to both text encoders: a funnel neck and brown rib appeared even from one reference
at scale 0.25–0.35, and weighting the change clauses at 1.5 restored the brown accents at 0.45 (2
of 2, against 0 of 2). Prompt structure mattered first, references second. Final setup: that sentence on
both encoders, 8 concatenated references, weight 1.5, per-style scale 0.35 (from a 0.15–0.45 sweep;
0.6–0.7 removes the changes).

**Two checks were missing.** Gate 3 asks, per briefed change, whether the garment has it. Against
my labels on 23 images (46 questions) the small local reader has recall 0.95 but specificity 0.58
(accuracy 0.74): it says yes too easily, so it supports the human check and does not replace it.
For integrity, a small vision model asked whether a picture is a coherent garment caught 0 of 3
malformed images. A reference-based floor works: the closest real reference must be at least as
similar as the 10th percentile of real nearest-sibling similarity (DINOv2). Calibrated per style it
catches 3 of 3 malformed and 5 of 5 known-bad images but passes only 8 of 9 known-good, and in a
near-identical style it becomes a similarity check (0.922 for the white top). Calibrated globally,
over 154 real photos in 8 styles (0.779), it passes 9 of 9 known-good and catches 4 of 5 known-bad
and 2 of 3 malformed, missing one sheer mesh (closest reference 0.811). Coherence is not a property
of one style, so the global floor decides and the per-style floor is advisory. That corrected my
framing, not a verdict: the white top (0.733) is below both.

**Judges and references.** With Groq's daily limit and Gemini's free quota spent, the panel is two
local models calibrated on 26 controls: SmolVLM-500M (gap 0.338) and Florence-2-base (0.373). Gemini
calibrates too (gap 0.522) but ran out of its 20 daily requests, so whether it agrees with
SmolVLM's "original" pattern call on the bikini is unresolved. Florence-2 is advisory: its
agreement with the API readers (kappa 0.36–0.48) is too low to carry a verdict, against 0.68 for
SmolVLM with Groq. That turns the white top's Gate 2 from fail to pass, though its overall verdict stays fail. I also widened the reference base from 4–6 to 17 (sweater), 25 (dress) and 19 (white
top; only 20 such articles exist), added a border check after a fabric close-up got through the
small judge, and regenerated the sweater. Confidence intervals on the DINOv2 limits shrank 2.5–4.4
times.

Sources: `prompt_lever_summary.md`, `integrity_global_validation.csv`, `judge_panel_kappa.csv`.

## 7. The agent layer

An orchestrator delegates to five sub-agents (see `agent_architecture.png`) and two skills, one that writes the brief and one that runs the quality checks. The MCP server exposes eight tools over stdio; `generate_concept`, `score_concept` and `forecast_concept` do live work.

Rebuilding the demonstration exposed two defects, fixed without changing any threshold.
`score_concept` had no integrity floor or Gate 3 and relied on a spent API key, so it could pass a
concept the final scoring fails; `forecast_concept` had no forecast-origin argument and scored the
summer bikini against the autumn table (rank 894 of 1,980 instead of 118 of 3,000).

## 8. Results

**Choosing the styles: the model surfaces candidates, a person judges them.** Before looking at
output I fixed the rules: three styles from the emerging table (guards applied), no intimates or
garments you cannot identify in a flat photo, no shared colour or product type. Red underwear
(ranked first) was skipped for category and a sweater (fourth) for a product-type collision. That
leaves the beige melange sweater (predicted 17.8, growth 1.59), the red dress (11.0, 1.41) and the
white jersey top (13.9, 1.17).

**Final concepts**, best of 8 seeds. Gate 2 needs both judges; "human" means every briefed change is
visible.

| Concept | G1 | G1b | Integrity (global) | G2 | G3 | Human | Closed-loop forecast |
|---|---|---|---|---|---|---|---|
| Sweater | pass | pass | pass | pass | pass | yes | Orange Solid (Divided): 7.9, rank 461, low |
| Dress | pass | pass | pass | pass | pass | yes | Red Dresses Ladies: 11.0, rank 282, high |
| White top | pass | pass | **fail** (0.733 < 0.779; per-style 0.922 fails too) | pass (Florence advisory: fail) | fail (YN) | mostly (narrow cuffs) | Divided Jersey Fancy: 6.4, rank 651, medium |
| Bikini top (summer) | pass | pass | pass | fail (SmolVLM 0.28) | pass | yes | Orange All-over: 37.9, rank 118 of 3,000, medium |

Two of four concepts pass every automatic check. The white top fails the integrity floor (closest reference 0.733 against the global floor of 0.779) and Gate 3. That is a matter of yield, not impossibility: 2 of its 8 seeds cleared every check (47 and 48; the kept image is seed 42), against 1 of 8 for the sweater, 4 of 8 for the dress and 0 of 8 for the summer bikini top, so yield is low everywhere at this scale. What is particular to the white top is its assortment: 19 near-identical real articles put the per-style floor at 0.922 and leave the global floor of 0.779 little room, and only 3 of 8 seeds showed both briefed changes (8 of 8 for the sweater and the dress). A visible change is harder to land here, not impossible, and I did not loosen any check. The bikini's Gate 2 failure is SmolVLM answering "original" for the pattern.

**The closed loop, rebuilt as retrieval (still a prototype).** My first version had a small vision
model caption the picture and mapped the words to a style; on 40 real photos it named the exact
style 12.5% of the time (Florence-2: 2.5%), a specification error, since garment group and
department are not visible in a photo. Now the picture is matched to the nearest real style by CLIP
and DINOv2 similarity, returning a top five and a confidence label, over all 1,980 forecast styles
(15,020 photos, at most 8 per style; 3,000 styles for the summer table).

| Test | Candidates | Photos | Exact style | Top 5 | Top 10 |
|---|---|---|---|---|---|
| Free-text caption baseline | 1,980 | 40 | 12.5% | n/a | n/a |
| Retrieval, same 40 photos, each left out | 1,980 | 40 | 27.5% | 62.5% | 75% |
| Retrieval, leave-one-out | 1,980 | 159 | 42.1% | 67.3% | 76.1% |
| Retrieval, leave-one-out, one photo per style | 1,980 | 1,980 | 24.1% | 48.5% | 60.7% |
| Earlier partial index (optimistic), leave-one-out | 415 | 159 | 72.3% | 86.8% | 91.8% |

On the 40 photos (chance 0.05%) retrieval gets product type right 82.5% of the time against 65%,
and colour 52.5% against 70%, which is worse. The exact-style gain is not established: retrieval
alone was right on 11 photos and the baseline alone on 5 (McNemar p = 0.21), so it keeps the
prototype label. Confidence carries signal: on the one-per-style set the exact style is first 45%
of the time at high confidence, 21% at medium and 10% at low. Two of the four concepts do not
map to their intended style: the sweater to an orange sibling 0.006 more similar than its
intended style, which is third, and the white top to a neighbouring Divided style, its intended
style outside the top five.

**Seasonal view.** With the same rules and a summer model trained under a 13-week embargo, I pick an
orange all-over-pattern bikini top, predicted at 37.88 and realised at 37.94. That is one data
point: across 127 emerging candidates the median absolute error is 6.0 units (correlation 0.82), so
the 0.06 gap is luck.

Sources: `final_selection.csv`, `final_three_selection_log.csv`, `retrieval_validation.csv`.

## 9. What did not work, and what I could not verify

**Neighbourhood features.** I tested whether a style's cohort is rising: 15 causal features over
three sibling groups, under the same embargoed protocol, with an adoption rule fixed beforehand
(the paired top-20 interval must exclude zero). The model uses them (15.9% of total mean |SHAP|),
but they gave no out-of-sample gain: Hit@3-in-top20 0.528 to 0.611 (+0.083, CI [−0.083, +0.250]),
and precision@10 fell 0.033 (CI [−0.050, −0.017]). I kept the simpler model.

**Buyer-mix and price features.** I built 36 features from the customer table (buyer age, club and
newsletter mix, repeat buyers, postal-code spread, each with slopes) and 6 price-elasticity
features. Under the same protocol they made ranking worse: Hit@3-in-top20 0.528 to 0.333 (CI
[−0.306, −0.139]) and NDCG@10 down 0.046 (CI [−0.064, −0.020]). In-sample the model leans on them
(25.1% of mean |SHAP| for buyer features, 2.1% for price), but mainly as a size proxy: distinct
postal codes scale with volume. My hypothesis that a broadening buyer base predicts sustained
growth came out mixed: 7 of 10 features agreed in SHAP after I fixed a sign I had coded backwards,
one short of the 8 I had required. Club status and newsletter frequency come from a single
end-of-data snapshot, so they are not verifiably causal; they sit in the rejected model, so
nothing in the final model depends on them.

**Hyperparameters.** The original grid was chosen under the leaky protocol, so I re-ran it under
the embargo, selecting on rolling origins only. It picked 15 leaves, learning rate 0.10 and 100
trees rather than my 63, 0.05 and 200, but the gain is 0.65% in validation error, no more than the
spread across the whole grid, so I kept the final configuration. Three validation origins have
label windows that overlap the first test origin's forecast period; no test-origin feature,
prediction or metric entered selection.

**Limits I still carry.** The image readers are small, so a person decided every concept. The
global integrity floor rests on 8 styles, the limits on 8–25 real photos per style, and Gate 2
penalises a brown trim on a beige garment. The forecast is for styles, not for whether these
pictures would sell.

Sources: `neighbourhood_decision.csv`, `buyer_price_decision.csv`, `embargoed_retune_selected.csv`.

