# next-season-styles — Write-up

## 1. Style definition and why

The unit of forecasting is not the H&M `article_id` (a single SKU/colourway) but a `style_key`: a
composite of 5 attribute columns from `articles.csv` — `index_group_name`, `product_type_name`,
`garment_group_name`, `perceived_colour_master_name`, `graphical_appearance_name`. The reasoning is
commercial: a buyer briefs a design team on "black jersey basic T-shirts," not on article
0554598001, and any one archetype is realised across many near-duplicate SKUs that turn over while
the archetype persists.

The evidence for this key's coherence is reported honestly rather than cherry-picked (A6,
`reports/tables/style_key_silhouette_comparison.csv`). Embedding each article's `detail_desc` and
measuring the 5-column key's silhouette in that space gives a weak observed silhouette of -0.453 —
but against a 50-shuffle permutation null (mean -0.700, sd 0.0017) that sits at z=145.5: far more
structure than chance despite the poor absolute number. The reason is that `detail_desc` describes
construction/fabric while `style_key` encodes merchandising attributes — related views of a garment,
not the same view. A reduced 4-column key (dropping `index_group_name`) retains more lifetime units
(90.33% vs. 85.07%) but scores worse on coherence (z≈89.0, silhouette -0.476). The recommendation,
report-only, was to keep the 5-column key.

## 2. Target definition

The target is `units_per_active_article` (intensity) rather than raw unit volume: the top-20 styles
by lifetime units and by mean intensity have zero overlap
(`reports/tables/intensity_comparison.csv`) — raw volume rewards assortment breadth, not per-SKU
demand. An empirical-Bayes shrinkage (`intensity_shrunk`, causal trailing-window) and a
`price_index` guard sit alongside it. The raw-intensity leaderboard is measurably not
markdown-driven: of its top-20 styles, 18 have a defined `price_index` and their mean is ≈1.043
(essentially full price), with only 1 of 18 below 0.9.

## 3. Evaluation methodology

This is the section that should be weighed most heavily, because the honest story is a correction,
not a straight line. The harness uses 20 rolling origins, 4-week step, 13-week burn-in/horizon. An
earlier comparison scored LightGBM against 4 baselines and a random-permutation floor (20 seeds) on
mismatched origin counts — LightGBM's own eval window covered only 12 of 20 origins while baselines
used all 20. `backtest_v2.py` fixes this by re-scoring every method, floor included, on the same
12-origin subset (`reports/tables/backtest_summary_v2.csv`, paired diffs in
`reports/tables/backtest_paired_diff.csv`).

On that corrected basis, the originally intended headline — Precision@3 — shows no demonstrated
signal: LightGBM's pooled Precision@3 is 0.056 (CI [0.000, 0.111]) vs. a random floor of 0.0014 (CI
[0.000, 0.004]) — overlapping intervals. Rather than call the model a failure, the headline metric
was replaced, not the model: Hit@3-in-top20 (does any of the top-3 picks land anywhere in the
realised top-20) scores 0.722 (CI [0.694, 0.917]) vs. a floor of 0.004 (CI [0.000, 0.007]), beating
all 4 baselines with paired-difference CIs that exclude zero throughout (vs. seasonal-naive: +0.400
CI [0.233, 0.700]; EWMA-persistence: +0.444 CI [0.444, 0.556]; parent-category-mean: +0.694 CI
[0.667, 0.889]; global-mean: +0.722 CI [0.694, 0.917]).

Why does a positive result at rank-20 evaporate at rank-3? The metric cascade shows why:
Hit@3-in-top20 0.722 → Hit@3-in-top10 0.500 → Precision@3 0.056. Ground-truth instability is
concentrated at the extreme head of the ranking — the model reliably finds the right neighbourhood
of styles but not the exact top-3 order. A1's turnover diagnostic
(`reports/tables/target_turnover_corrected.csv`) is consistent: raw vs. shrunk intensity's top-3
turnover paired difference is +0.123 (std 0.355, t=1.508, p=0.149) — directionally favouring
shrunk, not significant at n=19; top-10 is weaker still (+0.037, t=1.022, p=0.320). The production
target was deliberately NOT switched to shrunk intensity on this non-significant advantage; that
decision is locked.

## 4. The stock limitation

Every number here is derived from realised transactions, censored by what was actually stocked and
merchandised — not latent demand. The model inherits a bias toward whatever was well-stocked
historically. A drop-then-recover detector flags a stockout signature in 3.76% of eligible
style-weeks (commit `bbc9d8d`, `src/nss/viz/panel_eda.py`) — a lower bound, since it requires a
recovery to fire and so cannot see permanent stockouts or end-of-life truncation. No inventory data
was invented anywhere in this project.

## 5. COVID

Weekly transaction volume hits its series low in the week of 2020-03-16 (183,656 transactions),
roughly 26% below the mean of the preceding 8 weeks (248,446), before rebounding sharply
(`reports/figures/weekly_volume.png`). Every rolling origin carries an `is_covid` flag (true if its
13-week horizon overlaps 2020-03-01 to 2020-06-30), and results are reported both pooled and split.
The COVID (n=7) and non-COVID (n=5) splits are reported as directional-only, not with the weight of
the pooled n=12 (paired LightGBM comparison) or n=20 (full harness) numbers — too few origins for
their own confidence intervals to be trustworthy standalone.

## 6. Generation

This project caught and corrected its own measurement error in generation QC **twice**.

**First correction (B3→C2).** B3's absolute CLIP band (`[0.8906, 0.9401]`), from
catalogue-vs-catalogue pairs, doesn't transfer to generated-vs-catalogue pairs: across-style mean
cosine is already ≈0.799 there, since H&M's shared flat-lay photography inflates similarity between
unrelated styles. C2's margin (own-style minus control-pool similarity) cancels that: `[0.0325,
0.0975]` (CLIP), `[0.1436, 0.4307]` (DINOv2), from 23 styles/183 real images.

**Second correction (E1).** C2's anchors came from REAL images but were applied, unchanged, to score
GENERATED images — B3's defect class, one level up: never validated against SDXL+IP-Adapter's own
output distribution. E1 re-derived both anchors directly in generated-image space
(`copy_anchor_gen`: a literal reference-description prompt; `unrelated_anchor_gen`: same references,
a different garment), both at `ip_adapter_scale=1.0`. The gap is large on the unrelated anchor:
pooled DINOv2 mean is ≈0.097 real-space vs. ≈0.490 generated-space, a ~5x inflation
(`reports/tables/margin_anchor_realspace_vs_genspace_gap.csv`); the copy anchor gap is smaller and
opposite (0.574 real vs. 0.515 generated, DINOv2). Side finding: at scale=1.0, forcing a different
garment barely moved the margin (0.515 vs. 0.490) — IP-Adapter's image conditioning dominates the
text prompt at full strength. Gate 1 is now a sign-safe one-sided copy-check, `margin <
copy_anchor_gen - 0.10*|copy_anchor_gen|`, on BOTH CLIP and DINOv2, anchored in generated-image space
(the old band is diagnostic only); Gate 2 (VLM fidelity ≥0.75) is unchanged; `overall_pass` requires
both.

**The second catch changed the result.** Re-scoring the 9 already-logged C7 attempts under the
corrected gate (`reports/tables/concept_qc_rescored_under_new_gate.csv`, no new generation/scoring)
gives **0/9 passing** — identical to the old gate. That is instrument validation's value: a sudden
pass on old attempts would mean the original gate was simply too strict; instead it confirms the
defects were real, not a measurement artifact — the fix changes WHY each attempt fails, not THAT it
fails.

**E5: honest 0/3 on the corrected gate.** Regenerating the 3 final concepts (fixed sweater framing
prompt; novelty moved into the prompt, 2+ `applied_changes` per style — Section 8;
`ip_adapter_scale=0.45`, above 0.2, already shown by C3/E4 to lose defining attributes) gave 24
candidates (8 seeds × 3 styles, 2 retry rounds), all visually inspected — still **0/3 pass**, each
for a distinct mechanism: the T-shirt fails Gate 1 on DINOv2 over-similarity across every qualified
candidate (seeds 46–49 disqualified for showing a human model); the underwear's C6/C7 human-model
defect is FIXED (0/8 show a model), but colour/pattern drifted to floral lace instead of solid red
(confirmed by the judge and visual inspection); the sweater's failure root-causes to its single
reference image itself being a texture close-up — out of scope to fix by swapping references. Also
fixed: SDXL's 77-token CLIP truncation was silently dropping the novelty instructions from the first
drafted prompts.

**VLM panel (E6).** A second working judge now exists: `qwen/qwen3.8-27b` on Groq, found via a live
model-list query for `input_modalities` rather than a 5th guessed name; it passed calibration (gap
≈0.44 vs. Gemini's ≈0.68). Gemini's free daily quota (20 req/day) was exhausted during E5's
regeneration, so the final-concept scoring round is Groq-only — no Cohen's kappa on the final
numbers, though the 2-judge infrastructure is built and validated. All VLM output remains labelled
`LLM-consensus (NOT human ground truth)`.

## 7. Agent architecture

An orchestrator delegates to 5 sub-agents (`reports/figures/agent_architecture.png`): `data-analyst`
(sales/trend Q&A), `forecaster` (reads frozen forecast tables), `style-profiler` (builds a design
brief via the `style-brief` skill), `concept-designer` (`generate_concept`), and `critic`
(`score_concept` + QC gate). The orchestrator calls no MCP tool directly. Two reusable skills carry
the dataset-agnostic logic — `skills/style-brief/` and `skills/concept-qc/` — with H&M-specific
adapters kept outside each skill. The critic's retry loop is real and exercised: the D4 demo run
(`reports/agent_run_transcript.md`) drove all 3 final concepts through 1 original attempt + 2
retries each (9 logged attempts), escalating to the orchestrator only after the cap was exhausted.
The MCP server (`src/nss/mcp_server.py`) exposes 7 read-only tools over stdio —
`query_transactions`, `get_style_profile`, `forecast_styles`, `get_reference_images`,
`generate_concept`, `score_concept`, `compose_final_sheet` — with modelling frozen: only
`generate_concept` performs live work.

## 8. Results

The final forecast selects a T1 (incumbent) and T2 (emerging) pair, both guard-passing. T1 rank 1:
Black Jersey Basic T-shirt (`Ladieswear || T-shirt || Jersey Basic || Black || Solid`), the single
highest-predicted-intensity guard-passing style. T2 rank 1: Red underwear bottom, growth ratio ≈3.67;
T2 rank 2: Beige Melange sweater, growth ratio ≈1.59
(`reports/tables/top_styles_final_three.csv`). The T1/T2 split is the commercially useful framing:
T1 tells a buyer what to keep betting on, T2 what's accelerating and worth a larger allocation —
reporting both avoids collapsing two different business questions into one list. A seasonal bonus
table (`reports/tables/top_styles_by_season_v2.csv`) adds a diversity-constrained top-3 per season,
and the hero figure (`reports/figures/FINAL_concepts.png`, rebuilt in E8, clean of QC stamps)
composes the generated concepts, with `evidence_chain.png` carrying the honest 0/3 Gate 1/Gate 2
status per style. Each concept's `applied_changes` (`design_briefs.json`) are baked into its
prompt: charcoal topstitching/cropped hem (T-shirt), burgundy trim/raised waistband (underwear),
funnel neckline/camel ribbing (sweater). C4's SHAP analysis is reported honestly: the underwear style's prediction is
driven by `lag_1` (persistence, SHAP≈0.486), not any seasonal/Christmas feature, despite that being
a plausible narrative — no fourier or `lag_52` term appears in its top-5 at all. The sweater's
dominant driver differs: `n_active_articles_level` (SHAP≈0.254) outweighs `lag_1` (SHAP≈0.164)
(`reports/tables/final_three_shap_verdict.csv`).

## 9. Limitations and what I would do with more time

The most consequential bug found was cross-process prediction jitter in the final LightGBM forecast
— separate `uv run` invocations of the same trained model disagreeing by up to ~52% relative on some
predictions. Root-caused via controlled ablation: the primary cause was NOT the originally
hypothesized categorical-hashing nondeterminism alone (that contributed a negligible ~5.3e-15). The
dominant cause was a missing `maintain_order="left"` on a polars `features.join(targets, ...)` call
— without it, join output row order isn't guaranteed stable across processes, and LightGBM's
histogram gradient accumulation is a non-associative floating-point sum over row order, so a
shuffled order alone produced materially different predictions. Fixed and verified bit-identical
(max abs diff 0.0) across separate processes with a new subprocess regression test.

Other limitations: the C2/E1 margin-anchor calibrations both rest on small samples — C2's real-space
anchors on 23 styles/183 images (upper), 8 samples (lower); E1's on 3 seeds per style per anchor
type — Gate 1's thresholds are a first calibration in either space. The 2-year,
104-week span caps the backtest at 20 origins (12 paired) — a real ceiling on interval tightness.
`is_self_scoring_contamination` is tested but never exercised — every scored concept used
`local_sdxl`; Gemini's image-generation backend remains quota-blocked, not deliberately excluded.
The sweater's failure is concrete but out of scope here: its sole IP-Adapter reference image
(`data/images/0673677023.jpg`) is itself a close-up fabric-texture shot, and that framing dominates
generation regardless of prompt — a reference-image swap, not authorized in E5. Gemini's quota
exhaustion (Section 6) again limited final-round scoring to Groq alone, with no kappa on the
numbers that selected the final concepts. Finally, 0/3 concepts passing full QC is a genuine
generation-quality limitation: next would be the sweater's reference-image swap, a stronger
negative-prompt for the underwear's floral-lace drift, and a wider `ip_adapter_scale` sweep.
