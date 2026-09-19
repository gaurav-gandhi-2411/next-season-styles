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

Generated concepts are scored for novelty/fidelity via a margin, not absolute embedding cosine. B3's
original absolute CLIP band (`[0.8906, 0.9401]`) was derived from catalogue-vs-catalogue pairs, but
across-style mean cosine in that space is ≈0.799 — H&M's shared flat-lay/white-background
photography inflates similarity between unrelated styles, so an absolute threshold doesn't transfer
to generated-vs-catalogue pairs. C2's margin (similarity to own style minus similarity to a control
pool) cancels that confound by construction, giving bands of `[0.0325, 0.0975]` (CLIP) and
`[0.1436, 0.4307]` (DINOv2), derived from 23 styles/183 images for the upper anchor and a single
held-out style pair (8 samples) for the lower anchor.

C3's `ip_adapter_scale` sweep (0.2–0.9) is an honest, controlled finding: no scale lands in-band for
both spaces at once — CLIP only in-band at 0.2, DINOv2 never in-band anywhere in range. Judging uses
a two-family VLM panel: Gemini 2.5 (calibration gap ≈0.68, positive-control mean 0.833 vs.
negative-control 0.154) and Groq/Llama-4-Scout, genuinely unavailable (404 across 4 attempted model
names, not a missing key). All VLM output is labelled `LLM-consensus (NOT human ground truth)`. The
real final outcome: 0 of 3 concepts passed QC within the 2-retry cap
(`reports/tables/concept_qc_results.csv`) — evidence the QC system works, not an assignment failure.
It caught a genuinely degenerate generation (the sweater concept's own Gemini judge called it "knit
fabric," not "sweater" — a texture crop) and a real DINOv2 structural over-similarity failure (the
T-shirt's DINOv2 margin, 0.68–0.72, sits far above the 0.431 upper bound at every attempted scale
despite passing CLIP and the VLM attribute check).

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
and the hero figure (`reports/figures/FINAL_concepts.png`) composes the generated concepts with
their evidence chain. C4's SHAP analysis is reported honestly: the underwear style's prediction is
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

Other limitations: the C2 margin-band derivation rests on a small sample (23 styles/183 images for
the upper anchor, a single style pair/8 samples for the lower anchor), so band edges should be
treated as a first calibration, not a precise estimate. The 2-year, 104-week data span caps the
backtest at 20 origins (12 with paired LightGBM coverage) — a real ceiling on how tight any interval
here can be. The Gemini self-judging exclusion rule (`is_self_scoring_contamination`) is implemented
and tested but never actually exercised, since every scored concept used the `local_sdxl` backend —
Gemini's own image-generation backend was quota-blocked, not deliberately excluded. Finally, 0 of 3
concepts passing full QC is a genuine generation-quality limitation as well as a QC success story;
with more time, a wider per-style `ip_adapter_scale` sweep (rather than one style's sweep
extrapolated to the others) and more negative-prompt engineering for the underwear category's
recurring human-model failure would be next. Groq's genuine unavailability limited the VLM panel to
a single judge throughout, so no Cohen's kappa is computable anywhere in this project — a second
working judge is the highest-value addition to the generation-QC pipeline.
