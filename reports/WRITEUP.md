# next-season-styles — Write-up

## 1. Style definition and why

The unit of forecasting is not the H&M `article_id` (a single SKU/colourway) but a `style_key`: a
composite of 5 attribute columns from `articles.csv` — `index_group_name`, `product_type_name`,
`garment_group_name`, `perceived_colour_master_name`, `graphical_appearance_name`. The reasoning is
commercial: a buyer briefs a design team on "black jersey basic T-shirts," not on article
0554598001, and any one archetype is realised across many near-duplicate SKUs that turn over while
the archetype persists.

The evidence for this key's coherence is reported honestly rather than cherry-picked (A6,
`reports/tables/style_key_silhouette_comparison.csv`). Embedding each article's `detail_desc` gives a
weak observed silhouette of -0.453 for the 5-column key — but against a 50-shuffle permutation null
(mean -0.700, sd 0.0017) that sits at z=145.5: far more structure than chance despite the poor
absolute number. A reduced 4-column key (dropping `index_group_name`) retains more lifetime units
(90.33% vs. 85.07%) but scores worse on coherence (z≈89.0). The recommendation was to keep the
5-column key.

## 2. Target definition

The target is `units_per_active_article` (intensity) rather than raw unit volume: the top-20 styles
by lifetime units and by mean intensity have zero overlap
(`reports/tables/intensity_comparison.csv`) — raw volume rewards assortment breadth, not per-SKU
demand. An empirical-Bayes shrinkage (`intensity_shrunk`, causal trailing-window) and a
`price_index` guard sit alongside it. The raw-intensity leaderboard is measurably not
markdown-driven: 18 of its top-20 styles have a defined `price_index` averaging ≈1.043 (essentially
full price), with only 1 of 18 below 0.9.

## 3. Evaluation methodology

The headline of this section is a diagnosis, not a straight-line success story: three modelling
alternatives were tried under a proper paired evaluation, on top of finding the premise motivating
them was itself a measurement artifact — both reached by rigorous experimentation, not assumed. The
harness uses 20 rolling origins, 4-week step, 13-week burn-in/horizon; LightGBM's paired comparison
runs on the 12-origin subset every method shares (`reports/tables/backtest_summary_v2.csv`). Its
headline metric, Hit@3-in-top20 (does any of the top-3 picks land anywhere in the realised top-20),
is 0.722 (CI [0.694, 0.917]), beating a random floor of 0.004 and all 4 baselines with
paired-difference CIs excluding zero.

**Track G tested the premise directly.** An earlier diagnostic (A1) found substantial top-3/top-20
overlap (0.421) between consecutive 4-week-apart backtest origins — read at the time as evidence of
a genuine persistence signal LightGBM might be under-using. G1 built a strictly CAUSAL persistence
oracle instead: predicting origin `t` from the fully-realised target of the earliest EARLIER origin
whose own 13-week forward window is guaranteed already closed by `t` (a derived minimal lag of 4
origin-steps / 16 calendar weeks, not assumed). Scored on the identical 12 origins, the oracle's
Hit@3-in-top20 is 0.222 (CI [0.083, 0.472]) — well below LightGBM's 0.722, not above it
(`reports/tables/g1_diagnostics_summary.csv`). The honest read: A1's 0.421 overlap was inflated by
window-overlap autocorrelation between origins whose 13-week forward targets share 9 of 13 weeks,
not genuine forecastable persistence. The premise was wrong — and G1 found that by measuring it
properly, not by assuming it either way.

G2 tested the corrected hypothesis that a ranking objective, not L2 regression, might close the
remaining gap: `lambdarank`, retrained on the identical feature set/origins, underperforms L2 on
every head metric — pooled Hit@3-in-top20 0.611 (CI [0.528, 0.833]), paired diff -0.111 (CI
[-0.167, -0.056], excluding zero), Spearman correlation down 0.27
(`reports/tables/g2_lambdarank_vs_l2.csv`). G3 tried three further variants on the L2 base:
top-heavy-weighted L2 (0.639, diff -0.083, CI excludes zero), a two-stage classify-then-rank model
(0.583, diff -0.139, CI excludes zero, plus a wmape blowup to ~15 from L2's 0.14 — a real
regression-head defect), and a rank-averaged L2+lambdarank ensemble (0.694, diff -0.028, CI
[-0.111, 0.083], including zero — the closest challenger, but still numerically below L2)
(`reports/tables/g3_variants.csv`). None of the four alternatives beat L2's 0.722 with a favourable
CI excluding zero. G4 re-ran the locked selection pipeline against the frozen L2 model and confirmed
the final three are unchanged (Section 8).

Why does a positive result at rank-20 evaporate at rank-3? Hit@3-in-top20 0.722 → Hit@3-in-top10
0.500 → Precision@3 0.056 (CI [0.000, 0.111], overlapping a random floor's [0.000, 0.004] — no
demonstrated signal there). Ground-truth instability concentrates at the extreme head — the model
finds the right neighbourhood, not the exact top-3 order. A1's turnover diagnostic agrees: raw vs.
shrunk intensity's top-3 turnover paired diff is +0.123 (t=1.508, p=0.149) — directional, not
significant at n=19; the production target was deliberately NOT switched to shrunk intensity on
that non-significant advantage.

## 4. The stock limitation

Every number here is derived from realised transactions, censored by what was actually stocked and
merchandised — not latent demand. The model inherits a bias toward whatever was well-stocked
historically. A drop-then-recover detector flags a stockout signature in 3.76% of eligible
style-weeks (`src/nss/viz/panel_eda.py`) — a lower bound, since it requires a recovery to fire and so
cannot see permanent stockouts or end-of-life truncation. No inventory data was invented anywhere.

## 5. COVID

Weekly transaction volume hits its series low in the week of 2020-03-16 (183,656 transactions),
roughly 26% below the preceding 8 weeks' mean (248,446), before rebounding sharply
(`reports/figures/weekly_volume.png`). Every rolling origin carries an `is_covid` flag, and results
are reported both pooled and split. The COVID (n=7) and non-COVID (n=5) splits are directional-only —
too few origins for their own confidence intervals to be trustworthy standalone.

## 6. Generation

This project caught and corrected its own measurement error in generation QC FOUR times: absolute
cosine → margin → real-space anchors → generated-space anchors → corrected generated-space anchors.

**First two corrections (B3→C2, then E1).** B3's absolute CLIP band from catalogue-vs-catalogue pairs
didn't transfer to generated-vs-catalogue pairs (H&M's shared flat-lay photography inflates
similarity between unrelated styles); C2's margin (own-style minus control-pool similarity) fixed
that. C2's anchors then had the same defect one level up — REAL images validating GENERATED ones,
never checked against SDXL+IP-Adapter's own output distribution. E1 re-derived both anchors directly
in generated-image space, both at `ip_adapter_scale=1.0`.

**Third correction (F1).** E1's `unrelated_anchor_gen` was generated at the same
`ip_adapter_scale=1.0` as `copy_anchor_gen` — IP-Adapter's image conditioning dominated the
deliberately-different-garment text prompt at full strength, so the two calibration endpoints scored
almost identically: pooled gap CLIP +0.003, DINOv2 +0.025, neither clearing a 0.05
non-discriminative bar. F1 re-derived ONLY `unrelated_anchor_gen` at `ip_adapter_scale=0.0` (zero
image conditioning); corrected gap CLIP +0.140, DINOv2 +0.505
(`reports/tables/margin_anchor_realspace_vs_genspace_gap.csv`), both clearing the bar.

**Fourth stage (F2): per-judge Gate-2 thresholds.** The flat 0.75 attribute-fidelity threshold was
never reachable for the Groq judge — its own calibration ceiling is 0.584. F2 scaled the threshold to
each judge's ceiling (0.75×): Groq 0.438, Gemini 0.625 (ceiling 0.833). Re-scoring E5's 24
already-generated candidates leaves Gate 1 unchanged (4/24) but raises Gate 2 from 0/24 to 6/24 and
overall from 0/24 to 2/24 — real progress from fixing a miscalibrated gate.

**F5: the final generation run — 0/12 pass, distinct real mechanisms per style.** 12 candidates (4
seeds × 3 styles, no retries) under F1's corrected anchors, F2's thresholds, F3's screened
full-garment references, and F4's token-budget-safe prompts. The T-shirt passes Gate 1's CLIP check
but fails DINOv2 over-similarity on every seed; its selected candidate (seed 43, chosen by manual
visual QC over seed 42's higher raw score, since seed 42 rendered a two-tone grey/black colour-block
shirt, not the required solid black) still fails DINOv2. The underwear fails BOTH gates — DINOv2
over-similarity every seed, Gate 2 for want of a score — via a pattern-drift mechanism distinct from
E5's now-fixed human-model defect: all 8 reference images are themselves lace-constructed
(`detail_desc`, e.g. "Thong briefs in lace...") despite H&M's `graphical_appearance_name=Solid` label
(a colour-family tag, not a fabric-texture one) — IP-Adapter's conditioning pulls lace texture in
regardless of F4's negative-prompt rule correctly excluding "floral, lace, pattern, print,
embroidery" in text; the selected candidate is visibly a lace-mesh brief with only its trim solid
red. The sweater's E5 texture-close-up framing defect is genuinely fixed by F3's reference screening
— every F5 candidate is a clean, full-garment shot — but now fails Gate 1's copy-check on BOTH CLIP
and DINOv2 every seed, at the corrected `scale=0.45` conditioning strength.

**Judge availability was severely constrained.** Gemini's daily quota was already exhausted before F5
started, confirmed still exhausted immediately after; Groq's token budget had only ~2,275 tokens of
headroom, consumed by the first candidate scored. Exactly 1 of 12 candidates got a real judge score
(Groq, T-shirt seed 42 — the candidate visual QC later rejected); the rest have zero contributing
judges. No Cohen's kappa is computable, and most `fidelity_pass=False` reflects unavailability, not
measured poor fidelity.

## 7. Agent architecture

An orchestrator delegates to 5 sub-agents (`reports/figures/agent_architecture.png`): `data-analyst`,
`forecaster`, `style-profiler`, `concept-designer`, and `critic`. The orchestrator calls no MCP tool
directly. Two reusable skills carry the dataset-agnostic logic — `skills/style-brief/` and
`skills/concept-qc/` — with H&M-specific adapters kept outside each. The critic's retry loop is real
and exercised: the D4 demo run (`reports/agent_run_transcript.md`) drove all 3 final concepts through
1 original attempt + 2 retries each, escalating only after the cap was exhausted. The MCP server
(`src/nss/mcp_server.py`) exposes 7 read-only tools over stdio; only `generate_concept` does live
work.

## 8. Results

The final forecast selects a T1 (incumbent) and T2 (emerging) pair, both guard-passing. T1 rank 1:
Black Jersey Basic T-shirt; T2 rank 1: Red underwear bottom, growth ratio ≈3.67; T2 rank 2: Beige
Melange sweater, growth ratio ≈1.59 (`reports/tables/top_styles_final_three.csv`). Track G's
retraining work (Section 3) confirmed these three styles unchanged (G4): the T1/T2 split is the
commercially useful framing regardless of which model produces it. A seasonal bonus table
(`reports/tables/top_styles_by_season_v2.csv`) adds a diversity-constrained top-3 per season, and the
hero figure (`reports/figures/FINAL_concepts.png`, rebuilt in F6 from F5's final generation, clean of
QC stamps) composes the generated concepts, with `evidence_chain.png` carrying the honest 0/12
candidate (0/3 style) status and judge-availability gap. Each concept's `applied_changes`
(`design_briefs.json`) are baked into its prompt: charcoal topstitching/cropped hem (T-shirt),
burgundy trim/raised waistband (underwear), funnel neckline/camel ribbing (sweater). C4's SHAP
analysis is reported honestly: the underwear's prediction is driven by `lag_1` (persistence,
SHAP≈0.486), not any seasonal/Christmas feature; the sweater's dominant driver differs —
`n_active_articles_level` (SHAP≈0.254) outweighs `lag_1` (SHAP≈0.164)
(`reports/tables/final_three_shap_verdict.csv`).

## 9. Limitations and what I would do with more time

The most consequential bug found was cross-process prediction jitter in the final LightGBM forecast —
separate `uv run` invocations disagreeing by up to ~52% relative on some predictions. Root-caused via
controlled ablation: the dominant cause was a missing `maintain_order="left"` on a polars
`features.join(targets, ...)` call, not the hypothesized categorical-hashing nondeterminism
(negligible ~5.3e-15) — without it, join row order isn't stable across processes, and LightGBM's
histogram gradient accumulation is a non-associative sum over row order. Fixed and verified
bit-identical (max abs diff 0.0) with a subprocess regression test.

Track G's retraining is itself a reported limitation, not a suppressed one: neither a
ranking-objective retrain (`lambdarank`) nor three engineering variants (top-heavy weighting,
two-stage, ensemble) beat L2's headline metric significantly — the feature set and origin count
appear close to exhausted for this task, a real ceiling rather than an unexplored one. The
underwear's F5 pattern-drift defect is similarly concrete: its reference images are themselves
lace-constructed despite a "Solid" label, and no text-level negative-prompt fix fully overrides
IP-Adapter's image conditioning — reference-set curation (excluding lace-textured stock photography
for solid-colour styles) is the untried next lever. Other limitations: the C2/E1/F1 margin-anchor
calibrations rest on small samples; the 2-year span caps the backtest at 20 origins (12 paired); the
sweater's `scale=0.45` copy-check failure is now genuine over-similarity, not a framing defect, and a
wider `ip_adapter_scale` sweep is the natural next step; F5's judge-quota exhaustion leaves final
selection resting on visual QC and one real judge score, with no kappa computable.
