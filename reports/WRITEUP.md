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

## 3. Evaluation methodology

The harness uses 20 rolling origins, a 4-week step and a 13-week horizon; the LightGBM paired
comparison runs on the 12 origins every method shares (`reports/tables/backtest_summary_v2.csv`).
Headline metric Hit@3-in-top20 (do the top-3 picks land in the realised top-20; chosen after a random-floor check showed Precision@3 near chance, so both are reported) is 0.722 (CI
[0.694, 0.917]) against a random floor of 0.004, beating all 4 baselines with paired-difference CIs
excluding zero.

**The result is not leakage: a label-shuffle control collapses it to chance.** I permuted the
forward target across styles within each origin, retrained the identical model on the shuffled
labels (3 seeds, same locked config, same 12 test origins) and scored it on the real answers: **0
hits in 108 picks** (Hit@3-in-top20 0.000 vs floor 0.004; Spearman −0.012; NDCG@10 0.396 vs floor
0.398). The unshuffled control through the same code reproduces 0.722 / 0.500 / 0.803 / 0.900
(`reports/tables/label_shuffle_control.csv`).

**Track G tested the premise.** An earlier 0.421 top-3/top-20 overlap between consecutive origins
looked like persistence LightGBM under-used. A strictly causal persistence oracle (16-week lag)
scores 0.222 (CI [0.083, 0.472]) on the same 12 origins: the 0.421 was window-overlap
autocorrelation. Four alternative objectives (best: rank ensemble 0.694) did not beat L2 under paired bootstrap CIs. The model is frozen.

**Why Precision@3 is near-random while head-region ranking is not.** Precision@3 is 0.056 (CI
[0.000, 0.111]) vs the oracle's 0.028: the true #3 and #4 styles differ by 0.61% of #3's value on
average, so ordering inside a near-tie is noise for any method. The model finds the neighbourhood,
not the order, hence a T1/T2 triple.

## 4. The stock limitation

Every number derives from realised transactions, censored by what was stocked and merchandised — not
latent demand. A drop-then-recover detector flags a stockout signature in 3.76% of eligible
style-weeks (`src/nss/viz/panel_eda.py`) — a lower bound, since it cannot see permanent stockouts.
No inventory data was invented.

## 5. COVID

Weekly volume bottoms in the week of 2020-03-16 (183,656 transactions), ~26% below the preceding
8-week mean, then rebounds (`reports/figures/weekly_volume.png`). Every origin carries an `is_covid`
flag; results are reported pooled and split, but the splits (n=7, n=5) are directional only.

## 6. Generation: a five-stage instrument validation

The generation stage produced a quality-control gate that rejected everything; the finding is that the *gate* was mis-built, in five ways, each found by measuring it against a control, not by tuning until concepts passed.

1. **Absolute CLIP cosine — rejected.** H&M's shared flat-lay photography makes even unrelated styles score a mean 0.799.
2. **Margin scoring.** Own-style similarity minus a control-pool similarity cancels the shared photographic style.
3. **Anchors re-derived in generated space (E1, F1).** Real-image anchors did not transfer to SDXL output, so both calibration endpoints were regenerated; the "unrelated" one had to be remade at `ip_adapter_scale=0.0`, since at 1.0 image conditioning overrides the prompt (endpoint gaps +0.003/+0.025 → +0.140/+0.505).
4. **A real-article benchmark replaces the copy anchor (H1).** The copy anchor was SDXL's most reference-faithful image, so gating at 90% of it was a fidelity ceiling mislabelled as a plagiarism check. H1 passed a concept whose mean similarity to its references was at or below the *median* similarity between distinct real articles of the style, in CLIP and DINOv2.
5. **The median was itself wrong, a leave-one-out control shows it, and p90 replaces it (J1–J2).** A median puts about half of real products on the wrong side by construction. I measured that: each real reference article was scored against its siblings through the identical gate code (`leave_one_out_control.csv`). **Under the median rule only 6 of 16 real articles (37.5%) pass** (T-shirt 2/6, underwear 1/4, sweater 3/6): the gate rejected 62.5% of genuine products. The threshold became the p90 of within-style pairwise similarity. Sanity check: 16/16 real articles pass the shipped threshold and 14/16 (87.5%) a threshold recomputed without the held-out article. **Gate 1 on F5's 12 candidates moved 4/12 → 12/12; on the four H3 underwear candidates 4/4 → 4/4; on the three final selections 1/3 → 3/3.**

**What p90 does not do, and Gate 1b.** An exact copy of reference 0, scored as a candidate, passes
Gate 1 for the T-shirt and underwear (`clone_positive_control.csv`): a mean over n references dilutes
a copy of one, so Gate 1 is a range check, not a copy detector. A detector that passes an exact copy
is broken by that control alone, so **Gate 1b** checks the nearest reference: a concept's closest
single reference must be no closer than the p90 of the *real* nearest-sibling similarity
(leave-one-out, per style and measure; `gate1b_nearest_reference.csv`), not a raw cut. Two rules
were fixed before scoring any concept: the clone must fail (it does, in all three styles) and the
threshold would not move afterwards. With 4–6 references the real nearest-sibling values pair up, so
this p90 equals the closest real pair.

**Judge checklist (H2, L2).** `garment_group` ("Jersey Basic") has no visual referent, so only product type, colour and pattern are scored; non-visual pattern labels (Other structure, Other pattern, Unknown, Treatment) are excluded the same way, and fidelity is reported both ways (`fidelity_both_figures.csv`).

**Underwear defect (H3).** The style's "Solid" label is a colour tag and its references were all lace, so IP-Adapter reproduced lace. Four verified plain references fixed it.

**Final result, stated plainly.** Each final concept was judged three times; the fidelity is the median, with its spread (`j4_judge_repeats.csv`):

| Concept | Gate 1 (p90) | Gate 1b (nearest reference) | Fidelity median (range) | Gate 2 |
|---|---|---|---|---|
| T-shirt | pass | **fail** (DINOv2 0.913 vs 0.911; CLIP passes) | 0.900 (0.000) | pass |
| Underwear | pass | pass | 0.614 (0.006) | pass |
| Sweater | pass | pass | 0.567 (0.000) | pass |

Gate 2 uses a 0.513 threshold. **The T-shirt fails, and the failure is corroborated.** Gate 1b, calibrated on real article pairs and clone-validated, flagged it (by 0.0018 on DINOv2; limit not moved), and my own visual comparison independently found it undifferentiated: the briefed charcoal topstitching and cropped drop-shoulder cut are not visible, so it is a plain black tee. It is not a copy of one photo (its nearest reference has a different cut), just a basic adding nothing over the category. The sweater's Gate 2 clears by 0.053, inside the ±0.21
noise bound, and the judge read its graphical treatment as "solid" against "melange" (0.00 on that
attribute), so the pass rests on product type and colour. Underwear and sweater pass every check plus
my visual check.

## 7. Agent architecture

An orchestrator delegates to 5 sub-agents (`reports/figures/agent_architecture.png`): `data-analyst`,
`forecaster`, `style-profiler`, `concept-designer`, `critic`. The orchestrator calls no MCP tool
directly. Two reusable skills carry the dataset-agnostic logic — `skills/style-brief/` and
`skills/concept-qc/` — with H&M-specific adapters outside each. The critic and the MCP `score_concept` tool apply the shipped gates (1, 1b, 2, plus a mandatory
human check); the demo transcript (`reports/agent_run_transcript.md`) predates them. The MCP server
exposes 7 tools over stdio; `generate_concept` and `score_concept` do live work.

## 8. Results

The forecast selects a T1 (incumbent) and T2 (emerging) pair, both guard-passing. T1 rank 1: Black
Jersey Basic T-shirt; T2 rank 1: Red underwear bottom (growth ratio ≈3.67); T2 rank 2: Beige Melange
sweater (≈1.59) (`reports/tables/top_styles_final_three.csv`). The hero `reports/figures/FINAL_concepts.png`
shows the best candidate per style captioned with what is visible;
`evidence_chain.png` traces references → brief → concept → Gate 1/1b → fidelity → verdict. SHAP: the
underwear is driven by `lag_1` (≈0.486); the sweater by `n_active_articles_level` (≈0.254).

**Seasonal view.** The same pipeline run for Summer changes the garment, not just a table
(`seasonal_comparison.png`; tables in `top_styles_by_season_v2.csv`). The
autumn/winter forecast (21 Sep 2020) leads with black jersey basics (T-shirt: 33.8 units per product
per week). Re-run as of 1 Jun 2020 using only earlier data, the model ranks a black textured swim
bottom 1 of 2,763 (forecast 93.1; realised 50.0: the ranking held, the level was overshot by ~86%),
and the concept is a high-waisted ribbed swim bottom. Two features make this possible: the Fourier
terms encode time of year, lifting a style whose sales concentrate in one season when the window
falls there, and `lag_52` supplies last year's same-week level; the same mechanism lifts
autumn/winter layering (knitwear, tights). The summer concept passes both similarity checks, my visual check and, after the L2 exclusion of the non-visual label "Other structure", Gate 2 (visible-only median 0.562 vs 0.513, a 0.049 margin inside the noise bound; 0.375 counting the label; both reported). (Unrefined brief; references screened by eye.)

## 9. Limitations and what I would do with more time

**Automated QC catches drift, not incoherence.** Of the four regenerated underwear candidates, three were visually malformed: a cut-out defect (seed 42), sheer mesh panels (44) and an unrecognisable folded object (45). All three passed Gate 1 under both the median and p90 rule, and all three cleared Gate 2 (single judge calls 0.617, 0.600, 0.567 vs 0.513). The Summer run repeated it: two of its four candidates (a pinstripe pattern drift, straps on a bottom) passed both similarity gates. Similarity scoring flags images too close to their references and attribute scoring flags drift from the brief; neither sees whether the garment is a garment. This is a conclusion about the limits of automated QC, not a caveat: a human check remains necessary, and the missing instrument is a judge of garment integrity.

**Measurement limits.** n is 4–6 references per style, so the within-style limits are noisy, and Gate 1b's limit equals the closest real pair, a strict test for near-identical basics like the T-shirt. One T-shirt image scored 0.6375 then 0.425 across sessions, so every fidelity number carries a measured bound of about ±0.21. The three repeats per final concept agree within 0.006 because they were made minutes apart: repeatability, not accuracy. Fidelity comes from one judge model (Groq's `qwen3.8-27b`, calibrated on 3 positive controls); Gemini's free tier and an OpenRouter fallback were unusable. The sweater's melange miss (0.00 although the flecking is visible) is a genuine judge error and stays scored. Briefed changes are prompt inputs, not verified outputs, so captions describe what is visible. `run_pipeline.py --dry-run` covers all seven stages on CPU (judges off); full-mode generation was not re-run.

With more time: more references per style; a multi-reference IP-Adapter to stop `references[0]` dominating; a scale sweep for the T-shirt and sweater; and a second, independent judge.
