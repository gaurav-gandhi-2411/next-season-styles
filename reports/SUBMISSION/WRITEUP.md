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
3. **Anchors re-derived in generated space (E1, F1).** Real-image anchors did not transfer to SDXL output, so both calibration endpoints were regenerated; the "unrelated" one had to be remade at `ip_adapter_scale=0.0`, since at 1.0 image conditioning overrides the prompt.
4. **A real-article benchmark replaces the copy anchor (H1).** The copy anchor was SDXL's most reference-faithful image, so gating at 90% of it was a fidelity ceiling mislabelled as a plagiarism check.
5. **The median was itself wrong, a leave-one-out control shows it, and p90 replaces it (J1–J2).** Each real reference article was scored against its siblings through the identical gate code (`leave_one_out_control.csv`). **Under the median rule only 8 of 18 real articles (44.4%) pass** (T-shirt 2/6, sweater 3/6, dress 3/6): the gate rejected 55.6% of genuine products. The threshold became the p90 of within-style pairwise similarity: 18/18 real articles pass it, and 15/18 (83.3%) a threshold recomputed without the held-out article. 

**What p90 does not do, and Gate 1b.** An exact copy of reference 0 passes Gate 1 for the T-shirt and dress (`clone_positive_control.csv`): a mean over n references dilutes a copy of one, so Gate 1 is a range check, not a copy detector. **Gate 1b** therefore checks the nearest reference: a concept's closest single reference must be no closer than the p90 of the *real* nearest-sibling similarity (`gate1b_nearest_reference.csv`). The clone must fail (it does, in all three styles), and the threshold was fixed before scoring any concept.

**Judge checklist (H2, L2).** Only product type, colour and pattern are scored; non-visual pattern labels (Other structure, Other pattern, Unknown, Treatment) are excluded, and fidelity is reported both ways (`fidelity_both_figures.csv`).

**Why the briefed design changes never appeared (M3).** Root cause: the 77-token CLIP budget drops novelty clauses first, and the long descriptive clause never fit, so prompts silently became attribute-only ("T-shirt, jersey basic construction, black solid."). I rewrote the brief fields concisely and asserted that no clause is dropped (`final_three_briefs.py`). Two rounds of four seeds per style, scale 0.45, style-specific negative prompts, every image inspected: **still no briefed change in any of the 24 images.** IP-Adapter conditions on `references[0]` and its structure dominated a soft text hint.

## 7. Agent architecture

An orchestrator delegates to 5 sub-agents (`reports/figures/agent_architecture.png`): `data-analyst`,
`forecaster`, `style-profiler`, `concept-designer`, `critic`. The orchestrator calls no MCP tool
directly. Two reusable skills carry the dataset-agnostic logic — `skills/style-brief/` and
`skills/concept-qc/` — with H&M-specific adapters outside each. The critic and the MCP `score_concept` tool apply the shipped gates (1, 1b, 2, plus a mandatory
human check); the demo transcript predates them. The MCP server
exposes 7 tools over stdio; `generate_concept` and `score_concept` do live work.

## 8. Results

**Selection rule: the model surfaces, a human judges.** The frozen model's top emerging style was red underwear, which a buyer would not brief as a season's design story. I added a human editorial constraint on the model's output (no retraining), fixed before inspecting results and applied in order: (1) exclude intimates, underwear and nightwear product types (swimwear and tights stay eligible); (2) no two of the final three share a perceived colour; (3) at least two emerging styles, at most one incumbent; (4) no shared (product type, colour). All guards stay. It removed exactly one style (T2 rank 1, growth 3.67; `final_three_selection_log.csv`). The final three (`top_styles_final_three.csv`): T1 rank 1 Black Jersey Basic T-shirt (predicted 33.8; SHAP `lag_1` 0.729); T2 rank 2 Beige Melange Sweater (17.8, growth 1.59; `n_active_articles_level` 0.254); T2 rank 3 Red Dress (11.0, growth 1.41; `n_active_articles_level` 0.260, then a seasonal-recovery term).

**Colour base rate (M1, `colour_base_rate.csv`): why is everything black?** Black is 37.6% of panel units. Black is 7 of the T1 top 10 (70%, 1.86× its sales share; binomial P=0.039) but 3 of the T2 top 10 (30%, 0.80×). T1 ranks by per-product intensity, so the fair base rate is the market's own intensity ranking: the realised top 10 among the 268 guard-passing styles is 80% Black. The model under-concentrates relative to the market, so this is calibrated market reflection (black basics sell hardest per product), not model bias; the colour rule is an editorial diversity choice, not a correction. Black is not a coarse bucket: 99.2% of Black units carry colour group Black, 0.8% Dark Grey.

**Final concepts, stated plainly.** Best of eight candidates per style, chosen by eye; three judge readings per judge (`j4_judge_repeats.csv`); Gate 2 needs both judges (Groq threshold 0.513, Gemini 0.667):

| Concept | Gate 1 | Gate 1b (DINOv2, nearest ref) | Gate 2: Groq / Gemini median | Gate 2 | Briefed change visible (human)? |
|---|---|---|---|---|---|
| T-shirt | pass | pass (0.900 vs 0.911) | 0.567 / 0.333 | **fail** | No: grey-bodied colour-block, unbriefed |
| Sweater | pass | pass (0.916 vs 0.942) | 0.567 / 0.617 | **fail** | No: generic beige V-neck |
| Dress | pass | pass (0.797 vs 0.904) | 0.567 / 0.667 | pass | No: coral-pink, not red; generic |

**All three fail their brief, and the dress's automatic "pass" is not a success.** Gemini's 0.667 is exactly its threshold (one of three readings was 0.333), and a pink dress in a "Red" style still clears it (colour scored 0.00, but product type and pattern score 1.00). A concept indistinguishable from a generic example of its category has failed its brief whatever the gates say. The T-shirt does look different from a plain black tee, but by an accident the brief never asked for, with the black anchor lost. The dress's red candidate (seed 44) failed Gate 1b as a near-copy of reference 0.

**Seasonal view.** The same pipeline run for Summer changes the garment, not just a table
(`seasonal_comparison.png`; tables in `top_styles_by_season_v2.csv`). The
autumn/winter forecast (21 Sep 2020) leads with black jersey basics (T-shirt: 33.8 units per product
per week). Re-run as of 1 Jun 2020 using only earlier data, the model ranks a black textured swim
bottom 1 of 2,763 (forecast 93.1; realised 50.0: the ranking held, the level was overshot by ~86%),
and the concept is a high-waisted ribbed swim bottom. Fourier terms and `lag_52` encode time of year. The summer concept passes both similarity checks, my visual check and, after the L2 exclusion of "Other structure", Groq's Gate 2 (visible-only median 0.562 vs 0.513; 0.375 counting the label; both reported). It was not read by Gemini.

## 9. Limitations and what I would do with more time

**Automated QC catches drift, not incoherence or absent design.** In an earlier exploration (retired as a final concept by the category exclusion), three of four regenerated red-underwear candidates were malformed (a cut-out defect, sheer mesh, an unrecognisable folded object) and passed Gate 1 and Gate 2. In this session the dress passed every automatic gate with none of its requested design. Similarity scoring flags images too close to references and attribute scoring flags drift from the brief; neither sees whether the garment is a garment, or new. A human check remains necessary.

**Measurement limits.** n is 4–6 references per style, so within-style limits are noisy. One T-shirt image scored 0.6375 then 0.425 across sessions, so every fidelity number carries about ±0.21. Groq's three concepts score an identical 0.567 median, a sign of coarse quantised scoring. The two judges disagree (T-shirt 0.567 vs 0.333), and Gemini's free-tier quota reset was needed to complete them. Briefed changes are prompt inputs, not verified outputs, so captions describe what is visible. `run_pipeline.py --dry-run` covers all seven stages on CPU (judges off); full-mode generation was not re-run.

With more time: replace `references[0]`-only conditioning with a multi-reference IP-Adapter or inpainting/img2img edits so requested changes survive; a scale sweep per style; a garment-integrity judge; more references per style; a third judge.
