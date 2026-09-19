# next-season-styles — Write-up

## 1. Style definition and why

The unit of forecasting is not the H&M `article_id` (a single SKU/colourway) but a `style_key`: a
composite of 5 attribute columns from `articles.csv` — `index_group_name`, `product_type_name`,
`garment_group_name`, `perceived_colour_master_name`, `graphical_appearance_name`. The reasoning is
commercial: a buyer briefs a design team on "black jersey basic T-shirts," not on article
0554598001, and one archetype is realised across many near-duplicate SKUs that turn over while the
archetype persists.

Coherence is reported honestly (`reports/tables/style_key_silhouette_comparison.csv`): embedding each
article's `detail_desc` gives a weak observed silhouette of -0.453 for the 5-column key, but against
a 50-shuffle permutation null (mean -0.700, sd 0.0017) that is z=145.5 — far more structure than
chance despite the poor absolute number. A 4-column key (dropping `index_group_name`) keeps more
lifetime units (90.33% vs 85.07%) but is less coherent (z≈89.0); the 5-column key was kept.

## 2. Target definition

The target is `units_per_active_article` (intensity), not raw volume: the top-20 styles by lifetime
units and by mean intensity have zero overlap (`reports/tables/intensity_comparison.csv`) — raw
volume rewards assortment breadth, not per-SKU demand. The raw-intensity leaderboard is not
markdown-driven: 18 of its top-20 styles have a defined `price_index` averaging ≈1.043, only 1 of 18
below 0.9.

## 3. Evaluation methodology

The harness uses 20 rolling origins, a 4-week step and a 13-week horizon; the LightGBM paired
comparison runs on the 12 origins every method shares (`reports/tables/backtest_summary_v2.csv`).
Headline metric Hit@3-in-top20 (any of the top-3 picks lands in the realised top-20) is 0.722 (CI
[0.694, 0.917]) against a random floor of 0.004, beating all 4 baselines with paired-difference CIs
excluding zero.

**Track G tested the premise.** An earlier 0.421 top-3/top-20 overlap between consecutive origins
looked like persistence LightGBM under-used. G1 built a strictly causal persistence oracle (predict
origin `t` from the earliest earlier origin whose 13-week window had already closed: a derived lag
of 16 weeks). On the identical 12 origins it scores 0.222 (CI [0.083, 0.472]), far below LightGBM —
the 0.421 was window-overlap autocorrelation, not signal (`reports/tables/g1_diagnostics_summary.csv`).
Four alternative objectives were then tried with paired bootstrap CIs; none beat L2 (lambdarank 0.611,
diff -0.111, CI [-0.167, -0.056]; top-heavy L2 0.639; two-stage 0.583 with wmape blowing up to ~15;
rank-averaged ensemble 0.694, CI [-0.111, 0.083]). The model is frozen; G4 confirmed the final three
styles unchanged.

**Why Precision@3 is near-random while head-region ranking is not.** The model beats all four
baselines *and* the causal persistence oracle on head-region ranking (Hit@3-in-top20 0.722 vs
0.222; Hit@3-in-top10 0.500 vs 0.167). Exact-argmax at rank 3 is different: Precision@3 is 0.056
(CI [0.000, 0.111]) vs the oracle's 0.028, CIs overlapping. The reason is in the target: the true #3
and #4 styles differ by 0.61% of #3's value on average (median 0.56%; #3 vs #10, 4.7%). Ordering
inside a near-tie is measurement noise for any method, so the model finds the right neighbourhood,
not the exact order — which is why the deliverable is a T1/T2 triple, not a claimed exact top-3.

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

The generation stage produced a quality-control gate that rejected everything, and the finding is that the *gate* was mis-built, in five successive ways. Each was found by measuring the instrument against a control, not by tuning until concepts passed.

1. **Absolute CLIP cosine — rejected.** H&M's shared flat-lay photography makes even unrelated styles score a mean 0.799 across styles.
2. **Margin scoring.** Own-style similarity minus a control-pool similarity cancels the shared photographic style.
3. **Anchors re-derived in generated space (E1, F1).** Real-image anchors did not transfer to SDXL output, so both calibration endpoints were regenerated. The "unrelated" endpoint had to be remade at `ip_adapter_scale=0.0`: at 1.0 the image conditioning overrides the prompt, so the endpoints were indistinguishable (CLIP gap +0.003, DINOv2 +0.025; after: +0.140, +0.505; `margin_anchor_realspace_vs_genspace_gap.csv`).
4. **A real-article benchmark replaces the copy anchor (H1).** The copy anchor was an image generated at scale 1.0, SDXL's most reference-faithful rendering, so gating at 90% of it was a fidelity ceiling mislabelled as a plagiarism check. H1 passed a concept if its mean similarity to its references was at or below the *median* similarity between distinct real articles of the same style, in CLIP and DINOv2.
5. **The median was itself wrong, a leave-one-out control shows it, and p90 replaces it (J1–J2).** A median puts about half of real products on the wrong side by construction (the T-shirt failed by 0.0001). Rather than argue that, I scored each real reference article against its siblings through the identical gate code (`leave_one_out_control.csv`). **Under the median rule only 6 of 16 real articles (37.5%) pass** (T-shirt 2/6, underwear 1/4, sweater 3/6): the gate rejected 62.5% of genuine products. The threshold became the p90 of within-style pairwise similarity, so a concept must land inside the range real product pairs span. Sanity check: 16/16 real articles pass the shipped threshold and 14/16 (87.5%) a threshold recomputed without the held-out article. It is not "90% by construction", because a mean over siblings is smoother than a single pair. **Gate 1 on F5's 12 candidates moved 4/12 → 12/12; on the four H3 underwear candidates 4/4 → 4/4; on the three final selections 1/3 → 3/3.** Nothing was adjusted after seeing these; the threshold had already been corrected twice.

**What p90 does not do.** An exact copy of reference 0, scored as a candidate, passes Gate 1 for the T-shirt and underwear (`clone_positive_control.csv`): a mean over n references dilutes a copy of one. Gate 1 is a range check, not a copy detector. The nearest-reference statistic (closest single reference vs each real article's closest sibling) flags every clone but stays un-gated, because promoting it after seeing results would be a third threshold change. It flags the final T-shirt (DINOv2 0.913 vs 0.896) and sweater (DINOv2 0.918 vs 0.864, CLIP 0.963 vs 0.954), not the underwear.

**Judge checklist (H2).** `garment_group` ("Jersey Basic") has no visual referent, so only product type, colour and graphical treatment are scored (Groq threshold 0.438 → 0.513).

**Underwear defect (H3).** The style's "Solid" label is a colour tag and its references were all lace, so IP-Adapter reproduced lace. Four verified plain references fixed it; seed 43, a clean solid red brief, was selected.

**Final result, stated plainly.** Each final concept was judged three times; the fidelity is the median, with its spread (`j4_judge_repeats.csv`):

| Concept | Gate 1 (p90) | Fidelity median (range) | Threshold | Gate 2 | Nearest-reference check |
|---|---|---|---|---|---|
| T-shirt | pass | 0.900 (0.000) | 0.513 | pass | flags |
| Underwear | pass | 0.614 (0.006) | 0.513 | pass | clear |
| Sweater | pass | 0.567 (0.000) | 0.513 | pass | flags |

The sweater's Gate 2, unmeasured until now (both judges had been quota-blocked), clears by 0.053, inside the ±0.21 noise bound; the judge read its graphical treatment as "solid" against "melange" (0.00 on that attribute), so the pass rests on product type and colour. All three concepts pass both automated gates and my visual check. Two of three are flagged by the un-gated nearest-reference check, so the defensible claim is "inside the real range, faithful to the brief, coherent", not "demonstrably not derivative".

## 7. Agent architecture

An orchestrator delegates to 5 sub-agents (`reports/figures/agent_architecture.png`): `data-analyst`,
`forecaster`, `style-profiler`, `concept-designer`, `critic`. The orchestrator calls no MCP tool
directly. Two reusable skills carry the dataset-agnostic logic — `skills/style-brief/` and
`skills/concept-qc/` — with H&M-specific adapters outside each. The critic's retry loop is exercised:
the demo run (`reports/agent_run_transcript.md`) drove all 3 concepts through 1 attempt + 2 retries
each before escalating. The MCP server (`src/nss/mcp_server.py`) exposes 7 read-only tools over
stdio; only `generate_concept` does live work.

## 8. Results

The forecast selects a T1 (incumbent) and T2 (emerging) pair, both guard-passing. T1 rank 1: Black
Jersey Basic T-shirt; T2 rank 1: Red underwear bottom (growth ratio ≈3.67); T2 rank 2: Beige Melange
sweater (≈1.59) (`reports/tables/top_styles_final_three.csv`). A seasonal bonus table
(`top_styles_by_season_v2.csv`) adds a diversity-constrained top-3 per season. The hero
`reports/figures/FINAL_concepts.png` shows the best candidate per style regardless of gate outcome, captioned with what is visible in each image;
`evidence_chain.png` traces references → brief → concept → similarity vs the real benchmark →
per-attribute fidelity → verdict. SHAP (C4): the underwear's prediction is driven by `lag_1`
(≈0.486), not a Christmas feature; the sweater's by `n_active_articles_level` (≈0.254) over `lag_1`
(≈0.164).

## 9. Limitations and what I would do with more time

**Automated QC catches drift, not incoherence.** Of the four regenerated underwear candidates, three were visually malformed: a cut-out defect (seed 42), sheer mesh panels (44) and an unrecognisable folded object (45). All three passed Gate 1 under both the median and the p90 rule, and all three cleared Gate 2 (single judge calls of 0.617, 0.600 and 0.567 against a 0.513 threshold; the folded object had never been scored before this session). Similarity scoring flags images too close to their references; attribute scoring flags images that drift from the brief. Neither sees whether the garment is a garment: a malformed image is dissimilar to its references and reads as the right words. This is a conclusion about the limits of automated QC, not a caveat. A human check remains necessary, and the missing instrument is a judge of garment integrity.

**Measurement limits.** n is 4–6 references per style, so the within-style benchmarks are noisy. Judge noise is large: one T-shirt image scored 0.6375 and then 0.425 across sessions, so every fidelity number here carries a measured bound of about ±0.21 and none should be read as precise. The three repeats per final concept agree to within 0.006 (the T-shirt and sweater each returned three identical scores) because they were made minutes apart; they establish repeatability, not accuracy. Fidelity comes from one judge model (Groq's `qwen3.8-27b`, calibrated on 3 positive controls); Gemini's free tier (20 requests/day) was exhausted, and an OpenRouter fallback (upstream rate-limits, one degenerate extraction) was abandoned rather than shipped unvalidated. `applied_changes` are prompt inputs, not verified outputs (the underwear shows dark piping, not the briefed burgundy picot edge), so the hero captions describe what is visible. E5's 24 earlier candidates could not be re-scored (images no longer on disk). `run_pipeline.py`'s generation stage was not re-run in the final session (no new generation was permitted); the hero-stage wiring is covered by a regression test.

The most consequential bug elsewhere was cross-process prediction jitter (up to ~52% relative) from a missing `maintain_order="left"` on a polars join; fixed and verified bit-identical with a subprocess regression test. With more time: decide whether the nearest-reference check becomes a gate; more references per style; a multi-reference IP-Adapter to stop `references[0]` dominating; and a scale sweep for the T-shirt and sweater.
