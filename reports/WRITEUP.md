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

## 6. Generation: a four-stage instrument validation

The generation stage produced a quality-control gate that rejected everything, and the finding is
that the *gate* was mis-built, in four successive ways. Each was found by measuring the instrument
against a calibration endpoint, not by tuning until concepts passed.

1. **Absolute CLIP cosine — rejected.** A band from catalogue-vs-catalogue pairs does not transfer:
   H&M's shared flat-lay photography makes even unrelated styles score a mean 0.799 across styles.
2. **Margin scoring.** Own-style similarity minus a control-pool similarity cancels the shared
   photographic style.
3. **Anchors re-derived in generated space.** Real-image anchors did not transfer to SDXL output,
   so both calibration endpoints were regenerated (E1).
4. **The "unrelated" anchor re-derived at `ip_adapter_scale=0.0` (F1).** At 1.0 the image
   conditioning overrides the text prompt, so the null was a near-copy by construction: the two
   endpoints were indistinguishable (CLIP gap +0.003, DINOv2 +0.025). At 0.0 they separate (+0.140,
   +0.505; `margin_anchor_realspace_vs_genspace_gap.csv`).

**H1: the copy anchor was itself the wrong construct.** It is an image generated at scale 1.0 — SDXL's
most reference-faithful rendering, already a new image — so gating at 90% of it rejects anything less
than 90% as faithful as the most faithful generation possible: a fidelity ceiling mislabelled as a
plagiarism check. It is replaced by an external benchmark: a concept passes if its mean similarity to
its references is at or below the **median similarity between distinct real articles of the same
style**, in both CLIP and DINOv2 (per style; `within_style_benchmark.csv`). A concept no more similar
to its references than two real products in that assortment are to each other is as novel as a real
new product. Medians (CLIP / DINOv2): T-shirt 0.954 / 0.851, sweater 0.936 / 0.776 (6 references,
15 pairs each), underwear 0.843 / 0.905 on the H3 reference set (4 references, 6 pairs).

**H2: the fidelity checklist.** `garment_group` ("Jersey Basic") is a merchandising term with no
visual referent — the judge answered "top" and scored 0.0 — so it was dropped; only product type,
colour and graphical treatment are scored. Thresholds were recomputed from the persisted calibration
scores (Groq 0.438 → 0.513, Gemini 0.625 → 0.667). Judged on the same stored per-attribute scores,
8 of 11 judged candidates cleared the old threshold and 10 of 11 clear the new one (mean fidelity 0.504 → 0.672)
(`h2_judge_rescore.csv`).

**H3: the underwear defect.** The style's "Solid" label is a colour tag, and its references were all
lace, so IP-Adapter reproduced lace regardless of the negative prompt. Of the style's 123 articles,
7 have a pattern-free description; visual inspection cut that to 4 (two "microfibre" articles are
plainly lace; one pack contains lace briefs), fetched and used as references. All four regenerated
candidates are lace-free. But visual inspection also rejected 3 of them (a malformed cut-out, sheer
mesh panels, an unrecognisable folded object); seed 43, a clean solid red brief, was selected.

**Final result, stated plainly.** Gate 1 on F5's 12 original candidates moved from 0/12 to 4/12
(all underwear — the lace-drift images, which pass because Gate 1 measures similarity, not pattern).
On the final selections: **underwear** passes Gate 1 (CLIP 0.763 ≤ 0.843, DINOv2 0.890 ≤ 0.905)
and Gate 2 (Groq fidelity 0.650 against a 0.513 threshold; one judge only, Gemini being quota-blocked) — **1 of 3 styles passes both gates plus visual inspection**. **T-shirt** fails Gate 1 on all four seeds — DINOv2 0.851–0.884 against 0.851 (seed 45
misses by under 0.001; CLIP passes throughout); its selected seed has Gate 2 fidelity 0.900.
**Sweater** fails Gate 1 on all four seeds — DINOv2 0.808–0.842 against a 0.776 median (one seed also
fails CLIP); its Gate 2 is unmeasured (Groq's daily token cap and Gemini's quota). The mechanism
is consistent with, but not proven by, IP-Adapter conditioning at 0.45 importing the conditioning
reference's silhouette, which DINOv2 is sensitive to; the sweater's real siblings are diverse
(cardigan, off-shoulder, v-neck), so its benchmark is low. Lowering the scale trades this against
attribute loss (black rendering grey), the F-phase finding.

**Limits.** n is 4–8 references per style, so the medians are noisy (a 0.0001 miss is a coin-flip).
The gates are necessary, not sufficient: a malformed image is dissimilar to its references and passes
Gate 1, and the judge scored the folded-object candidate above threshold — visual inspection carried
that load. Repeat calls on one image varied by ~0.2 in fidelity (T-shirt seed 42: 0.64, then 0.43).
E5's 24 earlier candidates could not be re-scored (images no longer on disk). `applied_changes` are
prompt inputs, not verified outputs: the generated underwear shows dark piping, not the briefed
burgundy picot edge.

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
`reports/figures/FINAL_concepts.png` shows the best candidate per style regardless of gate outcome;
`evidence_chain.png` traces references → brief → concept → similarity vs the real benchmark →
per-attribute fidelity → verdict. SHAP (C4): the underwear's prediction is driven by `lag_1`
(≈0.486), not a Christmas feature; the sweater's by `n_active_articles_level` (≈0.254) over `lag_1`
(≈0.164).

## 9. Limitations and what I would do with more time

The most consequential bug was cross-process prediction jitter (up to ~52% relative): a missing
`maintain_order="left"` on a polars join made row order unstable, and LightGBM's histogram sums are
order-sensitive; fixed and verified bit-identical with a subprocess regression test. Track G's
retraining is a reported limit, not a suppressed one: the feature set and 20-origin span look close
to exhausted. Next steps: more references per style to firm up the within-style medians; a
multi-reference IP-Adapter to stop `references[0]` dominating; a scale sweep for the T-shirt and
sweater against the H1 gate; a judge that checks garment integrity; and re-running the sweater's
Gate 2 once judge quota allows.
