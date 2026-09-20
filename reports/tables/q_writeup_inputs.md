# Q write-up inputs (paste into WRITEUP.md; P owns that file)

Provenance: branch `feat/q-neighbourhood-retrieval`. Q1 numbers come from
`reports/tables/q1_decision.csv`, `q1_neighbourhood_paired_diff.csv`,
`q1_neighbourhood_per_origin.csv`, `q1_neighbourhood_shap.csv` (produced by
`python -m nss.models.q1_neighbourhood`, commit ceb6305). Q2 numbers come from
`reports/tables/q2_retrieval_validation.csv` and `q2_concept_forecast_retrieval.csv`
(`python -m nss.generate.concept_forecast_validation` / `concept_forecast_final`, commits 9598eb4 and
c8ef588, plus 54835e9 for the `n_indexed_styles` field).

## (i) Cross-style neighbourhood features: tested, rejected

Every existing feature comes from a style's own history, so we tested the one lever left untried:
whether a style's neighbours in the attribute lattice are rising. Fifteen causal features were built
(three sibling groups: same product type + garment group in other colours; same colour + appearance
in other product types; same department + garment group; each with the sibling mean intensity, the
4- and 13-week mean sibling slope, the style's intensity relative to its cohort, and the number of
active siblings). The features use only rows at or before the origin week; a test that shuffles every
post-origin row confirms they are unchanged. Both arms were retrained under the same 13-week
embargo protocol on the same 12 test origins with the frozen hyperparameters (no tuning); the
control reproduced the recorded embargoed headline (Hit@3 in the top 20 = 0.528).

Result, decided by a rule fixed in advance (adopt only if the paired improvement on Hit@3-in-top-20
has a block-bootstrap 95% CI excluding zero): Hit@3-in-top-20 rose from 0.528 to 0.611, a paired
difference of +0.083 with CI [-0.083, +0.250], which includes zero. The features are **not adopted**;
the current model stays. On the other metrics the treatment did not improve either: Hit@3-in-top-10
0.306 to 0.278 (diff -0.028, CI [-0.083, 0.000]), NDCG@10 0.872 to 0.855 (-0.017, CI [-0.032,
+0.002]), Spearman 0.712 to 0.715 (+0.004, CI [-0.003, +0.007]), WMAPE 0.170 to 0.170 (+0.0002,
CI [-0.003, +0.004]), and precision@10 fell 0.192 to 0.158 (-0.033, CI [-0.050, -0.017], the only
interval that excludes zero, on the worse side). Twelve origins give a wide interval, so this is
"no demonstrated gain", not "proven no effect".

The features are nonetheless used by the model: in an in-sample SHAP analysis of the treatment model
on the last embargoed fold (16 training origins, 48,003 rows), the cohort ratio for the
department + garment-group neighbourhood (`nb_igg_ratio`) ranks 5th of 41 features (mean |SHAP| 0.054), the cohort mean intensity 9th, the
13-week cohort slope 13th, and the 15 neighbourhood features together carry 15.9% of total mean
|SHAP|. That is consistent with cohort trend being informative, but it did not translate into
out-of-sample ranking accuracy here; in-sample importance is a statement about what the model uses,
not about what generalises. Paired differences of the treatment model against the four baselines
(all in `q1_neighbourhood_paired_diff.csv`; the control's own paired table is P1's
`backtest_embargo_paired_diff.csv`) include, e.g., Hit@3-in-top-20 vs EWMA persistence +0.333, CI
[+0.250, +0.556], and NDCG@10 vs parent-category mean +0.596, CI [+0.510, +0.701]. Note the seasonal-naive baseline
has no value for the first two origins (no 52-week history), so its pairings use 10 origins although
the table's `n_origins` column reads 12; and the global-mean baseline predicts a constant, so its
NDCG@10 depends on an arbitrary tie-break and differs slightly between re-runs (the LightGBM rows
are bit-identical across re-runs).

## (ii) Closed loop rebuilt as image retrieval: a prototype with a coverage limit

The first closed loop asked a small VLM to caption a concept and mapped the caption to an H&M style
key; on 40 real catalogue photos it named the exact style 12.5% of the time (SmolVLM; Florence-2
2.5%). That was a specification error: the style key includes merchandising labels (garment group,
department) that no photo shows. The loop now embeds the image (CLIP ViT-L/14 and DINOv2) and
returns the nearest style by the average cosine similarity to the mean embedding of that style's
real photos, with the top-5 styles and a confidence label (high = both views agree on the top-1 and
its similarity leads the 6th-ranked style by at least 0.02; medium = either view's top-1 is in the
other's top-5; low otherwise). The configuration and thresholds were fixed before any evaluation
number was computed.

What we can and cannot claim. The index holds 415 to 455 of the 1,980 forecast styles (548 to 592
photos: the screened reference sets, earlier exemplars, and 316 photos fetched before Kaggle began
returning HTTP 429; every retry over the following 50+ minutes failed). The 40 baseline photos are held out of the
index, and their styles are essentially not covered: on those exact 40 photos coverage is 0 of 40
(1 of 40 if the one extra on-disk photo is counted), so top-1, top-5 and top-10 accuracy are all
0/40. That number measures coverage, not retrieval quality, and it is NOT comparable to the 12.5%
baseline. Product type of the top-1 style matched for 65% of the 40 photos (SmolVLM 65%, Florence-2
62.5%) and colour for 20% (SmolVLM 70%, Florence-2 37.5%), both dragged down because the true style
is never in the candidate set.

Retrieval quality was therefore measured separately, on a supplementary leave-one-out protocol:
each of 159 on-disk catalogue photos (from styles with at least two photos) is matched against an
index whose own-style prototype excludes it, over 415 candidate styles. The averaged view gets the
exact style at rank 1 for 72.3% (115/159), in the top 5 for 86.8% (138/159) and in the top 10 for
91.8% (146/159), against chance of 0.2%, 1.2% and 2.4%; product type of the top-1 style is right
91.8% of the time and colour 79.9%. CLIP alone is comparable at the top (75.5% / 86.8% / 90.6%) and
DINOv2 alone weaker (62.9% / 80.5% / 86.8%); the pre-declared averaged configuration is reported as
the headline even though CLIP alone is 3 points better at rank 1. The confidence label is
informative: top-1 accuracy is 91.3% for `high` (104 queries), 41.3% for `medium` (46) and 11.1% for
`low` (9). Caveats: this is not the 40-photo set, near-duplicate articles inside a style make it
optimistic, and the candidate set (415 styles) is far smaller than the full catalogue, so it does not
support a like-for-like claim against 12.5%. The feature stays labelled a prototype; closing the gap
needs the full catalogue indexed (all ~1,980 forecast styles, at least one photo each) and the
40-photo comparison re-run against it.

Final concepts, scored by retrieval against the forecast table of their own origin (top-1 style,
forecast in units/product/week, rank, confidence): beige knit sweater -> Ladieswear / Sweater /
Knitwear / Beige / Melange, 17.8, rank 106 of 1,980, medium; red dress -> Ladieswear / Dress / Dresses
Ladies / Red / Solid, 11.0, rank 282 of 1,980, high; white jersey top -> Sport / Top / Jersey Fancy /
White / Solid, 15.8, rank 140 of 1,980, low (the intended Ladieswear / Top / Jersey Basic / White /
Solid is second by 0.001 similarity, rank 174); orange patterned bikini top (summer origin) ->
Ladieswear / Bikini top / Swimwear / Orange / All over pattern, 37.9, rank 118 of 3,000, high. Three
of four map to their intended style, but the index contains each intended style's own reference
photos (the ones the concept was generated from), so that is close to circular and is not evidence
of the method.

## How P switches the demo/figures to the retrieval results (nothing here overwrites P's inputs)

- `reports/tables/q2_concept_forecast_retrieval.csv` has the same columns as
  `concept_forecast_final.csv` (style_id, image_path, forecast_origin, mapped_style_key,
  maps_to_intended_style, forecast, rank, n_styles, match_level, confidence, judges, extraction) plus
  `top5`, `similarity`, `margin`, `n_index_styles`, `n_index_images`. To switch, point
  `FORECAST` in `src/nss/generate/h4_deliverables.py` (line 48) at the q2 file. `extraction` now
  holds the two views' top-1 styles (`{'clip': {'style_key': ...}, 'dino': {...}}`), not VLM
  attributes; `judges` is `clip,dino`.
- `reports/tables/q2_retrieval_validation.csv` has a NEW schema (`condition`, `row_type`, `config`,
  per-photo and `summary_*` rows); `src/nss/viz/demo_page.py` line 407 reads the OLD
  `concept_forecast_validation.csv` (type/colour/pattern/exact accuracy of the free-text judges) and
  will not parse it. Keep that panel on the old file as "before" and add the numbers above as
  "after", or read the `summary_*` rows filtered on `condition` and `config == 'avg'`.
