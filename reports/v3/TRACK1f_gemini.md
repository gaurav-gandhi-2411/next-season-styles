# 1f: Gemini re-read of the four submitted concepts

Rules: `reports/v3/PREREGISTRATION.md` 1b (judges, thresholds, kappa on binarised items). This supersedes the partial table in `TRACK1_concepts.md`.

**Coverage.** Gemini's free tier is 20 requests per day per project and it ran out again at 12 of the 18 readings needed. Those 12 are exactly the **four submitted concepts, 3 readings each**, so the four-concept comparison is complete. The white-top alternates (seeds 47 and 48) were not read (6 requests short); nothing about them is claimed here. SmolVLM and Florence-2 are re-scored from their stored greedy extractions (deterministic).

## Per-judge scores (mean attribute score, each judge against its own calibrated Gate 2 threshold)

| Concept | Gemini (threshold 0.667) | SmolVLM (0.384, gating) | Florence-2 (0.337, advisory) |
|---|---|---|---|
| Sweater (seed 45) | 0.617, **fails** | 0.567, passes | 0.567, passes |
| Dress (seed 44) | 1.000, passes | 0.850, passes | 0.567, passes |
| White top (seed 42) | 0.617, **fails** | 0.567, passes | 0.283, **fails** |
| Bikini top (seed 48) | 0.667, passes (on the threshold) | 0.283, **fails** | 0.574, passes |

Per-attribute scores (`v3_judge_items_wide_four.csv`): every judge gets colour right for all four; the differences are in the product-type and pattern dimensions.

## Agreement (Cohen's kappa on binarised per-attribute calls, n = 12 items)

| Pair | Raw agreement | Kappa |
|---|---|---|
| Gemini vs SmolVLM | 0.917 | **0.800** |
| Gemini vs Florence-2 | 0.833 | 0.636 |
| Florence-2 vs SmolVLM | 0.750 | 0.471 |

n is 12, so these are rough. They say Gemini and SmolVLM make mostly the same right/wrong calls on these four images (11 of 12), and that Florence-2 is the outlier, consistent with the earlier reason for making it advisory. Kappa is agreement on calls, not correctness.

## Which attribute drove the sweater's 0.617

**The pattern attribute, and it is not a Gemini "solid" misread.** Gemini's score is `(product 1.0 + colour 0.85 + pattern 0.0) / 3 = 0.617`. The label is **Melange**. Gemini's three readings for the pattern were "colour blocking", "colorblock" and "colourblocking": it saw the concept's dark-brown collar, cuffs and hem (the briefed contrast trim), never said solid, and never said melange. SmolVLM said "Solid." and Florence-2 wrote a long caption that also does not name a melange effect.

**Is that a judge limitation or a concept defect?** The evidence favours a judge limitation, with one caveat:

- SmolVLM was run on the **17 real catalogue photos of beige melange sweaters** (this style's own screened references): it answered "Solid." 13 times, "Melange." once, and something else three times. So SmolVLM cannot name melange on real melange sweaters either. Its 0 on the concept is what it gives the real thing (`v3_melange_reference_answers.csv`).
- The earlier human check on the concept said its body is "only faintly heathered". So part of the shortfall may be a weakly realised pattern, not only a judge that cannot see one.
- I could not run the same real-photo check for Gemini (the quota was spent), so Gemini's limitation on melange is not tested; its "colour blocking" answers are evidence that it describes the contrast trim in preference to the surface texture.

Read together: the sweater's low Gemini score comes from the Melange label, which none of the three judges reliably names even on real melange sweaters. That is a documented judge limitation and not evidence of a concept defect. It does mean the sweater's Gate 2 pass under SmolVLM (0.567) and its failure under Gemini (0.617 against a stricter threshold) rest on different thresholds, not on different readings of the garment: both give the pattern 0.

## Other things the four-way table shows

- **The white top also fails Gemini's threshold** (0.617), because its product-type dimension scores 0 for all three judges. The label is H&M's catch-all "Top"; Gemini answered "blouse" in all three readings, Florence-2 "a white blouse", SmolVLM "Sweater.". A garment with a square neckline and long sleeves is reasonably called a blouse, so this is a label-vocabulary mismatch ("Top" is a bucket, like "Jersey Basic"), not evidence that the concept is the wrong garment. It is a different mechanism from the bikini's "Bikini." vs "Bikini top".
- **The bikini** passes under Gemini (exactly at threshold), passes under Florence-2, and fails only under SmolVLM, the gating judge, by the vocabulary problem described in `TRACK1e_pattern.md`.

Tables: `v3_judge_scores_four.csv`, `v3_judge_items_four.csv`, `v3_judge_items_wide_four.csv`, `v3_judge_kappa_four.csv`, `v3_gemini_readings_four.csv`, `v3_melange_reference_answers.csv`.
