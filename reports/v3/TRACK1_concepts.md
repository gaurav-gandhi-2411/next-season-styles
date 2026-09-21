# Track 1: concepts

Branch `feat/v3-improvements` (from `main` at `119d187`). Rules were committed before results: `reports/v3/PREREGISTRATION.md` in `b500577`. `main` and `reports/SUBMISSION/` are untouched. **Nothing here is adopted**: no figure under `reports/figures/` changed. The white top awaits the human visual check.

## 1a. White top: what the rule picks

Applied as written (`candidate_selection.py`): pass every automatic gate, then most briefed changes visible, then highest SmolVLM fidelity, then Florence-2 fidelity, then lowest seed.

- On the submitted 8 seeds, exactly two pass every gate: seeds 47 and 48. Both show both changes (Gate 3 "YY"). SmolVLM fidelity is 0.85 for seed 47 and 0.567 for seed 48. **The rule picks seed 47.**
- With the pool at 24 seeds, four pass (47, 48, 51, 62). **The pick is still 47.**
- The submitted image (seed 42) fails the global integrity floor (closest reference 0.733 vs 0.779) and Gate 3.
- Disclosure: the 8-seed scores were read before the rule was committed, so this is a check, not a blind prediction. The rule reproduces the prior pass counts exactly (sweater 1/8, white top 2/8, dress 4/8, bikini 0/8).

Images for your check: `reports/v3/figures/white_top_candidates_8seed.png` (42, 47, 48 with references) and `reports/v3/figures/white_top_passers_24seed.png` (42, 47, 48, 51, 62). My own reading, which is not the human check: all four passers have a squared neckline and gathered cuffs; none has the full balloon volume of seed 42; seed 47's neckline is the shallowest of the four, seed 62's the most clearly square. The automatic gates cannot tell you whether "wide ribbed cuffs" or "balloon sleeves" are visible enough; that is your call.

## 1b. Gemini re-read: partial, quota-limited

Gemini's free tier is **20 requests per day per project** (`GenerateRequestsPerDayPerProject`, model `gemini-3.6-flash`). The quota had not reset when Track 1 began; a first run failed before caching anything and lost its readings, so the module now caches every reading and stops cleanly on a 429 (`v3_judge_reread.py`, tested). After the reset it produced 7 of the 18 readings needed and then stalled inside the SDK's rate-limit backoff for 15 minutes, and I stopped it. Cached: sweater (3 readings), dress (3), white top seed 42 (1). Resume with `uv run python -m nss.generate.v3_judge_reread`; it re-reads nothing already cached.

What the partial data shows (3 readings each, median per attribute; SmolVLM and Florence-2 re-scored from their stored greedy extractions, so identical by construction):

| Image | Gemini | SmolVLM | Florence-2 |
|---|---|---|---|
| Sweater (seed 45) | 0.617, **fails** its 0.667 threshold | 0.567, passes its 0.384 | 0.567, passes 0.337 |
| Dress (seed 44) | 1.000, passes | 0.850, passes | 0.567, passes |

All three judges score the sweater's "Melange" surface 0.0 (it reads as plain). So the submitted sweater passes Gate 2 under SmolVLM but would fail under Gemini at Gemini's own calibrated threshold. That is two images and six attribute items: **Cohen's kappa is not reported**, because it would be meaningless at n=6 (Gemini and SmolVLM agree on all 6 binarised items, which is agreement on very little). The four-concept, six-image comparison is not done.

## 1c. Yield at 24 seeds

Same config (scale 0.35, weight 1.5, 8 concatenated references), seeds 42-49 reused, 50-65 new. Cost: 64 new images at about 23 s each on the local RTX 3070 (measured), free.

| Style | Pooled 24 | Prior 8 (seeds 42-49) | New 16 (seeds 50-65) | Interval overlap |
|---|---|---|---|---|
| Sweater | **2/24** (8.3%, Wilson [2.3, 25.8]) | 1/8 | 1/16 | overlap |
| Dress | **10/24** (41.7%, [24.5, 61.2]) | 4/8 | 6/16 | overlap |
| White top | **4/24** (16.7%, [6.7, 35.9]) | 2/8 | 2/16 | overlap |
| Bikini top | **0/24** (0%, [0, 13.8]) | 0/8 | 0/16 | overlap |

**Pre-registered criterion: yield "changed" only if the prior-8 and new-16 Wilson intervals are disjoint. They overlap for every style, so the yield is consistent with the prior rate. More seeds did not raise it; they gave a tighter estimate of the same rate.** With more seeds the white top's pooled rate (16.7%) is below the earlier 2/8 (25%), which is what small-sample noise looks like.

**Floor.** Scoring each style's 24 images as if they were the next style (wrong references, attributes, changes) through the same gates: **0 false passes in 96** (0/24 for each style; Wilson upper bound 13.8% each). The gate stack does not pass wrong-style images.

**Reproducibility.** Re-scoring seeds 42-49 reproduced the committed gate results and fidelities 32/32 on every column.

**Which gate binds differs by style** (pass counts of 24):

| Style | Gate 1 | Gate 1b | Integrity | Gate 2 | Gate 3 |
|---|---|---|---|---|---|
| Sweater | 24 | 24 | 24 | **2** | 22 |
| Dress | 24 | 24 | **10** | 22 | 23 |
| White top | 24 | 24 | 19 | 22 | **6** |
| Bikini top | 24 | 22 | 22 | **0** | 11 |

**The bikini's zero is a judge artefact, but not the one assumed.** SmolVLM's fidelity is exactly 0.283 on all 24 seeds. It does **not** read the print as solid: it never answers "solid". It names a pattern every time (Melange 8, Floral 5, Checkered 5, Polka dot 3, other 3) and never uses the label "All over pattern". So Gate 2 gives the bikini no seed-level information at all, and the advisory Florence-2 passes 24/24. The bikini's 0/24 is therefore not evidence about generation quality. Gate 3 (changes visible) passes 11/24 for it.

**Selection from the 24-seed pool (the rule, unchanged).** Sweater: seed 45 (same as submitted). White top: seed 47. Dress: seed 43; **five dress seeds tie exactly** on the rule's three ranking criteria (changes visible, SmolVLM 0.85, Florence-2 0.567), the pre-registered tie-break (lowest seed) picks 43, and the submitted seed 44 is one of the tied five. The tie-break is arbitrary and I have not adopted a swap. Bikini: no passing candidate, so nothing is picked. In no style did the larger pool change the pick relative to the 8-seed pool.

## 1d. FLUX comparison: not started

Nothing was provisioned. Reasons to stop and ask, not to proceed:

- **Cost cannot be shown to be zero.** I cannot verify from the CLI that free credits remain, or that a new project has spot-GPU quota (new projects often have none). Rough order of magnitude, *assumed not measured*: a spot L4 or A100 for 1-3 hours is roughly $0.5-5.
- **The comparison would not be like-for-like.** The pipeline conditions on eight concatenated reference photos through SDXL's IP-Adapter. FLUX has different image-conditioning options, so "same briefs, same gates" would also change the conditioning method and confound any lift.
- **Licence and access.** FLUX.1-dev is non-commercial and gated on Hugging Face; FLUX.1-schnell is Apache-2.0 but a different model.
- **Policy.** GCP work here must run as the sole-owner identity with an explicit `--account`, and needs your confirmation first.

If you want it: state a budget cap and the model (dev or schnell), and the comparison rule goes into `PREREGISTRATION.md` before anything runs. My recommendation is to skip it: the bottleneck the data points to is the Gate 2 judge and the integrity/Gate 3 pass rates, not the generator's ceiling.

## Incidents during the run

- An orphaned generation process from my own `timeout`-wrapped probe (and a `bash` loop that survived `TaskStop`) held the GPU at 8 GB and made one image take over ten minutes; killed, after which the rate was 23 s/image. Sweater seeds 50-53 and dress seed 50 were written by the surviving loop; seeded generation does not depend on speed, and all 96 images opened without error.
- Quota loss on the first Gemini run, described above.
