---
name: concept-qc
description: Combine an already-computed margin-band score (e.g. CLIP + DINOv2 novelty/fidelity margins) with a blind multi-judge VLM attribute-fidelity panel into one QC verdict for a generated concept image, plus a generic retry-value strategy for re-generating a failing concept with an adjusted tunable parameter. Use when a generation pipeline needs an automated pass/fail gate before a generated concept image ships, with a bounded retry loop and a full, auditable retry history.
---

# Concept QC

Turns "here is a generated concept image, its already-computed margin-band score, and its target
style's attribute checklist" into "here is a pass/fail verdict, with per-judge blind attribute
scores, an inter-judge agreement statistic, and (if it failed) a reasoned next value to retry
generation at." The skill is dataset- and provider-agnostic: it never hardcodes a specific
embedding model, a specific VLM provider, or a specific product taxonomy's column names. Any
generation pipeline with (a) a margin/similarity-band check already built and (b) at least one
blind VLM judge can use it by writing a small adapter that supplies both as plain data/callables.

## When to use this skill

- A generation pipeline (text-to-image, image-to-image, or any generator with a tunable
  conditioning-strength parameter) needs an automated accept/reject gate before a generated
  concept ships, and that gate must combine an embedding-space novelty/fidelity check with an
  attribute-level correctness check -- neither signal alone is sufficient (a concept can be
  embedding-similar to its style's references while completely misrepresenting the garment/product
  itself, e.g. a texture close-up instead of a full product shot -- see Worked Example 2).
- You need the attribute-fidelity score to be a genuine BLIND measurement (the judge is never told
  the correct answer, only which attribute dimensions to report on), not a judge grading its own
  leaked answer key.
- A failing concept should be retried with an adjusted generation parameter, up to a hard cap, with
  the FULL retry history (not just the final attempt) preserved as an auditable record -- including
  the honest possibility that a concept never passes within the cap.

## Input: an already-computed `MarginBandResult` (DIAGNOSTIC ONLY, see Gate 1 below)

This skill does not compute embeddings or margins itself -- it reads an already-computed result:

```jsonc
{
  "clip_margin": 0.0948,       // or any embedding-space "novelty margin" metric
  "clip_in_band": true,        // whether that margin falls in the derived in-band range
  "dino_margin": 0.7245,       // a second, independent embedding-space metric (optional in
  "dino_in_band": false        // principle, but this project always supplies both)
}
```

`margin_band_pass` (computed by `qc_verdict`) requires **BOTH** `clip_in_band` AND `dino_in_band`
to be `true` -- a deliberately strict, joint reading, same convention Gate 1 below reuses. **As of
task E2, `margin_band_pass` no longer gates `overall_pass`** -- it is still computed and reported on
every `QCVerdict` for comparison, but the actual gate is the two one-sided tests described in "Gate
1: the sign-safe discount formula" and "Gate 2" below. See that section for why.

## Input: a per-style `CopyCheckResult` (Gate 1 -- the actual gate, task E2)

```jsonc
{
  "clip_margin": 0.0948,
  "clip_copy_anchor_gen": 0.1359,          // this style's copy_anchor_gen CLIP mean (task E1)
  "clip_copy_anchor_threshold": 0.1223,    // copy_anchor_threshold(0.1359, discount=0.10)
  "clip_below_copy_anchor": true,          // clip_margin < clip_copy_anchor_threshold
  "dino_margin": 0.7245,
  "dino_copy_anchor_gen": 0.6811,
  "dino_copy_anchor_threshold": 0.6130,
  "dino_below_copy_anchor": false
}
```

`copy_check_pass` (computed by `copy_check_pass`) requires **BOTH** `clip_below_copy_anchor` AND
`dino_below_copy_anchor` by default -- the same strict joint-AND convention the old
`margin_band_pass` used. `copy_check_pass(result, active_metrics={"clip"})` (task F1) narrows this
to only the metric(s) named in `active_metrics`, for when calling-code evidence shows a metric is
non-discriminative -- see "Gate 1: dropping a non-discriminative metric" below.

## Input: `JudgeCaller` -- a blind attribute-extraction callable

```python
class JudgeCaller(Protocol):
    def __call__(self, image_path: Path, attribute_dimensions: Sequence[str]) -> dict[str, str]:
        ...
```

A `JudgeCaller` is handed an image and the NAMES of the attribute dimensions to report on (e.g.
`"product_type"`, `"colour_family"`) -- **never** the target/ground-truth values, and never which
style, prompt, or design brief produced the image. Scoring the judge's raw free-text extraction
against ground truth happens entirely inside this skill, AFTER the judge call returns, so the
ground truth is never present in any judge's own context. This is what makes the resulting score a
genuine blind measurement.

Implementations should raise `JudgeUnavailableError` (defined in `run_qc.py`) for any recoverable
"this judge cannot score right now" condition (missing credentials, a decommissioned/unreachable
model, an exhausted quota). `run_judge` captures ANY exception from a `JudgeCaller` (not just
`JudgeUnavailableError`) into a reported `excluded_reason` rather than crashing the whole QC gate --
one judge's failure must never block scoring with the other judge, or block the margin-band check.

## Self-scoring contamination

`is_self_scoring_contamination(judge_name, generation_backend)` is checked automatically by
`run_judge` before every call: a judge must never score an image its own model family generated
(e.g. a `"gemini"` judge scoring a `"gemini"`-backend-generated concept). The rule is a plain
family-name match (`judge_name == generation_backend`), enforced unconditionally -- not only for
provider pairings that happen to be reachable today. In this project's C7 run, every scored image
was generated by the `"local_sdxl"` backend (Gemini's own image-generation backend was quota-
blocked in the prior task, C6), so this rule never actually excluded anything here -- it is
implemented and tested regardless, for when/if a Gemini-generated image needs scoring.

## Output schema: `QCVerdict`

```jsonc
{
  "style_id": "<opaque identifier, pass-through>",
  "margin": { "clip_margin": 0.0, "clip_in_band": true, "dino_margin": 0.0, "dino_in_band": false },
  "margin_band_pass": false,        // DIAGNOSTIC ONLY (task E2) -- does NOT gate overall_pass
  "copy_check": {
    "clip_margin": 0.0948, "clip_copy_anchor_gen": 0.1359, "clip_copy_anchor_threshold": 0.1223,
    "clip_below_copy_anchor": true,
    "dino_margin": 0.7245, "dino_copy_anchor_gen": 0.6811, "dino_copy_anchor_threshold": 0.6130,
    "dino_below_copy_anchor": false
  },
  "copy_check_pass": false,         // Gate 1: BOTH clip_below_copy_anchor AND dino_below_copy_anchor
  "judges": {
    "<judge_name>": {
      "judge_name": "<judge_name>",
      "available": true,
      "raw_extraction": {"product_type": "T-shirt", "...": "..."},  // null if unavailable
      "scores": {"product_type": 1.0, "...": 0.0},                  // null if unavailable
      "mean_score": 0.8125,                                        // null if unavailable
      "excluded_reason": null       // set iff unavailable/excluded (contamination or a caller error)
    }
  },
  "consensus_mean_attribute_fidelity": 0.8125,   // mean of "mean_score" across CONTRIBUTING judges
  "n_contributing_judges": 1,
  "fidelity_pass": true,            // Gate 2: consensus_mean_attribute_fidelity >= threshold (0.75)
  "overall_pass": false,            // copy_check_pass AND fidelity_pass (task E2)
  "label": "LLM-consensus (NOT human ground truth)"
}
```

`label` is a required, always-present field -- every `QCVerdict` this skill produces carries it
verbatim, so nothing downstream can present a VLM-judge score as if it were human-annotated ground
truth.

## Blind attribute-fidelity scoring

`score_attribute_match(extracted, ground_truth)` is the pure, deterministic core: exact match
(case/whitespace-insensitive) scores `1.0`; one string fully containing the other scores `0.85`;
otherwise word-level Jaccard overlap gives partial credit (e.g. extracted `"Basic Tops"` vs. ground
truth `"Jersey Basic"` scores `1/3`); no overlap scores `0.0`. `binarize_scores` thresholds a score
dict at (default) `0.5` for inter-judge agreement analysis.

## Inter-judge agreement: `cohens_kappa`

Given two judges' binarized (0/1) calls over the same set of attribute x concept pairs,
`cohens_kappa` computes the standard chance-corrected agreement statistic. If a project has only
one available judge (as this project's C7 run did -- see the task report), kappa is **not
computable** and must be reported as such explicitly, never silently omitted or defaulted to a
number that looks like a real measurement.

## Retry-value selection: `choose_next_retry_value`

A generic helper for QC gates whose generation pipeline exposes ONE scalar "conditioning
strength"-style tunable (this project's `ip_adapter_scale`) with a fixed, already-swept range of
tested values, and a PRIMARY band-check metric expected (from prior sweep evidence, supplied by the
CALLER, never computed by this skill) to move roughly monotonically with that tunable.

Rule, in priority order:
1. **Primary metric undershoots** (below its lower bound): move to the smallest untried value
   greater than the last tried value (smallest helpful step upward).
2. **Primary metric overshoots** (above its upper bound): move to the largest untried value less
   than the last tried value (smallest helpful step downward), if one exists. If the last tried
   value is already the FLOOR of the tested range (no smaller value can exist), there is no tested
   value expected to help -- the skill still returns the smallest untried value above the floor
   anyway, to gather direct evidence rather than concluding "impossible" without trying (the
   monotonic assumption may not hold for every concept).
3. **Primary metric already in-band**: the failure is the secondary metric and/or downstream
   fidelity. If secondary bounds are supplied, pick the untried value expected (per a
   caller-supplied `secondary_favors_higher` flag) to help the secondary metric most.

See `run_qc.py`'s docstring for the full parameter list; this project's concrete instantiation
(CLIP as primary, DINOv2 as secondary, `secondary_favors_higher=True`) and the evidence behind that
choice are documented in `nss.generate.concept_qc_pipeline`'s module docstring. This retry-DIRECTION
heuristic still uses the OLD two-sided real-space band (`clip_band`/`dino_band`) as its input --
that is an orthogonal concern (which way to move `ip_adapter_scale` next) from Gate 1's PASS/FAIL
decision below, and was left unchanged by task E2.

## Gate 1: the sign-safe discount formula (task E2)

**Why the old two-sided band was replaced.** `margin_band_pass` (the pre-E2 gate) checked a
generated image's CLIP/DINOv2 margin against a band estimated from REAL H&M catalogue images (task
C2, 6-23 real reference styles per embedding space) and cross-applied to a DIFFERENT distribution:
SDXL-generated images. Task E1 directly measured what the SAME margin metric looks like on
GENERATED images at two deliberately-designed calibration endpoints per style -- `copy_anchor_gen`
(a generated image intentionally made to look like a near-duplicate of a real reference) and
`unrelated_anchor_gen` (a generated image intentionally unrelated to the style). Gating against
endpoints measured on the SAME distribution being scored is more defensible than cross-applying a
band estimated from a handful of real photographs to a generative model's own output distribution.

**Two one-sided tests instead of one two-sided band.** Gate 1 tests only the UPPER concern ("is this
too similar to what a literal copy looks like?"): `margin < copy_anchor_threshold`, on BOTH CLIP and
DINOv2. The OLD band's LOWER concern ("is this too dissimilar / not recognizably the style?") is now
Gate 2 (VLM attribute fidelity, `>= 0.75`) -- a genuinely better signal for that specific question,
because attribute fidelity directly measures "did the model ignore the reference and produce
something off-style," rather than inferring it indirectly from a second cosine-similarity threshold.
A second margin-based lower bound would be REDUNDANT with a directly-validated, calibrated signal
that already answers the same question more legibly (see Worked Example 2, where the per-attribute
breakdown names the exact defect a bare similarity number cannot). This is also why `unrelated_anchor_gen`
is loaded by `nss.generate.concept_qc_pipeline` but not wired into `copy_check`'s THRESHOLD formula
-- Gate 1 only needs the copy-side endpoint for the threshold itself; the unrelated-side endpoint's
job is now done by Gate 2 for the "too dissimilar" question, AND (task F1, see below) as the
evidence input for whether a given metric even carries a usable copy-vs-unrelated signal at all.

**The sign-safe discount formula.** Gate 1's threshold is `copy_anchor_threshold(copy_anchor_gen,
discount=0.10) = copy_anchor_gen - discount * abs(copy_anchor_gen)`, NOT the naive
`copy_anchor_gen * (1 - discount)`. The two formulas agree whenever `copy_anchor_gen` is positive
(this project's T-shirt and Underwear bottom styles): `0.1359 * 0.90 == 0.1359 - 0.10 * 0.1359 ==
0.1223`. They DISAGREE for this project's real Sweater-style CLIP anchor, which is **negative**
(`-0.0472`, task E1) -- and disagree in exactly the direction that matters:

- Naive `copy_anchor_gen * 0.90` = `-0.0472 * 0.90` = **`-0.0425`** -- a HIGHER, LESS-negative
  value than the anchor itself. Multiplying a negative number by a fraction less than 1 moves it
  TOWARD zero, not away from it. A threshold of `-0.0425` would make Gate 1 *easier* to pass than an
  un-discounted `margin < -0.0472` check -- the opposite of what a "10% stricter" discount is
  supposed to do, and backwards relative to the positive-anchor case.
- Sign-safe `copy_anchor_gen - 0.10 * abs(copy_anchor_gen)` = `-0.0472 - 0.10 * 0.0472` =
  **`-0.0519`** -- a LOWER, MORE-negative value, strictly further from zero in the SAME direction
  the anchor itself already points. This is consistently stricter than the raw anchor, exactly like
  the positive-anchor case, regardless of sign.

The general, sign-independent invariant this formula guarantees: `copy_anchor_threshold(x, discount)
< x` for any nonzero `x` and any `discount > 0` -- verified for both a positive and a negative anchor
in `tests/test_concept_qc_run_qc.py`'s `test_copy_anchor_threshold_always_stricter_than_anchor_regardless_of_sign`.
`discount = 0.10` was chosen to match the original two-sided band's own implied 10% margin (the task
brief's own starting point of `copy_anchor * 0.90`), generalized to stay correct under a sign flip.

**A second, independent finding this task surfaced, reported as an observation, not "fixed":** the
Sweater style's CLIP anchors are themselves close to zero and NOISE-DOMINATED -- `copy_anchor_gen`
(-0.0472) is actually LOWER than `unrelated_anchor_gen` (-0.0411), the reverse of what both anchors'
names would suggest (a literal-copy calibration image scoring as LESS similar, by this metric, than
a deliberately-unrelated one). DINOv2 for the same style shows the expected ordering (copy 0.0735 >
unrelated 0.0530). Gate 1 is applied exactly as designed regardless -- the formula's job is to be
correct given whatever the anchor is, not to second-guess a noisy but honestly-measured input -- but
this ordering flip is worth flagging as a standing calibration-quality caveat specific to the
Sweater/CLIP combination, not something this task's gate-logic change resolves.

## Gate 1: dropping a non-discriminative metric (task F1)

**The defect this fixes.** E1's `unrelated_anchor_gen` (the "genuinely different garment"
calibration endpoint) was generated at the SAME `ip_adapter_scale=1.0` as `copy_anchor_gen` (the
"genuinely a copy" endpoint) -- full-strength IP-Adapter image conditioning, regardless of what the
text prompt asked for. At that scale, the reference IMAGE dominates generation so strongly that
even a deliberately-different-garment text prompt barely moved the embedding: E1's own measured
pooled copy-vs-unrelated gap was CLIP `0.0029`, DINOv2 `0.0251` -- both near zero, meaning
"genuinely a copy" and "genuinely unrelated" scored almost identically. A metric with that little
separation between its two calibration endpoints carries essentially no signal for Gate 1's actual
job (telling a copy apart from something unrelated).

**The fix (task F1).** `nss.generate.derive_unrelated_anchor_fix` regenerates ONLY
`unrelated_anchor_gen`, at `ip_adapter_scale=0.0` (pure text-to-image, zero IP-Adapter image
conditioning) -- the correct construction for "what does this unrelated-garment text prompt
produce with zero influence from the reference image." `copy_anchor_gen` is UNCHANGED (it is a
correct construction as-is; only the contaminated anchor is regenerated).

**Measured result (real numbers, `reports/tables/margin_anchors_generated_space.csv` +
`margin_anchor_realspace_vs_genspace_gap.csv`, `ALL_STYLES_POOLED` rows, pooled across the 3
final-three styles):**

| Metric | copy_anchor_gen mean | unrelated_anchor_gen mean (E1, contaminated) | unrelated_anchor_gen mean (F1, corrected) | Gap (E1, contaminated) | Gap (F1, corrected) |
|--------|----------------------|-----------------------------------------------|---------------------------------------------|--------------------------|------------------------|
| CLIP   | 0.0960               | 0.0931                                         | -0.0438                                      | +0.0029                  | **+0.1398**            |
| DINOv2 | 0.5150               | 0.4899                                         | 0.0097                                       | +0.0251                  | **+0.5053**            |

Both corrected gaps clear `nss.generate.derive_unrelated_anchor_fix.NON_DISCRIMINATIVE_GAP_THRESHOLD`
(`0.05`, chosen before this run, order-of-magnitude above both of E1's contaminated gaps) by a wide
margin -- **neither metric is dropped**; `GATE1_ACTIVE_METRICS` (`nss.generate.concept_qc_pipeline`)
stays `{"clip", "dinov2"}`, the original both-metrics default. Had a metric's corrected gap
remained under the bar, `copy_check_pass(result, active_metrics=...)` (task F1's generic mechanism,
see above) is how a caller drops it from the gate -- reported here as the mechanism that WOULD have
fired, not as something this run actually needed.

## Gate 2: per-judge calibrated threshold (task F2, supersedes the flat 0.75)

**Why the flat 0.75 was replaced.** Through task E2, `fidelity_pass` was
`consensus_mean_attribute_fidelity >= 0.75` -- one flat, judge-agnostic threshold applied to the
MEAN of whichever judges were available. `0.75` was picked as a round number, without reference to
what either judge actually scores on a genuine positive control (a real image checked, blind,
against its OWN true attributes) -- i.e. without checking whether `0.75` was even ACHIEVABLE for a
given judge. `reports/tables/vlm_calibration_results.csv` (this project's real calibration run)
shows it was not: Gemini's own positive-control mean is `0.8333` (3 real target-style images,
each checked blind against its own true attributes -- scores `1.0`, `0.5`, `1.0`), but Groq's is
only `0.5841` (`0.675`, `0.4397`, `0.6375`) -- BELOW the old flat `0.75`. A judge whose own
calibration ceiling sits below the pass bar can never pass Gate 2, even on a genuinely correct,
faithful concept -- that is a threshold-calibration bug, not evidence the judge (or the concepts it
scores) is unfaithful.

**The fix: `threshold = 0.75 x that judge's OWN positive-control mean`**
(`run_qc.judge_fidelity_threshold`, wired into `qc_verdict` via `per_judge_fidelity_thresholds`;
calling-code computation in `nss.generate.concept_qc_pipeline.compute_fidelity_thresholds`). Each
judge is compared only against its OWN achievable ceiling, scaled by the SAME `0.75` fraction the
old flat threshold used (unchanged methodology, correctly re-scoped per judge) -- not a looser or
stricter fraction chosen to hit a target pass rate.

**Measured resulting values, and which direction each moved (task F2, exact numbers, not
estimates -- from `reports/tables/vlm_calibration_results.csv`):**

| Judge  | Positive-control mean | Old flat threshold | New threshold (`0.75 x mean`) | Direction |
|--------|------------------------|---------------------|-------------------------------|-----------|
| gemini | 0.8333                 | 0.75                | 0.6250                        | DOWN (looser) |
| groq   | 0.5841                 | 0.75                | 0.4381                        | DOWN (looser), and now actually ACHIEVABLE |

**Both moved down -- this is the correctness fix, not a loosening for its own sake.** For Gemini,
the old flat `0.75` was already below its own ceiling (`0.8333`) and thus technically attainable,
but still an arbitrary absolute number unrelated to what Gemini actually achieves on ground truth;
the corrected `0.625` is 75% of that judge's real, measured ceiling. For Groq, the old flat `0.75`
was ABOVE its own ceiling (`0.5841`) -- mathematically impossible to pass regardless of how
faithful a scored concept actually was, silently making Gate 2 an automatic fail for any run where
Groq was the only available judge. The corrected `0.4381` is the first threshold Groq's own
calibration run shows is actually reachable. See `nss.generate.concept_qc_pipeline`'s
`compute_fidelity_thresholds` for the exact computation and `run_qc.fidelity_pass_from_per_judge`
for the combination rule: every AVAILABLE judge must independently clear ITS OWN threshold (the
same strict joint-AND convention Gate 1 uses) -- `run_qc.qc_verdict`'s
`per_judge_fidelity_thresholds` parameter opts into this; passing `None` (the default) preserves
the old flat-threshold-on-the-consensus-mean behavior for backward compatibility.

## Re-score of the 9 already-logged C7 attempts (task E2 part 3)

Re-evaluating the 9 already-computed `concept_qc_results.csv` rows (3 styles x 3 attempts each --
NO new generation, embedding, or VLM call) under the new gate, using each row's already-logged
`clip_margin`/`dino_margin`/`fidelity_pass` verbatim
(`nss.generate.concept_qc_pipeline.rescore_results_csv`, written to
`reports/tables/concept_qc_rescored_under_new_gate.csv`): **0/9 pass.** Reported honestly -- the new
gate does NOT retroactively pass any of C7's 9 logged attempts. Per style:

- **T-shirt (0/3 pass):** all 3 attempts fail Gate 1 (copy-check) -- every attempt's DINOv2 margin
  meets or exceeds the raw `copy_anchor_gen` itself (`0.6811`), not just its threshold. Attempts 1-2
  additionally pass Gate 2 (fidelity `1.0`, `0.75`), so those two fail SOLELY on Gate 1.
- **Underwear bottom (0/3 pass):** attempts 0-1 PASS Gate 1 (both margins comfortably below their
  thresholds) but fail Gate 2 (fidelity `0.25 < 0.75` both times -- the judge's blind extraction
  called the garment "boxer briefs"/an "all-over print," disagreeing with ground truth "Underwear
  bottom"/"Solid"). Attempt 2 fails BOTH gates (DINOv2 margin `0.8132` exceeds its `0.7114`
  threshold; fidelity `0.0`).
- **Sweater (0/3 pass):** all 3 attempts fail BOTH gates -- Gate 1 fails on CLIP specifically for
  all 3 (margins `-0.0408`/`-0.0364`/`-0.0412`, all above the sign-safe threshold `-0.0519`) while
  DINOv2 passes cleanly every time. NOTE: for these particular 3 margins, the naive (wrong)
  `* 0.90` threshold (`-0.0425`) would have reached the SAME fail verdict -- none of the 9 logged
  margins happen to land in the narrow zone where the two formulas actually disagree (see Worked
  Example 2's discussion for exactly where that zone is, `(-0.0519, -0.0425]`, and why the
  sign-safe formula must still be used regardless, since which margins a FUTURE retry lands on is
  not predictable in advance). Gate 2 fails at a flat `0.25` for all 3 (the judge consistently
  misreads the C6-known "degenerate fabric-texture close-up" defect as "knit fabric," not
  "Sweater" -- see Worked Example 2).

This is a genuinely different failure PROFILE than the old gate's (which failed 9/9 primarily on the
real-photo-band's DINOv2 lower bound being systematically miscalibrated for generated images) -- the
new gate isolates the T-shirt/Sweater failures to a real, single, legible cause each (T-shirt: too
copy-similar on DINOv2; Sweater: a marginal CLIP copy-check value plus a genuine texture-close-up
generation defect) rather than a band failure that conflated multiple possible causes into one
opaque "out of band" verdict. It does NOT, however, resolve C7's failures "for free" -- the honest
result is that generation quality (the Sweater's texture-close-up defect, the Underwear bottom's
garment-type misidentification) is the remaining blocker, not the QC gate's calibration.

## Re-score of E5's 24 already-generated `final_concepts_v2.csv` candidates (tasks F1 + F2)

Re-evaluating all 24 already-computed `final_concepts_v2.csv` rows (3 styles x up to 8 seeds/retry
rounds each -- NO new generation, embedding, or VLM call) under BOTH corrections at once
(`nss.generate.concept_qc_pipeline.rescore_final_concepts_v2_under_f1_f2`, written to
`reports/tables/final_concepts_v2_rescored_f1_f2.csv`):

| Gate | Before (original gate) | After (F1/F2-corrected) |
|------|--------------------------|----------------------------|
| Gate 1 (copy-check) | 4/24 pass | **4/24 pass -- unchanged** |
| Gate 2 (attribute fidelity) | 0/24 pass | **6/24 pass** |
| Overall (both gates) | 0/24 pass | **2/24 pass** |

**Gate 1 is unchanged, and this is the expected, correct result, not a gap in the fix.**
`copy_check_pass` reads `clip_below_copy_anchor`/`dino_below_copy_anchor`, which are derived from
`copy_anchor_gen` (task F1 confirms this anchor was ALREADY correct and does not touch it) and
`active_metrics` (task F1 confirms BOTH metrics remain discriminative after the fix, so
`GATE1_ACTIVE_METRICS` stays `{"clip", "dinov2"}`, identical to the pre-F1 default). Neither input
to Gate 1's actual pass/fail decision changed for these 24 rows -- task F1's contribution is
CONFIRMING Gate 1 remains trustworthy (its `unrelated_anchor_gen`-derived discriminativeness check
now rests on an uncontaminated measurement), not changing any individual verdict.

**Gate 2 moved substantially (0 -> 6/24) because task F2's threshold correction changes the actual
pass/fail arithmetic, not just its calibration provenance.** Several already-logged rows have a
Groq `mean_score` that clears the corrected per-judge threshold (`0.4381`) despite being below the
old, unachievable-for-Groq flat `0.75` -- see "Gate 2" above for why that old threshold was never
attainable for Groq in the first place. Overall pass count rose from 0/24 to 2/24 accordingly (both
gates must still pass -- Gate 1's unchanged 4/24 remains the binding constraint on how high overall
pass can go).

## Gate 1 as of H1: the within-style MEDIAN benchmark (superseded by the J2 p90 rule below)

**What was wrong with the copy anchor.** `copy_anchor_gen` is an image generated at
`ip_adapter_scale=1.0` -- SDXL's most reference-faithful *rendering*, itself already a new image.
Gating at 90% of it rejects anything less than 90% as faithful as the most faithful generation
possible: a fidelity ceiling mislabelled as a plagiarism check. It cannot be calibrated away; the
construct is wrong. (The sections above document it as the historical gate; `copy_check` /
`copy_check_pass` remain in `run_qc.py` for reproducing those results.)

**The replacement.** A concept passes Gate 1 iff its mean cosine similarity to its style's
reference images is **at or below the median pairwise similarity between DISTINCT real articles of
that same style**, in **both** CLIP and DINOv2 (`within_style_novelty_pass`; joint AND). Rationale:
a concept no more similar to its references than two real, commercially released products in that
style are to each other is, by construction, as novel as an actual new product in that assortment.
The threshold is an empirical, externally-grounded benchmark measured on real catalogue images, not
a number we chose. It is computed **per style** from the actual reference images
(`nss.generate.within_style_benchmark`) -- never a pooled global figure (the pooled B3 value 0.9401
hid a per-style CLIP range of 0.936-0.954 and a per-style DINOv2 range of 0.776-0.908).

Reported alongside (not gated): a leave-one-out-mean and a nearest-neighbour variant (concept's
closest reference vs. each real article's closest sibling -- the strictest plagiarism reading).

**Limits, stated.** (1) n is small (4-8 references -> 6-28 pairs), so a median carries sampling
noise: a 0.0001 DINOv2 miss is a coin-flip, not a finding. (2) The concept's mean includes the very
reference it was IP-Adapter-conditioned on, so it is structurally closer to its references than a
real article is to its siblings -- the gate is conservative on that axis. (3) Real-vs-real pairs
share catalogue photography, generated-vs-real pairs do not, biasing concept CLIP similarity low; a
CLIP pass alone is weak evidence. (4) **Both gates are necessary, not sufficient**: a malformed
or off-brief image is *dissimilar* to its references and so passes Gate 1 trivially, and the VLM
judge reads attribute words, not garment integrity (H3: the malformed cut-out and sheer-mesh
underwear candidates, seeds 42 and 44, scored 0.617 and 0.600 against a 0.513 threshold; the
0.68 once cited for the "folded object" was the F5 lace original's score, and the H3 folded
object, seed 45, was never scored by any judge before task J3). Visual inspection of every
candidate remains part of the procedure.

## CURRENT Gate 1: range-based within-style rule, p90 (task J2, supersedes the H1 median)

**What was wrong with the median.** H1's threshold was the *median* pairwise similarity between
distinct real articles of a style. A median sits at the middle of the observed distribution, so
about half of genuinely new real products fail it by construction -- a coin flip, not a copy
detector. This was measured, not argued: the **leave-one-out control** scores each real reference
article against the OTHER references of its own style through the identical code path candidates
use (`concept_similarity` + `within_style_novelty_pass`;
`nss.generate.leave_one_out_control`, `reports/tables/leave_one_out_control.csv`). Under the median
rule only **6 of 16** real articles pass jointly (T-shirt 2/6, underwear 1/4, sweater 3/6; CLIP
10/16, DINOv2 7/16) -- the gate rejected 62.5% of real products.

**The replacement.** The threshold is the **p90 of within-style pairwise similarity**, per style,
per embedding space (`pair_p90` in `within_style_benchmark.csv`; the max is reported alongside).
The defensible claim is that a concept is as novel as a genuinely new product in that assortment
when it lands *inside the observed distribution* of real product pairs, not below its midpoint.
Same statistic (mean cosine to the references), same joint-AND over CLIP and DINOv2, same function.

**Sanity check of the implementation.** Real-article pass rate under p90: **16/16** against the
threshold candidates face (computed over all references), **14/16 (87.5%)** against a threshold
recomputed from the other references only (T-shirt 5/6, underwear 4/4, sweater 5/6) -- the
held-out figure is the honest "genuinely new product" reading, and it sits near the intended 90%.
The full-set figure is 100% rather than ~90% because a real article's *mean* over several siblings
is a smoother statistic than a single *pair* and lands well inside the pairwise p90; the rule is
not "90% by construction", and the write-up says so.

**What the corrected gate does NOT do -- positive control.** `clone_positive_control.csv` scores
an exact copy of reference 0 exactly as a candidate would be scored. It **passes** Gate 1 for the
T-shirt and the underwear (fails only for the sweater): a mean over n references dilutes a copy of
one of them (mean 0.88 vs p90 0.896 in DINOv2 for the T-shirt), so an intended premise of the
p90 rule -- "a near-identical clone still fails" -- is **false** for this statistic. The
nearest-neighbour statistic (max cosine to any reference, benchmarked against real articles'
nearest sibling) does flag every clone (max 1.0 vs 0.86-0.97) and is reported beside the gate
(`*_nn_pass` in `gate1_rescored_within_style.csv`) but is deliberately NOT gated: adding it after
seeing results would be a third threshold change, and is left as a documented follow-up decision.
Gate 1 is therefore a *range* check ("looks like it could be one of this assortment"), not a
plagiarism detector.

**Effect.** Gate 1 on F5's 12 candidates: median 4/12 -> p90 12/12; H3's 4 underwear: 4/4 under
both; final selections: median 1/3 -> p90 3/3
(`gate1_rescored_within_style.csv`, `underwear_scored.csv`).

## CURRENT Gate 1b: nearest-reference check, calibrated on real articles (task K2)

The clone control showed Gate 1 (mean over references) cannot detect a copy of one reference, and
a copy detector that passes an exact copy is broken -- established by the control, independent of
any candidate. Gate 1b looks at the NEAREST reference instead: a concept's max cosine to any of its
style's references must be at or below the **p90 of the real-article nearest-sibling distribution**
(each real reference's max cosine to the OTHER references, the same leave-one-out set as J1), per
style, per space, joint AND, through the same `within_style_novelty_pass`
(`nss.generate.gate1b_nearest_reference`, `reports/tables/gate1b_nearest_reference.csv`). Not a raw
cut: a raw nearest-reference threshold would repeat the median error.

Verification (declared before scoring candidates): the exact clone of reference 0 **fails** in all
three styles (max cosine 1.000 vs thresholds 0.911-0.985), so the check works and is gated. Real
articles pass their own threshold 100% (n=4-6). Stated properties: (1) with n=4-6, nearest-sibling
values come in mutual pairs, so the p90 equals the max real nearest-sibling value in every
style/space -- Gate 1b reads "no closer to a reference than the closest real pair is"; (2) a real
article's nearest sibling is over n-1 references, a concept's over n (slightly stricter for
concepts); (3) the concept is IP-Adapter-conditioned on reference 0, so some closeness is structural.

Finals: underwear PASS (0.912/0.921 vs 0.950/0.955), sweater PASS (0.963/0.918 vs 0.985/0.942),
**T-shirt FAIL on DINOv2** (0.913 vs 0.911; CLIP 0.971 vs 0.978 passes). The threshold was not
adjusted after seeing this.

## CURRENT Gate 2 checklist: visually observable attributes only (task H2)

The judge is asked for `product_type`, `colour_family` and `graphical_treatment` only.
`garment_group` was **dropped**. It is an internal merchandising-taxonomy term ("Jersey Basic",
"Under-, Nightwear") with no visual referent: shown a plain black T-shirt, Groq correctly answered
`"top"` and scored 0.0 against `"Jersey Basic"`. Measured on the persisted calibration positives
(real catalogue images of the true style, `reports/tables/vlm_calibration_results.csv`),
`garment_group` scored 0.0 in 3 of 6 judge x style controls, capping the fidelity ceiling below 1.0
independent of image quality. *Dropped rather than mapped* to a visual descriptor (e.g. "Jersey
Basic" -> "lightweight knit jersey fabric") because a mapping needs one curated, unvalidated
descriptor per taxonomy value and would itself be a new construct to calibrate; dropping is the
simplest change that removes the unmeasurable component. Cost: the sweater's "Knitwear" was
observable and is no longer scored.

Per-judge thresholds are recomputed from the same persisted calibration scores (0.75 x positive
mean, `judge_calibration_recomputed.csv`): Groq 0.438 -> 0.513, Gemini 0.625 -> 0.667. Fidelity means
before/after for every candidate are in `reports/tables/judge_rescore.csv`, derived from one
stored per-attribute score set (`data/generated/judge_cache.jsonl`, judged once with the legacy
4-attribute prompt; the 3-attribute figure is a pure recomputation). **Judge noise:** a repeat call
on the same T-shirt image with the same checklist scored 0.425 where F5 had stored 0.6375 -- single
judge calls carry roughly +/-0.2 on one image; treat per-candidate fidelity as coarse.

## Gate 2 checklist, completed: non-visual pattern labels are excluded too (task L2)

H2 dropped `garment_group` ("Jersey Basic") because it has no visual referent. The same defect
remained in `graphical_appearance_name`: the Summer concept scored 0 on "Other structure", an
internal catch-all the judge cannot name, while correctly describing the image as "solid, ribbed".
The `graphical_treatment` attribute is therefore EXCLUDED when the style's pattern label is one of
(`nss.generate.fidelity.NON_VISUAL_GRAPHICAL_VALUES`, checked against all 30 values in
`articles.csv`):

| Excluded pattern label | Why it is not a describable visual pattern |
|---|---|
| `Other structure` | internal catch-all for any texture not otherwise listed |
| `Other pattern` | internal catch-all for any pattern not otherwise listed |
| `Unknown` | no information |
| `Treatment` | a fabric finishing process, not something visible |

Every other value is kept, including vague-sounding ones (`Contrast`, `Mixed solid/pattern`,
`Neps`, `Metallic`): a judge miss on a describable pattern is a real error. In particular the
sweater's `Melange` scored 0 although melange IS visible; that is a genuine judge error, stays in
the score and is listed in the limitations.

Fidelity is always reported BOTH ways (`reports/tables/fidelity_both_figures.csv`): all three
attributes, and visual-only (the gated figure). No new threshold: the per-judge threshold is the
calibrated `0.75 x positive mean` (0.513 for Groq) and is unchanged. Caveats stated, not hidden:
that threshold was calibrated on three-attribute means, and `product_type` is the attribute that
usually scores lowest (terse catalogue words vs the judge's longer descriptions), so a two-attribute
mean is not perfectly like-for-like. Recomputed from the persisted per-attribute scores (no judge
re-call): the three AW2020 concepts are unchanged (no excluded attribute applies; T-shirt 0.900,
underwear 0.614, sweater 0.567); the Summer concept moves from 0.375 (all attributes, fail) to
0.562 (visual-only, pass by 0.049, well inside the +/-0.21 noise bound).

**Agent layer / MCP (task L1):** `agents/critic.md` and the MCP `score_concept` tool now run these
shipped gates (`nss.generate.qc_gates`: Gate 1, Gate 1b with live clone validation, optional Gate 2,
human check always required). `qc_verdict` / `copy_check` in `run_qc.py` are retained only to
reproduce the historical margin-band results.

## Gate 2 fidelity is a median of repeated calls, with a stated noise bound (tasks J3/J4)

A single judge call is not a precise measurement: the same T-shirt image scored 0.6375 and then
0.425 across sessions (~0.21). Each FINAL concept is therefore judged 3 times per judge
(`nss.generate.judge_repeat`, raw calls in `data/generated/judge_repeat_j4.jsonl`, summary in
`reports/tables/judge_repeats.csv`); the per-judge MEDIAN is compared with that judge's own
calibrated threshold, and every fidelity number is reported with the +/-0.21 cross-call bound. Three
same-session repeats agreed to <= 0.006 (the T-shirt and sweater each returned three identical scores), so they show
repeatability, not accuracy -- they do not shrink the cross-session bound. A judge with 1..2 of 3
calls (quota ran out) yields Gate 2 = inconclusive, never a pass. Final medians (Groq
`qwen3.8-27b`): T-shirt 0.900, underwear 0.614, sweater 0.567, each vs a 0.513 threshold.

## Design notes

- **`copy_check_pass` requires BOTH metrics below their thresholds, deliberately stricter than a
  ranking rule** -- same reasoning the old `margin_band_pass` used (see "Gate 1" above for what
  replaced it and why). A SELECTION step (picking the best of several candidates) can reasonably
  treat one metric as primary and rank by it alone -- but a QC GATE re-checking an already-selected
  concept is a different job: its purpose is to catch problems a ranking rule's own tie-breaking
  logic would paper over. This project's C6 selection step used CLIP-primary ranking (documented in
  `nss.generate.final_concepts`); this skill's QC gate is intentionally NOT bound by that same
  looseness.
- **A single aggregate fidelity score can hide a real defect a per-attribute breakdown surfaces.**
  See Worked Example 2: the judge's raw extraction itself names a quality defect in one field
  while other fields score highly -- the per-attribute breakdown, not just the mean, is what makes
  that visible.
- **Judge unavailability is reported, never silently defaulted.** `n_contributing_judges` and each
  judge's `excluded_reason` make it explicit whether a `0.0`-adjacent score reflects a genuine low
  measurement or the simple absence of a working judge.

## Worked example 1 -- Gate-1 (copy-check) failure AND partial attribute-fidelity mismatch (T-shirt)

Actual `nss.generate.concept_qc_pipeline` C7 run output (2026-09-19) for style
`Ladieswear || T-shirt || Jersey Basic || Black || Solid`'s original (attempt 0) C6 candidate
(`data/generated/final_concepts/ladieswear_t-shirt_jersey-basic_black_solid_seed43.png`,
`ip_adapter_scale=0.2`) -- Gemini judge only; Groq was unreachable for this run (model
`meta-llama/llama-4-scout-17b-16e-instruct` returns `model_not_found` for this project's
`GROQ_API_KEY`, confirmed by direct API test, see the C7 task report). Re-scored under the task E2
gate (task E2 part 3's honest re-score, `reports/tables/concept_qc_rescored_under_new_gate.csv`):

```json
{
  "style_id": "Ladieswear || T-shirt || Jersey Basic || Black || Solid",
  "margin": {
    "clip_margin": 0.09480527128492089,
    "clip_in_band": true,
    "dino_margin": 0.7245466072644506,
    "dino_in_band": false
  },
  "margin_band_pass": false,
  "copy_check": {
    "clip_margin": 0.09480527128492089,
    "clip_copy_anchor_gen": 0.13587470962887718,
    "clip_copy_anchor_threshold": 0.12228723866598946,
    "clip_below_copy_anchor": true,
    "dino_margin": 0.7245466072644506,
    "dino_copy_anchor_gen": 0.6811162972024509,
    "dino_copy_anchor_threshold": 0.6130046674822058,
    "dino_below_copy_anchor": false
  },
  "copy_check_pass": false,
  "judges": {
    "gemini": {
      "judge_name": "gemini",
      "available": true,
      "raw_extraction": {
        "product_type": "t-shirt",
        "colour_family": "grey",
        "graphical_treatment": "solid",
        "garment_group": "tops"
      },
      "scores": {
        "product_type": 1.0,
        "colour_family": 0.0,
        "graphical_treatment": 1.0,
        "garment_group": 0.0
      },
      "mean_score": 0.5,
      "excluded_reason": null
    },
    "groq": {
      "judge_name": "groq",
      "available": false,
      "raw_extraction": null,
      "scores": null,
      "mean_score": null,
      "excluded_reason": "GROQ_API_KEY set, but the requested vision judge model is unreachable for this account: ... model_not_found"
    }
  },
  "consensus_mean_attribute_fidelity": 0.5,
  "n_contributing_judges": 1,
  "fidelity_pass": false,
  "overall_pass": false,
  "label": "LLM-consensus (NOT human ground truth)"
}
```

Two genuine, independent failure signals here, neither fabricated: (1) under Gate 1, this image's
DINOv2 margin (`0.7245`) actually EXCEEDS the real DINOv2 `copy_anchor_gen` itself (`0.6811`), let
alone its 10%-stricter threshold (`0.6130`) -- this generated T-shirt is, by DINOv2's own measure,
MORE similar to its real references than a calibration image deliberately constructed to look like a
near-duplicate copy is; `clip_below_copy_anchor` is `true` but `copy_check_pass` requires BOTH, so
Gate 1 fails on the joint requirement; (2) the judge, shown the image BLIND, independently called the
colour "grey" (ground truth: "Black") and the garment group "tops" (ground truth: "Jersey Basic") --
and visually inspecting the actual generated image confirms the judge is right to hedge: the
rendered garment reads as a dark charcoal/slate tone, not a true black, a genuine SDXL rendering
drift a margin check alone would never surface. `fidelity_pass` is `false` (mean `0.5 < 0.75`) on top
of the Gate-1 failure -- `overall_pass` is `false` under both the old gate and the new one here, but
for a more legible, SAME-distribution-calibrated reason under the new gate (a literal excess-of-copy
finding, not an out-of-a-real-photo-band finding).

## Worked example 2 -- a single aggregate score would have hidden a real defect (Sweater)

Actual C7 run output for style `Ladieswear || Sweater || Knitwear || Beige || Melange`'s original
(attempt 0) C6 candidate
(`data/generated/final_concepts/ladieswear_sweater_knitwear_beige_melange_seed42.png`). This is ALSO
the project's real negative-anchor edge case (task E2) -- see "Gate 1: the sign-safe discount
formula" above for the full derivation of the `-0.0519` threshold used below:

```json
{
  "style_id": "Ladieswear || Sweater || Knitwear || Beige || Melange",
  "margin": {
    "clip_margin": -0.04081239700317385,
    "clip_in_band": false,
    "dino_margin": 0.005258871614932992,
    "dino_in_band": false
  },
  "margin_band_pass": false,
  "copy_check": {
    "clip_margin": -0.04081239700317385,
    "clip_copy_anchor_gen": -0.04717115908861158,
    "clip_copy_anchor_threshold": -0.051888274997472738,
    "clip_below_copy_anchor": false,
    "dino_margin": 0.005258871614932992,
    "dino_copy_anchor_gen": 0.07352664197484653,
    "dino_copy_anchor_threshold": 0.06617397777736188,
    "dino_below_copy_anchor": true
  },
  "copy_check_pass": false,
  "judges": {
    "gemini": {
      "judge_name": "gemini",
      "available": true,
      "raw_extraction": {
        "product_type": "knit fabric",
        "colour_family": "brown",
        "graphical_treatment": "solid",
        "garment_group": "knitwear"
      },
      "scores": {
        "product_type": 0.0,
        "colour_family": 0.0,
        "graphical_treatment": 0.0,
        "garment_group": 1.0
      },
      "mean_score": 0.25,
      "excluded_reason": null
    },
    "groq": {
      "judge_name": "groq",
      "available": false,
      "raw_extraction": null,
      "scores": null,
      "mean_score": null,
      "excluded_reason": "GROQ_API_KEY set, but the requested vision judge model is unreachable for this account: ... model_not_found"
    }
  },
  "consensus_mean_attribute_fidelity": 0.25,
  "n_contributing_judges": 1,
  "fidelity_pass": false,
  "overall_pass": false,
  "label": "LLM-consensus (NOT human ground truth)"
}
```

This is C6's independently documented "degenerate fabric-texture close-up" quality problem
(`nss.generate.final_concepts` module docstring / C6 task report), and the blind judge's OWN
extraction names the defect without ever being told about it: asked (blind) what `product_type` it
sees, it answers `"knit fabric"` -- not `"Sweater"` -- because the actual image (visually confirmed)
is a macro close-up of woven fabric texture with no garment silhouette at all. `garment_group`
still scores `1.0` (the fabric genuinely does look like knitwear), which is exactly why the
PER-ATTRIBUTE breakdown matters: a caller reading only `consensus_mean_attribute_fidelity = 0.25`
would correctly reject this concept, but a caller reading only a coarser pass/fail summary (or a
single "is this the right category" check) could easily miss that the failure is specifically "not
a garment shot at all," a much more actionable defect description than a bare low score.

**Gate 1 on the real negative-anchor number.** `clip_copy_anchor_gen` is `-0.0472` here; the
sign-safe threshold is `-0.0519` (10% further from zero in the SAME, negative, direction). This
image's own CLIP margin, `-0.0408`, is HIGHER (less negative) than `-0.0519`, so
`clip_below_copy_anchor` is `false` -- Gate 1 correctly reads this as "not clearly below the
copy-anchor threshold" and fails it, exactly the behavior a positive-anchor style would get from the
same formula. For THIS specific image, the naive (wrong) `copy_anchor_gen * 0.90` formula happens to
land on the same fail verdict (`-0.0472 * 0.90 = -0.0425`, and `-0.0408` is still not below
`-0.0425` either) -- but that agreement is a coincidence of this one margin value, not a property of
the naive formula: any margin in the range `(-0.0519, -0.0425]` would INCORRECTLY pass the naive
gate while correctly failing the sign-safe one, silently admitting an image more copy-like than the
anchor itself. `dino_margin` (`0.0053`) IS below its own threshold (`0.0662`), so
`dino_below_copy_anchor` is `true` -- but `copy_check_pass` requires BOTH, so this attempt still
fails Gate 1 on CLIP alone. `fidelity_pass` independently fails too (`0.25 < 0.75`) --
`overall_pass` is `false` under both gates for this attempt, but Gate 1's failure reason changed
from "out of the real-photo band on both metrics" to "not below either the DINOv2-clean-pass or the
CLIP-marginal-fail copy threshold," a more legible per-metric story.

## Adapting to a new project

Write a small adapter (calling code, not part of this skill) that:

1. Supplies `MarginBandResult` dicts from whatever embedding/similarity scoring the project already
   has (this skill never computes an embedding).
2. Implements one or more `JudgeCaller`s against whatever VLM provider(s) the project uses --
   `nss.generate.vlm_judges` is this project's concrete Gemini + Groq example, including exactly
   how each converts a provider-specific "this model is unreachable" condition into
   `JudgeUnavailableError`.
3. Supplies a ground-truth `dict[str, str]` per concept (the ATTRIBUTE dimensions and their known-
   correct values) -- `nss.generate.concept_qc_pipeline.parse_style_attributes` is this project's
   H&M-`style_id`-specific example.
4. Calls `run_judge` once per judge, `qc_verdict` to combine them with the margin result, and (on
   failure, up to a retry cap) `choose_next_retry_value` to pick the next generation parameter --
   `nss.generate.concept_qc_pipeline.run_qc_with_retries` is this project's full retry-loop example,
   including how it injects fakes for `judge_panel_fn`/`margin_fn`/`generate_fn` to stay fully unit
   testable without any real API/GPU call.

`skills/concept-qc/run_qc.py`'s `qc_verdict`, `run_judge`, `score_attributes`, `cohens_kappa`, and
`choose_next_retry_value` are the entry points calling code needs.

## Update (M-session): the token budget can silently delete the design change

`prompt_budget.fit_prompt_to_token_budget` drops novelty clauses first. A long descriptive
clause (~100 CLIP tokens) never fits, so it is dropped and the prompt becomes attribute-only:
every generated concept then looks like the reference, whatever the brief says. Keep brief
fields concise and assert that no clause was dropped (`nss.generate.final_three_briefs`).
Even with every clause intact, IP-Adapter conditioned on `references[0]` at scale 0.45 kept
the reference structure in 24 of 24 images: passing Gate 1/1b/2 does not show a design change,
so record a per-concept human judgment of whether the briefed change is visible. The red
underwear examples above are historical: the final selection excludes intimates by an
editorial rule (see WRITEUP section 8).

## Update (N-session): Gate 3, the integrity floor and local judges

**Gate 3 -- design-change verification (first-class gate).** Gates 1/1b test "not a copy" and Gate 2
tests "still the style"; nothing tested "implements the brief". For each briefed change ask every
judge one binary question ("Does this garment have <change>?"); a judge passes iff a strict majority
of changes are present; the panel passes iff every judge does. `nss.generate.gate3`. Validate the
judge first: the local SmolVLM-500M scored accuracy 0.74, recall 0.95, specificity 0.58 on 46
human-labelled questions (`evals/fixtures/gate3_sweater_labels.json`), i.e. it says yes too easily;
Gate 3 supports the human check, it does not replace it.

**Integrity.** A generic "is this a coherent garment?" question to a small VLM caught 0 of the 3
known-malformed images (`evals/fixtures/integrity_labels.json`): a check that does not fail known-bad
cases does not work. The operative check is a reference-based FLOOR: the closest real reference
(DINOv2) must be at least as close as the 10th percentile of real nearest-sibling similarity
(`gate3.integrity_floor`): 3/3 malformed and 5/5 known-bad caught, 8/9 known-good passed. It cannot
be met by a design that departs from a style whose real articles are near-identical (the white top:
floor 0.922); report that as a mechanism, do not loosen it.

**Local judges.** Free-tier API judges are quota-limited and can disappear (Gemini 401, Groq daily
token limit). `nss.generate.local_vlm` runs SmolVLM-500M and Florence-2-base locally; calibrate them
exactly as the API judges (positive minus negative mean >= 0.3; threshold 0.75 x positive mean):
SmolVLM gap 0.338, Florence-2 gap 0.373. Greedy decoding is deterministic, so repeated readings are
identical (spread 0): one reading per judge. Also check the framing screen itself: the small judge
answered "yes" to "is this one complete garment?" for 100% of candidates, so a fabric close-up got
into the reference base until a border-variance check was added.

**The prompt is part of the gate's surface.** The token budget and an attribute-first prompt (texture
words, attribute-only second-encoder prompt) hid every briefed change; a plain sentence naming the
garment and its changes on both text encoders fixed it (`prompt_lever_summary.md`).

## Update (P-session): two calibration decisions

**Integrity: the GLOBAL floor gates, the per-style floor is advisory (R2 correction).** Both are
reported for every concept; neither choice changes a verdict. The per-style floor is a similarity
gate in near-identical styles (white top: p10 0.922) and false-alarmed on 1 of 9 known-good images.
Garment coherence is a global property, so one floor calibrated over all real photos on disk (154
photos, 8 styles, p10 0.779) gates: it passes 9/9 known-good, catches 4/5 known-bad and 2/3
malformed images (`integrity_global_validation.csv`). **The one case the global floor misses is
seed 44 (sheer mesh, closest reference 0.811); the per-style floor catches it.** This is a
correction of mechanism and framing, not an outcome change: the white top (0.733) is below both
floors. Earlier text here said the global floor was rejected for missing that image; that
reasoning is superseded, because the per-style floor's failure mode (unmeetable for any design
change in a near-identical style) is the worse one for a generation gate. The human check remains
mandatory either way. The pool is small (8 styles): a larger one is untested.

**Florence-2 is an ADVISORY judge.** Its agreement with the API judges on binarised attribute calls
(kappa 0.36-0.48) is too low to carry a verdict; SmolVLM's is 0.68 with Groq. With Gemini's key
invalid and Groq's daily budget spent, SmolVLM is the gating Gate 2 judge and Florence-2 is reported,
never gating (`concept_scoring.GATING_JUDGES` / `ADVISORY_JUDGES`). This was decided on measured agreement,
not to pass concepts; it changes one verdict (white top Gate 2: fail -> pass) and no overall verdict.
Restore Florence-2 as a gate only if its kappa against a live API judge rises above ~0.6.
