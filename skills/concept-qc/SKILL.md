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
`dino_below_copy_anchor` -- the same strict joint-AND convention the old `margin_band_pass` used.

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
is loaded by `nss.generate.concept_qc_pipeline` but not wired into `copy_check` -- Gate 1 only needs
the copy-side endpoint; the unrelated-side endpoint's job is now done by Gate 2.

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

## Gate 2: attribute fidelity (unchanged by task E2)

`fidelity_pass` is the same `consensus_mean_attribute_fidelity >= 0.75` check this skill has always
used (see "Blind attribute-fidelity scoring" below) -- task E2 did not touch Gate 2's threshold or
scoring logic, only Gate 1's replacement and `overall_pass`'s new definition
(`copy_check_pass AND fidelity_pass`, replacing `margin_band_pass AND fidelity_pass`).

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
