# 1e: the "All over pattern" label: rule built, control failed, mapping NOT adopted

Rule and controls were committed before any control ran (`8fbe6dd`, `reports/v3/PREREGISTRATION.md` 1e). Result: **the mapping is not applied.** Production scoring (`concept_scoring.judge_rows`) is unchanged from `main`; the rule module `src/nss/generate/pattern_label.py` and its 18 tests remain, unused.

## The mandatory negative control could not be run

The rule needed 6 real catalogue photos of SOLID-colour bikini tops (373 exist in the catalogue; none is on disk). The Kaggle API returned **HTTP 429 (Too Many Requests) on every attempt** for the six pre-registered articles, across roughly two hours of retries with 90 s spacing (`v3_pattern_control fetch` and a slower loop). This is a rate limit I could not get past, not a result. So the specified control has no outcome.

## A supplementary control found a false pass

Because the specified control was blocked, I ran the same rule, on the same judge, on 16 real catalogue photos of SOLID garments that are on disk (the screened references of the white top and the red dress; not bikinis), asking whether SmolVLM's answer would satisfy "All over pattern". **All 16 should fail.**

- 15 of 16 answered "Solid." and correctly failed.
- **1 of 16 (a real solid red dress) answered "Melange." and wrongly satisfied the label.** False-pass rate 6.3% (Wilson 95% [1.1%, 28.3%]).

That is exactly the risk noted in the pre-registration when `melange` was included. The rule as written says any solid garment that passes means the mapping is broken. The case is a dress, not a bikini, but it is the same judge and the same rule, so I treated it as the failure it is and **removed the hook** (fail closed). Table: `reports/tables/v3_pattern_control_supplementary.csv`.

## What the rule would have produced (provisional, not adopted)

- **No side effects, as required:** re-scoring the 96 candidates from their stored extractions changed **0 of 72** rows for the sweater, dress and white top (any gate or fidelity). The rule is inactive for every label except "All over pattern". `v3_pattern_rescore_summary.csv`.
- **Bikini, mapped:** Gate 2 passes 23 of 24 (unmapped: 0 of 24); all gates pass 8 of 24 (unmapped: 0 of 24). **These numbers should not be used.** Of the 8 all-gate passers, **6 rest on the answer "Melange."**, the answer that just false-passed a solid dress (the others are "Checked." and "Dots."). Answers over all 24 seeds: Melange 8, Floral 5, Checkered 5, Polka dot 3, Checked 1, Dots 1, "ORIGINAL." 1.

## The second defect, still open

The bikini's product-type dimension also scores 0 for a different reason: SmolVLM answers "Bikini." and the label is "Bikini top", and the scorer keeps the trailing period, so neither string contains the other. This is out of the 1e scope and unchanged. Even with the pattern label satisfied the fidelity would be (0 + 0.85 + 1.0) / 3 = 0.617, which clears the 0.384 threshold, but the product-type score is understating a correct answer.

## Decision needed (a rule change, so it is yours to make)

**Proposal, not run:** drop `melange` from the pattern list. SmolVLM uses "Melange" for plain heathered or flat fabric, which is what it did on the solid dress, so it is not evidence of a visible print. Everything else in the rule stays. Required before adopting: the pre-registered solid-bikini control (6 photos, all must fail), which needs Kaggle to stop rate-limiting (or another source of real solid bikini-top photos).

Illustrative only, computed after the fact and so not pre-registered: without `melange` the bikini would have Gate 2 passes on 15 of 24 seeds and all-gate passes on 2 of 24 ("Checked.", "Dots."). That is still a large change from 0/24, but it rests on 2 seeds and one judge's vocabulary, so treat the bikini's yield as still unmeasured.
