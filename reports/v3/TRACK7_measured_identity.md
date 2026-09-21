# Track 7: measured colour, retrieval product type, restructured Gate 2, blind re-grade, the scale habit (L1-L5)

Rules committed in `c5799c3` before any L1-L3 scoring, the L4 judge calls and the L5 re-run. `main`, the README and `reports/SUBMISSION/` untouched. **Plan utilisation before the one small LLM run (L5): seven-day 0.90, five-hour 0.51** (three Sonnet calls; L4 used Gemini and Groq, not the Claude plan).

## L1. Measured colour

**Method (pre-registered).** Border-sampled background mask (rembg is not installed and needs a model download), dominant Lab colour by k-means (k=3), CIEDE2000 nearest-reference distance, per-style threshold = p90 of the real nearest-sibling distances (calibrated like Gate 1b). Code `src/nss/generate/colour_check.py`.

**Real-article leave-one-out (nested; threshold never sees the held-out article):** sweater 15/17 (0.882), dress 23/25 (0.920), white top 16/19 (0.842), bikini top 7/8 (0.875); **overall 61/69 = 0.884**. The stop rule (below 0.80 overall, or 0.75 on a final style) was not triggered. Caveats: all 19 white-top references hit the central-box fallback (a white garment on a near-white background gives a tiny mask), so that style's threshold is 1.06 CIEDE2000, very tight; the bikini has 8 references and a threshold of 16.4, so its colour check is loose; underwear (4 references) is 3/4 and reported apart. `v3_colour_loo.csv`, `v3_colour_reference_colours.csv`.

**Required outcomes, all met** (`v3_l_identity_cases.csv`, `..._required_extras.csv`; nearest reference distance against the style's threshold):

| Case | Required | Distance | Threshold | Result |
|---|---|---|---|---|
| emerald-green dress | FAIL | 54.18 | 4.50 | fails, by a mile |
| coral dress, M3 attempt 1 | FAIL | 5.52 | 4.50 | fails, **narrowly** |
| coral dress, M3 attempt 2 | FAIL | 5.38 | 4.50 | fails, **narrowly** |
| submitted dress (seed 44) | PASS | 1.80 | 4.50 | passes |
| submitted sweater (seed 45) | PASS | 2.75 | 3.44 | passes |
| red dress 0.35 seed 47 (K4 false reject) | PASS | 3.44 | 4.50 | passes |
| red dress 0.45 seed 43 (K4 false reject) | PASS | 1.67 | 4.50 | passes |

**Fragility, plainly:** the coral dresses fail by about 1 CIEDE2000 (20% over), and the nearest passing real-looking red dress is 3.44 (23% under); the threshold sits between them on 25 references. A different reference set could move it. The coral candidate at 0.35 seed 45 (5.13) and the salmon one at 0.15 (12.33) also fail, in the order my own look gave earlier. Live check through `score_gates`: the submitted dress passes Gate 2 (colour 1.80) and the green dress fails it (colour 54.18).

## L2. Product type from retrieval

Top-1 style of the CLIP+DINOv2 retrieval over 3,003 styles, exact catalogue string, no synonyms (`product_retrieval.py`); the 82.5% (40 real photos) is the submission's figure and is not re-measured. **All four submitted concepts pass:** sweater retrieved as Sweater, dress as Dress, **white top as Top** (SmolVLM read it as "Sweater.", which is why K4 failed it), **bikini as Bikini top** (colour also passes, 7.19 against 16.44).

- **White top, plainly:** its Gate 2 passes again (K4 failed it); the verdict stays REJECT on integrity and Gate 3. Its colour passes narrowly (0.91 against 1.06, fallback mask).
- **Bikini, plainly:** both identity checks pass, and **Gate 2 still fails on fidelity 0.283 against 0.384**, the judge's pattern and product-string problem; L1-L3 cannot flip it and do not touch it.
- **Cost seen:** retrieval reads all three malformed underwear images and the human-coherent underwear image (seed 43) as "Swimwear bottom", not "Underwear bottom" (a red solid brief that is genuinely ambiguous between the two), so that coherent image still fails the product check, as under K4.

## L3. Restructured Gate 2 (`critic_rule`, `critic.md`, `SKILL.md`, `concept_scoring`, `qc_gates` updated together)

Gate 2 = the unchanged SmolVLM fidelity AND measured colour AND retrieval product type; pattern stays averaged and Gate 3 is the only VLM yes/no question. K4's VLM-reading constraints are removed (`identity_match` is kept only for its evidence).

**All-gate passes on the 54 recorded candidates: 9 before K4, 5 under K4, 7 under L3** (dress 6, 2, 4; white top 2, 2, 2; sweater 1, 1, 1; bikini 0). The two red dresses K4 rejected come back (colour 3.44 and 1.67); the salmon (12.33) and coral (5.13) candidates stay out. (`v3_l_identity_candidates.csv`.)

**Verdicts that change over the 129 cases (`v3_l_identity_cases.csv`):** against the pre-K4 verdicts, **2**: the green dress PASS_PENDING_HUMAN to REJECT (colour distance 54.18), and the human-coherent underwear image INCONCLUSIVE to REJECT (retrieval "Swimwear bottom"; the same reason K4 rejected it, on a different signal). **Against K4, none.** Gate 2 itself differs from K4 in three cases, none changing a verdict: `clone_Top` and the submitted white top now pass Gate 2 (K4 failed them on the VLM's "V neck." and "Sweater." readings), and `H2_change_absent_Dress` now fails it (colour 5.95; the striped dress, already REJECT). 37 of the 129 measurements used the central-box fallback.

## L4. Blind independent re-grade of K2 (corrected in M4)

**Labelled LLM consensus, not human ground truth.** Blind protocol as pre-registered: per case one request with the scenario, the tool result, the rubric and the case's decisions under shuffled anonymous ids (seed 42), no arm, run number or Claude grade. Ten cases, 59 parseable decisions (the one malformed K2 reply is excluded). `v3_k2_blind_{counts,kappa,disagreements}.csv`, raw replies `v3_k2_blind_raw_*.jsonl`.

**How far each judge got.** Groq-hosted Qwen (`qwen/qwen3.8-27b`): all 10 cases, 59 decisions. Gemini: **12 of 59** (cases S01 and S02, `gemini-2.5-flash`); the other 47 are blocked by the free tier's 20-requests-a-day limit (checked again in M4: still exhausted; it resets at midnight Pacific, about 12:30 IST the next day). `gemini-3.6-flash` had spent its quota, `gemini-3.7-flash` graded S01 only, and `gemini-3.5-flash` and `gemini-3-flash-preview` returned HTTP 503 on every retry. No Claude model was substituted. `python scripts/k2_blind_regrade.py run gemini` resumes at S03 once the quota resets.

**The defensible measure: acceptable versus not** (a decision is acceptable if graded better or equivalent):

| Judge | Graded | Acceptable | Not acceptable |
|---|---|---|---|
| Claude (mine, K2) | 59 | 53 | 6 |
| Qwen | 59 | 51 | 8 |
| Gemini 2.5 Flash (S01, S02 only) | 12 | 9 | 3 |

**Claude-Qwen kappa on acceptable-versus-not is 0.84 (n = 59).** Both flag the same cases as not acceptable (S02 arm A, S04 arm A). Four-way agreement on better / equivalent / worse / other was much lower (kappa 0.51; 69% raw agreement), and **the fine-grained counts are not stable across graders** (the "better" count was 18 for me and 29 for Qwen), so they are not reported as a finding. Gemini's 12 decisions are two cases and are anecdote; pairwise kappa with a second complete independent judge waits for the Gemini quota.

**Correction.** An earlier version of this section re-read the disagreements after seeing Qwen's grades, concluded that Qwen was right on most of them and that my grades "understated" the model. That is withdrawn: revising a grader's grades toward the other grader's more favourable ones after seeing them is not independent evidence. The blind grades stand as they were.

## L5. The scale habit against the K3 rule

Utilisation checked first (0.90 seven-day, 0.51 five-hour). S09 only, three Sonnet runs, current `critic.md` (K3 in context), no evidence note. **The habit did not persist: 3 of 3 runs REJECT and retry with a new seed (43) and the scale kept at 0.35**, each quoting the rule ("nothing evidence-based moves a judge's reading with the scale", "attempt 1 of 3, so no earlier reading to compare and no basis yet to escalate"). In K2, 6 of 6 runs raised it to 0.45. So the habit was a gap in the written rule, not something the model resists. Caveat: the S09 fixture predates L3 and lacks the new identity fields; all three runs noticed and said so. `v3_s09_rerun_runs.jsonl`.

## Flags for you

1. The colour check is a heuristic calibrated on 8 to 25 real references per style; the coral margin is narrow and the white top runs on a fallback mask. It is better than reading colour from the small VLM, but it is not robust enough to trust on a style with few references.
2. The coherent underwear image is still rejected (twice now, by two different signals): decide whether a red solid underwear brief should accept "Swimwear bottom".
3. The bikini still cannot pass Gate 2 (fidelity 0.283, unchanged).

Files: `src/nss/generate/{colour_check,product_retrieval}.py`, `scripts/{colour_loo,gate2_measured_identity,k2_blind_regrade,agent_llm_s09_rerun}.py`, tests `tests/test_{colour_check,product_retrieval,qc_gates}.py`.
