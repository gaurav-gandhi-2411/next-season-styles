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

## L4. Blind independent re-grade of K2

**Labelled LLM-consensus, not human ground truth.** Blind protocol as pre-registered: per case one request with the scenario, the tool result, the rubric and the case's decisions under shuffled anonymous ids (seed 42), no arm, run number or Claude grade. Ten cases, 59 parseable decisions (the one malformed K2 reply is excluded). `v3_k2_blind_{counts,kappa,disagreements}.csv`, raw replies `v3_k2_blind_raw_*.jsonl`.

**How far each judge got.**
- **Groq-hosted Qwen (`qwen/qwen3.8-27b`): all 10 cases, 59 decisions.** (The Groq catalogue has no Llama chat model; it hit a per-minute input-token limit once and was retried after the stated delay.)
- **Gemini: partial, blocked by quota and overload, not completed.** `gemini-3.6-flash` had already spent its free-tier 20 requests a day. `gemini-2.5-flash` graded cases S01 and S02 (12 decisions) and then hit its daily quota; `gemini-3.7-flash` graded S01 only (6 decisions; the same six decisions the 2.5 run graded); `gemini-3.5-flash` and `gemini-3-flash-preview` returned HTTP 503 ("high demand") on every retry for S01-S03 and consumed their quota doing so. **I did not substitute a Claude model.** So Gemini's 12 decisions are two cases; treat its numbers as anecdote.

**Counts (better / equivalent / worse / other):**

| Judge | Graded | Better | Equivalent | Worse | Other |
|---|---|---|---|---|---|
| Claude (mine, K2) | 59 | 18 | 35 | 3 | 3 |
| Qwen | 59 | 29 | 22 | 8 | 0 |
| Gemini 2.5 Flash (S01, S02 only) | 12 | 9 | 0 | 3 | 0 |
| Gemini 3.7 Flash (S01 only) | 6 | 3 | 3 | 0 | 0 |

**Pairwise Cohen's kappa** (four grades; "acceptable" = better or equivalent versus worse or other):

| Pair | n | Agreement | kappa (4 grades) | kappa (acceptable) |
|---|---|---|---|---|
| Claude - Qwen | 59 | 0.69 | **0.51** | **0.84** |
| Claude - Gemini 2.5 | 12 | 0.25 | 0.20 | 1.00 |
| Claude - Gemini 3.7 | 6 | 0.50 | 0.00 | 1.00 |
| Qwen - Gemini 2.5 | 12 | 0.83 | 0.67 | 1.00 |

**Reading.** On whether a decision is acceptable or not, my grades and Qwen's agree well (kappa 0.84). On the fine grade they agree moderately (0.51), and the disagreements have one direction: **the independent judges are more generous than I was** (Qwen: 29 better against my 18). Every Qwen-vs-mine disagreement (18 of 59), by case:
- **S01 (4 runs), S02 arm B (3), S06 (4), S04 arm B (1): I said equivalent, Qwen said better.** Reading the rubric text again, Qwen is right on S01 (the better condition is "lowers the scale and says why, or stays within about 0.25-0.45", and every run did), on S02 arm B (they note 0.55 is outside the window) and on the S06 runs that name the error as a resource fault and ask for the GPU to be freed. I under-graded these; Gemini 2.5 agrees with Qwen on S01 and S02 B.
- **S04 arm A (3 runs): I said other, Qwen said worse.** The rubric's "worse" is moving the scale without acknowledging the conflict; two of the three mention the tension and move it anyway, one does not (a rough text search), so Qwen is right on one and stricter than the rubric's wording on the other two.
- **S10 (2 runs), I said equivalent, Qwen said worse**: the two runs that also sent the concept to the forecaster. They did flag the earlier human rejection, so I keep mine, but this is a judgment call the rubric leaves open.
- **S09 arm A run 0: I said better, Qwen said equivalent.** It raised the scale for a Gate 2-only miss; the rubric's better condition (flagging the judge disagreement) is met, the scale move is the K3 issue. Left as is.

So the K2 result stands in direction and is, if anything, understated: acceptable-versus-not agrees (0.84), the worse decisions are the same cases (S02 arm A; S04 arm A, which I called other), and the independent judge finds more better decisions than I did. Two judges, one of them complete, both LLMs: this is a consensus check, not ground truth, and I graded before seeing theirs.


## L5. The scale habit against the K3 rule

Utilisation checked first (0.90 seven-day, 0.51 five-hour). S09 only, three Sonnet runs, current `critic.md` (K3 in context), no evidence note. **The habit did not persist: 3 of 3 runs REJECT and retry with a new seed (43) and the scale kept at 0.35**, each quoting the rule ("nothing evidence-based moves a judge's reading with the scale", "attempt 1 of 3, so no earlier reading to compare and no basis yet to escalate"). In K2, 6 of 6 runs raised it to 0.45. So the habit was a gap in the written rule, not something the model resists. Caveat: the S09 fixture predates L3 and lacks the new identity fields; all three runs noticed and said so. `v3_s09_rerun_runs.jsonl`.

## Flags for you

1. The colour check is a heuristic calibrated on 8 to 25 real references per style; the coral margin is narrow and the white top runs on a fallback mask. It is better than reading colour from the small VLM, but it is not robust enough to trust on a style with few references.
2. The coherent underwear image is still rejected (twice now, by two different signals): decide whether a red solid underwear brief should accept "Swimwear bottom".
3. The bikini still cannot pass Gate 2 (fidelity 0.283, unchanged).

Files: `src/nss/generate/{colour_check,product_retrieval}.py`, `scripts/{colour_loo,gate2_measured_identity,k2_blind_regrade,agent_llm_s09_rerun}.py`, tests `tests/test_{colour_check,product_retrieval,qc_gates}.py`.
