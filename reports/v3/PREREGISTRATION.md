# v3 improvements: pre-registered rules

Branch `feat/v3-improvements`, cut from `main` at `119d187`. `main` and `reports/SUBMISSION/` are not touched.

**Rule for the whole pass.** Before any experiment that produces a comparison runs, its decision rule is committed here. The commit that adds a section predates every result that section governs (check `git log -- reports/v3/PREREGISTRATION.md` against the result commits). A rule is not edited after results are seen; if a rule turns out to be badly specified, that is reported as a defect in the next section, not fixed silently. Every comparison carries a floor and a paired block-bootstrap CI or, for proportions, a Wilson interval. Measured versus assumed is stated explicitly.

---

## Track 1: concepts

### 1a. White-top selection (rule is the repo's existing one, restated as code)

The rule is implemented in `src/nss/generate/candidate_selection.py` and unit-tested. It restates what `final_selection_figures` already documented ("passes every automatic gate -> most briefed changes visible -> highest fidelity, then confirmed by a human"); it does not change it.

- **Pass:** every automatic gate passes: Gate 1, Gate 1b, the global integrity floor, Gate 2 (SmolVLM gates; Florence-2 is advisory) and Gate 3. A missing or null gate is a fail. The per-style integrity floor is advisory.
- **Rank among passers:** (1) number of briefed changes Gate 3 judged visible, (2) highest SmolVLM fidelity, (3) highest Florence-2 fidelity (tie-break), (4) lowest seed.
- **Human visual check:** a veto applied afterwards by a person looking at the image. It is never a rank input and never inferred from a score.
- **No passer:** the rule returns nothing; it never promotes a failure.

*Disclosure:* the 8-seed scores for the white top existed and had been read before this rule was committed, so 1a is not a blind prediction. It is a check that the rule, applied as written, reproduces the prior pass counts (sweater 1/8, white top 2/8, dress 4/8, bikini 0/8; it does) and reports what it picks. Nothing is adopted: the user does the human check.

### 1b. Second-opinion re-read with Gemini

- **Images:** the four submitted concepts, plus the white-top rule pick and runner-up from 1a.
- **Judges:** Gemini (3 readings per image, median per attribute dimension), SmolVLM (gating, one greedy reading), Florence-2 (advisory, one reading).
- **Scores:** each judge's mean attribute score per image, and each judge's Gate 2 pass at its own calibrated threshold (unchanged, `judge_repeat.judge_thresholds`).
- **Agreement:** pairwise Cohen's kappa on binarised per-(image, attribute) items (score >= 0.5), with n, computed as in `judge_panel_kappa`. Kappa says whether two judges make the same calls, not whether either is right; with a handful of images it is a rough figure and will be reported with n.
- **No verdict changes** are adopted from this. If the Gemini reading contradicts a gate, that is reported, not acted on.

### 1c. Yield at 24 seeds

- **Config unchanged:** IP-Adapter scale 0.35, compel weight 1.5, multi-reference concatenation of the 8 best screened references, same prompts, negatives and references as the submitted run. Seeds 42-49 already exist and are reused; seeds 50-65 are new (24 in total per style, four styles).
- **Pass:** exactly the 1a definition.
- **Reported per style:** passes/24 pooled (the headline yield estimate), and the prior 8 seeds versus the new 16 seeds, each with a Wilson 95% interval.
- **"Yield changed" criterion:** claimed only if the prior-8 and new-16 Wilson intervals are disjoint. Otherwise the result is "consistent with the prior rate", never "improved".
- **Floor:** cross-style mismatch. Each style's 24 images are scored as if they belonged to the next style in registry order (wrong references, wrong attributes, wrong briefed changes) through the same gate stack. The false-pass rate is reported next to the yield. Expected near zero; any pass is a defect of the gate stack and is reported as one.
- **Selection:** the 1a rule applied to each style's pool of 24. The pick is a candidate only. The human check is the user's. No figure under `reports/figures/` and nothing in `reports/SUBMISSION/` is changed.

### 1d. FLUX comparison (optional)

Not started. Requires a GCP GPU, and I cannot verify from the CLI that free credits cover it. The cost estimate and a request for confirmation go to the user first; nothing is provisioned before that. If approved, its comparison rule will be committed in this file before it runs.
