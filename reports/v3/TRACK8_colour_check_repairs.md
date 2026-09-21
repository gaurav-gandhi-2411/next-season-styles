# Track 8: shrunk colour thresholds, rembg masks, the borderline band, the K2 statement (M1-M4)

Rules committed in `3b63aa9` before any M1-M3 scoring; the band width was measured and committed with the calibration (`e02114e`) before any concept was scored. `main`, the README and `reports/SUBMISSION/` untouched.

## M2. rembg mask

**Install.** rembg 2.0.69, onnxruntime 1.30.0, pooch, pymatting, flatbuffers into the project venv with `pillow` held at the project's pin (10.4; unpinned, the install would have upgraded it to 12.3); the `u2net` model is **175.997 MB** (checked with a HEAD request first; `data/rembg_models/`, not committed).

- **Fallbacks:** 0 of the 73 reference articles now fall back to the central box (the border mask: 19 of 19 white-top references and 1 sweater). Of the 129 stored cases, 2 fall back (37 before); of the 54 recorded candidates, 0 (14 before).
- **White top's threshold: 1.06 to 1.12 (raw).** It barely moved, so **the 1.06 was not a fallback artifact**: the 19 white references are genuinely near-identical, and the tight threshold is real. That premise of M2 was only partly right.
- **Did other thresholds move materially?** By my pre-registered rule (more than 25% or more than 2 CIEDE2000 units): **the bikini's raw threshold, 16.44 to 13.82 (-2.6, -16%)**. Sweater 3.44 to 3.20, dress 4.50 to 4.30, underwear 23.72 to 23.72 did not.

## M1. Shrinkage (K = 17, the median reference count 17, 25, 19, 8, 4; global median of the raw thresholds 4.298)

| Style | n | w = n/(n+17) | L1 threshold (border mask) | rembg raw | **Shrunk** | Real-article LOO: L1 / rembg raw / **shrunk** |
|---|---|---|---|---|---|---|
| Sweater | 17 | 0.50 | 3.44 | 3.20 | **3.75** | 0.882 / 0.882 / 0.882 |
| Dress | 25 | 0.60 | 4.50 | 4.30 | **4.30** | 0.920 / 0.880 / 0.880 |
| White top | 19 | 0.53 | 1.06 | 1.12 | **2.62** | 0.842 / 0.842 / **1.000** |
| Bikini top | 8 | 0.32 | 16.44 | 13.82 | **7.35** | 0.875 / 0.750 / **0.500** |
| Underwear (reported apart) | 4 | 0.19 | 23.72 | 23.72 | **8.00** | 0.750 / 0.750 / 0.750 |
| **Four final styles** | 69 | | | | | **0.884 / 0.855 / 0.870** |

Nested leave-one-out as pre-registered (held-out article removed, style p90 and the global median recomputed). **Plainly:** shrinkage helps the white top (0.84 to 1.00) and hurts the bikini (0.75 to 0.50: the shrunk threshold rejects half of the real bikini tops). The global median is built from three solid-colour styles and cannot describe a patterned garment; shrinkage is the wrong prior for the bikini, and I did not tune it. **Required outcomes, all met** (binary; `v3_m_identity_cases.csv`, `..._required_extras.csv`):

| Case | Required | Distance | Shrunk threshold | Band |
|---|---|---|---|---|
| emerald-green dress | FAIL | 54.19 | 4.30 | fail |
| coral dress, M3 attempt 1 | FAIL | 6.01 (was 5.52) | 4.30 | fail (fail region starts at 4.70) |
| coral dress, M3 attempt 2 | FAIL | 5.77 (was 5.38) | 4.30 | fail |
| submitted dress | PASS | 1.67 | 4.30 | pass |
| submitted sweater | PASS | 2.75 | 3.75 | pass (pass region ends at 3.35) |
| red dress 0.35 seed 47 (K4 false reject) | PASS | 3.43 | 4.30 | pass (pass region ends at 3.90; margin 0.47) |
| red dress 0.45 seed 43 (K4 false reject) | PASS | 1.64 | 4.30 | pass |

The coral margin is now 1.5 to 1.7 (was about 1). Also: the submitted white top passes colour (1.02 against 2.62) and the submitted bikini top passes (2.34 against 7.35), no fallback masks.

## M3. Borderline band from measured mask noise

**Band half-width 0.403 CIEDE2000.** Measured, not chosen: 552 jittered crops (8 per real reference of the four final styles, 90% windows at seeded random offsets) through the full pipeline; the deviation of each crop's dominant colour from the full image's has median 0.045, p90 0.221, **p95 0.403**, max 50.3 (`v3_colour_mask_noise.csv`). Per style, p95: sweater 0.19, dress 0.20, white top 0.23, **bikini 28.9**. The band is below one just-noticeable difference, so it does not swallow the coral dresses. **Caveat that matters:** for the patterned bikini the dominant colour flips between crops (p95 28.9), so the pooled 0.403 is not representative of that style; a patterned garment needs a different colour statistic, not a band.

**Verdict:** pass at or below `T - 0.403`, fail at or above `T + 0.403`, ESCALATE between (Gate 2 unmeasured, INCONCLUSIVE unless something else already failed). **Cases in the band:** none of the required ones. Of the 129 stored cases, 7 fall in it (four wrong-style images scored against the sweater style, the coherent clean sweater `single-0.35-nat`, the M3 sweater, the "brief absent" sweater), and of the 54 recorded candidates 6 (three bikini seeds 42, 43, 47 at 7.04 to 7.69 against 7.35; sweater 0.25 seeds 42 and 43 and 0.35 seed 46 at 3.52 to 4.07 against 3.75). **All 13 already fail another gate (the fidelity or the product check), so the band changes no verdict in the stored data**; it is a safeguard for cases not yet seen.

## Effect on the all-gate count and verdicts

All-gate passes on the 54 recorded candidates: **9 before K4, 5 under K4, 7 after L3, 7 now** (with or without the band). Over the 129 cases, verdicts against L3: **none change**; against the pre-K4 verdicts: the same two as before (the green dress PASS to REJECT; the human-coherent underwear image INCONCLUSIVE to REJECT, retrieval "Swimwear bottom"). The binary colour result moves in three cases against L3: two wrong-style sweater images now fail (3.1 and 3.19 to 5.05 and 4.07), and the "brief absent" sweater now passes (3.45 to 3.48 against 3.75).

## M4. The K2 statement, and the Gemini progress

The K2 conclusion is restated on the defensible measure only (`TRACK6`, `TRACK7`, `IMPROVEMENTS_SINCE_SUBMISSION.md` corrected): **acceptable-versus-not, Claude-Qwen kappa 0.84, n = 59, LLM consensus**; four-way agreement was 0.51 and the fine-grained counts are not stable across graders, so no "better" count is reported. The post-hoc re-reading of the disagreements after seeing Qwen's grades is withdrawn from every document. **Gemini: 12 of 59 decisions graded** (`gemini-2.5-flash`, cases S01 and S02); the other 47 (S03 to S10) are still blocked: the free-tier daily quota (20 requests per model per day, several models) is exhausted, and it resets at midnight Pacific, about 12:30 IST tomorrow. The runner is set to resume at S03 with the same model: `uv run --no-sync python scripts/k2_blind_regrade.py run gemini`, then `... analyse` for the pairwise kappa. No Claude model was substituted.

## Flags for you

1. Shrinkage hurts the bikini (0.75 to 0.50 real-article pass); the pre-registered rule was applied as written. A patterned style needs its own colour statistic (or a per-style exemption), which I did not add.
2. The band is measured but tiny (0.403) and empty in the stored data; its value is protective.
3. The 47 Gemini decisions cannot be graded until the quota resets.

Files: `src/nss/generate/colour_check.py`, `scripts/{colour_loo_m,colour_mask_noise,gate2_measured_identity,k2_blind_regrade}.py`, tests `tests/test_{colour_check,qc_gates}.py`; tables `v3_colour_loo_m.csv`, `v3_colour_mask_noise.csv`, `v3_colour_reference_colours_rembg.csv`, `v3_m_identity_*.csv`.
