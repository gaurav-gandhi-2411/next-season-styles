# Track 4: the agent layer, evaluated (H3)

Rules committed in `14a0ae7` before any of this ran. Nothing here says anything about an LLM agent: **the agent layer is role definitions (`agents/*.md`) plus a scripted driver, and no LLM makes any decision in it.** What was evaluated is the deterministic decision rule in `agents/critic.md`, the routing in `agents/orchestrator.md` (coded as `route()` in `src/nss/generate/agent_eval.py`), and the gates behind them, against human-derived labels. Every case was scored through the real MCP `score_concept` tool over stdio (CPU, allowlist enforced), except the 96 wrong-style rows, which use their stored gate columns plus a live parity sample of 6 (**6 of 6 agree with the stored values on all five gates**, `v3_agent_eval_parity.csv`).

## 4b. The critic's decisions (accept = PASS_PENDING_HUMAN)

| Set | Positives | Negatives | Accepted | Precision [Wilson 95%] | Recall [Wilson 95%] | Accept-all floor (precision = prevalence) |
|---|---|---|---|---|---|---|
| **Primary: P1 (submitted four, human-approved), hard negatives** | 4 | 9 | 2 | **2/2 = 1.00 [0.34, 1.00]** | **2/4 = 0.50 [0.15, 0.85]** | 0.308 |
| P1, all negatives (adds the 96 wrong-style) | 4 | 105 | 2 | 2/2 = 1.00 [0.34, 1.00] | 2/4 = 0.50 [0.15, 0.85] | 0.037 |
| P1+P2 (adds 6 coherent images), hard negatives | 10 | 9 | 2 | 2/2 = 1.00 [0.34, 1.00] | **2/10 = 0.20 [0.06, 0.51]** | 0.526 |
| P1+P2, all negatives | 10 | 105 | 2 | 2/2 = 1.00 [0.34, 1.00] | 2/10 = 0.20 [0.06, 0.51] | 0.087 |

Random-accept recall floor = prevalence (same as the last column). Source: `v3_agent_eval_metrics.csv`, `v3_agent_eval_cases.csv`; harness `scripts/agent_eval.py` at this branch's commit; scores cached verbatim in `v3_agent_eval_scores.jsonl`.

**Read these numbers with four cautions.**

1. **Zero false accepts among 105 negatives, but the interval on precision is [0.34, 1.00] from only 2 accepts.** Against the accept-everything floor, the interval's lower end (0.34) is only just above it for P1 with hard negatives (0.31) and below it for P1+P2 with hard negatives (0.53). Precision does not demonstrably beat "accept everything" on the hard sets; it clears the floor only when the 96 easy wrong-style negatives are included (0.04 and 0.09).
2. **P1 is not independent of the gates.** Final selection was "passes every automatic gate, then the human check" (`final_selection_figures`), so the two P1 concepts the critic accepts (sweater, dress) were chosen partly *because* they passed. The two P1 concepts chosen without a gate pass (white top: fails integrity and Gate 3; bikini: fails Gate 2) were both **rejected by the critic although a person approved them**. Only that half of P1 is unbiased evidence, and it is 0/2.
3. **P2 recall is 0/6** (5 rejected, 1 inconclusive), lower than P1. P2 is human-labelled *coherent*, not "brief met", and the M3 images' own brief is not recorded (they were scored against `CHANGES[style]`), so some P2 rejects may be Gate 3 disagreeing about a brief the image never had. Reported separately for that reason; it is not the primary.
4. **The malformed and swatch negatives are not held out, and the known false-accept mode is not in the set.** The three malformed underwear images and the two swatches are the human-labelled cases the integrity check was validated on (`integrity_global_validation.csv`, `integrity_labels.json`), so its catches on them are not independent evidence. `critic.md` itself says the automatic gates have passed visibly malformed garments; under the current stack none of my 9 hard negatives reproduces that (the older stack's Gate 1 passed all three malformed underwear images, and its API Gate 2 passed two of them, `underwear_scored.csv`). "0 false accepts" is a property of this case set, not a bound on the critic.

### Every case except the 96 wrong-style rows (all 96: REJECT, 96/96 fail the integrity floor)

| Case | Truth | Failing gates | Verdict, as `critic.md` reads literally | Verdict if a definite failure decides | Routed to |
|---|---|---|---|---|---|
| underwear seed 42 (malformed) | reject | integrity | **INCONCLUSIVE** | REJECT | escalate / concept-designer |
| underwear seed 44 (malformed) | reject | Gate 2 | INCONCLUSIVE | REJECT | same |
| underwear seed 45 (malformed) | reject | integrity | INCONCLUSIVE | REJECT | same |
| swatch (concat 0.15, s43) | reject | integrity, G2, G3 | REJECT | REJECT | concept-designer (scale) |
| swatch (style-only 1.0, s42) | reject | integrity, G2, G3 | REJECT | REJECT | concept-designer (scale) |
| exact clone: sweater, dress, top | reject | Gate 1b, G3 | REJECT | REJECT | concept-designer (scale) |
| exact clone: bikini | reject | Gate 1b, G2, G3 | REJECT | REJECT | concept-designer (scale) |
| submitted sweater | accept | none | **PASS_PENDING_HUMAN** | same | forecaster |
| submitted dress | accept | none | **PASS_PENDING_HUMAN** | same | forecaster |
| submitted white top | accept | integrity, G3 | REJECT | REJECT | concept-designer (scale) |
| submitted bikini | accept | Gate 2 | REJECT | REJECT | concept-designer (scale) |
| H3 underwear seed 43 (coherent) | accept | none (Gate 3 not run) | INCONCLUSIVE | INCONCLUSIVE | escalate |
| 3 clean n1 sweaters | accept | G2 (all three); G3 also for one | REJECT | REJECT | concept-designer (scale) |
| M3 sweater | accept | G2, G3 | REJECT | REJECT | concept-designer (scale) |
| M3 dress | accept | G3 | REJECT | REJECT | concept-designer (scale) |

**Why the underwear cases are INCONCLUSIVE:** the H3 underwear images were generated without a brief (their sidecar has no `changes`, and the style is not in `CHANGES`), so Gate 3 comes back `not_run`. `critic.md` calls a `not_run` gate malformed data and says INCONCLUSIVE, but does not say what to do when *another* gate has already failed. I read it literally (INCONCLUSIVE); the alternative (a definite failure decides, as `qc_gates.verdict_from_gates` does) is in the fourth column. The choice changes no accept, so precision and recall are identical; it changes the route (escalate vs retry).

## Routing (`v3_agent_eval_routing.csv`): 0 failures, plus a spec/driver disagreement

| Check | Result |
|---|---|
| (a) `route(verdict)` equals the pre-registered hop for every case's actual verdict (115 cases) | 0 failures |
| (b) retry cap on the three malformed underwear images, literal reading | attempt 1 is INCONCLUSIVE, so the cap is never reached: only the escalate branch is exercised |
| (b) same, definite failure decides | attempt 1 REJECT -> concept-designer, attempt 2 REJECT -> concept-designer, attempt 3 REJECT -> FAILED, no fourth call: 0 failures |
| (c) stubbed `score_concept`: raises twice / raises once then succeeds / Gate 2 `not_run` / Gate 3 null / clone control not validated | INCONCLUSIVE after exactly 2 calls / normal verdict after exactly 2 calls / INCONCLUSIVE / INCONCLUSIVE / INCONCLUSIVE: 0 failures |
| (d) allowlists from `agents/*.md`: concept-designer lists `generate_concept`, forecaster lists `forecast_concept`; critic omits `generate_concept`, `forecast_concept`, `forecast_styles` | 0 failures |

**Finding (a defect in the spec and driver, not in the router):** three implementations disagree on the same input, "a gate failed and Gate 3 was not run". `agents/critic.md` read literally: INCONCLUSIVE, escalate. `scripts/agent_demo.py::critic_replay` (null counts as failed): REJECT, retry until the cap, then FAILED. `qc_gates.verdict_from_gates`: `REJECT: failed integrity`. On the three underwear images the driver's replay says FAILED (retry cap exhausted) where the spec's reading says escalate. I did not edit `critic.md` or the driver; which reading is intended is your decision. (This sensitivity was added after seeing the INCONCLUSIVE result, and is decision-neutral for accepts.)

## 4c. Gate ablation (`v3_agent_eval_ablation.csv`)

Catches per gate (a gate "catches" a case when it is among the failing gates). **Unique catch** = removing the gate would newly let the case reach PASS_PENDING_HUMAN.

| Gate | N1 malformed (3) | N2 swatch (2) | N3 clone (4) | N4 wrong style (96) | **Unique catches, all 105 bad** | P1 (4) catches / unique false rejects | P2 (6) catches / unique false rejects |
|---|---|---|---|---|---|---|---|
| Gate 1 (within-style p90) | 0 | 0 | 0 | 0 | **0** | 0 / 0 | 0 / 0 |
| Gate 1b (nearest reference, clone control) | 0 | 0 | 4 | 0 | **0** | 0 / 0 | 0 / 0 |
| Integrity floor (global) | 2 | 2 | 0 | 96 | **0** | 1 / 0 | 0 / 0 |
| Gate 2 (SmolVLM attributes) | 1 | 2 | 1 | 95 | **0** | 1 / **1** | 4 / **2** |
| Gate 3 (briefed changes visible) | 0 (not run) | 2 | 4 | 94 | **0** | 1 / 0 | 3 / **2** |

- **No gate uniquely catches any known-bad case on this set.** Every bad case fails two or more gates (or, for the three underwear images, is also blocked by an unrunnable Gate 3), so removing any single gate changes no decision on the negatives. That is what overlap looks like; it says nothing about a case set with a single-gate failure. **I removed nothing and am not proposing to; that is your decision.**
- **Gate 1 caught 0 of the 115 cases**, including the exact clones (it passes them by design: the p90 rule is a "within the real spread" check, which is why Gate 1b exists). On this case set Gate 1 has no measured catch; whether it catches anything a real generation run produces is not tested here.
- **Gate 1b's only catches are the 4 clones**, and Gate 3 also fails all 4. Gate 3's clone catch is incidental (a photo of an unchanged product does not show the briefed changes), untested for a clone whose original already has the briefed features. Gate 1b's clone validation is what makes it fail closed.
- **The integrity floor alone stops all 96 wrong-style images**, Gates 2 and 3 stop 95 and 94 of them.
- **Gates 2 and 3 are where the cost is.** Removing Gate 2: P1 recall 2/4 -> 3/4, P1+P2 (hard) 2/10 -> 5/10, false accepts still 0. Removing Gate 3: P1+P2 (hard) 2/10 -> 4/10, false accepts still 0, and the INCONCLUSIVE cases resolve (the 3 malformed underwear images become REJECTs, the brief-less coherent underwear image becomes an accept; that image is one of the two Gate 3 "unique false rejects" in P2, blocked because Gate 3 could not run). Removing Gates 1, 1b or the integrity floor changes no count. This is measured on 9 hard negatives; it does not show the false-accept rate stays 0 without them.
- Every known-bad case is stopped by at least one automatic gate, so no case here relies on the human check alone.

## 4a. The workflow, fully live (`reports/agent_run_transcript_live.md`, `v3_agent_live_attempts.csv`)

Scope: the three autumn/winter styles (the summer style has no live forecast source, and nothing may be replayed). 336 s wall time, $0. Step labels: 16 LIVE MCP calls (1 `forecast_styles`, 3 `get_style_profile`, 5 `generate_concept`, 5 `score_concept`, 2 `forecast_concept`), 3 INPUT (the human-authored briefs), 9 LOCAL (prompt builder, decision rule, router), 3 **NOT PERFORMED** (the human visual check). The word REPLAY appears nowhere in the transcript.

| Style | Attempts (scale, seed) | Failing gates per attempt | Outcome |
|---|---|---|---|
| Beige melange sweater | 3 (0.35, 0.25, 0.15; seed 42) | G2+G3; G2; G2+G3 | **FAILED, retry cap exhausted** (1 original + 2 retries, no fourth call) |
| Red dress | 1 (0.35, seed 42) | none | PASS_PENDING_HUMAN |
| White jersey top | 1 (0.35, seed 42) | none | PASS_PENDING_HUMAN |

The human check is not performed and I did not open these images; the white-top result is an automatic-gate result only, and no white-top pick is finalised. The closed loop (`forecast_concept`) was live for the two passing concepts: the dress's top-1 retrieval is an orange blouse-dress (the intended red dress is 2nd), yet the tool labels the match `confidence: high`; the top's intended style is 4th. That is the prototype's stated caveat (it reads the archetype, not the design), visible here.

**Findings, whatever the outcome:**

1. **The live MCP generation path is not the configuration that made the deliverables.** `mcp_server.generate_concept` -> `backends.generate_concept` exposes no negative prompt and no concat mode and does not use the weighted prompt; the deliverables came from `levers.generate_variant` (concat, negative prompt, weighted prompt). These images are a weaker configuration and are **not evidence about the submitted concepts**, and none is a candidate for the submission. The live sweater's 0/3 and the dress/top 1/1 are not comparable to the recorded candidate pass rates (sweater 1 of 14, dress 6 of 14, top 2 of 18).
2. **`generate_concept` names files `seed<seed>_<i>.png` with no style or scale**, so a retry with the same seed, or a second style at the same seed, overwrites the previous image. This run's own sweater retries kept seed 42 (only the scale changed), and all three styles used seed 42. The driver archived each image under a unique name in `data/generated/agent_live/`, but the tool overwrote `data/generated/local_sdxl/seed42_00.{png,json}`, an earlier run's output. I had copied those to a scratch backup first and restored them byte-identically after (sha256 checked); no file under `data/` was lost. The defect is in the tool and I did not change it.
3. The pre-registered retry rule (integrity-only miss changes the seed, otherwise the scale) was followed; every live failure involved Gate 2, so the scale stepped down each time (0.35 -> 0.25 -> 0.15, the floor).

## Deviations and disclosures

- The M3 images named in `integrity_labels.json` under `reports/concepts/` no longer exist there; the same bytes (sha256 identical at commit `41f302c`) are in `data/generated/final_concepts_m3/attempt1|2/`, which is what was scored.
- My first ablation counted "the only failing gate" as a unique catch and reported integrity 2 and Gate 2 1 unique catches on the malformed set. That does not match the pre-registered meaning (removal newly lets the case pass): those images also have Gate 3 unrun. I corrected `ablate()` and reran; both counts are kept (`unique` and `sole_failing` in the CSV) and the corrected ones are used above.
- My first routing run flagged one check (c) failure that was an over-specified expected tuple in my harness (it also expected an empty failing-gate list, which the rule never stated); the pre-registered expectation (verdict INCONCLUSIVE, one scoring call) holds. The comparison is now verdict and call count only.
- The 96 wrong-style gate columns were known before the rule was written (stated in the prereg).
- Two positives (P2) and the underwear negatives depend on briefs that were never recorded or never existed (see above).

Tables: `v3_agent_eval_{cases,metrics,ablation,routing,parity}.csv`, `v3_agent_eval_scores.jsonl`, `v3_agent_live_attempts.csv`. Code: `src/nss/generate/agent_eval.py`, `scripts/agent_eval.py`, `scripts/agent_live.py`; tests: `tests/test_agent_eval.py`, `tests/test_agent_live.py`.
