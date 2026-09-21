# Track 6: the LLM where the rule is silent, the scale rule, Gate 2 identity constraints, Gate 1 advisory (K1-K5)

Rules committed in `f14d7a2` before K2 ran and before K4 was scored. `main` untouched; `reports/SUBMISSION/` untouched. Nothing was pushed to `main`.

## K1. Complete LLM-orchestrated transcript

**`reports/agent_run_transcript_llm.md`, complete.** Sonnet orchestrator and sub-agents, headless Claude Code, Max login, 492 s, all three styles to an outcome, against the K3/K4/K5 critic rules. The session limit was not hit (the stream does carry plan warnings: seven-day utilisation 0.88 to 0.89, so the plan is close to its weekly limit). Raw stream `reports/tables/v3_agent_llm_stream_run_sonnet.jsonl`. The earlier Opus run that the limit cut off is kept as `..._attempt4_opus_session_limit.md`.

| Style | Outcome | Critic verdicts |
|---|---|---|
| Beige melange sweater | **escalated to a person** (one retry unused) | attempt 1 and 2 (scale 0.35, seeds 42, 43): REJECT on Gate 2 only, the identical SmolVLM reading (0.283, pattern read "Solid") both times; the critic applied the new rule and escalated instead of spending the last retry |
| Red dress | **PASS_PENDING_HUMAN**, attempt 2 (seed 43), live closed-loop forecast (intended style 1st of 5, confidence low) | attempt 1: integrity (0.751 vs 0.779) and Gate 2, the colour check (read "Orange"); attempt 2: all gating gates pass |
| White top | **FAILED at the cap** | attempt 1: integrity, Gate 2 (read "Sweater"), Gate 3; attempt 2: Gate 2 (read "Blanket"), Gate 3; attempt 3 (scale 0.25): integrity, Gate 3 |

The human visual check was done by nobody. All four retry decisions the critics made follow the K3 rule: seed changes for the Gate 2-only sweater miss and for the two conflicting failure sets (dress: integrity and Gate 2; top: integrity, Gate 2 and Gate 3), then a scale step down (0.35 to 0.25) for the top's Gate 3 miss.

## K2. Judgment where the rule is silent

Ten scenarios (`evals/fixtures/silent_rule/`), 3 runs each in two arms (A: spec as at `896b10d`, silent; B: spec plus a fixed project-evidence note), Sonnet; 60 runs, **1 unparseable reply** (the model's own malformed JSON) and 8 earlier parse failures re-run after I fixed my parser (`raw_decode`, `strict=False`). I graded every run against the rubric I wrote before running; **the reference for "what a person would choose" is that rubric, my stand-in for you, so overrule any grade** (`v3_silent_rule_grades.csv`, runs `v3_silent_rule_runs.jsonl`).

| Case | Arm A (spec only) | Arm B (spec + evidence) | Reading |
|---|---|---|---|
| S01 Gate 3 miss at scale 0.35 | equivalent x3 (scale down 0.25-0.30) | equivalent x3 | The model lowers the scale on its own; the direction is not the problem here |
| S02 Gate 3 miss at scale 0.55 | **worse x3**: kept 0.55, changed only the seed ("scale is not the lever") | equivalent x3 (to 0.25, past 0.45) | Without the lever evidence the model does not see that 0.55 is itself too high |
| S03 integrity-only miss | equivalent x3 (seed, scale kept) | equivalent x3 | |
| S04 integrity and Gate 3 pull opposite ways | **other x3**: raised the scale to 0.45 (one run names the conflict and does it anyway) | equivalent x2 (seed, scale kept; 1 unparseable) | Defensible without the evidence, worse than keeping the scale |
| S05 judge reads the same on every seed, last attempt | **better x3**: FAILED + escalate to a human, cites the constant 0.283 and the Florence-2 pass | better x3 | Better than the plain rule (which says only FAILED) |
| S06 gate returns a transient error | equivalent x3 (INCONCLUSIVE, retry the scoring call, GPU freed by a person) | equivalent x3 | |
| S07 deterministic error (missing image) | equivalent x3 (retries scoring once, flags the path) | equivalent x3 | The ideal was to escalate at once; the spec says retry once |
| S08 gating judge passes, advisory fails | **better x3**: PASS_PENDING_HUMAN + names the disagreement | better x3 | |
| S09 gating judge fails, advisory passes | **better x3**: REJECT + flags the disagreement; **but raised the scale to 0.45 in all six runs** for a Gate 2-only miss | same | The scale move has no evidence (the reading is identical across seeds and scales); even the run given that fact raised it |
| S10 every gate passes but a person said the brief is not met | equivalent x3 (PASS_PENDING_HUMAN, collar named, Gate 3's earlier false pass named) | equivalent x3 | Suggested a lower scale next time but did not send it back |

**Result, on the defensible measure (corrected in M4):** of the 59 parseable decisions, 53 are acceptable (better or equivalent to the rubric's good decision) and 6 are not (S02 arm A x3: kept the scale at 0.55; S04 arm A x3: raised the scale to resolve conflicting failures). A blind re-grade by a Qwen model agrees on acceptable-versus-not at kappa 0.84 (n = 59), labelled LLM consensus (`TRACK7`, L4). The four-way grades (better, equivalent, worse, other) agree only at kappa 0.51 and their counts differ between graders, so the per-case labels above are my reading of each run, not a stable finding. **What this says.** On judgment about *process* (errors, conflicting judges, a human's earlier rejection, a judge that cannot represent the style) the model's decisions are acceptable; the two places they are not are where the right call depends on *project evidence* it does not have (the scale window), fixed once it is told (arm B). One weakness the evidence note did not fix: for a Gate 2-only failure it reaches for the scale (S09, 6 of 6), which K3 now rules out (and L5 shows it no longer does with the rule in context). Caveats: ten cases, three runs, one model.

## K3. The scale-direction rule (in `agents/critic.md`)

Window **0.25 to 0.45**, from `prompt_lever_summary.md` and the WRITEUP (briefed changes appear at 0.25-0.35, weaken at 0.45, disappear at 0.6-0.7; with the older prompt 0.15-0.25 collapsed into fabric swatches). Exactly one parameter changes per retry; if the move would leave the window, change the seed.

| Failure | Move | Why |
|---|---|---|
| Gate 3 (change absent) or Gate 1b (near-copy) | scale **down** 0.10 (0.55 goes to 0.45) | reference influence suppresses briefed changes |
| Integrity alone | **seed**; scale **up** 0.10 only if the previous attempt also failed integrity | seed-dependent near-floor miss (6 of 18 tops at 0.35) |
| Integrity with Gate 3 or 1b | **seed**, scale kept | opposite pulls |
| Gate 2 alone | **seed**, scale kept; escalate to a human if the reading repeats | nothing evidence-based moves a judge's reading with scale |

**Decisions it would have changed** (`v3_scale_rule_replay.csv`): scripted live run 1 of 2 (sweater retry 2: 0.25 to 0.15 was chosen, now a seed change); Opus LLM run 4 of 5 (sweater 0.35 to 0.55 for a Gate 2-only miss, now seed 43; sweater 0.55 to 0.35+seed 43 changed two parameters, now 0.45; dress 0.35 to 0.55 for integrity+Gate 2, now seed 43; white top 0.35 to 0.50 for integrity+Gate 3, now seed 43), and its unscored top attempt 2 to 3 (0.50, now 0.40). The `adjust` reference the J2 evaluation used (integrity-only = seed, else scale) differs for 103 of 112 RETRY cases (mostly the wrong-style set, where three gates fail so the new rule keeps the scale); J2's 335/336 is relative to the old rule and is not re-run. `route()` now names the parameter that actually changes.

## K4. Gate 2 hard identity constraints

Gate 2 = the unchanged averaged fidelity and threshold, **AND** the gating judge's product type and colour each match the style (`identity_match`, list committed before scoring). An AND on the old rule: it can only tighten. Applied offline to every stored reading (129 cases plus the 54 recorded candidates; no judge re-run). One correction before any scoring: my first `_match` treated a word shared by two families (tee, t-shirt) as a conflict; fixed and tested, no scoring had happened.

**Required outcomes, all met:** emerald-green dress FAILS (read "Green."); both coral dresses FAIL (read "Orange."); submitted sweater PASSES ("Sweater."/"Beige."); submitted dress PASSES ("Dress."/"Red.").

**White top, plainly:** now also fails Gate 2 (SmolVLM reads "Sweater.", not a Top). Verdict unchanged (already REJECT on integrity and Gate 3); it now fails three gates instead of two.
**Bikini, plainly:** unchanged, still fails Gate 2. Its product type reads "Bikini." and the entry **`bikini` (in the Bikini top family) is what makes the product check pass**; without it the check would have failed too. Colour ("Orange.") passes. But fidelity 0.283 is under 0.384 whatever the synonyms say, so **no synonym flips the bikini**; the AND makes that impossible. The scorer's trailing-period defect that gives "Bikini." a 0 in the average is untouched (fixing it would move the calibration).

**Effect:** 10 of the 129 cases change Gate 2; **2 change verdict**: the green dress (PASS_PENDING_HUMAN to REJECT, intended) and the human-coherent H3 underwear image (INCONCLUSIVE to REJECT, because the judge read "Bikini." for an underwear bottom: a judge misread now failing a good image). The other 8 were already rejected. Of the 54 recorded candidates, Gate 2 passes fall from 29 to 20 and all-gate passes from 9 to 5 (dress 6 to 2, top 2, sweater 1, bikini 0). **Cost of the constraint, checked by my own look at the four dress candidates it removed:** two are right to fail (salmon-pink at 0.15, coral at 0.35 seed 45), **two are red dresses the judge read as "Orange" (0.35 seed 47, 0.45 seed 43): false rejects**. So on the dress, the hard colour constraint caught 2 wrong-colour dresses and rejected 2 right-coloured ones because the judge misread the colour; that is the risk of hard constraints on a small judge, and it is your call whether it is worth it. Tables: `v3_gate2_identity_rescore.csv`, `..._candidates.csv`.

## K5. Gate 1 advisory

Gate 1 is still computed and reported (`role: advisory`); `critic_rule` (`ADVISORY = ("gate1",)`), `qc_gates.verdict_from_gates`, `critic.md` and `SKILL.md` updated together, with tests. **Verdicts changed across the 129 cases by K5 alone: 0** (Gate 1 failed in none). The ablation still lists Gate 1 (0 catches). The only changes in `gate2_identity_rescore.csv`'s verdict columns are K4's two.

## Flags for you

1. K4's cost (2 of the 4 removed dress candidates were right-coloured) and the human-coherent underwear image it now rejects.
2. K2's grades are mine against my rubric; S09's scale-up for a Gate 2-only miss is the recurring LLM habit K3 now forbids in writing.
3. The plan is at 0.89 seven-day utilisation: another large LLM run will hit the limit.

Files: `src/nss/generate/identity_match.py`, `scripts/{agent_llm_silent,gate2_identity_rescore,scale_rule_replay}.py`, `evals/build_silent_rule_fixtures.py`; tests `tests/test_{identity_match,agent_eval,qc_gates}.py`.
