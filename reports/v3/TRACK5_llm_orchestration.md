# Track 5: real LLM orchestration, one critic rule, production-config generation, hard negatives (J1-J7)

Rules for J2, J6, J7 committed in `bcd1357` before those ran. `main` untouched; `reports/SUBMISSION/` untouched.

## J1. LLM orchestration through headless Claude Code

**Blocker check (J1c): none.** Headless Claude Code (`claude -p`, Claude Max login, `ANTHROPIC_API_KEY` removed from the child environment and asserted absent by a test) runs the sub-agent pattern with MCP tools attached. **(a)** `agents/*.md` had no frontmatter, so they were not sub-agent files; all six now carry `name` / `description` / `tools` frontmatter, role text unchanged. **(b)** `scripts/agent_llm.py` stages them byte-for-byte into a throwaway `.claude/agents/`, attaches two MCP servers (`nss_gpu` generation only, `nss_cpu` everything else with the GPU hidden), and runs the orchestrator (Opus) with `--tools Agent` (no Bash/Read/Write), `--permission-mode dontAsk` and an exact tool allowlist; sub-agents run Sonnet. The models make every routing, retry and escalation decision; every tool is the same deterministic MCP tool.

**Transcript: `reports/agent_run_transcript_llm.md`** (labelled the LLM-orchestrated path; raw stream `reports/tables/v3_agent_llm_stream_run.jsonl`). **It is incomplete: attempt 4 was cut by the Max session limit** (HTTP 429, 2,087 s in, list-price $6.83, not billed) while the white top's third attempt was being scored, and I did not re-run it to avoid spending the quota again. What it contains, all live: forecast read; three production-config generations; sweater REJECT x3 (Gate 2; Gate 2+3; Gate 2), i.e. the retry cap; red dress REJECT x2 then **PASS_PENDING_HUMAN** on attempt 3 (scale 0.55, seed 43) followed by the live closed-loop `forecast_concept`; white top REJECT x2 (integrity+Gate 3, then Gate 3). The human visual check is done by nobody.

**Three failed attempts came first, and each found a real defect** (their transcripts are kept: `agent_run_transcript_llm_attempt{1,2,3}_*.md`):

1. Every tool call failed with "Error executing tool". Claude Code starts an MCP server in its own working directory and ignores a `cwd` key, so the tools' relative `data/` paths broke. The orchestrator then spawned a built-in `Explore` agent for "infrastructure triage", which probed the servers with invented inputs (`definitely_not_a_table`, `malformed-no-separators`). Fixed: the server chdirs itself; the orchestrator's frontmatter now allows only `Agent(forecaster, data-analyst, style-profiler, concept-designer, critic)`. (Separately, `orchestrator.md`'s description had a colon, invalid YAML, so Claude Code silently did not register it; a test now parses every frontmatter.)
2. The orchestrator dispatched three concept-designers in parallel: all three `generate_concept` calls failed (reproduced outside Claude Code with three concurrent MCP calls; one alone succeeds).
3. Then three critics scored in parallel: all `score_concept` calls failed. Fixed in `mcp_server` (commit `cd1469e`): `generate_concept`, `score_concept` and `forecast_concept` share a process-wide lock. The scripted driver never hit this because it calls tools one at a time; an LLM does not.

**Things the LLM did that the script cannot,** worth reading in the transcript: it skipped `style-profiler` because the briefs were human-authored ("re-deriving a brief would risk overriding a human design decision"); with `SendMessage` disabled it spawned fresh critics and handed each the attempt history so the critic still owned the cap arithmetic (the retry state therefore lives in the orchestrator's hand-off text, not in the critic); it **refused a critic's request for a prompt override** because the buyer had mandated the production configuration and re-asked for an in-scope parameter. **Where it differs from the scripted rule:** the critics adjusted `ip_adapter_scale` *upward* (0.35 -> 0.55 / 0.50), the script steps it *down*; `critic.md` says "a different scale" and no direction, so this is a spec gap (the direction is not evidence-based on either side).

## J2. LLM decisions against the deterministic rule

134 (case, attempt) pairs x 3 runs = **402 decisions** by headless Claude (Sonnet) playing critic and orchestrator, none unparsed: the 115 H3 cases at attempt 1 and the 19 live cases at the retry cap (`v3_llm_routing_{runs,cases,summary}.csv`).

| Stratum (attempt 1) | Cases | Runs | Verdict = rule | Outcome = rule | Next hop = rule | `adjust` = rule | Cases identical in all 3 runs |
|---|---|---|---|---|---|---|---|
| N1 malformed | 3 | 9 | 9/9 | 9/9 | 9/9 | 9/9 | 3/3 |
| N2 swatch | 2 | 6 | 6/6 | 6/6 | 6/6 | 6/6 | 2/2 |
| N3 clone | 4 | 12 | 12/12 | 12/12 | 12/12 | 12/12 | 4/4 |
| N4 wrong style | 96 | 288 | 288/288 | 288/288 | 288/288 | 288/288 | 96/96 |
| P1 approved | 4 | 12 | 12/12 | 12/12 | 12/12 | **5/6** | **3/4** |
| P2 coherent | 6 | 18 | 18/18 | 18/18 | 18/18 | 15/15 | 6/6 |
| **All** | 115 | 345 | **345/345** (Wilson 95% [0.989, 1.000]) | 345/345 | 345/345 | 335/336 | **114/115** |
| Retry cap (attempt 3), 19 live cases | 19 | 57 | 57/57 | 57/57 | 57/57 | n/a (no retry) | 19/19 |

**The one disagreement.** `submitted_Top` (integrity and Gate 3 failed), run 1 of 3: the model asked for a new `seed` where the rule says `ip_adapter_scale`. Its reason: "the integrity and Gate 3 misses together look like a one-off sampling artifact". **The rule is right per the text:** `critic.md` says a Gate 3 miss calls for a scale change and a seed change only for an integrity-floor miss "while sibling seeds clear it", and no sibling-seed evidence existed. Runs 2 and 3 got it right.

**Read this as an easy test.** The model is shown pass flags, not measured values, and `critic.md` now states the unified rule it is scored against (J3), so it shows the model follows the written spec, not that it would derive it. It also does not test open-ended orchestration; the J1 failures above are what that looks like.

## J3. One rule for "a gate failed and another was not run"

**Recommendation: your prior is right for the failed-plus-unrun case, and escalation is right for the other one.** Rule (`nss.generate.critic_rule.decide`, now called by `critic.md`'s spec, `qc_gates.verdict_from_gates` and `agent_demo.critic_replay`):

1. A gate is *unmeasured* if its `pass` is null/missing, or it is Gate 1b and its clone control did not fail as required (its `False` then says nothing about the concept).
2. Any definite failure -> **REJECT**, even with an unmeasured gate: it cannot pass, and a retry can fix the failure. The verdict text names the unrun gate.
3. Else any unmeasured gate -> **INCONCLUSIVE**: escalate, never regenerate. Nothing failed, so a new image faces the same unmeasured gate; the cause (a missing brief, a broken control, a tool error) needs a person. **That is the real purpose of escalation you asked about**, and it is why REJECT-always would be wrong: a concept with no brief would burn its retries and end as FAILED, hiding the cause.
4. Else PASS_PENDING_HUMAN.

**Verdicts that change** (115 H3 cases): exactly **3**, the malformed underwear images (INCONCLUSIVE/escalate -> REJECT/retry; Gate 3 was unrun because they had no brief). tp/fp/fn/tn are unchanged in every metric row; only the inconclusive counts move. Other behaviour changes: `critic_replay` on a row with nothing failed and a null gate (was REJECT, now INCONCLUSIVE); `verdict_from_gates` on an unvalidated clone control (was REJECT, now not-a-pass unless another gate failed); the H3 routing check (b) now exercises the retry cap on the real underwear images (REJECT -> retry, retry, FAILED) and the driver's replay agrees. Two tests that encoded the old semantics were replaced.

## J4. `generate_concept` reproduces the submission

With `style_key`, the MCP tool now calls `concept_generation.generate_candidate` (extracted from `generate()`, the code that made the deliverables: concat mode over the 8 best references, brief negative prompt, weighted prompt) and writes under `data/generated/agent_tool/`. Through the real MCP server, the submitted sweater (scale 0.35, seed 45) and dress (0.35, seed 44) regenerate **byte-identical** (`v3_generation_reproduction.csv`: sha256 equal, max pixel difference 0) to the recorded candidates, and those are sha256-identical to the committed `reports/concepts/beige-knit-sweater_s0.35_seed45.png` and `red-dress_s0.35_seed44.png`. Passing `prompt`/`reference_images` with `style_key` raises rather than being ignored.

## J5. Filename collision

Names are now `<style>_s<scale>_seed<seed>_<i>.png` (`backends.output_stem`; `style_key` recorded in the sidecar). Tests (`tests/test_generate_backends.py`): two styles at one seed and scale do not collide, one style at two scales or seeds does not, and the name carries all three. Everything under `data/generated/` I might touch was copied to a scratch backup first; nothing was overwritten or deleted.

## J6. Hard negatives and the ablation

Built as pre-registered; 14 scored, **4 excluded by my label-QA look** (not the human check; reasons in `v3_hard_negatives_labels.json`): the colour-swap sweater came out beige (IP-Adapter overrode the colour word) and is effectively a valid concept; the colour-swap top is still white; the "brief absent" dress shows the belt; the averaged white top is a coherent white top. The two M3 coral-pink dresses are attempts 1 and 2, seed 45, **identified by me from the eight M3 dress images; please confirm**. **Correction to H3:** the attempt-2 file is the same as H3's P2 `m3_dress`, counted there as an ACCEPT; it is coral-pink, so P2 recall is 2/9 = 0.22 [0.06, 0.55] without it, not 2/10.

Catches per gate / **unique** catches (removal newly lets the case pass), 10 kept hard negatives (`v3_hard_negatives_{cases,ablation,recall}.csv`):

| Class (kept n) | Gate 1 | Gate 1b | Integrity | Gate 2 | Gate 3 | Passes every gate |
|---|---|---|---|---|---|---|
| H1 wrong attribute (4: coral dress x2, green dress, blue bikini) | 0/0 | 0/0 | 0/0 | 1/**0** | 3/**2** | **1 (the green dress)** |
| H2 briefed change absent (3: bikini, sweater, top) | 0/0 | 1/**0** | 0/0 | 2/**0** | 3/**1** | 0 |
| H3 averaged garment (3: bikini, dress, sweater) | 0/0 | 0/0 | 0/0 | 2/**0** | 3/**1** | 0 |

- **Gate 3 is the unique catcher** of 4 of 10: both coral dresses (H1), the "brief absent" top (H2), the averaged red dress (H3). **No other gate uniquely catches anything**, in these classes or in the H3 set (Gate 1 caught 0 of the 129 cases scored so far; the integrity floor passes every averaged garment, since it is close to the references, and caught no hard negative).
- **A real false accept:** the emerald-green dress passes every automatic gate. SmolVLM read "Green." correctly, yet fidelity is 0.567 against a threshold of 0.384: two of three attributes right (product type, pattern) clears the bar, so a colour miss alone is invisible to Gate 2. The coral dress passes Gate 2 the same way (read "Orange.", 0.567). Precision with these hard negatives: 2/3 = 0.67 (was 2/2). The human check is what stops it.
- The excluded colour-swap sweater is a valid-looking concept that Gate 2 alone rejects (read "Turtleneck." / "Brown.", fidelity 0.0), another false reject.
- **Gate 2 uniquely catches nothing in any set and costs recall (H3) and passes a wrong colour; Gate 1 catches nothing.** No gate was removed, and I am not recommending it: this is 10 constructed cases; whether Gate 1 and the integrity floor matter for real generations is untested.

## J7. Why the critic rejected the two human-approved concepts (report only)

Values from the cached live scores (`v3_critic_strictness.csv`, context `v3_critic_strictness_context.csv`). No threshold, judge or gate changed.

| Concept | Gate that drove it | Measured vs threshold | Marginal (within 10%)? | Threshold or judge? |
|---|---|---|---|---|
| White top | **Integrity floor** | closest-reference DINOv2 0.733 vs global floor 0.779 (-0.045, 5.8% below) | **yes** | **Threshold-marginal, not a misread.** The floor is a similarity measure, not a judge; the design's departure from the references (square neckline, balloon sleeves) pulls it just below. 6 of 18 recorded top candidates fail it (median 0.817). |
| White top | **Gate 3** | answers "YN": square neckline yes; "long balloon sleeves with wide ribbed cuffs" no; strict majority of 2 needs both | n/a (a count) | **Judge consistent with the human record**: the recorded check says the "cuffs are narrower than the briefed 'wide ribbed' cuffs", so the brief was partly missed. It says "N" on 14 of 18 recorded top candidates, so the compound item is rarely satisfied by this generator; the human approved it anyway. |
| Bikini top | **Gate 2** (SmolVLM) | fidelity 0.283 vs 0.384 (-0.101, 26% below) | **no** | **Judge/scorer, not threshold.** The score is exactly 0.283 for all 8 recorded bikini candidates, so Gate 2 cannot pass any bikini at any seed. Per the 1e pre-registration, SmolVLM never says "All over pattern" (it answered "ORIGINAL." here) and answers "Bikini." for "Bikini top" (scorer keeps the period, so 0). The recorded human check says orange all-over print, one coherent top. Florence-2 (advisory) reads the pattern correctly and passes (0.574). |

So the white top is rejected by a marginal threshold plus a stricter-than-the-human brief check, the bikini by a judge that cannot represent its attributes. Context, not asked: Gate 2 also fails 13 of 14 recorded sweater candidates (SmolVLM fidelity 0 to 0.57), and the two hard-negative results above (green dress passes, valid sweater fails) point the same way: Gate 2's failure pattern tracks the judge's vocabulary more than the concept's validity. Whether to change it is your decision.

## Decisions and flags for you

1. Confirm the coral-pink dress identification (M3 attempts 1 and 2, seed 45), and that the recall correction (2/9) is right.
2. Gate 2's threshold/judge (green dress passes, coral dress passes, bikini can never pass, valid sweater fails) and Gate 1 (0 catches): report only, nothing changed.
3. `critic.md` gives no direction for the scale adjustment; the LLM went up, the script goes down.
4. The LLM run is incomplete (quota). If you want a complete transcript, re-run `scripts/agent_llm.py run` with a cheaper orchestrator model.

Files: `scripts/{agent_llm,agent_llm_eval,hard_negatives,critic_strictness,verify_generation_reproduction}.py`, `src/nss/generate/critic_rule.py`; tests `tests/test_{agent_llm,agent_eval,agent_demo,qc_gates,mcp_server,generate_backends}.py`.
