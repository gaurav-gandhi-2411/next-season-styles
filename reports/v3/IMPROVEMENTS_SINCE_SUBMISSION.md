# What changed since the submission, and what it found

DRAFT for the README, written in the register of "Corrections since submission". Not pushed to `main`; the README is unchanged. Everything below is on the branch `feat/v3-improvements`; the submitted concepts and `reports/SUBMISSION/` are untouched, and no concept was regenerated. Every rule was committed before the run it governs (`reports/v3/PREREGISTRATION.md`: `b500577`, `8fbe6dd`, `339ac47`, `a0a13ad`, `48ca1a0`, `e22a0ed`, `0394197`, `14a0ae7`, `bcd1357`, `f14d7a2`). The per-track reports named in brackets carry the tables and the code paths.

## Concept generation

Running the submitted setup at 24 seeds instead of 8 did not raise the yield. All-gate passes: sweater 2/24, dress 10/24, white top 4/24, bikini top 0/24; for every style the interval of the new 16 seeds overlaps the interval of the first 8, so the earlier rates were noise around the same rate, not an underestimate (`TRACK1_concepts.md`). Applying my own selection rule to the white top picks seed 47, not the submitted seed 42, which fails the integrity floor (closest reference 0.733 against 0.779) and Gate 3; I read those scores before writing the rule, so this is a check, not a blind test, and the pick awaits the human check, which is not mine to make.

A Gemini re-read of the four submitted concepts (three readings each, free tier) agrees with the local judge on 11 of 12 binarised calls (kappa 0.80, n = 12, so rough). It scores the sweater 0.617 against its own 0.667 threshold, so the sweater passes Gate 2 under one judge and fails under the other. The cause is the "Melange" label, which none of the three judges names even on real catalogue melange sweaters (the local judge answers "Solid." on 13 of 17), so this is a judge limitation and not evidence against the concept (`TRACK1f_gemini.md`).

Two things I tried did not work. A rule to map the bikini's "All over pattern" label onto the judge's vocabulary passed a solid red dress on a supplementary control (1 of 16 solid garments answered "Melange."), and the mandatory solid-bikini control could not be run because the catalogue download was rate-limited (HTTP 429 on every attempt), so the mapping is not applied and the bikini's 0/24 stands (`TRACK1e_pattern.md`). A four-season concept set was dropped for the same selection problem as the growth ranking below.

## Prediction

The score that picks the two emerging concepts (predicted intensity divided by the trailing 13-week mean) beats chance by a wide margin and does not beat any simple baseline. Hit@3-in-top20 over 10 embargoed origins: model 0.767 [0.733, 0.867], seasonal naive 0.800 [0.733, 0.900], random floor 0.177 (`TRACK2_prediction.md`, `v3_growth_decision_P2.csv`). Most of the skill is a base effect: a constant forecast scores 0.733, and realised growth correlates -0.42 with trailing intensity, so low-base styles grow faster whatever the model says. I built a version with the base removed (forecast minus the seasonal-naive forecast); it removes the effect (the constant forecast falls to the random floor) and is anti-predictive of the deployed target (Hit@3 0.100 against seasonal naive's 0.800, difference -0.700 [-0.800, -0.533]), so it is not adopted (`G1_G2_emerging_redesign.md`). No growth target I can define is free of a base effect.

The validated fallback, the intensity ranking, gives a different three: a black jersey T-shirt, a white knitwear sweater and grey trousers, all solid basics. Its Hit@3 is 0.528 at full origin density, comparable to seasonal naive (paired +0.095 [-0.040, +0.278]) and better on NDCG, Spearman and WMAPE. That is the honest cost of selecting only what has a validated ranking, and it is why I did not regenerate: the submitted three stay, and the README says which ranking chose them.

Split by season of the test window (one outcome window per season, so a cell describes that stretch, not the season in general), the model's top-3 edge over seasonal naive changes sign: positive in winter and spring, and negative in summer (-0.121 [-0.242, -0.030], all in the COVID months). 22 of 140 cells are flagged thin, and the rule that flags them is more lenient than the real overlap (`TRACK3a_seasons.md`).

## The agent layer

The agent layer in the submission was role definitions plus a scripted driver, with no language model making any decision. I evaluated that rule and then built the missing half.

**The critic's rule.** Against human-derived labels the critic accepted 2 of 4 human-approved concepts and no known-bad one (precision 2/2 [0.34, 1.00], recall 2/4 [0.15, 0.85]); the two it rejected are exactly the two concepts chosen without a gate pass, so its recall on concepts it did not help select is 0/2. Adding hard negatives (a green dress in a red style, the coral-pink dress, a brief left out, an averaged garment), no gate but Gate 3 uniquely catches any of them, and the green dress passed every gate: two of three attributes right cleared the Gate 2 threshold (`TRACK4_agent_layer.md`, `TRACK5_llm_orchestration.md`).

**Defects the evaluation found, all fixed with tests.** Three places disagreed on "a gate failed and another was not run"; there is now one rule (failed beats unmeasured; nothing failed and something unmeasured escalates). The generation tool did not use the configuration that made the submission and named files by seed alone, so it overwrote an earlier image; it now reproduces the submitted sweater and dress byte-for-byte (sha256 equal) and names files by style, scale and seed. The tools failed under concurrent calls, which the scripted driver never made and a language model does at once; they are now serialised.

**Language-model orchestration.** Headless Claude Code runs the sub-agent pattern. Against the deterministic rule a model agreed on 402 of 402 routing decisions, which tests fidelity to a written rule and little else. On ten cases where the rule is silent (which way to move the scale, a gate returning an error, conflicting judges, a concept that passes but misses the brief), 60 runs, 59 parseable, graded against a rubric written beforehand: 53 acceptable and 6 not. The model's decisions are acceptable on process (it escalates when a judge reads the same thing on every seed, and it flags judge disagreements) and not acceptable where the answer needs project evidence it was not given (the safe scale window), until it is told. It took five live runs of the three-style workflow before one finished: they exposed a wrong working directory, an orchestrator that improvised with an unlisted agent, and the concurrency failures above (`TRACK5_llm_orchestration.md`, `TRACK6_silent_rule_and_gate_changes.md`, `agent_run_transcript_llm.md`).

**Rule changes made from that evidence.** The critic now has a written scale rule (window 0.25 to 0.45, from the lever experiments; down for a missing briefed change, seed for a lone integrity miss, seed when failures conflict, seed for a Gate 2-only miss). With that rule in its context the model no longer raises the scale for a Gate 2-only miss (0 of 3 runs, against 6 of 6 before). Gate 1, which failed in none of 129 cases, is advisory: computed and reported, no longer part of the verdict, and no verdict changes.

Gate 2 went through three versions. First it required the small judge's product-type and colour readings to match the style; that failed the green and both coral dresses and kept the submitted sweater and dress, but it also rejected two correctly red dresses that the judge read as orange, because a hard constraint amplifies whatever error its input carries. Second, the inputs were measured or retrieved instead of read: the garment's colour from pixels (a border-sampled mask, dominant colour in CIELAB, CIEDE2000 distance to the style's real reference articles, threshold at the p90 of real nearest-sibling distances, the same calibration as Gate 1b) and the product type from the top-1 image-retrieval match, compared as an exact catalogue string.

Third, that colour check was repaired where it was weak. The mask now comes from rembg (a 176 MB model; the border mask had fallen back to the centre of the image for all 19 white-top references, and now falls back for none of the references and 2 of the 129 stored cases). That did not change the white top's threshold (1.06 to 1.12): the near-identical white references were the cause, not the fallback. Thresholds are shrunk toward the global median with weight n / (n + 17), where 17 is the median reference count, the same convention as the intensity shrinkage; that moves the white top to 2.62 and the bikini's from 13.8 to 7.35, but it also cuts the bikini's real-article leave-one-out pass rate from 0.75 to 0.50, because a patterned garment legitimately varies more than the solid styles the median is built from. Across the four final styles, nested leave-one-out on real articles passes 0.884 with the first version, 0.855 with rembg alone and 0.870 with shrinkage. A distance within 0.403 CIEDE2000 of a threshold, the 95th percentile of measured mask noise over 552 jittered crops of the real references, is neither pass nor fail: Gate 2 is unmeasured and the verdict is INCONCLUSIVE, so a person looks. The green dress fails (54.19 against 4.30), both coral dresses fail with a wider margin than before (6.01 and 5.77), and the submitted dress (1.67), sweater (2.75 against 3.75) and both red dresses (3.43, 1.64) pass; none of those falls in the band. Over the 54 recorded candidates all-gate passes go 9 before the constraint, 5 under the judge-reading version and 7 now. Seven of the 129 stored cases and 6 of the 54 candidates fall in the band, all of them already failing another gate, so no verdict changes; two verdicts differ from before the first version (the green dress goes to REJECT, and a human-coherent underwear image goes from INCONCLUSIVE to REJECT because retrieval reads it as a swimwear bottom). The mask noise for the bikini is large (p95 28.9), because its multi-tone print makes the dominant colour flip between crops, so the pooled band is not representative there: the check is measured and better than reading colour from the small judge, but it is not robust on a patterned style or on a style with few references (`TRACK6`, `TRACK7`, `TRACK8`).

**A blind re-grade of the ten judgment cases.** I graded the sixty K2 decisions myself against a rubric I had written, which is a weak check, so they were re-graded blind (shuffled, without the arm or my grades) by a Groq-hosted Qwen model on all ten cases; Gemini graded two before its free-tier quota stopped it (no Claude model was substituted). On whether a decision is acceptable, Qwen and I agree at kappa 0.84 (n = 59); on the four fine grades we agree at 0.51 and the counts differ between graders, so I do not report them. This is LLM consensus, not human ground truth, and it is one complete independent judge, not two (`TRACK7_measured_identity.md`).

**What the agent evaluation found, in plain terms.** The model follows explicit rules faithfully, improvises where the rules are silent, and putting the evidence or the rule in front of it corrects most of the poor improvisation. Given the pass flags and a written rule, it made the same routing decision as the deterministic rule in all 402 runs (J2). Where the rule was silent (K2), 53 of 59 decisions were acceptable, and the 6 that were not were all made without the project's lever evidence (it left a scale at 0.55, or raised it to resolve conflicting failures); the same cases were acceptable once the evidence note was in the prompt. One habit survived the evidence note: for a Gate 2-only miss it raised the scale in every run, and once the K3 rule was written into `critic.md` it changed the seed instead in every run (L5: 3 of 3, against 6 of 6). So the reliable ways to get good behaviour from it are to write down the rule and the evidence it needs, and to check the result, not to expect it to derive them.

## Process, catalogue-wide priors, and a second spec gap (N1-N6, A1-A3)

A process gap from the previous session was closed first: a test's expectation
(`test_band_gives_pass_fail_or_escalate`) had been corrected without re-running the full suite
afterward. The OLD expectation (`verdict_for(4.6, 4.5, band=None) == FAIL`) was simply wrong
against the code as written (`band=None` resolves to the measured `W_BAND`, giving ESCALATE, not
FAIL, confirmed by calling the function directly); the code was correct, so nothing was reverted.
The full suite is green (913 passed, 3 skipped, 1 xfailed).

The K2 blind re-grade finished once Gemini's quota reset: all 59 decisions are now graded by all
three judges. On the defensible binary measure (acceptable versus not), the three pairwise Cohen's
kappas are claude-qwen 0.838 and gemini-qwen 0.838 (2,000-resample bootstrap 95% CI [0.550, 1.0]),
and claude-gemini a literal 59/59 raw agreement, for which the percentile bootstrap is degenerate
(it cannot show disagreement that is not in the sample) and is replaced by the exact
Clopper-Pearson interval on raw agreement, [0.939, 1.0]. The base rate of "not acceptable" is a
real but not vanishing minority (6-8 of 59, 10-14%), stated here so the kappa is not read without
that context. This is LLM consensus, not human ground truth.

Qwen's only two dissents from both other graders are the same scenario, S10 (a concept whose
attempt 1 a human REJECTed for a missing funnel neck; attempt 2 passes every automatic gate). The
orchestrator's reply is `PASS_PENDING_HUMAN` + `FORWARD` to the forecaster + a second human
escalation, while its own reasoning says the concept "stays unshippable" -- action and reasoning
contradict. Graded on the executed action rather than the prose reasoning (a rubric principle
added for all future grading, since being persuaded by well-written reasoning over a contradictory
action is a known LLM-judge failure mode), this is the rubric's `worse`: Qwen's grade was right,
and Claude's and Gemini's `equivalent` grades on these two decisions are adjudicated to `worse`.
This is an explicit, reported correction, not a quiet one, and the raw grader records are
untouched on disk (`scripts/k2_adjudicate.py` applies the override on top of them). Revised: all
three graders now agree exactly (51 acceptable, 8 not, was 53/6 for Claude and Gemini), and all
three pairwise kappas become 59/59 raw agreement (Clopper-Pearson CI [0.939, 1.0]).

The cause was a real spec gap: nothing in `critic_rule.decide` or `agents/critic.md` said what to
do when a person has already rejected an earlier attempt of the same concept -- the same class of
gap as the K3 scale-direction fix, on the verdict side instead of the retry-parameter side. Fixed:
**a prior human REJECT on a concept request is terminal** (`critic_rule.decide(...,
prior_human_reject=True)` returns REJECT before any gate; `agents/critic.md` and
`skills/concept-qc/SKILL.md` state the same rule). No existing caller passes the new parameter, so
every previously-scored verdict (129 H3/J6 cases, 54 candidates, S01-S09) is unaffected by
construction, and the deterministic rule now gives REJECT on S10 (was PASS_PENDING_HUMAN).
**A live Sonnet re-run of S10 (3 runs, required: REJECT with no FORWARD and no escalation in all
three) has NOT been run yet** -- it needs a live model call, Max plan utilisation was last known
at ~0.90 seven-day, and this session has no tool to check the current figure directly, so it is
paused for explicit confirmation before spending that quota rather than guessed at.

A catalogue-wide colour-threshold prior was computed to replace M1's single global median (built
from only three solid-colour calibrated styles, which is why shrinkage hurt the bikini): for each
of the 1,980 autumn forecast-eligible styles, up to 8 real catalogue photos through the unchanged
M2 rembg dominant-colour pipeline, split solid (`Solid`+`Melange`, decided from the already-measured
M3 mask-noise evidence, not from this result: 1,219 styles, median raw threshold 7.879) versus
patterned (everything else: 761 styles, median raw threshold 9.445); every style had at least 2
photos, so neither class needed the pooled fallback. This run first crashed the machine (12 worker
processes, each loading its own ~1.1 GB rembg model copy, pushed the system into low memory while
another local process needed priority) and was killed by the harness; the retry used at most 2
worker processes with multithreaded onnxruntime sessions instead, BELOW_NORMAL priority, CPU
affinity capped at half the physical cores, a system-RAM guard, and per-shard checkpointing
verified by an actual kill-and-resume rehearsal (bit-for-bit identical output). Peak RSS was 2.18
GB against a 6 GB target; the full run took 2h26m. These medians describe real (unscreened)
catalogue photos, noisier than the five calibrated styles' screened flat-lay references, so they
are not directly comparable in absolute terms to M1's 4.30 -- they are still the correct per-class
shrinkage target, since solid and patterned are measured within the same population on both sides.

For the bikini (the one patterned calibrated style), dominant colour was replaced with a
Wasserstein colour-histogram distance (sum of three per-channel 1-D earth-mover distances over the
garment's Lab histogram), thresholded and shrunk the same way as before but toward the new
patterned-class prior. The four solid-class styles are untouched, and all seven of M1's required
outcomes still hold under this mixed pipeline. The two outcomes specific to the bikini did **not**
meet their bar, and are reported as such rather than adjusted: the nested leave-one-out pass rate
is 0.50, identical in count to the old dominant-colour result (independently verified as different
underlying numbers, not a bug) and short of the 0.80 target; the noise band (p95 of 64 jittered
crops) is 36.96, higher than the old statistic's 28.9, not "well below" it as hoped -- one
reference photo has a real bimodal crop sensitivity (6 of 8 crop offsets deviate 33-37, 2 of 8
deviate under 1, no mask fallback involved), a genuine property of a distribution statistic on an
asymmetrically-laid-out print. The diagnosis: the bikini's raw histogram threshold (24.64) would
pass most held-out references on its own; empirical-Bayes shrinkage toward *any* external prior
(the old 4.30 or the new 9.445) pulls it down to 14.3, which is what causes the misses. This is
the same mechanism M1 already found for dominant colour -- the shrinkage step hurts the bikini,
not the underlying statistic -- so changing the statistic without changing how it is shrunk did
not fix it, and the method is not tuned further to force a pass.

## Not done

The white top's human check is still open. The solid-bikini control and the Gemini reading of the white-top alternates were blocked by rate limits. The bikini can still not pass Gate 2: its score is exactly 0.283 on all 24 seeds because the judge cannot name its pattern and reads "Bikini." for "Bikini top", and a fix to that scoring would move the calibration, so I left it and reported it.

The live Sonnet re-run of S10 under the new terminal-REJECT rule has not been run (paused for
explicit confirmation to spend Max plan quota; utilisation was last known at ~0.90 and this
session has no way to check the current figure directly). The bikini's Gate 2 colour check still
does not meet its own bar under either statistic: neither the dominant-colour version (M1, 0.50
shrunk leave-one-out) nor the histogram version (N4, also 0.50) reaches 0.80, and the mechanism
in both cases is the shrinkage step pulling the threshold toward an external prior the bikini's
own colour spread does not resemble, not the choice of statistic. No further attempt was made to
fix this within this session; it is reported, not resolved.
