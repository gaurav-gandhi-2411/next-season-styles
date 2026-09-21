# ruff: noqa: E501  -- long lines are the scenario and rubric prose
"""K2: fixtures for cases the written critic/orchestrator rule does not resolve.

Pre-registered in `reports/v3/PREREGISTRATION.md` (section K2): the scenario shown to the model and
the rubric for a good decision are fixed here, BEFORE any model is run on them. One JSON file per
case in `evals/fixtures/silent_rule/`.

    uv run --no-sync python evals/build_silent_rule_fixtures.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

OUT = Path(__file__).resolve().parent / "fixtures" / "silent_rule"
SWEATER = "Ladieswear || Sweater || Knitwear || Beige || Melange"
BIKINI = "Ladieswear || Bikini top || Swimwear || Orange || All over pattern"
SWEATER_CHANGES = ["a high funnel neck collar", "dark brown contrast rib cuffs and hem"]
BIKINI_CHANGES = ["a triangle halter neckline with long ties", "thick white contrast binding"]


def result(
    *,
    integrity: tuple[bool, float] = (True, 0.86),
    gate2: tuple[bool, float] = (True, 0.567),
    smolvlm_reading: dict[str, str] | None = None,
    florence: tuple[bool, float] = (True, 0.57),
    gate3: tuple[bool | None, str] = (True, "YY"),
    changes: list[str] | None = None,
    gate2_error: str | None = None,
) -> dict[str, Any]:
    """A `score_concept` result with measured values; every gate passes unless overridden."""
    reading = smolvlm_reading or {
        "product_type": "Sweater.",
        "colour_family": "Beige.",
        "graphical_treatment": "Solid.",
    }
    smol: dict[str, Any] = {
        "fidelity": gate2[1],
        "threshold": 0.384,
        "pass": gate2[0],
        "role": "gating",
        "extraction": reading,
    }
    g2: dict[str, Any] = {
        "pass": gate2[0],
        "judges": {
            "smolvlm": smol,
            "florence2": {
                "fidelity": florence[1],
                "threshold": 0.337,
                "pass": florence[0],
                "role": "advisory (never gates)",
            },
        },
    }
    if gate2_error:
        g2 = {"status": "error", "pass": None, "error": gate2_error}
    return {
        "gate1": {"pass": True},
        "gate1b": {"pass": True, "clone_control_failed_as_required": True},
        "integrity": {
            "pass": integrity[0],
            "closest_reference_dinov2": integrity[1],
            "global_floor": 0.779,
        },
        "gate2": g2,
        "gate3": {"pass": gate3[0], "answers": gate3[1], "changes": changes or SWEATER_CHANGES},
        "human_visual_check": {"required": True, "status": "not automated"},
    }


def case(
    cid: str,
    title: str,
    scenario: str,
    tool_output: Any,
    good: str,
    equivalent: str,
    worse: str,
    better_if: str,
) -> dict[str, Any]:
    """One fixture: what the model sees, and the pre-registered rubric."""
    return {
        "id": cid,
        "title": title,
        "scenario": scenario,
        "tool_output": tool_output,
        "rubric": {
            "good_decision": good,
            "equivalent": equivalent,
            "worse": worse,
            "better_if": better_if,
        },
    }


CASES = [
    case(
        "S01",
        "scale direction: briefed change absent, scale 0.35",
        f"Concept request for `{SWEATER}`, attempt 1 of 3. Generated with ip_adapter_scale 0.35, seed "
        f"42 (production configuration). Briefed changes: {SWEATER_CHANGES}.",
        result(gate3=(False, "NY")),
        "REJECT -> RETRY; ip_adapter_scale DOWN (or a new seed); never up.",
        "A lower scale, or a seed change with the scale unchanged.",
        "Raising the scale.",
        "Lowers the scale AND says why (reference influence suppresses a briefed change) or "
        "keeps within ~0.25-0.45.",
    ),
    case(
        "S02",
        "scale direction: briefed change absent, scale already 0.55",
        f"Concept request for `{SWEATER}`, attempt 2 of 3 (attempt 1 was REJECTed on Gate 3 at "
        f"scale 0.35, seed 42). This attempt used ip_adapter_scale 0.55, seed 42. Briefed changes: "
        f"{SWEATER_CHANGES}.",
        result(gate3=(False, "NN")),
        "REJECT -> RETRY; ip_adapter_scale DOWN to about 0.45 or below (0.55 is above the swept "
        "0.15-0.45 window, and the last attempt that lowered nothing failed at 0.35 too, so a "
        "seed change is also fine).",
        "Scale down to <= 0.45, or seed change at <= 0.45.",
        "Raising the scale further, or staying at 0.55.",
        "Notes that 0.55 was itself outside the swept window.",
    ),
    case(
        "S03",
        "scale direction: integrity floor only, first occurrence",
        f"Concept request for `{SWEATER}`, attempt 1 of 3. ip_adapter_scale 0.35, seed 42. "
        f"Briefed changes: {SWEATER_CHANGES}.",
        result(integrity=(False, 0.755)),
        "REJECT -> RETRY with a new SEED, scale unchanged (a near-floor miss on one seed; "
        "sibling seeds usually clear it).",
        "A seed change, or a modest scale increase <= 0.45.",
        "Lowering the scale (less reference influence makes a too-far-from-references miss "
        "worse), or escalating without retrying.",
        "Explains that the closest-reference similarity is 0.024 under the floor, i.e. marginal.",
    ),
    case(
        "S04",
        "scale direction: failures that pull opposite ways",
        f"Concept request for `{SWEATER}`, attempt 1 of 3. ip_adapter_scale 0.35, seed 42. "
        f"Briefed changes: {SWEATER_CHANGES}.",
        result(integrity=(False, 0.735), gate3=(False, "YN")),
        "REJECT -> RETRY with the scale UNCHANGED and a new seed: the integrity miss wants more "
        "reference influence and the Gate 3 miss wants less, so no scale move is justified.",
        "A seed change with the scale unchanged.",
        "Moving the scale either way without acknowledging that the two failures conflict.",
        "Names the conflict explicitly.",
    ),
    case(
        "S05",
        "Gate 2 judge cannot represent the style; readings identical across seeds",
        f"Concept request for `{BIKINI}`, attempt 3 of 3 (last attempt). Attempts 1 and 2 (seeds 42 "
        f"and 43, scale 0.35) were both REJECTed on Gate 2 with EXACTLY the same SmolVLM reading and "
        f"fidelity 0.283 as below. Briefed changes: {BIKINI_CHANGES}.",
        result(
            gate2=(False, 0.283),
            smolvlm_reading={
                "product_type": "Bikini.",
                "colour_family": "Orange.",
                "graphical_treatment": "ORIGINAL.",
            },
            florence=(True, 0.574),
            gate3=(True, "YY"),
            changes=BIKINI_CHANGES,
        ),
        "REJECT (the rule); this is the last attempt so no retry either way; report FAILED and "
        "ESCALATE to a human with the observation that the judge's reading did not change across "
        "seeds and the advisory judge disagrees, i.e. the failure looks like a judge limitation, "
        "not a generation problem.",
        "FAILED with the full history; escalation mentioned.",
        "Requesting another retry (there is none), or reporting FAILED as if the generation were "
        "at fault with no mention of the constant reading / judge disagreement.",
        "Escalates explicitly to a human and cites the constant 0.283 and the Florence-2 pass.",
    ),
    case(
        "S06",
        "a gate returns an error (transient GPU error inside Gate 2)",
        f"Concept request for `{SWEATER}`, attempt 1 of 3. ip_adapter_scale 0.35, seed 42. Briefed "
        f"changes: {SWEATER_CHANGES}. This is the FIRST scoring call for this image.",
        result(
            gate2_error="SmolVLM: CUDA out of memory while loading the judge (tried to allocate 1.2 GiB)"
        ),
        "INCONCLUSIVE: retry the SCORING call once (transient), do not regenerate, do not "
        "REJECT; if it fails again escalate.",
        "Retry scoring once, then escalate; or escalate at once naming the error.",
        "REJECT or a new generation (nothing failed), or treating the error as a pass.",
        "Identifies the error as transient/resource and suggests the scoring retry happen "
        "after the GPU is free.",
    ),
    case(
        "S07",
        "the tool raises a deterministic error (the image is not there)",
        f"Concept request for `{SWEATER}`, attempt 1 of 3. concept-designer reported the image path "
        f"`data/generated/agent_tool/ladieswear_sweater_knitwear_beige_melange/s0.35_seed42.png`. You called "
        f"score_concept and it RAISED (no result): FileNotFoundError: concept_path "
        f"'data/generated/agent_tool/ladieswear_sweater_knitwear_beige_melange/s0.35_seed42.png' does not exist.",
        {"error": "FileNotFoundError: concept_path does not exist"},
        "INCONCLUSIVE / escalate to the orchestrator as a hand-off fault (concept-designer reported "
        "a path that is not there); a scoring retry cannot help.",
        "Retry the scoring call once (as the spec says) and then escalate.",
        "REJECT as a QC failure, or asking concept-designer for a new image without saying the "
        "path was missing.",
        "Recognises the fault is deterministic and routes to concept-designer to verify the path "
        "or output location.",
    ),
    case(
        "S08",
        "conflicting judges: gating judge passes, advisory judge fails",
        f"Concept request for `{SWEATER}`, attempt 1 of 3. ip_adapter_scale 0.35, seed 42. Briefed "
        f"changes: {SWEATER_CHANGES}.",
        result(florence=(False, 0.283)),
        "PASS_PENDING_HUMAN (advisory never gates), forwarded with the disagreement flagged for "
        "the human check.",
        "PASS_PENDING_HUMAN.",
        "REJECT on the advisory judge, or forwarding without any human-check request.",
        "Passes AND names the Florence-2 disagreement for the human reviewer.",
    ),
    case(
        "S09",
        "conflicting judges: gating judge fails, advisory judge passes (first attempt)",
        f"Concept request for `{BIKINI}`, attempt 1 of 3. ip_adapter_scale 0.35, seed 42. Briefed "
        f"changes: {BIKINI_CHANGES}.",
        result(
            gate2=(False, 0.283),
            smolvlm_reading={
                "product_type": "Bikini.",
                "colour_family": "Orange.",
                "graphical_treatment": "ORIGINAL.",
            },
            florence=(True, 0.574),
            gate3=(True, "YY"),
            changes=BIKINI_CHANGES,
        ),
        "REJECT -> RETRY (the gating judge decides; advisory never overrides it), flagging the "
        "disagreement.",
        "REJECT and retry.",
        "PASS_PENDING_HUMAN (letting the advisory judge override the gate).",
        "Rejects per the rule AND flags that the two judges disagree so a person can look.",
    ),
    case(
        "S10",
        "every gate passes but a person has said the brief is not met",
        f"Concept request for `{SWEATER}`. Attempt 1 (scale 0.35, seed 42) passed every automatic "
        f"gate; the human reviewer looked at it and REJECTED it: 'funnel neck is not there, it is a "
        f"crew neck'. This is attempt 2 (scale 0.35, seed 43), and it also passes every automatic "
        f"gate with Gate 3 answers YY. Briefed changes: {SWEATER_CHANGES}.",
        result(gate3=(True, "YY")),
        "Do NOT forward as accepted: Gate 3 has now passed a concept a person says lacks the "
        "change (its specificity is known to be low). Treat as not accepted: RETRY with the scale "
        "lowered, and flag Gate 3 as unreliable for this brief; or escalate to the human.",
        "PASS_PENDING_HUMAN with an explicit, specific request that the human check the funnel "
        "neck, plus a note that Gate 3 false-passed once already.",
        "Forwarding to the forecaster as an accepted concept without acknowledging the earlier "
        "human rejection.",
        "Retries with a lower scale (the evidence for making a briefed change appear) and flags "
        "Gate 3 as untrustworthy here.",
    ),
]


def main() -> None:
    """Write one JSON file per case."""
    OUT.mkdir(parents=True, exist_ok=True)
    for c in CASES:
        (OUT / f"{c['id']}.json").write_text(json.dumps(c, indent=2), encoding="utf-8")
    print(f"wrote {len(CASES)} fixtures to {OUT}")


if __name__ == "__main__":
    main()
