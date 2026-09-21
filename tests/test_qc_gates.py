"""Unit tests for the shipped-gate verdict logic behind the MCP `score_concept` tool.

Models and image embeddings are never loaded: only the pure verdict/brief-resolution logic and the
wiring of `concept_scoring.judge_rows` are exercised. The gates' thresholds are not touched here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from nss.generate import concept_generation, concept_scoring, final_registry, qc_gates


def _gates(**passes: bool | None) -> dict[str, dict[str, Any]]:
    """All five gates passing unless overridden by `name=False` / `name=None` (not run)."""
    return {name: {"pass": passes.get(name, True)} for name in qc_gates.GATE_NAMES}


def test_verdict_all_pass_still_requires_human() -> None:
    """All gates run and passing -> automated True; the verdict still demands the human check."""
    verdict, automated = qc_gates.verdict_from_gates(_gates())
    assert automated is True
    assert "human visual check is still REQUIRED" in verdict


def test_verdict_integrity_and_gate3_failures_are_named() -> None:
    """The white-top failure shape: integrity floor and Gate 3 are both named in the REJECT."""
    verdict, automated = qc_gates.verdict_from_gates(_gates(integrity=False, gate3=False))
    assert automated is False
    assert verdict == "REJECT: failed integrity, gate3"


def test_verdict_gate2_only_failure() -> None:
    """The summer bikini shape: a Gate-2-only failure rejects and names only gate2."""
    verdict, automated = qc_gates.verdict_from_gates(_gates(gate2=False))
    assert (verdict, automated) == ("REJECT: failed gate2", False)


def test_verdict_not_run_is_never_a_pass() -> None:
    """Fail-closed: Gates 2/3 not run -> automated is None (not True) and they are listed."""
    verdict, automated = qc_gates.verdict_from_gates(_gates(gate2=None, gate3=None))
    assert automated is None
    assert "gate2, gate3 not run" in verdict


def test_verdict_failure_beats_not_run() -> None:
    """A gate that already failed rejects even when other gates were not run."""
    _verdict, automated = qc_gates.verdict_from_gates(_gates(gate1b=False, gate2=None))
    assert automated is False


def test_briefed_changes_prefers_sidecar_then_registry(tmp_path: Path) -> None:
    """Sidecar `changes` win; a registry style falls back to CHANGES; an unknown style is empty."""
    image = tmp_path / "c.png"
    image.write_bytes(b"x")
    assert (
        qc_gates.briefed_changes(final_registry.DRESS, image)
        == (concept_generation.CHANGES[final_registry.DRESS]["applied_changes"])
    )
    assert qc_gates.briefed_changes("Nope || X || Y || Z || W", image) == []
    image.with_suffix(".json").write_text(json.dumps({"changes": ["a bow"]}), encoding="utf-8")
    assert qc_gates.briefed_changes(final_registry.DRESS, image) == ["a bow"]


def test_judge_rows_uses_row_changes_without_a_sidecar(tmp_path: Path) -> None:
    """Scoring an image outside the candidate tree: row `changes` replace the missing sidecar."""
    image = tmp_path / "c.png"
    image.write_bytes(b"x")
    row: dict[str, Any] = {
        "image_path": str(image),
        "style_id": final_registry.DRESS,
        "changes": ["a bow", "puff sleeves"],
    }
    with (
        patch.object(concept_scoring.local_vlm, "load"),
        patch.object(concept_scoring.local_vlm, "unload"),
        patch.object(
            concept_scoring.local_vlm,
            "extract_attributes_local",
            return_value={"product_type": "dress", "colour_family": "red"},
        ),
        patch.object(
            concept_scoring.gate3,
            "gate3_local",
            return_value={"answers": [True, False], "pass": False},
        ) as g3,
        patch.object(concept_scoring.gate3, "integrity_local", return_value=(True, "yes")),
        patch.object(
            concept_scoring.colour_check,
            "check",
            return_value={"pass": True, "nearest_delta_e": 1.0, "threshold": 4.5},
        ),
        patch.object(
            concept_scoring.product_retrieval,
            "check",
            return_value={"pass": True, "retrieved": "Dress"},
        ),
    ):
        concept_scoring.judge_rows("smolvlm", [row], {"smolvlm": 0.384})
    g3.assert_called_once_with(image, ["a bow", "puff sleeves"])
    assert row["smolvlm_gate3_answers"] == "YN" and row["smolvlm_gate3_pass"] is False


def test_apply_panel_rule_smolvlm_gates_florence_advises() -> None:
    """Panel rule: a Florence-2 Gate-2 fail is advisory and does not fail Gate 2."""
    row: dict[str, Any] = {
        "floor_max_sim": 0.9,
        "integrity_floor_pass": True,
        "integrity_style_pass": True,
        "smolvlm_gate2_pass": True,
        "smolvlm_gate3_pass": True,
        "florence2_gate2_pass": False,
    }
    concept_scoring.apply_panel_rule(row)
    assert row["gate2_pass"] is True and row["gate2_advisory_pass"] is False
    assert row["gate3_pass"] is True


def test_verdict_names_the_unrun_gate_beside_a_failure() -> None:
    """J3: failure beats not-run, and the REJECT still says which gate was not run."""
    verdict, automated = qc_gates.verdict_from_gates(_gates(integrity=False, gate3=None))
    assert (verdict, automated) == ("REJECT: failed integrity; gate3 not run", False)


def test_unvalidated_clone_control_is_not_a_definite_failure() -> None:
    """Gate 1b's `False` from a failed clone control says nothing about the concept: escalate."""
    gates = _gates(gate1b=False)
    gates["gate1b"]["clone_control_failed_as_required"] = False
    _verdict, automated = qc_gates.verdict_from_gates(gates)
    assert automated is None
    gates["gate2"]["pass"] = False
    verdict, automated = qc_gates.verdict_from_gates(gates)
    assert automated is False and verdict.startswith("REJECT: failed gate2")


def test_failed_advisory_gate1_is_named_but_does_not_reject() -> None:
    """K5: Gate 1 fails, everything gating passes -> still a pass, with the advisory noted."""
    verdict, automated = qc_gates.verdict_from_gates(_gates(gate1=False))
    assert automated is True and "advisory gate1 failed" in verdict
    verdict, automated = qc_gates.verdict_from_gates(_gates(gate1=False, gate3=False))
    assert automated is False and verdict.startswith("REJECT: failed gate3")


def _judge_with(colour_ok: bool, product_ok: bool, tmp_path: Path) -> dict[str, Any]:
    image = tmp_path / "c.png"
    image.write_bytes(b"x")
    row: dict[str, Any] = {
        "image_path": str(image),
        "style_id": final_registry.DRESS,
        "changes": ["a bow"],
    }
    with (
        patch.object(concept_scoring.local_vlm, "load"),
        patch.object(concept_scoring.local_vlm, "unload"),
        patch.object(
            concept_scoring.local_vlm,
            "extract_attributes_local",
            return_value={
                "product_type": "Dress.",
                "colour_family": "Red.",
                "graphical_treatment": "Solid.",
            },
        ),
        patch.object(
            concept_scoring.gate3, "gate3_local", return_value={"answers": [True], "pass": True}
        ),
        patch.object(concept_scoring.gate3, "integrity_local", return_value=(True, "yes")),
        patch.object(
            concept_scoring.colour_check,
            "check",
            return_value={"pass": colour_ok, "nearest_delta_e": 9.0, "threshold": 4.5},
        ),
        patch.object(
            concept_scoring.product_retrieval,
            "check",
            return_value={"pass": product_ok, "retrieved": "Top"},
        ),
    ):
        concept_scoring.judge_rows("smolvlm", [row], {"smolvlm": 0.384})
    return row


def test_gate2_needs_fidelity_and_measured_colour_and_retrieved_product(tmp_path: Path) -> None:
    """L3: a fidelity pass alone is not enough; either identity check failing fails Gate 2."""
    assert _judge_with(True, True, tmp_path)["smolvlm_gate2_pass"] is True
    assert _judge_with(False, True, tmp_path)["smolvlm_gate2_pass"] is False
    assert _judge_with(True, False, tmp_path)["smolvlm_gate2_pass"] is False
