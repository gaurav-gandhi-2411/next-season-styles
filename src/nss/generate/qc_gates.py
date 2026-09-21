"""The shipped concept-QC gates as one callable -- what the MCP `score_concept` tool runs.

Gate 1   within-style range: mean similarity to the style's references must be <= the p90 of
         similarity between distinct REAL articles of that style, in CLIP and DINOv2.
Gate 1b  nearest reference: the closest single reference must be <= the p90 of the real
         nearest-sibling similarity; validated live by an exact-clone control that must FAIL (a
         check that passes a clone is broken and is reported UNVALIDATED, never as a pass).
Integrity  GLOBAL floor (gates, `integrity_global`): the closest real reference (DINOv2) must be at
         least the p10 of real nearest-sibling similarity pooled over every style. The per-style
         floor (`gate3.integrity_floor`) is reported beside it as ADVISORY.
Gate 2   attribute fidelity of a blind local-judge read of the picture against the style's VISIBLE
         attributes (`nss.generate.fidelity`), each judge against its own calibrated threshold.
         SmolVLM GATES; Florence-2 is ADVISORY (reported, never gating) --
         `concept_scoring.GATING_JUDGES` / `ADVISORY_JUDGES`, the same panel rule that scored the
         final concepts.
Gate 3   are the briefed changes visible? One yes/no question per change to the gating local judge;
         a strict majority must be present. Needs the briefed changes (see `briefed_changes`).
Human    a mandatory visual check. It is never automated: the automatic gates passed visibly
         malformed candidates (WRITEUP s9), so a pass here is "pending human check", not "ship".

Gates 2 and 3 load a local VLM (`include_fidelity=True`; no network, no API quota). This supersedes
the earlier Groq single-reading Gate 2 (Groq's key/quota was spent; a single reading varied by
about +/-0.21) and the earlier verdict that had no integrity floor and no Gate 3.

Everything reuses the functions that scored the final concepts (`within_style_benchmark`,
`gate1b_nearest_reference`, `integrity_global`, `gate3`, `concept_scoring`); no threshold is
re-derived here and none was changed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nss.generate import (
    clip_scoring,
    concept_generation,
    critic_rule,
    dino_scoring,
    gate3,
    integrity_global,
)
from nss.generate.gate1b_nearest_reference import gate1b_pass, gate1b_threshold
from nss.generate.vlm_judges import SKILL
from nss.generate.within_style_benchmark import concept_similarity, style_benchmark

SPACES = ("clip", "dinov2")
HUMAN_CHECK_NOTE = (
    "REQUIRED and never automated: look at the image. The automatic gates have passed visibly "
    "malformed garments (cut-out defects, pattern drift, straps on a bottom)."
)


def reference_paths_for_style(style_key: str) -> list[Path]:
    """The screened full-garment references the gates are calibrated on, best-selling first.

    Underwear uses the verified plain-solid references (the screened set is the lace one they
    replaced). Every other style uses `concept_generation.load_refs()` -- the SAME widened base the
    final concepts were scored on (autumn/winter screened manifest plus the summer style's widened
    one), so this tool and `concept_scoring` gate against identical references. The earlier summer
    manifest is kept only as a fallback.

    Raises:
        ValueError: no screened references exist for `style_key` (the gates are undefined without
            real same-style articles to calibrate on).
    """
    from nss.generate import screen_references, seasonal_concept, underwear_generate, underwear_refs

    if style_key == underwear_refs.STYLE_ID:
        return underwear_generate.reference_paths()
    final_refs = concept_generation.load_refs()
    if style_key in final_refs:
        return final_refs[style_key]
    if seasonal_concept.SCREENED_PATH.exists():
        import polars as pl

        if pl.read_csv(seasonal_concept.SCREENED_PATH)["style_id"][0] == style_key:
            return seasonal_concept.screened_references()
    screened = screen_references.load_screened_references()
    if style_key not in screened:
        raise ValueError(f"No screened reference images found for style_key {style_key!r}")
    return screened[style_key]


def _embed_all(paths: list[Path]) -> dict[str, list[Any]]:
    return {
        "clip": [clip_scoring.embed_image(p) for p in paths],
        "dinov2": [dino_scoring.embed_image(p) for p in paths],
    }


GATE_NAMES = ("gate1", "gate1b", "integrity", "gate2", "gate3")


def briefed_changes(style_key: str, concept: Path) -> list[str]:
    """The design changes the concept was briefed with, for Gate 3.

    Order: the sidecar next to the image (`<image>.json`, key `changes`), else the final
    concepts' registry (`concept_generation.CHANGES`). Empty if neither knows: Gate 3 is then NOT
    RUN (never a pass), because "are the briefed changes visible" is undefined without the brief.
    """
    sidecar = concept.with_suffix(".json")
    if sidecar.exists():
        changes = json.loads(sidecar.read_text(encoding="utf-8")).get("changes")
        if changes:
            return list(changes)
    spec = concept_generation.CHANGES.get(style_key)
    return list(spec["applied_changes"]) if spec else []


def verdict_from_gates(gates: dict[str, dict[str, Any]]) -> tuple[str, bool | None]:
    """Combine the per-gate results into `(verdict text, automated_gates_pass)`.

    The rule is `critic_rule.decide` (the same one `agents/critic.md` and the drivers use): a gate
    that definitely failed rejects even if another gate was not run; with no failure, an
    unmeasured gate (`pass` is `None`, or Gate 1b's clone control did not fail as required) means
    the verdict is never a pass (fail-closed) and `automated_gates_pass` is `None`.
    """
    passes = {name: gates[name].get("pass") for name in GATE_NAMES}
    clone_ok = critic_rule.clone_ok_from_gates(gates)
    failed = critic_rule.failing(passes, clone_ok)
    not_run = critic_rule.unmeasured(passes, clone_ok)
    advisory = [g for g in critic_rule.ADVISORY if gates[g].get("pass") is False]
    adv_note = f" (advisory {', '.join(advisory)} failed)" if advisory else ""
    if failed:
        note = f"; {', '.join(not_run)} not run" if not_run else ""
        return "REJECT: failed " + ", ".join(failed) + note + adv_note, False
    if not_run:
        passed = [n for n in GATE_NAMES if n not in not_run]
        return (
            f"{', '.join(passed)} passed; {', '.join(not_run)} not run; "
            f"human visual check REQUIRED{adv_note}",
            None,
        )
    return (
        f"PASS on all gating gates{adv_note}; the human visual check is still REQUIRED",
        True,
    )


def _local_panel(
    concept: Path, style_key: str, changes: list[str], floor: dict[str, Any], global_limit: float
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Gate 2 and Gate 3 from the local judge panel, via the code that scored the final concepts.

    Runs `concept_scoring.judge_rows` for every gating and advisory judge, then
    `concept_scoring.apply_panel_rule` (SmolVLM gates, Florence-2 is advisory). Returns
    `(gate2, gate3)` result dicts.
    """
    from nss.generate import concept_scoring

    backends = [*concept_scoring.GATING_JUDGES, *concept_scoring.ADVISORY_JUDGES]
    thresholds = concept_scoring.judge_thresholds(backends)
    row: dict[str, Any] = {
        "image_path": str(concept),
        "style_id": style_key,
        "changes": changes,
        "floor_max_sim": floor["max_sim"],
        "integrity_style_pass": floor["pass"],
        "integrity_floor_pass": floor["max_sim"] >= global_limit,
    }
    for backend in backends:
        concept_scoring.judge_rows(backend, [row], thresholds)
    concept_scoring.apply_panel_rule(row)
    judges = {
        b: {
            "fidelity": row[f"{b}_fidelity"],
            "threshold": thresholds[b],
            "pass": row[f"{b}_gate2_pass"],
            "role": "gating" if b in concept_scoring.GATING_JUDGES else "advisory",
            "colour_ok": row[f"{b}_colour_ok"],
            "colour_delta_e": row[f"{b}_colour_delta_e"],
            "colour_threshold_p90": row[f"{b}_colour_threshold"],
            "product_type_ok": row[f"{b}_product_ok"],
            "product_type_retrieved": row[f"{b}_product_retrieved"],
            "extraction": row[f"{b}_extraction"],
        }
        for b in backends
    }
    gate2 = {
        "status": "scored (local judges, greedy decoding: one reading is the reading)",
        "pass": row["gate2_pass"],
        "advisory_pass": row["gate2_advisory_pass"],
        "judges": judges,
    }
    if row["gate3_pass"] is None:
        gate3_result: dict[str, Any] = {
            "status": "not_run",
            "pass": None,
            "note": "no briefed changes known for this concept (no sidecar, not in CHANGES)",
        }
    else:
        judge = concept_scoring.GATING_JUDGES[0]
        gate3_result = {
            "status": "scored (local judge)",
            "pass": row["gate3_pass"],
            "judge": judge,
            "changes": changes,
            "answers": row[f"{judge}_gate3_answers"],
        }
    return gate2, gate3_result


def score_gates(
    concept_path: str | Path,
    style_key: str,
    include_fidelity: bool = False,
    changes: list[str] | None = None,
) -> dict[str, Any]:
    """Run every shipped gate on one image: 1, 1b (clone-validated), integrity, and 2 / 3.

    Args:
        concept_path: The generated concept image.
        style_key: `" || "`-joined style key it was generated for.
        include_fidelity: Also run the LOCAL judge panel for Gate 2 and Gate 3 (loads SmolVLM and
            Florence-2; no network). Without it Gates 2 and 3 are reported `not_run` and the
            verdict can never be a pass. Name kept for compatibility with existing callers.
        changes: The briefed design changes for Gate 3. Default: `briefed_changes` (sidecar JSON
            next to the image, else the final-concept registry).

    Returns:
        A dict with `gate1`, `gate1b`, `integrity`, `gate2`, `gate3`, `human_visual_check`,
        `automated_gates_pass` (`None` until every gate has been run) and a plain `verdict`.

    Raises:
        FileNotFoundError: `concept_path` does not exist.
        ValueError: no screened references exist for `style_key`.
    """
    concept = Path(concept_path)
    if not concept.exists():
        raise FileNotFoundError(f"concept_path {str(concept_path)!r} does not exist")
    refs = reference_paths_for_style(style_key)
    if len(refs) < 2:
        raise ValueError(f"need >= 2 reference images for {style_key!r}, found {len(refs)}")
    ref_embs = _embed_all(refs)
    emb = {"clip": clip_scoring.embed_image(concept), "dinov2": dino_scoring.embed_image(concept)}

    sims = {s: concept_similarity(emb[s], ref_embs[s])["mean"] for s in SPACES}
    limits = {s: style_benchmark(ref_embs[s])["pair_p90"] for s in SPACES}
    gate1 = {
        "role": "advisory (K5: reported, never gates)",
        "pass": SKILL.within_style_novelty_pass(sims, limits),
        **{
            s: {
                "similarity": sims[s],
                "limit_p90_of_real_pairs": limits[s],
                "pass": SKILL.within_style_novelty_pass({s: sims[s]}, {s: limits[s]}),
            }
            for s in SPACES
        },
    }

    b = gate1b_pass(emb, ref_embs)
    clone = gate1b_pass({s: ref_embs[s][0] for s in SPACES}, ref_embs)
    validated = not clone["joint_pass"]
    gate1b = {
        "pass": bool(b["joint_pass"]) and validated,
        "clone_control_failed_as_required": validated,
        **{
            s: {
                "closest_reference_similarity": b[f"{s}_max_sim"],
                "limit_p90_of_real_nearest_sibling": gate1b_threshold(ref_embs[s]),
                "pass": b[f"{s}_pass"],
            }
            for s in SPACES
        },
    }
    if not validated:
        gate1b["note"] = "UNVALIDATED: an exact clone passed this check, so it cannot gate."

    floor = gate3.integrity_floor(emb["dinov2"], ref_embs["dinov2"])
    global_limit = integrity_global.global_floor()
    integrity = {
        "pass": bool(floor["max_sim"] >= global_limit),
        "closest_reference_dinov2": floor["max_sim"],
        "global_floor": global_limit,
        "per_style_floor_advisory": {"limit": floor["floor"], "pass": floor["pass"]},
    }

    gate2: dict[str, Any] = {
        "status": "not_run",
        "pass": None,
        "note": "pass include_fidelity=True to run the local judge panel (no network)",
    }
    gate3_result: dict[str, Any] = {"status": "not_run", "pass": None}
    if include_fidelity:
        briefed = changes if changes is not None else briefed_changes(style_key, concept)
        gate2, gate3_result = _local_panel(concept, style_key, briefed, floor, global_limit)

    verdict, automated = verdict_from_gates(
        {
            "gate1": gate1,
            "gate1b": gate1b,
            "integrity": integrity,
            "gate2": gate2,
            "gate3": gate3_result,
        }
    )
    return {
        "concept_path": str(concept),
        "style_key": style_key,
        "n_references": len(refs),
        "gate1": gate1,
        "gate1b": gate1b,
        "integrity": integrity,
        "gate2": gate2,
        "gate3": gate3_result,
        "human_visual_check": {
            "required": True,
            "status": "not automated",
            "note": HUMAN_CHECK_NOTE,
        },
        "automated_gates_pass": automated,
        "verdict": verdict,
    }
