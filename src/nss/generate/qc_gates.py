"""The shipped concept-QC gates as one callable (task L1) -- what the MCP `score_concept` tool runs.

Gate 1   within-style range: mean similarity to the style's references must be <= the p90 of
         similarity between distinct REAL articles of that style, in CLIP and DINOv2.
Gate 1b  nearest reference: the closest single reference must be <= the p90 of the real
         nearest-sibling similarity; validated live by an exact-clone control that must FAIL (a
         check that passes a clone is broken and is reported UNVALIDATED, never as a pass).
Gate 2   VLM attribute fidelity (optional here: needs a judge call): fidelity of a blind read of
         the picture against the style's VISIBLE attributes (non-visual catch-alls excluded,
         `nss.generate.fidelity`) must reach the judge's own calibrated threshold. Reported both
         with and without the excluded attributes.
Human    a mandatory visual check. It is never automated: the automatic gates passed visibly
         malformed candidates (WRITEUP s9), so a pass here is "pending human check", not "ship".

Everything reuses the functions that scored the final concepts (`within_style_benchmark`,
`gate1b_nearest_reference`, `SKILL.within_style_novelty_pass`); nothing is re-implemented.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nss.generate import clip_scoring, dino_scoring
from nss.generate.fidelity import fidelity_both
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

    Underwear uses H3's verified plain-solid references (the screened set is the lace one H3
    replaced); the Summer style uses its own screened manifest; the other styles use
    `screen_references.load_screened_references`.

    Raises:
        ValueError: no screened references exist for `style_key` (the gates are undefined without
            real same-style articles to calibrate on).
    """
    from nss.generate import h3_generate, h3_underwear_refs, screen_references, seasonal_concept

    if style_key == h3_underwear_refs.STYLE_ID:
        return h3_generate.reference_paths()
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


def score_gates(
    concept_path: str | Path, style_key: str, include_fidelity: bool = False
) -> dict[str, Any]:
    """Run Gate 1, Gate 1b (with its clone validation) and optionally Gate 2 on one image.

    Args:
        concept_path: The generated concept image.
        style_key: `" || "`-joined style key it was generated for.
        include_fidelity: Also make ONE blind Groq judge call for Gate 2 (network; a single
            reading varies by about +/-0.21, so it is indicative; a verdict needs the median of 3).

    Returns:
        A dict with `gate1`, `gate1b`, `gate2`, `human_visual_check`, `automated_gates_pass`
        (`None` until every gate has been run) and a plain `verdict` string.

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

    gate2: dict[str, Any] = {
        "status": "not_run",
        "note": "pass include_fidelity=True (one Groq call) or run `nss.generate.judge_repeat`",
    }
    if include_fidelity:
        gate2 = _fidelity(concept, style_key)

    failed = [
        name
        for name, g in (("gate1", gate1), ("gate1b", gate1b), ("gate2", gate2))
        if g.get("pass") is False
    ]
    all_run = gate2.get("pass") is not None
    if failed:
        verdict = "REJECT: failed " + ", ".join(failed)
        automated: bool | None = False
    elif all_run:
        verdict = "PASS on all automatic gates; the human visual check is still REQUIRED"
        automated = True
    else:
        verdict = "Gate 1 and Gate 1b passed; Gate 2 not run; human visual check REQUIRED"
        automated = None
    return {
        "concept_path": str(concept),
        "style_key": style_key,
        "n_references": len(refs),
        "gate1": gate1,
        "gate1b": gate1b,
        "gate2": gate2,
        "human_visual_check": {
            "required": True,
            "status": "not automated",
            "note": HUMAN_CHECK_NOTE,
        },
        "automated_gates_pass": automated,
        "verdict": verdict,
    }


def _fidelity(concept: Path, style_key: str) -> dict[str, Any]:
    """One blind Groq reading; fidelity is reported both with and without the excluded attributes.

    The judge is asked for all three attributes so BOTH figures can be computed from one call; the
    gated figure is the visual-only one (`nss.generate.fidelity`).
    """
    from nss.generate import h2_rejudge, vlm_judges
    from nss.generate.concept_qc_pipeline import parse_style_attributes

    truth_all = parse_style_attributes(style_key)
    dims = vlm_judges.ATTRIBUTE_DIMENSIONS
    truth = {d: truth_all[d] for d in dims}
    res = SKILL.run_judge(
        "groq", vlm_judges.extract_attributes_groq, concept, dims, truth, "local_sdxl"
    )
    if not res["available"]:
        return {"status": "judge_unavailable", "reason": res["excluded_reason"], "pass": None}
    both = fidelity_both(res["scores"], truth_all["graphical_treatment"])
    threshold = h2_rejudge.recompute_calibration()[2]["groq"]
    return {
        "status": "scored (1 reading)",
        "fidelity_visual_only": both["visual_only"],
        "fidelity_all_attributes": both["all_attributes"],
        "excluded_attributes": both["excluded"],
        "threshold": threshold,
        "pass": both["visual_only"] >= threshold,
        "note": "single reading varies by about +/-0.21; a verdict needs the median of 3 readings",
    }
