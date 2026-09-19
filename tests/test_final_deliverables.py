from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from nss.generate.final_deliverables import (
    STYLE_ORDER,
    build_brief_excerpt,
    build_judge_scores_text,
    build_margin_text,
    build_rationale,
    display_name,
    main,
    select_final_row,
)

# A blank/near-blank PNG is a few KB; a real multi-panel composed figure at dpi=150 is well over
# 50KB -- a cheap, meaningful non-triviality floor (rule: light sanity check, not a formal
# visual-regression test), matching `tests/test_agent_architecture.py`'s convention.
MIN_NON_TRIVIAL_BYTES = 50_000

_BRIEF = {
    "silhouette": "Relaxed, straight-body silhouette.",
    "colour_direction": "Black as the anchor colour.",
    "preserve": [
        "garment category: T-shirt",
        "construction/fabrication family: Jersey Basic",
        "anchor colour: Black",
        "surface treatment: Solid",
    ],
    "change": ["a subtle graphic motif"],
    "applied_changes": [
        "adjacent-colourway (charcoal) contrast topstitching at the crew neckline and cuffs -- "
        "trim/construction detail",
        "cropped, dropped-shoulder hem falling at hip length -- proportion tweak",
    ],
}


def _make_v3_df(rows: list[dict[str, object]]) -> pl.DataFrame:
    """Build a minimal synthetic `final_concepts_v3.csv`-shaped frame for unit tests."""
    return pl.DataFrame(rows)


def _row(
    style_id: str,
    seed: int,
    is_selected: bool,
    selection_passed: bool,
    fidelity: float,
) -> dict[str, object]:
    return {
        "style_id": style_id,
        "seed": seed,
        "retry_round": 0,
        "image_path": f"data/generated/{style_id}_{seed}.png",
        "clip_margin": 0.1,
        "clip_copy_anchor_threshold": 0.12,
        "clip_below_copy_anchor": True,
        "dino_margin": 0.7,
        "dino_copy_anchor_threshold": 0.6,
        "dino_below_copy_anchor": False,
        "copy_check_pass": False,
        "gemini_available": False,
        "gemini_mean_score": None,
        "gemini_excluded_reason": "gemini judge call failed: 429 RESOURCE_EXHAUSTED",
        "groq_available": True,
        "groq_mean_score": fidelity,
        "groq_excluded_reason": None,
        "mean_attribute_fidelity": fidelity,
        "n_contributing_judges": 1,
        "fidelity_pass": fidelity >= 0.75,
        "overall_pass": False,
        "is_selected": is_selected,
        "selection_passed": selection_passed,
        "selection_mode": "gated_pass" if selection_passed else "fallback_no_pass",
        "n_retry_rounds_used": 2,
        "all_disqualified": False,
    }


def test_select_final_row_returns_the_is_selected_row() -> None:
    """`select_final_row` returns the one row flagged `is_selected`, not any other candidate."""
    df = _make_v3_df(
        [
            _row("style-a", 42, is_selected=False, selection_passed=False, fidelity=0.2),
            _row("style-a", 45, is_selected=True, selection_passed=False, fidelity=0.4),
        ]
    )

    row = select_final_row(df, "style-a")

    assert row["seed"] == 45
    assert row["mean_attribute_fidelity"] == pytest.approx(0.4)


def test_select_final_row_no_match_raises() -> None:
    """A style_id with zero `is_selected` rows raises rather than silently returning nothing."""
    df = _make_v3_df([_row("style-a", 42, is_selected=False, selection_passed=False, fidelity=0.2)])

    with pytest.raises(ValueError, match="Expected exactly 1 is_selected row"):
        select_final_row(df, "style-a")


def test_select_final_row_multiple_matches_raises() -> None:
    """More than one `is_selected` row for the same style is a data-integrity error, not silent."""
    df = _make_v3_df(
        [
            _row("style-a", 42, is_selected=True, selection_passed=False, fidelity=0.2),
            _row("style-a", 45, is_selected=True, selection_passed=False, fidelity=0.4),
        ]
    )

    with pytest.raises(ValueError, match="Expected exactly 1 is_selected row"):
        select_final_row(df, "style-a")


def test_display_name_known_style() -> None:
    """Every real winning style_id has a registered human-readable display name."""
    for style_id in STYLE_ORDER:
        name = display_name(style_id)
        assert name
        assert "||" not in name


def test_display_name_unknown_style_raises() -> None:
    with pytest.raises(ValueError, match="No display name registered"):
        display_name("Some || Unregistered || Style")


def test_build_rationale_names_actual_applied_changes() -> None:
    """The rationale is drawn from the brief's own preserve/applied_changes fields, not invented."""
    rationale = build_rationale(_BRIEF)

    assert "Black" in rationale
    assert "Jersey Basic" in rationale
    assert "T-shirt" in rationale
    assert "charcoal" in rationale
    assert "dropped-shoulder hem" in rationale
    # the brainstormed `change` axis list must NOT leak into the rationale -- only applied_changes.
    assert "subtle graphic motif" not in rationale


def test_build_brief_excerpt_shows_kept_and_changed() -> None:
    """The evidence-chain brief excerpt separates what was preserved from what was changed."""
    excerpt = build_brief_excerpt(_BRIEF)

    assert "PRESERVED" in excerpt
    assert "CHANGED" in excerpt
    assert "Black" in excerpt
    assert "charcoal" in excerpt


def test_build_margin_text_labels_gate1_and_diagnostic_band_separately() -> None:
    """Both anchor systems appear, clearly labeled, with real numbers -- never just a checkmark."""
    row = _row("style-a", 45, is_selected=True, selection_passed=False, fidelity=0.425)

    text = build_margin_text(
        row, real_space_clip_band=(0.03, 0.10), real_space_dino_band=(0.14, 0.43)
    )

    assert "GATE 1" in text
    assert "DIAGNOSTIC ONLY" in text
    assert "0.1000" in text  # this candidate's real clip_margin, printed
    assert "[0.0300, 0.1000]" in text  # the C2 diagnostic band, printed


def test_build_judge_scores_text_states_gemini_unavailable_plainly() -> None:
    """A quota-exhausted Gemini judge is stated explicitly, not hidden behind a bare score."""
    row = _row("style-a", 45, is_selected=True, selection_passed=False, fidelity=0.425)

    text = build_judge_scores_text(row)

    assert "Gemini: UNAVAILABLE" in text
    assert "quota exhausted" in text
    assert "Single-judge" in text
    assert "0.425" in text
    assert "DID NOT PASS" in text


def test_build_judge_scores_text_reports_passing_style() -> None:
    """A passing style's QC status block says PASSED, with the real fidelity number."""
    row = _row("style-a", 45, is_selected=True, selection_passed=True, fidelity=0.9)
    row["fidelity_pass"] = True
    row["overall_pass"] = True

    text = build_judge_scores_text(row)

    assert "PASSED" in text
    assert "0.900" in text


def test_main_writes_non_trivial_pngs(tmp_path: Path) -> None:
    """`main()` runs end-to-end against the repo's real F5 artifacts and writes real PNGs."""
    hero_out = tmp_path / "FINAL_concepts.png"
    evidence_out = tmp_path / "evidence_chain.png"

    hero_result, evidence_result = main(hero_out_path=hero_out, evidence_out_path=evidence_out)

    assert hero_result == hero_out
    assert evidence_result == evidence_out
    for path in (hero_out, evidence_out):
        assert path.exists()
        assert path.stat().st_size > MIN_NON_TRIVIAL_BYTES


def test_main_creates_missing_parent_dirs(tmp_path: Path) -> None:
    """`main()` creates any missing parent directories for both output paths."""
    hero_out = tmp_path / "nested" / "dir" / "FINAL_concepts.png"
    evidence_out = tmp_path / "nested" / "dir2" / "evidence_chain.png"

    hero_result, evidence_result = main(hero_out_path=hero_out, evidence_out_path=evidence_out)

    assert hero_result.exists()
    assert evidence_result.exists()
