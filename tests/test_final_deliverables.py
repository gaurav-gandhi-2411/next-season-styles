from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from nss.generate.final_deliverables import (
    STYLE_ORDER,
    build_rationale,
    build_scores_text,
    display_name,
    main,
    select_best_attempt,
)

# A blank/near-blank PNG is a few KB; a real multi-panel composed figure at dpi=150 is well over
# 50KB -- a cheap, meaningful non-triviality floor (rule: light sanity check, not a formal
# visual-regression test), matching `tests/test_agent_architecture.py`'s convention.
MIN_NON_TRIVIAL_BYTES = 50_000


def _make_qc_df(rows: list[dict[str, object]]) -> pl.DataFrame:
    """Build a minimal synthetic `concept_qc_results.csv`-shaped frame for unit tests.

    Column dtypes are inferred from the (already homogeneous-per-column) row dicts -- no explicit
    schema needed since every row here is built via the `_row` helper below.
    """
    return pl.DataFrame(rows)


def _row(
    style_id: str,
    attempt_number: int,
    clip_in_band: bool,
    dino_in_band: bool,
    fidelity: float,
) -> dict[str, object]:
    return {
        "style_id": style_id,
        "attempt_number": attempt_number,
        "seed": 42,
        "ip_adapter_scale": 0.2,
        "image_path": f"data/generated/{style_id}_{attempt_number}.png",
        "clip_margin": 0.1,
        "clip_in_band": clip_in_band,
        "dino_margin": 0.5,
        "dino_in_band": dino_in_band,
        "mean_attribute_fidelity": fidelity,
        "n_contributing_judges": 1,
        "fidelity_pass": fidelity >= 0.75,
        "overall_pass": clip_in_band and dino_in_band and fidelity >= 0.75,
        "style_final_pass": False,
        "n_attempts_for_style": 3,
    }


def test_select_best_attempt_prefers_higher_composite_score() -> None:
    """A clip+dino in-band, high-fidelity attempt beats a low-fidelity, out-of-band one."""
    df = _make_qc_df(
        [
            _row("style-a", 0, clip_in_band=False, dino_in_band=False, fidelity=0.25),
            _row("style-a", 1, clip_in_band=True, dino_in_band=True, fidelity=1.0),
        ]
    )

    best = select_best_attempt(df, "style-a")

    assert best["attempt_number"] == 1
    assert best["composite_score"] == pytest.approx(3.0)


def test_select_best_attempt_ties_break_on_lower_attempt_number() -> None:
    """Equal composite scores across attempts resolve to the EARLIEST attempt."""
    df = _make_qc_df(
        [
            _row("style-b", 0, clip_in_band=False, dino_in_band=False, fidelity=0.25),
            _row("style-b", 1, clip_in_band=False, dino_in_band=False, fidelity=0.25),
            _row("style-b", 2, clip_in_band=False, dino_in_band=False, fidelity=0.0),
        ]
    )

    best = select_best_attempt(df, "style-b")

    assert best["attempt_number"] == 0
    assert best["composite_score"] == pytest.approx(0.25)


def test_select_best_attempt_unknown_style_raises() -> None:
    """A style_id with no rows raises ValueError rather than silently returning nothing."""
    df = _make_qc_df([_row("style-a", 0, clip_in_band=True, dino_in_band=True, fidelity=1.0)])

    with pytest.raises(ValueError, match="No QC attempts found"):
        select_best_attempt(df, "does-not-exist")


def test_display_name_known_style() -> None:
    """Every real winning style_id has a registered human-readable display name."""
    for style_id in STYLE_ORDER:
        name = display_name(style_id)
        assert name
        assert "||" not in name


def test_display_name_unknown_style_raises() -> None:
    with pytest.raises(ValueError, match="No display name registered"):
        display_name("Some || Unregistered || Style")


def test_build_rationale_reflects_brief_preserve_change() -> None:
    """The rationale text is drawn from the brief's own preserve/change fields, not invented."""
    brief = {
        "preserve": [
            "garment category: T-shirt",
            "construction/fabrication family: Jersey Basic",
            "anchor colour: Black",
            "surface treatment: Solid",
        ],
        "change": ["a subtle graphic motif"],
    }

    rationale = build_rationale(brief)

    assert "Black" in rationale
    assert "Jersey Basic" in rationale
    assert "T-shirt" in rationale
    assert "Solid" in rationale


def test_build_scores_text_reports_real_numbers_and_fail_status() -> None:
    """A failing attempt's scores block states FAIL explicitly, with the actual numbers."""
    attempt = {
        "attempt_number": 0,
        "seed": 43,
        "ip_adapter_scale": 0.2,
        "clip_margin": 0.0948,
        "clip_in_band": True,
        "dino_margin": 0.7245,
        "dino_in_band": False,
        "mean_attribute_fidelity": 0.5,
        "n_contributing_judges": 1,
        "fidelity_pass": False,
        "overall_pass": False,
        "style_final_pass": False,
        "n_attempts_for_style": 3,
        "composite_score": 1.5,
    }

    text = build_scores_text(attempt)

    assert "0.0948" in text
    assert "0.7245" in text
    assert "0.500" in text
    assert "DID NOT PASS" in text
    assert "OUT of band" in text


def test_main_writes_non_trivial_pngs(tmp_path: Path) -> None:
    """`main()` runs end-to-end against the repo's real C6/C7 artifacts and writes real PNGs."""
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
