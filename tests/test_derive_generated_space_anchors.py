from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from nss.generate.derive_generated_space_anchors import (
    ALL_STYLES_POOLED_KEY,
    COPY_ANCHOR,
    UNRELATED_ANCHOR,
    UNRELATED_GARMENT_DESCRIPTIONS,
    build_copy_anchor_prompt,
    build_gap_table,
    build_unrelated_anchor_prompt,
    compute_gap,
    load_realspace_anchor_summary,
    summarize_margins_by_style_and_pooled,
)

_TSHIRT_STYLE = "Ladieswear || T-shirt || Jersey Basic || Black || Solid"


def test_build_copy_anchor_prompt_is_a_literal_restatement() -> None:
    """Matches the exact wording task E1 gives as the worked example for the T-shirt style."""
    assert build_copy_anchor_prompt(_TSHIRT_STYLE) == (
        "a black solid jersey basic t-shirt, product photography, plain background"
    )


def test_build_unrelated_anchor_prompt_uses_a_different_garment() -> None:
    prompt = build_unrelated_anchor_prompt(_TSHIRT_STYLE)
    assert prompt == "a pair of denim trousers, product photography, plain background"
    assert "t-shirt" not in prompt


def test_build_unrelated_anchor_prompt_has_no_doubled_article() -> None:
    """Regression test: UNRELATED_GARMENT_DESCRIPTIONS values already carry their own leading
    article -- the prompt builder must not prepend a second one ("a a pair of ...")."""
    for style_key in UNRELATED_GARMENT_DESCRIPTIONS:
        assert not build_unrelated_anchor_prompt(style_key).startswith("a a ")


def test_unrelated_garment_descriptions_cover_all_three_final_styles_distinctly() -> None:
    """Every final-three style has an unrelated-garment override, and all 3 are mutually distinct
    garment categories (no accidental overlap between the swap-in garments)."""
    assert len(UNRELATED_GARMENT_DESCRIPTIONS) == 3
    assert len(set(UNRELATED_GARMENT_DESCRIPTIONS.values())) == 3


def test_summarize_margins_by_style_and_pooled_hand_computed() -> None:
    """2 styles x 1 anchor type x 1 embedding space, 2 margins each -> per-style means plus a
    pooled row averaging all 4 values together."""
    scored_rows = [
        {
            "style_key": "style_a",
            "anchor_type": COPY_ANCHOR,
            "embedding_space": "clip",
            "margin": 0.2,
        },
        {
            "style_key": "style_a",
            "anchor_type": COPY_ANCHOR,
            "embedding_space": "clip",
            "margin": 0.4,
        },
        {
            "style_key": "style_b",
            "anchor_type": COPY_ANCHOR,
            "embedding_space": "clip",
            "margin": 0.6,
        },
        {
            "style_key": "style_b",
            "anchor_type": COPY_ANCHOR,
            "embedding_space": "clip",
            "margin": 0.8,
        },
    ]

    results = summarize_margins_by_style_and_pooled(scored_rows)
    by_key = {(r["style_key"], r["anchor_type"], r["embedding_space"]): r for r in results}

    assert by_key[("style_a", COPY_ANCHOR, "clip")]["mean"] == pytest.approx(0.3)
    assert by_key[("style_a", COPY_ANCHOR, "clip")]["n"] == 2
    assert by_key[("style_b", COPY_ANCHOR, "clip")]["mean"] == pytest.approx(0.7)
    pooled = by_key[(ALL_STYLES_POOLED_KEY, COPY_ANCHOR, "clip")]
    assert pooled["mean"] == pytest.approx(0.5)
    assert pooled["n"] == 4


def test_compute_gap_hand_computed() -> None:
    assert compute_gap(genspace_mean=0.10, realspace_mean=0.13) == pytest.approx(-0.03)
    assert compute_gap(genspace_mean=0.13, realspace_mean=0.13) == pytest.approx(0.0)


def test_build_gap_table_joins_copy_to_upper_and_unrelated_to_lower() -> None:
    genspace_summaries = [
        {
            "style_key": ALL_STYLES_POOLED_KEY,
            "anchor_type": COPY_ANCHOR,
            "embedding_space": "clip",
            "n": 9,
            "mean": 0.10,
            "median": 0.10,
            "std": 0.01,
        },
        {
            "style_key": ALL_STYLES_POOLED_KEY,
            "anchor_type": UNRELATED_ANCHOR,
            "embedding_space": "clip",
            "n": 9,
            "mean": 0.02,
            "median": 0.02,
            "std": 0.01,
        },
    ]
    realspace_summaries = {
        "clip": {
            "upper_anchor": {"mean": 0.13, "std": 0.04, "n": 183},
            "lower_anchor": {"mean": -0.02, "std": 0.01, "n": 8},
        }
    }

    gap_df = build_gap_table(genspace_summaries, realspace_summaries)
    rows = {r["anchor_type"]: r for r in gap_df.iter_rows(named=True)}

    assert rows[COPY_ANCHOR]["realspace_row_type"] == "upper_anchor"
    assert rows[COPY_ANCHOR]["realspace_mean"] == pytest.approx(0.13)
    assert rows[COPY_ANCHOR]["gap"] == pytest.approx(0.10 - 0.13)

    assert rows[UNRELATED_ANCHOR]["realspace_row_type"] == "lower_anchor"
    assert rows[UNRELATED_ANCHOR]["realspace_mean"] == pytest.approx(-0.02)
    assert rows[UNRELATED_ANCHOR]["gap"] == pytest.approx(0.02 - (-0.02))


def test_load_realspace_anchor_summary_reads_mean_std_n(tmp_path: Path) -> None:
    path = tmp_path / "margin_anchors_clip.csv"
    pl.DataFrame(
        [
            {"row_type": "upper_anchor", "n": 183, "mean": 0.13, "median": 0.136, "std": 0.039},
            {"row_type": "lower_anchor", "n": 8, "mean": -0.016, "median": -0.015, "std": 0.013},
        ]
    ).write_csv(path)

    result = load_realspace_anchor_summary(path)

    assert result["upper_anchor"] == {
        "mean": pytest.approx(0.13),
        "std": pytest.approx(0.039),
        "n": 183,
    }
    assert result["lower_anchor"]["n"] == 8
