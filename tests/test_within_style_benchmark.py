from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from nss.generate import judge_rescore, vlm_judges
from nss.generate.underwear_refs import is_plain_description
from nss.generate.within_style_benchmark import (
    concept_similarity,
    gate1_within_style,
    style_benchmark,
)


def _unit(*xs: float) -> np.ndarray:
    v = np.array(xs, dtype=float)
    return v / np.linalg.norm(v)


def test_style_benchmark_known_values() -> None:
    """Three orthogonal-ish vectors: pair cosines are hand-computable."""
    embs = [_unit(1, 0), _unit(1, 1), _unit(0, 1)]
    b = style_benchmark(embs)
    # pairs: cos(0,1)=1/sqrt2, cos(0,2)=0, cos(1,2)=1/sqrt2 -> median 1/sqrt2
    assert b["n_pairs"] == 3
    assert b["pair_median"] == pytest.approx(1 / np.sqrt(2))
    assert b["nn_median"] == pytest.approx(1 / np.sqrt(2))


def test_style_benchmark_needs_two_articles() -> None:
    with pytest.raises(ValueError, match="2 distinct"):
        style_benchmark([_unit(1, 0)])


def test_concept_similarity_mean_and_max() -> None:
    sim = concept_similarity(_unit(1, 0), [_unit(1, 0), _unit(0, 1)])
    assert sim == {"mean": pytest.approx(0.5), "max": pytest.approx(1.0)}


def test_gate1_at_or_below_median_passes_boundary_inclusive() -> None:
    assert gate1_within_style(0.90, 0.90)
    assert gate1_within_style(0.80, 0.90)
    assert not gate1_within_style(0.9001, 0.90)


def test_within_style_gate_refuses_mismatched_spaces() -> None:
    """Hostile input: a missing benchmark space must raise, never silently pass (rule 98a)."""
    skill = vlm_judges.SKILL
    with pytest.raises(ValueError):
        skill.within_style_novelty_pass({"clip": 0.1}, {"clip": 0.9, "dinov2": 0.9})
    with pytest.raises(ValueError):
        skill.within_style_novelty_pass({}, {})
    assert not skill.within_style_novelty_pass(
        {"clip": 0.5, "dinov2": 0.95}, {"clip": 0.9, "dinov2": 0.9}
    )


def test_checklist_drops_garment_group_only() -> None:
    assert "garment_group" not in vlm_judges.ATTRIBUTE_DIMENSIONS
    assert vlm_judges.LEGACY_ATTRIBUTE_DIMENSIONS[-1] == "garment_group"
    assert vlm_judges.LEGACY_ATTRIBUTE_DIMENSIONS[:-1] == vlm_judges.ATTRIBUTE_DIMENSIONS


def test_mean_over_recomputes_from_stored_scores() -> None:
    scores = {
        "product_type": 0.85,
        "colour_family": 1.0,
        "graphical_treatment": 0.85,
        "garment_group": 0.0,
    }
    assert judge_rescore.mean_over(scores, vlm_judges.LEGACY_ATTRIBUTE_DIMENSIONS) == pytest.approx(
        0.675
    )
    assert judge_rescore.mean_over(scores, vlm_judges.ATTRIBUTE_DIMENSIONS) == pytest.approx(0.9)


def test_recompute_calibration_thresholds_move_as_expected(tmp_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for ctype, s in (
        (
            "positive",
            {
                "product_type": 1.0,
                "colour_family": 1.0,
                "graphical_treatment": 1.0,
                "garment_group": 0.0,
            },
        ),
        (
            "negative",
            {
                "product_type": 0.0,
                "colour_family": 0.0,
                "graphical_treatment": 0.0,
                "garment_group": 0.0,
            },
        ),
    ):
        rows.append(
            {
                "judge_name": "j",
                "control_type": ctype,
                "style_id_checked_against": "s",
                "image_path": "x",
                "available": True,
                "raw_extraction_json": None,
                "scores_json": json.dumps(s),
                "mean_score": sum(s.values()) / 4,
                "excluded_reason": None,
            }
        )
    p = tmp_path / "cal.csv"
    pl.DataFrame(rows).write_csv(p)
    _, old_t, new_t = judge_rescore.recompute_calibration(p)
    assert old_t["j"] == pytest.approx(0.75 * 0.75)
    assert new_t["j"] == pytest.approx(0.75 * 1.0)


def test_plain_description_screen() -> None:
    assert is_plain_description("Thong briefs in microfibre with a low waist and string back.")
    assert not is_plain_description("Hipster briefs in cotton jersey with lace trims.")
    assert not is_plain_description(None)
