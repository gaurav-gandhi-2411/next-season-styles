from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from nss.models.diversity_forecast import (
    apply_diversity_constraint,
    build_final_three,
    build_t2_candidate_frame,
    select_t1_incumbent,
    select_t2_emerging,
    t2_absolute_intensity_floor,
)

# ---------------------------------------------------------------------------
# apply_diversity_constraint -- pure skip logic on a hand-built, already-sorted ranking.
# ---------------------------------------------------------------------------


def _diversity_ranking() -> pl.DataFrame:
    """10 rows, already sorted best-first, with 2 deliberate collisions:
    - rank 2 (B) collides with rank 1 (A): both (T-shirt, Black).
    - rank 5 (E) collides with rank 3 (C): both (Sweater, White).
    Everything else is a distinct (product_type_name, perceived_colour_master_name) pair.
    """
    return pl.DataFrame(
        {
            "style_key": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
            "product_type_name": [
                "T-shirt",
                "T-shirt",
                "Sweater",
                "Cardigan",
                "Sweater",
                "Blazer",
                "Top",
                "Dress",
                "Skirt",
                "Jeans",
            ],
            "perceived_colour_master_name": [
                "Black",
                "Black",
                "White",
                "Black",
                "White",
                "Black",
                "Blue",
                "Red",
                "Green",
                "Black",
            ],
            "predicted_intensity": [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        }
    )


def test_apply_diversity_constraint_skips_collisions_keeps_rest() -> None:
    ranking = _diversity_ranking()

    result = apply_diversity_constraint(ranking, top_n=10)

    # B (rank 2) and E (rank 5) are skipped as pair-collisions; everything else survives.
    assert result["style_key"].to_list() == ["A", "C", "D", "F", "G", "H", "I", "J"]
    assert result["rank"].to_list() == list(range(1, result.height + 1))


def test_apply_diversity_constraint_respects_top_n_after_skipping() -> None:
    ranking = _diversity_ranking()

    result = apply_diversity_constraint(ranking, top_n=3)

    # Walk: A (keep, 1), B (skip, collides w/ A), C (keep, 2), D (keep, 3) -- stop at 3.
    assert result["style_key"].to_list() == ["A", "C", "D"]
    assert result["rank"].to_list() == [1, 2, 3]


def test_apply_diversity_constraint_returns_fewer_rows_when_pool_exhausted() -> None:
    """Diversity-compliant pool smaller than top_n -- never an error, just fewer rows."""
    ranking = _diversity_ranking().head(3)  # A, B, C -- B collides with A, so only A, C survive.

    result = apply_diversity_constraint(ranking, top_n=10)

    assert result.height == 2
    assert result["style_key"].to_list() == ["A", "C"]


def test_apply_diversity_constraint_empty_input() -> None:
    ranking = _diversity_ranking().clear()

    result = apply_diversity_constraint(ranking, top_n=10)

    assert result.height == 0
    assert "rank" in result.columns


# ---------------------------------------------------------------------------
# select_t1_incumbent -- diversity constraint composed with the guard filter + sort.
# ---------------------------------------------------------------------------


def _guarded_ranking(overrides: dict[str, list[object]] | None = None) -> pl.DataFrame:
    base = _diversity_ranking()
    n = base.height
    frame = base.with_columns(
        guard1_pass=pl.Series([True] * n),
        guard2_pass=pl.Series([True] * n),
        guard3_pass=pl.Series([True] * n),
        rank_unguarded=pl.Series(list(range(1, n + 1))),
    )
    if overrides:
        frame = frame.with_columns(**{k: pl.Series(v) for k, v in overrides.items()})
    return frame


def test_select_t1_incumbent_applies_diversity_and_drops_failing_guards() -> None:
    ranking = _guarded_ranking(
        overrides={"guard1_pass": [True, True, True, False, True, True, True, True, True, True]}
    )

    t1 = select_t1_incumbent(ranking, top_n=10)

    # D (guard1 fail) excluded; B and E excluded as diversity collisions.
    assert "D" not in t1["style_key"].to_list()
    assert "B" not in t1["style_key"].to_list()
    assert "E" not in t1["style_key"].to_list()
    assert "rank_unguarded" not in t1.columns
    # Still sorted by predicted_intensity descending.
    assert t1["predicted_intensity"].to_list() == sorted(
        t1["predicted_intensity"].to_list(), reverse=True
    )


# ---------------------------------------------------------------------------
# T2: zero-trailing-mean edge case + absolute-intensity floor.
# ---------------------------------------------------------------------------


def _t2_panel_and_ranking() -> tuple[pl.DataFrame, pl.DataFrame]:
    """3 styles at a shared forecast origin:
    - "ZERO": trailing_13w_mean_intensity == 0.0 (no recent sales) -- must be excluded from T2.
    - "GROWER": modest trailing mean, high predicted_intensity -- big growth ratio, above floor.
    - "TINY": near-zero (but > 0) trailing mean, LOW predicted_intensity (below the floor) -- huge
      growth ratio but must be excluded by the absolute-intensity floor.
    """
    origin = date(2020, 9, 21)
    weeks = [origin]
    panel = pl.DataFrame(
        {
            "style_key": ["ZERO", "GROWER", "TINY"],
            "week_start": weeks * 3,
            "units_per_active_article": [0.0, 5.0, 0.01],
        }
    )
    ranking = pl.DataFrame(
        {
            "style_key": ["ZERO", "GROWER", "TINY"],
            "product_type_name": ["A", "B", "C"],
            "perceived_colour_master_name": ["Black", "White", "Blue"],
            "predicted_intensity": [50.0, 20.0, 1.0],
            "guard1_pass": [True, True, True],
            "guard2_pass": [True, True, True],
            "guard3_pass": [True, True, True],
            "rank_unguarded": [1, 2, 3],
            "price_index_level": [1.0, 1.0, 1.0],
            "guard1_n_active_articles_trailing_mean": [20.0, 20.0, 20.0],
            "guard3_n_weeks_active_trailing": [52, 52, 52],
        }
    )
    return panel, ranking


def test_t2_absolute_intensity_floor_is_median_of_guard_passing_predicted_intensity() -> None:
    _panel, ranking = _t2_panel_and_ranking()

    floor = t2_absolute_intensity_floor(ranking)

    assert floor == pytest.approx(20.0)  # median of [50.0, 20.0, 1.0]


def test_build_t2_candidate_frame_excludes_zero_trailing_mean_style() -> None:
    """ZERO (trailing_13w_mean_intensity == 0.0) must never appear in T2 -- see module docstring
    T2 ZERO-TRAILING-MEAN HANDLING."""
    panel, ranking = _t2_panel_and_ranking()

    candidates = build_t2_candidate_frame(panel, ranking, forecast_origin=panel["week_start"][0])

    assert "ZERO" not in candidates["style_key"].to_list()


def test_build_t2_candidate_frame_excludes_below_floor_style() -> None:
    """TINY has a huge growth ratio (1.0 / 0.01 = 100x) but predicted_intensity (1.0) is below the
    floor (median = 20.0) -- must be excluded. See module docstring T2 ABSOLUTE-INTENSITY FLOOR."""
    panel, ranking = _t2_panel_and_ranking()

    candidates = build_t2_candidate_frame(panel, ranking, forecast_origin=panel["week_start"][0])

    assert "TINY" not in candidates["style_key"].to_list()
    assert candidates["style_key"].to_list() == ["GROWER"]
    row = candidates.filter(pl.col("style_key") == "GROWER")
    assert row["growth_ratio"][0] == pytest.approx(20.0 / 5.0)


def test_select_t2_emerging_end_to_end_with_zero_and_floor_edge_cases() -> None:
    panel, ranking = _t2_panel_and_ranking()

    t2 = select_t2_emerging(panel, ranking, top_n=10, forecast_origin=panel["week_start"][0])

    assert t2["style_key"].to_list() == ["GROWER"]
    assert "rank_unguarded" not in t2.columns
    assert "rank" in t2.columns


# ---------------------------------------------------------------------------
# build_final_three -- T1 rank 1 + T2 ranks 1-2, skipping a T1/T2 duplicate.
# ---------------------------------------------------------------------------


class _StubModel:
    """A no-op stand-in for the SHAP call path -- `build_final_three` only needs a model object to
    pass through to `compute_local_shap_drivers`, which we bypass via monkeypatch in this test."""


def _t1_t2_tables_with_duplicate() -> tuple[pl.DataFrame, pl.DataFrame]:
    common_cols = {
        "product_type_name": None,
        "perceived_colour_master_name": None,
        "garment_group_name": None,
        "index_group_name": None,
        "graphical_appearance_name": None,
        "price_index_level": 1.0,
        "guard1_pass": True,
        "guard1_n_active_articles_trailing_mean": 20.0,
        "guard2_pass": True,
        "guard3_pass": True,
        "guard3_n_weeks_active_trailing": 52,
    }
    t1 = pl.DataFrame(
        [
            {"style_key": "DUPE", "rank": 1, "predicted_intensity": 40.0, **common_cols},
        ]
    )
    # T2 rank 1 duplicates T1's pick (DUPE) -- must be skipped, falling through to ranks 2 and 3.
    t2 = pl.DataFrame(
        [
            {
                "style_key": "DUPE",
                "rank": 1,
                "predicted_intensity": 40.0,
                "growth_ratio": 5.0,
                **common_cols,
            },
            {
                "style_key": "SECOND",
                "rank": 2,
                "predicted_intensity": 22.0,
                "growth_ratio": 4.0,
                **common_cols,
            },
            {
                "style_key": "THIRD",
                "rank": 3,
                "predicted_intensity": 21.0,
                "growth_ratio": 3.0,
                **common_cols,
            },
        ]
    )
    return t1, t2


def test_build_final_three_skips_t2_entry_duplicating_t1_pick(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    t1, t2 = _t1_t2_tables_with_duplicate()

    def _fake_shap(
        model: object, style_keys: list[str], *_args: object, **_kwargs: object
    ) -> pl.DataFrame:
        return pl.DataFrame({"style_key": style_keys})

    monkeypatch.setattr("nss.models.diversity_forecast.compute_local_shap_drivers", _fake_shap)

    final_three = build_final_three(t1, t2, _StubModel(), pl.DataFrame(), [])

    # DUPE from T1, then SECOND + THIRD from T2 (T2's own DUPE row skipped).
    assert final_three["style_key"].to_list() == ["DUPE", "SECOND", "THIRD"]
    assert final_three["source_table"].to_list() == [
        "T1_incumbent",
        "T2_emerging",
        "T2_emerging",
    ]
    assert final_three["growth_ratio"].to_list()[0] is None
