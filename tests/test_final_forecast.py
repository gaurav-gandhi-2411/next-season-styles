from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from nss.models.backtest import generate_origin_schedule
from nss.models.final_forecast import (
    GUARD1_WINDOW_WEEKS,
    GUARD3_MIN_WEEKS_ACTIVE,
    GUARD3_WINDOW_WEEKS,
    SEASON_MONTHS,
    build_forecast_frame,
    build_guard_frame,
    build_ranking_frame,
    build_seasonal_ranking,
    compute_local_shap_drivers,
    final_training_origin_weeks,
    predict_intensity,
    select_markdown_excluded,
    select_top_styles,
    train_final_model,
    verify_forecast_origin,
)
from nss.models.lightgbm_model import (
    build_model_frame,
    feature_columns,
    predict_lightgbm,
    train_lightgbm,
)

_WEEK0 = date(2018, 1, 1)
_N_WEEKS = 90

# A tiny config so end-to-end tests train fast -- accuracy is not the point, structural
# correctness of the guard/ranking/SHAP plumbing is (same rationale as
# tests/test_lightgbm_model.py's _FAST_CONFIG).
_FAST_CONFIG: dict[str, int | float] = {
    "num_leaves": 7,
    "learning_rate": 0.2,
    "n_estimators": 5,
    "min_child_samples": 2,
}


def _weeks(n: int, start: date = _WEEK0) -> list[date]:
    return [start + timedelta(weeks=i) for i in range(n)]


def _synthetic_panel(n_weeks: int = _N_WEEKS) -> pl.DataFrame:
    """3 styles, full n_weeks history each -- same shape as test_lightgbm_model.py's fixture."""
    weeks = _weeks(n_weeks)
    first_seen = weeks[0]
    last_seen = weeks[-1]

    def _style(style_key: str, index_group: str, base: float) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "style_key": [style_key] * n_weeks,
                "index_group_name": [index_group] * n_weeks,
                "product_type_name": ["Trousers"] * n_weeks,
                "garment_group_name": ["GG"] * n_weeks,
                "perceived_colour_master_name": ["Black"] * n_weeks,
                "graphical_appearance_name": ["Solid"] * n_weeks,
                "week_start": weeks,
                "first_week_seen": [first_seen] * n_weeks,
                "last_week_seen": [last_seen] * n_weeks,
                "units": [float(base + i) for i in range(n_weeks)],
                "n_active_articles": [float(1 + (i % 5)) for i in range(n_weeks)],
                "price_index": [1.0 + 0.01 * i for i in range(n_weeks)],
                "intensity_shrunk": [base + 0.5 * i for i in range(n_weeks)],
                "units_per_active_article": [base + 0.3 * i for i in range(n_weeks)],
            }
        )

    return pl.concat(
        [
            _style("A", "G1", base=10.0),
            _style("B", "G1", base=100.0),
            _style("C", "G2", base=50.0),
        ]
    )


def _train_fast_model(panel: pl.DataFrame) -> tuple[object, list[str]]:
    origins = generate_origin_schedule(panel)
    frame = build_model_frame(panel, [o.origin_week for o in origins])
    columns = feature_columns(frame)
    model = train_lightgbm(frame, _FAST_CONFIG, columns)
    return model, columns


# ---------------------------------------------------------------------------
# final_training_origin_weeks / verify_forecast_origin
# ---------------------------------------------------------------------------


def test_final_training_origin_weeks_is_weekly_from_panel_start() -> None:
    """The WIDE training set steps weekly (not 4-weekly) and starts at the panel's first week
    (no burn-in) -- see module docstring TRAINING DATA WIDTH."""
    panel = _synthetic_panel()

    origin_weeks = final_training_origin_weeks(panel)

    assert origin_weeks[0] == _WEEK0
    assert origin_weeks[1] - origin_weeks[0] == timedelta(weeks=1)
    # Same schedule as generate_origin_schedule(step_weeks=1, burn_in_weeks=0).
    expected = [
        o.origin_week for o in generate_origin_schedule(panel, step_weeks=1, burn_in_weeks=0)
    ]
    assert origin_weeks == expected


def test_verify_forecast_origin_passes_on_match() -> None:
    panel = _synthetic_panel()
    max_week = panel["week_start"].max()
    verify_forecast_origin(panel, forecast_origin=max_week)  # must not raise


def test_verify_forecast_origin_raises_on_mismatch() -> None:
    panel = _synthetic_panel()
    wrong_origin = panel["week_start"].max() - timedelta(weeks=1)

    with pytest.raises(ValueError, match="does not match"):
        verify_forecast_origin(panel, forecast_origin=wrong_origin)


def test_train_final_model_trains_on_the_wide_origin_set() -> None:
    """`train_final_model` pools rows from EVERY wide-set origin, not just the narrow
    4-week-spaced backtest schedule -- more rows than `build_model_frame` over the narrow
    schedule would produce for the same panel."""
    panel = _synthetic_panel()

    _model, model_frame, columns = train_final_model(panel)

    narrow_origins = generate_origin_schedule(panel)
    narrow_frame = build_model_frame(panel, [o.origin_week for o in narrow_origins])
    assert model_frame.height > narrow_frame.height
    assert "style_key" not in columns
    assert "y_true" not in columns


# ---------------------------------------------------------------------------
# Guard windows (build_guard_frame) -- hand-constructed pass/fail cases
# ---------------------------------------------------------------------------


def test_build_guard_frame_guard1_trailing_mean_is_inclusive_and_windowed() -> None:
    """Guard 1's trailing mean is over the last GUARD1_WINDOW_WEEKS weeks, INCLUSIVE of the
    origin row itself (see module docstring GUARD WINDOW CONVENTION)."""
    n_weeks = GUARD1_WINDOW_WEEKS + 5
    weeks = _weeks(n_weeks)
    # First 5 weeks have n_active_articles=1 (would drag the mean below 10 if included); the
    # trailing 13 weeks (indices 5..17) all have n_active_articles=20.
    n_active = [1.0] * 5 + [20.0] * (n_weeks - 5)
    panel = pl.DataFrame(
        {
            "style_key": ["A"] * n_weeks,
            "week_start": weeks,
            "units": [5.0] * n_weeks,
            "n_active_articles": n_active,
        }
    )
    origin = weeks[-1]

    guard_frame = build_guard_frame(panel, forecast_origin=origin)

    row = guard_frame.filter(pl.col("style_key") == "A")
    assert row["guard1_n_active_articles_trailing_mean"][0] == pytest.approx(20.0)


def test_build_guard_frame_guard3_counts_active_weeks_in_trailing_window() -> None:
    """Guard 3 counts weeks with units > 0 in the trailing GUARD3_WINDOW_WEEKS window."""
    n_weeks = GUARD3_WINDOW_WEEKS
    weeks = _weeks(n_weeks)
    # Exactly 26 active (units>0) weeks, alternating -- style should sit right at the guard 3
    # threshold (GUARD3_MIN_WEEKS_ACTIVE = 26).
    units = [10.0 if i % 2 == 0 else 0.0 for i in range(n_weeks)]
    n_active_weeks = sum(1 for u in units if u > 0)
    assert n_active_weeks == GUARD3_MIN_WEEKS_ACTIVE  # sanity on the fixture itself

    panel = pl.DataFrame(
        {
            "style_key": ["A"] * n_weeks,
            "week_start": weeks,
            "units": units,
            "n_active_articles": [10.0] * n_weeks,
        }
    )
    origin = weeks[-1]

    guard_frame = build_guard_frame(panel, forecast_origin=origin)

    row = guard_frame.filter(pl.col("style_key") == "A")
    assert row["guard3_n_weeks_active_trailing"][0] == GUARD3_MIN_WEEKS_ACTIVE


# ---------------------------------------------------------------------------
# select_top_styles / select_markdown_excluded -- pure guard-selection logic,
# hand-constructed synthetic styles that should pass/fail each guard individually.
# ---------------------------------------------------------------------------


def _ranking_row(
    style_key: str,
    predicted_intensity: float,
    guard1_pass: bool,
    guard2_pass: bool,
    guard3_pass: bool,
    rank_unguarded: int,
) -> dict[str, object]:
    return {
        "style_key": style_key,
        "index_group_name": "G1",
        "product_type_name": "Trousers",
        "garment_group_name": "GG",
        "perceived_colour_master_name": "Black",
        "graphical_appearance_name": "Solid",
        "price_index_level": 1.0 if guard2_pass else 0.5,
        "predicted_intensity": predicted_intensity,
        "guard1_n_active_articles_trailing_mean": 20.0 if guard1_pass else 1.0,
        "guard3_n_weeks_active_trailing": 30 if guard3_pass else 5,
        "guard1_pass": guard1_pass,
        "guard2_pass": guard2_pass,
        "guard3_pass": guard3_pass,
        "rank_unguarded": rank_unguarded,
    }


def _hand_built_ranking() -> pl.DataFrame:
    """10 styles, ranked 1..10 by predicted_intensity (descending), covering every individual
    guard pass/fail combination the selection rule needs to distinguish:
      - style_all_pass: passes all 3 guards (should reach the guarded top list)
      - style_fail_guard1_only: fails guard 1 only (simply excluded, no special table)
      - style_fail_guard2_only: fails guard 2 only (markdown-driven-excluded table)
      - style_fail_guard3_only: fails guard 3 only (simply excluded, no special table)
      - style_fail_guard1_and_guard2: fails both (simply excluded -- NOT guard-2-specific)
      - remaining 5 all-pass styles at lower predicted_intensity, to exercise top-N clipping.
    """
    rows = [
        _ranking_row("style_all_pass_1", 100.0, True, True, True, 1),
        _ranking_row("style_fail_guard1_only", 95.0, False, True, True, 2),
        _ranking_row("style_fail_guard2_only", 90.0, True, False, True, 3),
        _ranking_row("style_fail_guard3_only", 85.0, True, True, False, 4),
        _ranking_row("style_fail_guard1_and_guard2", 80.0, False, False, True, 5),
        _ranking_row("style_all_pass_2", 75.0, True, True, True, 6),
        _ranking_row("style_all_pass_3", 70.0, True, True, True, 7),
        _ranking_row("style_all_pass_4", 65.0, True, True, True, 8),
        _ranking_row("style_all_pass_5", 60.0, True, True, True, 9),
        _ranking_row("style_all_pass_6", 55.0, True, True, True, 10),
    ]
    return pl.DataFrame(rows)


def test_select_top_styles_keeps_only_all_three_guards_passing() -> None:
    ranking = _hand_built_ranking()

    top = select_top_styles(ranking, top_n=10)

    assert set(top["style_key"].to_list()) == {
        "style_all_pass_1",
        "style_all_pass_2",
        "style_all_pass_3",
        "style_all_pass_4",
        "style_all_pass_5",
        "style_all_pass_6",
    }
    # Ranked 1..N by predicted_intensity descending among the passing styles.
    assert top["rank"].to_list() == list(range(1, top.height + 1))
    assert top.sort("rank")["predicted_intensity"].to_list() == sorted(
        top["predicted_intensity"].to_list(), reverse=True
    )


def test_select_top_styles_respects_top_n() -> None:
    ranking = _hand_built_ranking()

    top = select_top_styles(ranking, top_n=3)

    assert top.height == 3
    assert top["style_key"].to_list() == [
        "style_all_pass_1",
        "style_all_pass_2",
        "style_all_pass_3",
    ]


def test_select_markdown_excluded_includes_only_guard2_specific_failures() -> None:
    ranking = _hand_built_ranking()

    excluded = select_markdown_excluded(ranking, top_n=10)

    # Only the style that fails guard 2 SPECIFICALLY (passes guards 1 and 3) appears.
    assert excluded["style_key"].to_list() == ["style_fail_guard2_only"]
    assert excluded["rank_if_included"][0] == 3


def test_select_markdown_excluded_excludes_multi_guard_failures() -> None:
    """A style failing guard 2 AND another guard is simply excluded, not in the markdown table --
    see module docstring."""
    ranking = _hand_built_ranking()

    excluded = select_markdown_excluded(ranking, top_n=10)

    assert "style_fail_guard1_and_guard2" not in excluded["style_key"].to_list()
    assert "style_fail_guard1_only" not in excluded["style_key"].to_list()
    assert "style_fail_guard3_only" not in excluded["style_key"].to_list()


def test_select_markdown_excluded_respects_top_n_window() -> None:
    """A guard-2-only failure ranked OUTSIDE the raw top N is not included."""
    ranking = _hand_built_ranking()

    excluded = select_markdown_excluded(ranking, top_n=2)  # style_fail_guard2_only is rank 3

    assert excluded.height == 0


# ---------------------------------------------------------------------------
# predict_intensity -- expm1(log1p prediction)
# ---------------------------------------------------------------------------


def test_predict_intensity_is_expm1_of_log_prediction() -> None:
    panel = _synthetic_panel()
    model, columns = _train_fast_model(panel)
    forecast_frame = build_forecast_frame(panel, forecast_origin=panel["week_start"].max())

    predicted = predict_intensity(model, forecast_frame, columns)
    log_preds = predict_lightgbm(model, forecast_frame, columns)

    assert np.all(np.isfinite(predicted))
    np.testing.assert_allclose(predicted, np.expm1(log_preds))


# ---------------------------------------------------------------------------
# build_ranking_frame -- end-to-end smoke
# ---------------------------------------------------------------------------


def test_build_ranking_frame_end_to_end_smoke() -> None:
    panel = _synthetic_panel()
    forecast_origin = panel["week_start"].max()
    model, columns = _train_fast_model(panel)

    ranking = build_ranking_frame(panel, model, columns, forecast_origin=forecast_origin)

    assert set(ranking["style_key"].to_list()) == {"A", "B", "C"}
    assert ranking["rank_unguarded"].to_list() == sorted(ranking["rank_unguarded"].to_list())
    # Sorted descending by predicted_intensity, rank_unguarded ascending.
    sorted_by_pred = ranking.sort("predicted_intensity", descending=True)
    assert ranking["rank_unguarded"].to_list() == sorted_by_pred["rank_unguarded"].to_list()
    for col in ("guard1_pass", "guard2_pass", "guard3_pass"):
        assert ranking[col].dtype == pl.Boolean


# ---------------------------------------------------------------------------
# compute_local_shap_drivers
# ---------------------------------------------------------------------------


def test_compute_local_shap_drivers_preserves_order_and_top_n() -> None:
    panel = _synthetic_panel()
    forecast_origin = panel["week_start"].max()
    model, columns = _train_fast_model(panel)
    forecast_frame = build_forecast_frame(panel, forecast_origin=forecast_origin)

    shap_df = compute_local_shap_drivers(model, ["C", "A", "B"], forecast_frame, columns, top_n=3)

    assert shap_df["style_key"].to_list() == ["C", "A", "B"]
    for i in range(1, 4):
        assert f"shap_driver_{i}_feature" in shap_df.columns
        assert f"shap_driver_{i}_value" in shap_df.columns
    assert "shap_driver_4_feature" not in shap_df.columns


# ---------------------------------------------------------------------------
# build_seasonal_ranking
# ---------------------------------------------------------------------------


def test_build_seasonal_ranking_surfaces_the_seasonally_strong_style() -> None:
    """A style with elevated units_per_active_article ONLY in summer weeks, across multiple
    years, should win the summer season's top rank."""
    n_years = 3
    weeks = _weeks(n_years * 52, start=date(2016, 1, 4))  # Monday
    n_weeks = len(weeks)
    first_seen = weeks[0]
    last_seen = weeks[-1]

    def _summer_boosted_row(w: date) -> float:
        return 200.0 if w.month in SEASON_MONTHS["summer"] else 20.0

    summer_style = pl.DataFrame(
        {
            "style_key": ["SUMMER_STAR"] * n_weeks,
            "index_group_name": ["Ladieswear"] * n_weeks,
            "product_type_name": ["Swimwear"] * n_weeks,
            "garment_group_name": ["Swimwear"] * n_weeks,
            "perceived_colour_master_name": ["Black"] * n_weeks,
            "graphical_appearance_name": ["Solid"] * n_weeks,
            "week_start": weeks,
            "first_week_seen": [first_seen] * n_weeks,
            "last_week_seen": [last_seen] * n_weeks,
            "units": [50.0] * n_weeks,
            "n_active_articles": [15.0] * n_weeks,
            "units_per_active_article": [_summer_boosted_row(w) for w in weeks],
        }
    )
    flat_style = pl.DataFrame(
        {
            "style_key": ["FLAT"] * n_weeks,
            "index_group_name": ["Ladieswear"] * n_weeks,
            "product_type_name": ["Trousers"] * n_weeks,
            "garment_group_name": ["GG"] * n_weeks,
            "perceived_colour_master_name": ["Black"] * n_weeks,
            "graphical_appearance_name": ["Solid"] * n_weeks,
            "week_start": weeks,
            "first_week_seen": [first_seen] * n_weeks,
            "last_week_seen": [last_seen] * n_weeks,
            "units": [50.0] * n_weeks,
            "n_active_articles": [15.0] * n_weeks,
            "units_per_active_article": [30.0] * n_weeks,
        }
    )
    panel = pl.concat([summer_style, flat_style])

    # A minimal "ranking" frame standing in for build_ranking_frame's output -- both styles pass
    # all 3 (non-seasonal) guards, matching a real ranking frame's schema.
    ranking = pl.DataFrame(
        {
            "style_key": ["SUMMER_STAR", "FLAT"],
            "index_group_name": ["Ladieswear", "Ladieswear"],
            "product_type_name": ["Swimwear", "Trousers"],
            "garment_group_name": ["Swimwear", "GG"],
            "perceived_colour_master_name": ["Black", "Black"],
            "graphical_appearance_name": ["Solid", "Solid"],
            "price_index_level": [1.0, 1.0],
            "guard2_pass": [True, True],
            "guard3_n_weeks_active_trailing": [52, 52],
            "guard3_pass": [True, True],
        }
    )

    seasonal = build_seasonal_ranking(panel, ranking, top_n=1)

    summer_row = seasonal.filter(pl.col("season") == "summer")
    assert summer_row["style_key"][0] == "SUMMER_STAR"
    winter_row = seasonal.filter(pl.col("season") == "winter")
    assert winter_row["style_key"][0] == "FLAT"
