from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from nss.models.backtest import Origin, generate_origin_schedule
from nss.models.lightgbm_model import (
    HYPERPARAM_GRID,
    _expanding_train_origin_weeks,
    build_model_frame,
    feature_columns,
    predict_lightgbm,
    run_lightgbm_walk_forward,
    select_hyperparameters,
    summarize_lightgbm,
    train_lightgbm,
)
from nss.models.metrics import METRIC_KEYS

_WEEK0 = date(2018, 1, 1)
_N_WEEKS = 90

# A tiny config (few trees, shallow) so tests train fast -- accuracy is not the point of these
# tests, structural correctness of the split/training/scoring plumbing is.
_FAST_CONFIG: dict[str, int | float] = {
    "num_leaves": 7,
    "learning_rate": 0.2,
    "n_estimators": 5,
    "min_child_samples": 2,
}


def _weeks(n: int, start: date = _WEEK0) -> list[date]:
    return [start + timedelta(weeks=i) for i in range(n)]


def _synthetic_panel(n_weeks: int = _N_WEEKS) -> pl.DataFrame:
    """3 styles, full n_weeks history each, enough for the real (13, 13, 4) origin schedule to
    produce several origins. Same shape/columns as `test_backtest.py`'s synthetic fixture."""
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


# ---------------------------------------------------------------------------
# _expanding_train_origin_weeks -- the causal-safety check on the SPLIT logic itself
# ---------------------------------------------------------------------------


def test_expanding_train_origin_weeks_never_includes_test_origin_or_later() -> None:
    """For every test_index, the returned training weeks are exactly origins[:test_index]'s own
    weeks -- never the test origin itself, and never any origin at or after it."""
    origins = [
        Origin(origin_week=_WEEK0 + timedelta(weeks=4 * i), has_52w_lag=False, is_covid=False)
        for i in range(10)
    ]

    for test_index in range(1, len(origins)):
        train_weeks = _expanding_train_origin_weeks(origins, test_index)
        test_week = origins[test_index].origin_week

        assert test_week not in train_weeks
        assert all(w < test_week for w in train_weeks)
        assert train_weeks == {o.origin_week for o in origins[:test_index]}


def test_expanding_train_origin_weeks_grows_with_test_index() -> None:
    """The expanding window strictly grows by one origin each step (no origin dropped)."""
    origins = [
        Origin(origin_week=_WEEK0 + timedelta(weeks=4 * i), has_52w_lag=False, is_covid=False)
        for i in range(6)
    ]

    sizes = [len(_expanding_train_origin_weeks(origins, i)) for i in range(1, len(origins))]

    assert sizes == list(range(1, len(origins)))


def test_expanding_train_origin_weeks_empty_at_the_first_index() -> None:
    origins = [
        Origin(origin_week=_WEEK0 + timedelta(weeks=4 * i), has_52w_lag=False, is_covid=False)
        for i in range(3)
    ]

    assert _expanding_train_origin_weeks(origins, 0) == set()


# ---------------------------------------------------------------------------
# build_model_frame / feature_columns
# ---------------------------------------------------------------------------


def test_build_model_frame_drops_null_target_rows() -> None:
    panel = _synthetic_panel()
    origins = generate_origin_schedule(panel)
    origin_weeks = [o.origin_week for o in origins]

    frame = build_model_frame(panel, origin_weeks)

    assert frame.height > 0
    assert frame["y_true"].null_count() == 0


def test_feature_columns_excludes_identifiers_and_target() -> None:
    panel = _synthetic_panel()
    origins = generate_origin_schedule(panel)
    frame = build_model_frame(panel, [o.origin_week for o in origins])

    cols = feature_columns(frame)

    assert "style_key" not in cols
    assert "origin_week" not in cols
    assert "y_true" not in cols
    assert "lag_1" in cols
    assert "index_group_name" in cols  # categorical attribute is a real feature, not excluded


# ---------------------------------------------------------------------------
# train_lightgbm / predict_lightgbm
# ---------------------------------------------------------------------------


def test_train_and_predict_lightgbm_smoke() -> None:
    panel = _synthetic_panel()
    origins = generate_origin_schedule(panel)
    frame = build_model_frame(panel, [o.origin_week for o in origins])
    columns = feature_columns(frame)

    model = train_lightgbm(frame, _FAST_CONFIG, columns)
    preds = predict_lightgbm(model, frame, columns)

    assert preds.shape[0] == frame.height
    assert np.all(np.isfinite(preds))
    assert len(np.unique(preds)) > 1  # not degenerate/constant


def test_train_lightgbm_is_bit_identical_across_repeated_runs() -> None:
    """See module docstring DETERMINISM (A5): `random_state` alone was observed NOT to make
    repeated training runs bit-identical. `LGBM_DETERMINISM_PARAMS` (`deterministic=True`,
    `force_row_wise=True`, `num_threads=1`, plus explicit `bagging_seed`/`feature_fraction_seed`/
    `data_random_seed`) must make two trains on the SAME data produce EXACTLY the same predictions,
    not merely close ones -- `np.array_equal`, not `np.allclose`."""
    panel = _synthetic_panel()
    origins = generate_origin_schedule(panel)
    frame = build_model_frame(panel, [o.origin_week for o in origins])
    columns = feature_columns(frame)

    model1 = train_lightgbm(frame, _FAST_CONFIG, columns)
    preds1 = predict_lightgbm(model1, frame, columns)
    model2 = train_lightgbm(frame, _FAST_CONFIG, columns)
    preds2 = predict_lightgbm(model2, frame, columns)

    assert np.array_equal(preds1, preds2)


# ---------------------------------------------------------------------------
# select_hyperparameters
# ---------------------------------------------------------------------------


def test_select_hyperparameters_tries_every_grid_config_and_picks_one_of_them() -> None:
    panel = _synthetic_panel()
    origins = generate_origin_schedule(panel)
    frame = build_model_frame(panel, [o.origin_week for o in origins])
    columns = feature_columns(frame)
    pool_origins = origins[:4]

    result = select_hyperparameters(frame, pool_origins, columns)

    assert len(result.all_results) == len(HYPERPARAM_GRID)
    assert result.best_config in HYPERPARAM_GRID
    assert all(r["val_rmse"] >= 0.0 for r in result.all_results)


def test_select_hyperparameters_requires_at_least_two_pool_origins() -> None:
    panel = _synthetic_panel()
    origins = generate_origin_schedule(panel)
    frame = build_model_frame(panel, [o.origin_week for o in origins])
    columns = feature_columns(frame)

    with pytest.raises(ValueError, match="at least 2 pool origins"):
        select_hyperparameters(frame, origins[:1], columns)


# ---------------------------------------------------------------------------
# run_lightgbm_walk_forward / summarize_lightgbm -- end-to-end smoke
# ---------------------------------------------------------------------------


def test_run_lightgbm_walk_forward_and_summarize_end_to_end() -> None:
    panel = _synthetic_panel()
    origins = generate_origin_schedule(panel)
    assert len(origins) >= 5  # sanity: the fixture actually produces a usable schedule

    per_origin, config_used, grid_results = run_lightgbm_walk_forward(
        panel, origins, pool_size=3, config=_FAST_CONFIG
    )

    n_test_origins = len(origins) - 3
    assert per_origin.height == n_test_origins
    assert grid_results == []  # config was passed in directly -- no search should have run
    assert config_used == _FAST_CONFIG
    assert set(per_origin["method"].unique().to_list()) == {"lightgbm"}
    for metric in METRIC_KEYS:
        assert metric in per_origin.columns

    summary = summarize_lightgbm(per_origin)
    assert summary.height == 3  # pooled / covid / non_covid
    assert set(summary["method"].unique().to_list()) == {"lightgbm"}
    for metric in METRIC_KEYS:
        assert f"{metric}_mean" in summary.columns
        assert f"{metric}_ci_low" in summary.columns
        assert f"{metric}_ci_high" in summary.columns


def test_run_lightgbm_walk_forward_runs_hyperparameter_search_when_config_omitted() -> None:
    panel = _synthetic_panel()
    origins = generate_origin_schedule(panel)

    _per_origin, config_used, grid_results = run_lightgbm_walk_forward(panel, origins, pool_size=3)

    assert len(grid_results) == len(HYPERPARAM_GRID)
    assert config_used in HYPERPARAM_GRID


def test_run_lightgbm_walk_forward_rejects_pool_size_leaving_no_test_origins() -> None:
    panel = _synthetic_panel()
    origins = generate_origin_schedule(panel)

    with pytest.raises(ValueError, match="pool_size"):
        run_lightgbm_walk_forward(panel, origins, pool_size=len(origins), config=_FAST_CONFIG)
