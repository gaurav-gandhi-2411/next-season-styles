from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

from nss.models.backtest import generate_origin_schedule
from nss.models.lightgbm_model import build_model_frame, feature_columns
from nss.models.metrics import METRIC_KEYS
from nss.models.ranking_variants import (
    VARIANT_A_METHOD,
    VARIANT_B_METHOD,
    VARIANT_C_METHOD,
    rank_percentile_weights,
    run_variant_a_top_heavy,
    run_variant_b_two_stage,
    run_variant_c_ensemble,
    train_lightgbm_weighted,
)

_WEEK0 = date(2018, 1, 1)
_N_WEEKS = 90

# Same tiny, fast-training config as tests/test_lightgbm_model.py's `_FAST_CONFIG` -- accuracy is
# not the point of these tests, structural correctness of the weighting/staging/ensembling plumbing
# is. Duplicated here (not imported from that test module) since `tests/` files are not meant to
# import from each other -- see project convention (no existing cross-test-file imports).
_FAST_CONFIG: dict[str, int | float] = {
    "num_leaves": 7,
    "learning_rate": 0.2,
    "n_estimators": 5,
    "min_child_samples": 2,
}


def _weeks(n: int, start: date = _WEEK0) -> list[date]:
    return [start + timedelta(weeks=i) for i in range(n)]


def _synthetic_panel(n_weeks: int = _N_WEEKS) -> pl.DataFrame:
    """4 styles, full n_weeks history each -- one extra style vs.
    `tests/test_lightgbm_model.py::_synthetic_panel` (3) so a rank-percentile-weighted origin has a
    non-degenerate spread of ranks to test against."""
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
            _style("D", "G2", base=5.0),
        ]
    )


# ---------------------------------------------------------------------------
# rank_percentile_weights -- pure, deterministic weighting logic
# ---------------------------------------------------------------------------


def test_rank_percentile_weights_top_style_gets_max_weight() -> None:
    """3 styles, one origin: the highest-y_true row gets weight `1 + SCALE` (percentile 1.0), the
    lowest gets weight `1.0` (percentile 0.0), matching module docstring VARIANT (a)'s exact
    definition."""
    frame = pl.DataFrame(
        {
            "origin_week": [date(2019, 1, 1)] * 3,
            "y_true": [1.0, 5.0, 3.0],  # B(5.0) highest, A(1.0) lowest, C(3.0) middle
        }
    )

    weights = rank_percentile_weights(frame)

    assert weights[1] == max(weights)  # B: highest y_true -> max weight
    assert weights[0] == min(weights)  # A: lowest y_true -> min weight
    assert weights[0] == 1.0
    assert weights[1] == 10.0  # 1 + 9.0 * 1.0


def test_rank_percentile_weights_computed_per_origin_independently() -> None:
    """Two origins with different population sizes/scales: each origin's own min/max weight must
    still land at exactly 1.0 / 10.0, not be distorted by the other origin's rows."""
    frame = pl.DataFrame(
        {
            "origin_week": [date(2019, 1, 1)] * 2 + [date(2019, 2, 1)] * 4,
            "y_true": [1.0, 2.0, 10.0, 20.0, 30.0, 40.0],
        }
    )

    weights = rank_percentile_weights(frame)

    origin1 = weights[:2]
    origin2 = weights[2:]
    assert min(origin1) == 1.0
    assert max(origin1) == 10.0
    assert min(origin2) == 1.0
    assert max(origin2) == 10.0


def test_rank_percentile_weights_single_row_origin_does_not_divide_by_zero() -> None:
    """A degenerate 1-row origin (denom guarded to >= 1.0, see implementation) must not raise or
    produce nan/inf."""
    frame = pl.DataFrame({"origin_week": [date(2019, 1, 1)], "y_true": [5.0]})

    weights = rank_percentile_weights(frame)

    assert np.all(np.isfinite(weights))


# ---------------------------------------------------------------------------
# train_lightgbm_weighted -- smoke + determinism
# ---------------------------------------------------------------------------


def test_train_lightgbm_weighted_smoke() -> None:
    panel = _synthetic_panel()
    origins = generate_origin_schedule(panel)
    frame = build_model_frame(panel, [o.origin_week for o in origins])
    columns = feature_columns(frame)
    weights = rank_percentile_weights(frame)

    model = train_lightgbm_weighted(frame, _FAST_CONFIG, columns, weights)
    from nss.models.lightgbm_model import predict_lightgbm

    preds = predict_lightgbm(model, frame, columns)

    assert preds.shape[0] == frame.height
    assert np.all(np.isfinite(preds))


def test_train_lightgbm_weighted_is_bit_identical_across_repeated_runs() -> None:
    """Same determinism guarantee as the unweighted `train_lightgbm` (see
    `tests/test_lightgbm_model.py::test_train_lightgbm_is_bit_identical_across_repeated_runs`) --
    `sample_weight` must not reintroduce nondeterminism."""
    panel = _synthetic_panel()
    origins = generate_origin_schedule(panel)
    frame = build_model_frame(panel, [o.origin_week for o in origins])
    columns = feature_columns(frame)
    weights = rank_percentile_weights(frame)
    from nss.models.lightgbm_model import predict_lightgbm

    model1 = train_lightgbm_weighted(frame, _FAST_CONFIG, columns, weights)
    preds1 = predict_lightgbm(model1, frame, columns)
    model2 = train_lightgbm_weighted(frame, _FAST_CONFIG, columns, weights)
    preds2 = predict_lightgbm(model2, frame, columns)

    assert np.array_equal(preds1, preds2)


# ---------------------------------------------------------------------------
# run_variant_a/b/c -- end-to-end smoke on the synthetic panel + fast config
# ---------------------------------------------------------------------------


def test_run_variant_a_top_heavy_end_to_end_smoke() -> None:
    panel = _synthetic_panel()
    origins = generate_origin_schedule(panel)
    assert len(origins) >= 5  # sanity: the fixture actually produces a usable schedule

    per_origin = run_variant_a_top_heavy(panel, origins, config=_FAST_CONFIG, pool_size=3)

    assert per_origin.height == len(origins) - 3
    assert (per_origin["method"] == VARIANT_A_METHOD).all()
    for metric in METRIC_KEYS:
        assert metric in per_origin.columns


def test_run_variant_b_two_stage_end_to_end_smoke() -> None:
    panel = _synthetic_panel()
    origins = generate_origin_schedule(panel)

    per_origin = run_variant_b_two_stage(
        panel, origins, config=_FAST_CONFIG, pool_size=3, head_size=2
    )

    assert per_origin.height == len(origins) - 3
    assert (per_origin["method"] == VARIANT_B_METHOD).all()
    for metric in METRIC_KEYS:
        assert metric in per_origin.columns


def test_run_variant_c_ensemble_end_to_end_smoke() -> None:
    panel = _synthetic_panel()
    origins = generate_origin_schedule(panel)

    per_origin = run_variant_c_ensemble(
        panel,
        origins,
        lgbm_config=_FAST_CONFIG,
        lambdarank_config=_FAST_CONFIG,
        pool_size=3,
    )

    assert per_origin.height == len(origins) - 3
    assert (per_origin["method"] == VARIANT_C_METHOD).all()
    for metric in METRIC_KEYS:
        assert metric in per_origin.columns
