from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from nss.models.backtest import generate_origin_schedule
from nss.models.lambdarank_model import (
    LAMBDARANK_TRUNCATION_CANDIDATES,
    RELEVANCE_GRADE_BANDS,
    add_relevance_grades,
    compute_relevance_grades,
    predict_lambdarank,
    run_lambdarank_walk_forward,
    select_truncation_level,
    summarize_lambdarank,
    train_lambdarank,
)
from nss.models.lightgbm_model import build_model_frame, feature_columns
from nss.models.metrics import METRIC_KEYS

_WEEK0 = date(2018, 1, 1)
_N_WEEKS = 90

# Same tiny config as tests/test_lightgbm_model.py's _FAST_CONFIG -- fast to train, not the point
# of these tests (structural correctness of the split/training/scoring plumbing is).
_FAST_CONFIG: dict[str, int | float] = {
    "num_leaves": 7,
    "learning_rate": 0.2,
    "n_estimators": 5,
    "min_child_samples": 2,
}


def _weeks(n: int, start: date = _WEEK0) -> list[date]:
    return [start + timedelta(weeks=i) for i in range(n)]


def _synthetic_panel(n_weeks: int = _N_WEEKS) -> pl.DataFrame:
    """3 styles, full n_weeks history each -- same fixture shape as test_lightgbm_model.py's."""
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
# compute_relevance_grades -- hand-verifiable
# ---------------------------------------------------------------------------


def test_compute_relevance_grades_hand_verified_bands() -> None:
    """250 distinct descending values -> exactly 3/7/40/150/50 styles in grades 4/3/2/1/0, per
    `RELEVANCE_GRADE_BANDS` = ((200, 1), (50, 2), (10, 3), (3, 4))."""
    y_true = np.arange(250, 0, -1, dtype=float)  # y_true[0]=250 (best) .. y_true[249]=1 (worst)

    grades = compute_relevance_grades(y_true)

    assert list(grades[:3]) == [4, 4, 4]  # true rank 1-3
    assert list(grades[3:10]) == [3] * 7  # true rank 4-10
    assert list(grades[10:50]) == [2] * 40  # true rank 11-50
    assert list(grades[50:200]) == [1] * 150  # true rank 51-200
    assert list(grades[200:250]) == [0] * 50  # true rank 201-250
    counts = {g: int((grades == g).sum()) for g in (0, 1, 2, 3, 4)}
    assert counts == {0: 50, 1: 150, 2: 40, 3: 7, 4: 3}


def test_compute_relevance_grades_ties_broken_by_stable_original_order() -> None:
    """Equal y_true values are ranked by original row order (earlier index wins the lower/better
    rank) -- the same `kind="stable"` convention `nss.models.metrics` uses elsewhere."""
    y_true = np.array([5.0, 5.0, 5.0, 1.0])  # 3-way tie for rank 1/2/3

    grades = compute_relevance_grades(y_true)

    # All 3 tied values fall within the top-3 band regardless of tie order -> all grade 4.
    assert list(grades[:3]) == [4, 4, 4]
    assert grades[3] == RELEVANCE_GRADE_BANDS[-2][1]  # rank 4 falls in the (10, 3) band -> grade 3


def test_compute_relevance_grades_small_n_all_within_top_band() -> None:
    """With fewer styles than the tightest cutoff, every style can still land in the top grade."""
    y_true = np.array([3.0, 2.0, 1.0])

    grades = compute_relevance_grades(y_true)

    assert list(grades) == [4, 4, 4]


def test_add_relevance_grades_matches_compute_relevance_grades_per_origin() -> None:
    """The polars `.over("origin_week")` implementation must agree with the numpy per-origin
    reference implementation, including tie-breaking, for a frame spanning multiple origins."""
    rng = np.random.default_rng(42)
    n_per_origin = 30
    origin_weeks = [_WEEK0, _WEEK0 + timedelta(weeks=4)]
    frames = []
    for ow in origin_weeks:
        y = rng.integers(0, 10, size=n_per_origin).astype(float)  # small range -> forces ties
        frames.append(
            pl.DataFrame(
                {
                    "style_key": [f"s{i}" for i in range(n_per_origin)],
                    "origin_week": [ow] * n_per_origin,
                    "y_true": y,
                }
            )
        )
    frame = pl.concat(frames)

    graded = add_relevance_grades(frame)

    for ow in origin_weeks:
        # Deliberately NOT re-sorted by style_key: "s10" < "s2" lexicographically, so a string
        # sort would silently permute rows relative to the ORIGINAL insertion order the reference
        # numpy function below assumes -- filtering alone preserves that original row order.
        sub = graded.filter(pl.col("origin_week") == ow)
        expected = compute_relevance_grades(sub["y_true"].to_numpy())
        assert list(sub["relevance_grade"]) == list(expected)


# ---------------------------------------------------------------------------
# train_lambdarank / predict_lambdarank
# ---------------------------------------------------------------------------


def _graded_frame_and_columns(panel: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    origins = generate_origin_schedule(panel)
    frame = build_model_frame(panel, [o.origin_week for o in origins])
    columns = feature_columns(frame)
    graded = add_relevance_grades(frame)
    return graded, columns


def _synthetic_panel_wide(n_weeks: int = _N_WEEKS) -> pl.DataFrame:
    """10 styles (vs. `_synthetic_panel`'s 3) -- with only 3 styles per origin, EVERY style falls
    in the true top-3 (`RELEVANCE_GRADE_BANDS`'s tightest band), so relevance grades are constant
    within every origin and there is no gradient signal to learn a non-degenerate ranking from.
    10 styles spans grade 4 (top-3) AND grade 3 (rank 4-10) per origin, giving `lambdarank` an
    actual (if tiny) ranking signal -- used only by the non-degeneracy smoke test below."""
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
        [_style(f"S{i}", "G1" if i % 2 == 0 else "G2", base=10.0 * (i + 1)) for i in range(10)]
    )


def test_train_and_predict_lambdarank_smoke() -> None:
    panel = _synthetic_panel_wide()
    frame, columns = _graded_frame_and_columns(panel)

    model = train_lambdarank(frame, _FAST_CONFIG, truncation_level=10, columns=columns)
    preds = predict_lambdarank(model, frame, columns)

    assert preds.shape[0] == frame.height
    assert np.all(np.isfinite(preds))
    assert len(np.unique(preds)) > 1  # not degenerate/constant


def test_train_lambdarank_is_bit_identical_across_repeated_runs() -> None:
    """Same bar as `tests/test_lightgbm_model.py`'s L2 determinism test: two trains on the SAME
    data must produce EXACTLY the same predictions (`np.array_equal`, not `np.allclose`)."""
    panel = _synthetic_panel()
    frame, columns = _graded_frame_and_columns(panel)

    model1 = train_lambdarank(frame, _FAST_CONFIG, truncation_level=10, columns=columns)
    preds1 = predict_lambdarank(model1, frame, columns)
    model2 = train_lambdarank(frame, _FAST_CONFIG, truncation_level=10, columns=columns)
    preds2 = predict_lambdarank(model2, frame, columns)

    assert np.array_equal(preds1, preds2)


# ---------------------------------------------------------------------------
# select_truncation_level
# ---------------------------------------------------------------------------


def test_select_truncation_level_tries_every_candidate_and_picks_one_of_them() -> None:
    panel = _synthetic_panel()
    origins = generate_origin_schedule(panel)
    frame, columns = _graded_frame_and_columns(panel)
    pool_origins = origins[:4]

    result = select_truncation_level(frame, pool_origins, columns, config=_FAST_CONFIG)

    assert len(result.all_results) == len(LAMBDARANK_TRUNCATION_CANDIDATES)
    assert result.best_truncation_level in LAMBDARANK_TRUNCATION_CANDIDATES


def test_select_truncation_level_requires_at_least_two_pool_origins() -> None:
    panel = _synthetic_panel()
    origins = generate_origin_schedule(panel)
    frame, columns = _graded_frame_and_columns(panel)

    with pytest.raises(ValueError, match="at least 2 pool origins"):
        select_truncation_level(frame, origins[:1], columns, config=_FAST_CONFIG)


# ---------------------------------------------------------------------------
# run_lambdarank_walk_forward / summarize_lambdarank -- end-to-end smoke
# ---------------------------------------------------------------------------


def test_run_lambdarank_walk_forward_and_summarize_end_to_end() -> None:
    panel = _synthetic_panel()
    origins = generate_origin_schedule(panel)
    assert len(origins) >= 5

    per_origin, truncation_used, search_results = run_lambdarank_walk_forward(
        panel, origins, pool_size=3, config=_FAST_CONFIG, truncation_level=10
    )

    n_test_origins = len(origins) - 3
    assert per_origin.height == n_test_origins
    assert search_results == []  # truncation_level was passed in directly -- no search should run
    assert truncation_used == 10
    assert set(per_origin["method"].unique().to_list()) == {"lambdarank"}
    for metric in METRIC_KEYS:
        assert metric in per_origin.columns

    summary = summarize_lambdarank(per_origin)
    assert summary.height == 3  # pooled / covid / non_covid
    assert set(summary["method"].unique().to_list()) == {"lambdarank"}
    for metric in METRIC_KEYS:
        assert f"{metric}_mean" in summary.columns
        assert f"{metric}_ci_low" in summary.columns
        assert f"{metric}_ci_high" in summary.columns


def test_run_lambdarank_walk_forward_runs_truncation_search_when_level_omitted() -> None:
    panel = _synthetic_panel()
    origins = generate_origin_schedule(panel)

    _per_origin, truncation_used, search_results = run_lambdarank_walk_forward(
        panel, origins, pool_size=3, config=_FAST_CONFIG
    )

    assert len(search_results) == len(LAMBDARANK_TRUNCATION_CANDIDATES)
    assert truncation_used in LAMBDARANK_TRUNCATION_CANDIDATES


def test_run_lambdarank_walk_forward_rejects_pool_size_leaving_no_test_origins() -> None:
    panel = _synthetic_panel()
    origins = generate_origin_schedule(panel)

    with pytest.raises(ValueError, match="pool_size"):
        run_lambdarank_walk_forward(
            panel, origins, pool_size=len(origins), config=_FAST_CONFIG, truncation_level=10
        )
