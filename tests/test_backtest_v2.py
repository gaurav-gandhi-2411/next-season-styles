from __future__ import annotations

import math
from datetime import date

import polars as pl
import pytest

from nss.models.backtest import block_bootstrap_ci
from nss.models.backtest_v2 import (
    add_signal_flags,
    identify_lightgbm_origins,
    paired_diff_table,
)
from nss.models.lightgbm_model import INITIAL_POOL_SIZE

# ---------------------------------------------------------------------------
# identify_lightgbm_origins
# ---------------------------------------------------------------------------


def _real_extent_panel() -> pl.DataFrame:
    """Only `week_start`'s min/max matter to `generate_origin_schedule` -- same trick as
    `test_backtest.py::test_generate_origin_schedule_matches_real_panel_extent`."""
    return pl.DataFrame({"week_start": [date(2018, 9, 17), date(2020, 9, 21)]})


def test_identify_lightgbm_origins_matches_checked_in_csv(tmp_path) -> None:
    from nss.models.backtest import generate_origin_schedule

    panel = _real_extent_panel()
    all_origins = generate_origin_schedule(panel)
    expected = all_origins[INITIAL_POOL_SIZE:]
    assert len(expected) == 12  # sanity, matches the documented 12 walk-forward origins

    csv_path = tmp_path / "backtest_per_origin_lightgbm.csv"
    pl.DataFrame({"origin_week": [o.origin_week for o in expected]}).write_csv(csv_path)

    out = identify_lightgbm_origins(panel, per_origin_csv_path=csv_path)

    assert [o.origin_week for o in out] == [o.origin_week for o in expected]


def test_identify_lightgbm_origins_raises_on_mismatch(tmp_path) -> None:
    panel = _real_extent_panel()

    csv_path = tmp_path / "backtest_per_origin_lightgbm.csv"
    # Deliberately wrong: a single, clearly-not-in-schedule origin week.
    pl.DataFrame({"origin_week": [date(2000, 1, 1)]}).write_csv(csv_path)

    with pytest.raises(ValueError, match="do not match"):
        identify_lightgbm_origins(panel, per_origin_csv_path=csv_path)


# ---------------------------------------------------------------------------
# add_signal_flags
# ---------------------------------------------------------------------------


def test_add_signal_flags_overlap_no_overlap_and_nan_cases() -> None:
    summary = pl.DataFrame(
        {
            "method": ["random_floor", "baseline_overlap", "baseline_beats_floor", "baseline_nan"],
            "split": ["pooled", "pooled", "pooled", "pooled"],
            "n_origins": [12, 12, 12, 5],
            "precision_at_3_mean": [0.10, 0.15, 0.35, float("nan")],
            "precision_at_3_ci_low": [0.05, 0.10, 0.30, float("nan")],
            "precision_at_3_ci_high": [0.15, 0.20, 0.40, float("nan")],
        }
    )

    out = add_signal_flags(summary, metric_keys=("precision_at_3",))

    rows = {row["method"]: row for row in out.iter_rows(named=True)}
    assert rows["random_floor"]["precision_at_3_no_demonstrated_signal"] is None
    assert rows["baseline_overlap"]["precision_at_3_no_demonstrated_signal"] is True
    assert rows["baseline_beats_floor"]["precision_at_3_no_demonstrated_signal"] is False
    assert rows["baseline_nan"]["precision_at_3_no_demonstrated_signal"] is None


def test_add_signal_flags_directional_only_below_min_origins() -> None:
    summary = pl.DataFrame(
        {
            "method": ["random_floor", "random_floor", "baseline_a", "baseline_a"],
            "split": ["covid", "pooled", "covid", "pooled"],
            "n_origins": [7, 12, 7, 12],
            "precision_at_3_mean": [0.1, 0.1, 0.1, 0.1],
            "precision_at_3_ci_low": [0.05, 0.05, 0.05, 0.05],
            "precision_at_3_ci_high": [0.15, 0.15, 0.15, 0.15],
        }
    )

    out = add_signal_flags(summary, metric_keys=("precision_at_3",))

    rows = out.filter(pl.col("method") == "baseline_a").sort("split")
    covid_row = rows.filter(pl.col("split") == "covid").row(0, named=True)
    pooled_row = rows.filter(pl.col("split") == "pooled").row(0, named=True)
    assert covid_row["directional_only"] is True  # n_origins=7 < 8
    assert pooled_row["directional_only"] is False  # n_origins=12 >= 8


def test_add_signal_flags_missing_random_floor_split_fails_closed_to_none() -> None:
    """A random_floor row missing for a split baseline_a reports (a malformed/incomplete summary
    that should never happen in production, since `summarize_backtest` always emits all 3 splits
    for every method present) must fail CLOSED to `None` ("can't verify"), never raise and never
    silently read as `False`/`True` -- see rule 98a (guard fail-closed, not open)."""
    summary = pl.DataFrame(
        {
            "method": ["baseline_a"],
            "split": ["pooled"],
            "n_origins": [12],
            "precision_at_3_mean": [0.1],
            "precision_at_3_ci_low": [0.05],
            "precision_at_3_ci_high": [0.15],
        }
    )

    out = add_signal_flags(summary, metric_keys=("precision_at_3",))

    assert out.row(0, named=True)["precision_at_3_no_demonstrated_signal"] is None


# ---------------------------------------------------------------------------
# paired_diff_table
# ---------------------------------------------------------------------------


def test_paired_diff_table_matches_hand_verifiable_block_bootstrap() -> None:
    """4 origins, 2 covid + 2 non-covid. lightgbm precision_at_3=[0.5,0.6,0.7,0.8],
    baseline_x precision_at_3=[0.1,0.1,0.1,0.1] -> diff=[0.4,0.5,0.6,0.7] pooled.

    The pooled diff series/CI must exactly match calling `block_bootstrap_ci` directly on
    [0.4, 0.5, 0.6, 0.7] with the same (block_size=4, seed=42, n_resamples=2000) defaults --
    `paired_diff_table` must not be silently using a different bootstrap configuration.
    """
    weeks = [date(2019, 1, 1), date(2019, 1, 29), date(2019, 2, 26), date(2019, 3, 26)]
    per_origin = pl.DataFrame(
        {
            "origin_week": weeks + weeks,
            "method": ["lightgbm"] * 4 + ["baseline_x"] * 4,
            "has_52w_lag": [True] * 8,
            "is_covid": [False, False, True, True, False, False, True, True],
            "n_eval_set": [100] * 8,
            "n_eval": [100] * 8,
            "precision_at_3": [0.5, 0.6, 0.7, 0.8, 0.1, 0.1, 0.1, 0.1],
        }
    )

    out = paired_diff_table(
        per_origin,
        treatment_method="lightgbm",
        baseline_methods=("baseline_x",),
        metric_keys=("precision_at_3",),
    )

    pooled_row = out.filter(pl.col("split") == "pooled").row(0, named=True)
    expected_mean, expected_lo, expected_hi = block_bootstrap_ci([0.4, 0.5, 0.6, 0.7])
    assert pooled_row["mean_diff"] == pytest.approx(expected_mean)
    assert pooled_row["diff_ci_low"] == pytest.approx(expected_lo)
    assert pooled_row["diff_ci_high"] == pytest.approx(expected_hi)
    assert pooled_row["n_origins"] == 4
    assert pooled_row["directional_only"] is True  # 4 < MIN_ORIGINS_FOR_NON_DIRECTIONAL (8)

    covid_row = out.filter(pl.col("split") == "covid").row(0, named=True)
    assert covid_row["n_origins"] == 2
    expected_covid_mean, _, _ = block_bootstrap_ci([0.6, 0.7])
    assert covid_row["mean_diff"] == pytest.approx(expected_covid_mean)
    assert math.isclose(covid_row["mean_diff"], (0.6 + 0.7) / 2)


def test_paired_diff_table_covers_every_baseline_method() -> None:
    weeks = [date(2019, 1, 1), date(2019, 1, 29)]
    per_origin = pl.DataFrame(
        {
            "origin_week": weeks * 3,
            "method": ["lightgbm"] * 2 + ["baseline_a"] * 2 + ["baseline_b"] * 2,
            "has_52w_lag": [True] * 6,
            "is_covid": [False, True] * 3,
            "n_eval_set": [100] * 6,
            "n_eval": [100] * 6,
            "precision_at_3": [0.5, 0.6, 0.1, 0.2, 0.3, 0.4],
        }
    )

    out = paired_diff_table(
        per_origin,
        treatment_method="lightgbm",
        baseline_methods=("baseline_a", "baseline_b"),
        metric_keys=("precision_at_3",),
    )

    assert set(out["method_b"].unique().to_list()) == {"baseline_a", "baseline_b"}
    assert out.height == 2 * 3  # 2 baselines x 3 splits (pooled/covid/non_covid)
