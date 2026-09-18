from __future__ import annotations

import math
from datetime import date, timedelta

import polars as pl
import pytest

from nss.features.targets import HORIZON_WEEKS, compute_forward_target

_WEEK0 = date(2018, 1, 1)


def _weeks(n: int) -> list[date]:
    """`n` consecutive Monday week_start dates starting at `_WEEK0`."""
    return [_WEEK0 + timedelta(weeks=i) for i in range(n)]


def _single_style_panel(n_weeks: int, style_key: str = "A") -> pl.DataFrame:
    """One style, `n_weeks` consecutive weeks.

    `units_per_active_article` == the week's index; `intensity_shrunk` == the week's index * 2 (a
    deliberately different value per row, so a test corrupting one column can't accidentally pass
    by reading the other).
    """
    weeks = _weeks(n_weeks)
    return pl.DataFrame(
        {
            "style_key": [style_key] * n_weeks,
            "week_start": weeks,
            "units_per_active_article": [float(i) for i in range(n_weeks)],
            "intensity_shrunk": [float(i) * 2 for i in range(n_weeks)],
        }
    )


def test_compute_forward_target_matches_hand_computed_mean() -> None:
    """Target = log1p(mean(units_per_active_article)) over the 13 weeks strictly after origin.

    Panel has 20 weeks (index 0-19); origin = week index 5. The window is weeks index 6-18
    (13 weeks), values 6..18, mean = 156/13 = 12.0 -- hand-verified below via `sum(range(6, 19))`.
    """
    panel = _single_style_panel(n_weeks=20)
    origin_week = _weeks(20)[5]

    result = compute_forward_target(panel, origin_week)

    expected_mean = sum(range(6, 19)) / 13
    assert expected_mean == 12.0  # sanity check on the hand computation itself
    row = result.filter(pl.col("style_key") == "A").row(0, named=True)
    assert row["n_weeks_in_window"] == HORIZON_WEEKS
    assert row["target"] == math.log1p(expected_mean)
    assert row["origin_week"] == origin_week


def test_compute_forward_target_null_when_window_incomplete() -> None:
    """Fewer than HORIZON_WEEKS future weeks available -> null target, not a partial-window mean.

    Panel has only 25 weeks (index 0-24); an origin near the end (index 20) has only 4 future
    weeks (21-24), far short of the 13 required.
    """
    panel = _single_style_panel(n_weeks=25)
    origin_week = _weeks(25)[20]

    result = compute_forward_target(panel, origin_week)

    row = result.filter(pl.col("style_key") == "A").row(0, named=True)
    assert row["n_weeks_in_window"] == 4
    assert row["target"] is None


def test_compute_forward_target_full_window_present_for_every_style() -> None:
    """`compute_forward_target` returns exactly one row per distinct style_key in the panel."""
    panel = pl.concat(
        [
            _single_style_panel(n_weeks=20, style_key="A"),
            _single_style_panel(n_weeks=20, style_key="B"),
        ]
    )
    origin_week = _weeks(20)[3]

    result = compute_forward_target(panel, origin_week)

    assert set(result["style_key"]) == {"A", "B"}
    assert result.height == 2


@pytest.mark.parametrize("target_column", ["units_per_active_article", "intensity_shrunk"])
def test_compute_forward_target_is_causally_safe(target_column: str) -> None:
    """The target reads only weeks strictly after origin, and only within the horizon.

    Parametrized over both supported `target_column` values (raw `units_per_active_article` and
    EB-shrunk `intensity_shrunk`) -- the causal window filter is applied identically regardless of
    which column feeds the aggregation, so this same positive/negative-control rigor must hold for
    both.

    Panel: 20 weeks (index 0-19). Origin = index 5, so the target window is index 6-18. Corrupting
    a pre-origin row (index 2) or a beyond-the-window future row (index 19) must NOT change the
    target. Corrupting a genuinely in-window row (index 10) MUST change it -- this is the "would
    fail if wrong" half of the test: a function that silently ignored its own window filter would
    also pass the two "unaffected" assertions, so this positive-control assertion is required to
    prove the window filter is actually being applied, not just absent-mindedly harmless.
    """
    weeks = _weeks(20)
    origin_week = weeks[5]

    baseline = _single_style_panel(n_weeks=20)
    baseline_target = compute_forward_target(
        baseline, origin_week, target_column=target_column
    ).row(0, named=True)["target"]

    # Corrupt a pre-origin row (index 2, week_start <= origin_week) -- must not move the target.
    corrupted_pre_origin = baseline.with_columns(
        pl.when(pl.col("week_start") == weeks[2])
        .then(pl.lit(999_999.0))
        .otherwise(pl.col(target_column))
        .alias(target_column)
    )
    pre_origin_target = compute_forward_target(
        corrupted_pre_origin, origin_week, target_column=target_column
    ).row(0, named=True)["target"]
    assert pre_origin_target == baseline_target

    # Corrupt a row beyond the 13-week window (index 19, week_start > origin_week + 13 weeks) --
    # must not move the target either.
    corrupted_post_window = baseline.with_columns(
        pl.when(pl.col("week_start") == weeks[19])
        .then(pl.lit(999_999.0))
        .otherwise(pl.col(target_column))
        .alias(target_column)
    )
    post_window_target = compute_forward_target(
        corrupted_post_window, origin_week, target_column=target_column
    ).row(0, named=True)["target"]
    assert post_window_target == baseline_target

    # Corrupt a genuinely in-window row (index 10, inside origin+1..origin+13) -- MUST move it.
    corrupted_in_window = baseline.with_columns(
        pl.when(pl.col("week_start") == weeks[10])
        .then(pl.lit(999_999.0))
        .otherwise(pl.col(target_column))
        .alias(target_column)
    )
    in_window_target = compute_forward_target(
        corrupted_in_window, origin_week, target_column=target_column
    ).row(0, named=True)["target"]
    assert in_window_target != baseline_target
