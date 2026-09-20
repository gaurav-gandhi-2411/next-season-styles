from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from nss.features.targets import HORIZON_WEEKS
from nss.models.backtest import STEP_WEEKS, Origin
from nss.models.backtest_embargo_check import compare_arms, embargoed_train_origin_weeks
from nss.models.lightgbm_model import _expanding_train_origin_weeks
from nss.models.metrics import METRIC_KEYS

_FIRST = date(2019, 7, 29)


def _origins(n: int = 6) -> list[Origin]:
    return [
        Origin(
            origin_week=_FIRST + timedelta(weeks=STEP_WEEKS * i), has_52w_lag=True, is_covid=False
        )
        for i in range(n)
    ]


def _window(origin: date) -> tuple[date, date]:
    """The forward-target window of `origin` per `compute_forward_target` (both ends inclusive)."""
    return origin + timedelta(weeks=1), origin + timedelta(weeks=HORIZON_WEEKS)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "DOCUMENTS A KNOWN FLAW: the shipped walk-forward has no embargo, so training origins "
        "t-4w/t-8w/t-12w have target windows extending past test origin t (label leakage). "
        "Remove this xfail when _expanding_train_origin_weeks is fixed."
    ),
)
def test_shipped_expanding_window_has_no_label_overlap_with_test_window() -> None:
    origins = _origins()
    test_index = len(origins) - 1
    test_week = origins[test_index].origin_week
    test_start, _ = _window(test_week)
    for o in _expanding_train_origin_weeks(origins, test_index):
        _, train_end = _window(o)
        assert train_end < test_start, f"train origin {o} labels reach {train_end} >= {test_start}"


def test_shipped_leak_concrete_example() -> None:
    """Shipped function includes t-4w, whose window overlaps t's window for 9 weeks."""
    origins = _origins()
    t = origins[-1].origin_week
    train = _expanding_train_origin_weeks(origins, len(origins) - 1)
    leaking = t - timedelta(weeks=STEP_WEEKS)
    assert leaking in train
    _, leak_end = _window(leaking)
    t_start, _ = _window(t)
    overlap_weeks = (leak_end - t_start).days // 7 + 1
    assert overlap_weeks == HORIZON_WEEKS - STEP_WEEKS == 9


def test_embargoed_train_origins_have_closed_windows() -> None:
    origins = _origins(8)
    for test_index in range(len(origins)):
        t = origins[test_index].origin_week
        for o in embargoed_train_origin_weeks(origins, test_index):
            assert o + timedelta(weeks=HORIZON_WEEKS) <= t


def test_embargoed_train_origins_exact_and_empty_cases() -> None:
    origins = _origins(8)
    # step 4w, horizon 13w -> t-4/8/12w excluded, t-16w is the latest eligible origin
    assert embargoed_train_origin_weeks(origins, 4) == {origins[0].origin_week}
    assert embargoed_train_origin_weeks(origins, 3) == set()  # nothing eligible -> skipped upstream
    assert embargoed_train_origin_weeks(origins, 7) == {o.origin_week for o in origins[:4]}


def test_compare_arms_pairs_only_shared_origins_and_signs_diff() -> None:
    weeks = [_FIRST + timedelta(weeks=STEP_WEEKS * i) for i in range(5)]

    def _table(ws: list[date], base: float) -> pl.DataFrame:
        cols: dict[str, list[object]] = {
            "origin_week": list(ws),
            "is_covid": [False] * len(ws),
        }
        for m in METRIC_KEYS:
            cols[m] = [base] * len(ws)
        return pl.DataFrame(cols)

    shipped = _table(weeks, 0.75)
    embargoed = _table(weeks[1:], 0.5)  # first origin skipped in the embargoed arm
    out = compare_arms(shipped, embargoed).filter(pl.col("split") == "pooled")
    assert set(out["n_origins"].to_list()) == {4}
    row = out.filter(pl.col("metric") == METRIC_KEYS[0]).row(0, named=True)
    assert row["diff"] == pytest.approx(0.25)
    assert row["shipped"] == pytest.approx(0.75)
    assert row["embargoed"] == pytest.approx(0.5)
