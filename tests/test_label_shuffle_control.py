from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from nss.models.label_shuffle_control import shuffle_within_origin

WEEKS = [date(2020, 1, 6)] * 5 + [date(2020, 2, 3)] * 5
FRAME = pl.DataFrame(
    {"origin_week": WEEKS, "y_true": [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0]}
)


def test_shuffle_preserves_each_origins_target_multiset() -> None:
    out = shuffle_within_origin(FRAME, seed=42)
    assert sorted(out[:5]) == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert sorted(out[5:]) == [10.0, 20.0, 30.0, 40.0, 50.0]


def test_shuffle_never_mixes_origins() -> None:
    out = shuffle_within_origin(FRAME, seed=7)
    assert set(out[:5]) <= {1.0, 2.0, 3.0, 4.0, 5.0}
    assert set(out[5:]) <= {10.0, 20.0, 30.0, 40.0, 50.0}


def test_shuffle_is_reproducible_and_seed_dependent() -> None:
    a = shuffle_within_origin(FRAME, seed=42)
    assert np.array_equal(a, shuffle_within_origin(FRAME, seed=42))
    assert any(not np.array_equal(a, shuffle_within_origin(FRAME, seed=s)) for s in (43, 44, 45))


def test_shuffle_destroys_row_correspondence() -> None:
    y = FRAME["y_true"].to_numpy()
    assert any(not np.array_equal(shuffle_within_origin(FRAME, seed=s), y) for s in range(5))
