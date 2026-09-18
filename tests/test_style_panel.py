from __future__ import annotations

import datetime
from datetime import timedelta
from pathlib import Path

import polars as pl
import pytest

from nss.features.style_panel import (
    PRICE_INDEX_TRAILING_WEEKS,
    STYLE_KEY_COLS,
    add_intensity_shrunk,
    add_price_index,
    build_style_week_panel,
    densify_panel,
    filter_by_support,
)

_STYLE_A = {
    "index_group_name": "Ladieswear",
    "product_type_name": "Trousers",
    "garment_group_name": "Trousers",
    "perceived_colour_master_name": "Black",
    "graphical_appearance_name": "Solid",
}
_STYLE_B = {
    "index_group_name": "Menswear",
    "product_type_name": "Shirt",
    "garment_group_name": "Shirts",
    "perceived_colour_master_name": "White",
    "graphical_appearance_name": "Stripe",
}


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Write a tiny synthetic articles.csv + partitioned transactions Parquet for testing.

    Style A has 2 articles (1, 2) with sales in 3 ISO weeks (2018-09-17, 2018-09-24, 2018-10-08)
    -- note the gap at 2018-10-01, deliberately left with no sales so densification has a
    zero-sale week to fill -- 5 total units. Style B has 1 article (3), 1 unit, in the
    2018-09-24 week. Manually verified expected aggregates are asserted in the tests below.
    """
    articles_path = tmp_path / "articles.csv"
    articles = pl.DataFrame(
        [
            {"article_id": 1, **_STYLE_A},
            {"article_id": 2, **_STYLE_A},
            {"article_id": 3, **_STYLE_B},
        ]
    )
    articles.write_csv(articles_path)

    transactions_dir = tmp_path / "transactions"
    transactions_dir.mkdir()
    txns = pl.DataFrame(
        {
            "t_dat": [
                "2018-09-20",  # Thu -> week_start 2018-09-17
                "2018-09-20",
                "2018-09-24",  # Mon -> week_start 2018-09-24
                "2018-09-24",
                "2018-09-25",  # Tue -> week_start 2018-09-24
                "2018-10-09",  # Tue -> week_start 2018-10-08 (gap week 2018-10-01 has no sales)
            ],
            "customer_id": ["cust1", "cust2", "cust1", "cust3", "cust1", "cust2"],
            "article_id": [1, 1, 2, 2, 3, 1],
            "price": [10.0, 10.0, 20.0, 20.0, 5.0, 15.0],
            "sales_channel_id": [1, 2, 1, 1, 2, 1],
        }
    ).with_columns(pl.col("t_dat").str.to_date())
    part_dir = transactions_dir / "part"
    part_dir.mkdir()
    txns.write_parquet(part_dir / "data.parquet")

    return transactions_dir, articles_path


def test_build_style_week_panel_aggregates(tmp_path: Path) -> None:
    """Weekly aggregates (units, revenue, channel split, lifetime bounds) match a hand tally."""
    transactions_dir, articles_path = _write_fixture(tmp_path)

    panel, lifetime = build_style_week_panel(transactions_dir, articles_path)

    assert panel.height == 4  # style A x 3 sparse weeks + style B x 1 week
    assert set(panel["style_key"]) == {
        " || ".join(_STYLE_A[c] for c in STYLE_KEY_COLS),
        " || ".join(_STYLE_B[c] for c in STYLE_KEY_COLS),
    }

    style_a_week1 = panel.filter(
        (pl.col("index_group_name") == "Ladieswear")
        & (pl.col("week_start") == datetime.date(2018, 9, 17))
    )
    row = style_a_week1.row(0, named=True)
    assert row["units"] == 2
    assert row["revenue"] == 20.0
    assert row["n_active_articles"] == 1
    assert row["units_per_active_article"] == 2.0
    assert row["mean_price"] == 10.0
    assert row["median_price"] == 10.0
    assert row["n_customers"] == 2
    assert row["units_online"] == 1  # channel 2, per the VALIDATED FINDING
    assert row["units_store"] == 1  # channel 1, per the VALIDATED FINDING
    assert row["first_week_seen"] == datetime.date(2018, 9, 17)
    assert row["last_week_seen"] == datetime.date(2018, 10, 8)  # style A's 3rd sparse week

    style_b = panel.filter(pl.col("index_group_name") == "Menswear")
    row_b = style_b.row(0, named=True)
    assert row_b["units"] == 1
    assert row_b["revenue"] == 5.0
    assert row_b["units_online"] == 1  # its 1 txn is channel 2, per the VALIDATED FINDING
    assert row_b["units_store"] == 0

    # Total units in the panel must equal total transaction row count (no double counting).
    assert panel["units"].sum() == 6
    assert lifetime["lifetime_units"].sum() == 6


def test_filter_by_support_retention(tmp_path: Path) -> None:
    """Style B (1 article, 1 unit) is dropped; style A (2 articles, 5 units) is kept."""
    transactions_dir, articles_path = _write_fixture(tmp_path)
    panel, lifetime = build_style_week_panel(transactions_dir, articles_path)

    filtered_panel, stats = filter_by_support(panel, lifetime, min_articles=2, min_units=3)

    assert stats["n_styles_before"] == 2
    assert stats["n_styles_after"] == 1
    assert stats["pct_units_retained"] == pytest.approx(100.0 * 5 / 6)  # 5 kept / 6 total units
    assert set(filtered_panel["index_group_name"]) == {"Ladieswear"}


def test_densify_panel_inserts_zero_sale_weeks(tmp_path: Path) -> None:
    """A known no-sales week appears as an explicit units=0 row, not an absent one.

    Style A has sales in 2018-09-17, 2018-09-24, and 2018-10-08, with a deliberate gap at
    2018-10-01 (no transactions that week). Densifying over style A's own lifetime
    (first_week_seen=2018-09-17, last_week_seen=2018-10-08) must insert an explicit zero row for
    2018-10-01, and each style's row count must equal its lifetime span in weeks + 1.
    """
    transactions_dir, articles_path = _write_fixture(tmp_path)
    panel, lifetime = build_style_week_panel(transactions_dir, articles_path)
    filtered_panel, _ = filter_by_support(panel, lifetime, min_articles=2, min_units=3)

    dense = densify_panel(filtered_panel)

    # Style A only (style B was dropped by the filter); 4 ISO weeks span the lifetime.
    assert dense.height == 4

    gap_week = dense.filter(pl.col("week_start") == datetime.date(2018, 10, 1))
    assert gap_week.height == 1
    gap_row = gap_week.row(0, named=True)
    assert gap_row["units"] == 0
    assert gap_row["revenue"] == 0.0
    assert gap_row["n_active_articles"] == 0
    assert gap_row["n_customers"] == 0
    assert gap_row["units_online"] == 0
    assert gap_row["units_store"] == 0
    assert gap_row["units_per_active_article"] == 0
    assert gap_row["mean_price"] is None  # never fabricate a price for an unobserved week
    assert gap_row["median_price"] is None
    # Lifetime bounds are unchanged style-level constants, even on the inserted zero row.
    assert gap_row["first_week_seen"] == datetime.date(2018, 9, 17)
    assert gap_row["last_week_seen"] == datetime.date(2018, 10, 8)

    # Per-style row count must equal (last_week_seen - first_week_seen) in weeks + 1.
    per_style_counts = dense.group_by(STYLE_KEY_COLS).agg(
        n_rows=pl.len(),
        first_week_seen=pl.col("first_week_seen").first(),
        last_week_seen=pl.col("last_week_seen").first(),
    )
    for row in per_style_counts.iter_rows(named=True):
        expected_weeks = (row["last_week_seen"] - row["first_week_seen"]).days // 7 + 1
        assert row["n_rows"] == expected_weeks


def test_densify_panel_no_nan_or_inf_anywhere(tmp_path: Path) -> None:
    """No numeric column contains NaN or +/-inf, on zero-sale rows or otherwise.

    Regression test for C3: `units_per_active_article` is filled with the literal 0.0 on
    zero-sale rows (not computed via a 0/0 division), so it must never be NaN. `mean_price` /
    `median_price` are expected to be *null* (no price observed) on zero-sale rows -- that is a
    distinct, correct-by-design condition, not the NaN/inf this test guards against.
    """
    transactions_dir, articles_path = _write_fixture(tmp_path)
    panel, lifetime = build_style_week_panel(transactions_dir, articles_path)
    filtered_panel, _ = filter_by_support(panel, lifetime, min_articles=2, min_units=3)
    dense = densify_panel(filtered_panel)

    numeric_cols = [
        "units",
        "revenue",
        "n_active_articles",
        "units_per_active_article",
        "mean_price",
        "median_price",
        "n_customers",
        "units_online",
        "units_store",
    ]
    for col in numeric_cols:
        series = dense[col]
        if series.dtype in (pl.Float32, pl.Float64):
            assert series.is_nan().fill_null(False).sum() == 0, f"{col} has NaN"
            assert series.is_infinite().fill_null(False).sum() == 0, f"{col} has inf"

    # units_per_active_article is exactly 0.0 (not NaN) on every zero-sale row.
    zero_sale = dense.filter(pl.col("units") == 0)
    assert zero_sale.height > 0  # the fixture's gap week must actually be present
    assert zero_sale["units_per_active_article"].unique().to_list() == [0.0]
    assert zero_sale["units_per_active_article"].null_count() == 0


def _weeks(n: int, start: datetime.date = datetime.date(2018, 1, 1)) -> list[datetime.date]:
    """`n` consecutive Monday week_start dates starting at `start`."""
    return [start + timedelta(weeks=i) for i in range(n)]


def test_add_price_index_requires_full_trailing_window() -> None:
    """price_index is null until PRICE_INDEX_TRAILING_WEEKS prior rows exist, then computed.

    53 weeks of a single style_key, mean_price constant at 100.0 for the first 52 weeks and 200.0
    on the 53rd (index 52). Row index 51 has only 51 prior rows -> null. Row index 52 has exactly
    52 prior rows -> price_index = 200.0 / median(100.0, ..., 100.0) = 2.0.
    """
    weeks = _weeks(PRICE_INDEX_TRAILING_WEEKS + 1)
    mean_price = [100.0] * PRICE_INDEX_TRAILING_WEEKS + [200.0]
    panel = pl.DataFrame(
        {"style_key": ["A"] * len(weeks), "week_start": weeks, "mean_price": mean_price}
    )

    out = add_price_index(panel)

    row_51 = out.filter(pl.col("week_start") == weeks[PRICE_INDEX_TRAILING_WEEKS - 1]).row(
        0, named=True
    )
    assert row_51["price_index"] is None

    row_52 = out.filter(pl.col("week_start") == weeks[PRICE_INDEX_TRAILING_WEEKS]).row(
        0, named=True
    )
    assert row_52["price_index"] == pytest.approx(2.0)


def test_add_price_index_excludes_current_week_from_its_own_denominator() -> None:
    """Corrupting the current row's own mean_price must not change its own price_index.

    A row's price_index numerator is that row's mean_price, but the trailing-median denominator
    must be built only from strictly-prior rows -- so changing the numerator changes price_index
    proportionally, but the denominator (recomputed from a fresh, larger current-week value) must
    stay fixed. This test isolates that: two panels differing ONLY in the target row's own
    mean_price must produce price_index values in exact proportion to that row's mean_price, i.e.
    the denominator did not itself pick up the new value.
    """
    weeks = _weeks(PRICE_INDEX_TRAILING_WEEKS + 1)
    mean_price_a = [100.0] * PRICE_INDEX_TRAILING_WEEKS + [200.0]
    mean_price_b = [100.0] * PRICE_INDEX_TRAILING_WEEKS + [400.0]  # only the LAST row differs
    panel_a = pl.DataFrame(
        {"style_key": ["A"] * len(weeks), "week_start": weeks, "mean_price": mean_price_a}
    )
    panel_b = pl.DataFrame(
        {"style_key": ["A"] * len(weeks), "week_start": weeks, "mean_price": mean_price_b}
    )

    last_week = weeks[PRICE_INDEX_TRAILING_WEEKS]
    price_index_a = (
        add_price_index(panel_a).filter(pl.col("week_start") == last_week).row(0, named=True)
    )["price_index"]
    price_index_b = (
        add_price_index(panel_b).filter(pl.col("week_start") == last_week).row(0, named=True)
    )["price_index"]

    # If the denominator (trailing median) had picked up the new 400.0 value, price_index_b would
    # NOT be exactly double price_index_a (the denominator would also have shifted).
    assert price_index_b == pytest.approx(2.0 * price_index_a)


def test_add_intensity_shrunk_trailing_group_mean() -> None:
    """Shrinkage weight, trailing (not same-week) group mean, and zero-evidence collapse.

    Two styles (A, B) share one group across 3 weeks. Hand-computed with k=2.0:
    - week0: no prior group history -> intensity_shrunk is null for both A and B.
    - week1: group_mean_trailing = week0's group mean (8.0) = mean(10.0, 6.0).
      A: n_active=5, w=5/7, raw=20.0 -> shrunk = (5/7)*20 + (2/7)*8 = 116/7.
      B: n_active=1, w=1/3, raw=4.0 -> shrunk = (1/3)*4 + (2/3)*8 = 20/3.
    - week2: group_mean_trailing = mean(week0's 8.0, week1's 12.0) = 10.0.
      A: n_active=3, w=3/5, raw=12.0 -> shrunk = 0.6*12 + 0.4*10 = 11.2.
      B: n_active=0 (zero-sale week) -> w=0 -> shrunk collapses fully to trailing (10.0), and B's
        zero-sale week is excluded from feeding week2's own group-mean contribution (it has no
        active articles, so it cannot inform "how intensely does this group sell when stocked").
    """
    weeks = _weeks(3)
    panel = pl.DataFrame(
        {
            "style_key": ["A", "B", "A", "B", "A", "B"],
            "index_group_name": ["G"] * 6,
            "garment_group_name": ["GG"] * 6,
            "week_start": [weeks[0], weeks[0], weeks[1], weeks[1], weeks[2], weeks[2]],
            "n_active_articles": [4, 2, 5, 1, 3, 0],
            "units_per_active_article": [10.0, 6.0, 20.0, 4.0, 12.0, 0.0],
        }
    )

    out = add_intensity_shrunk(panel, k=2.0)

    def _shrunk(style_key: str, week: datetime.date) -> float | None:
        row = out.filter((pl.col("style_key") == style_key) & (pl.col("week_start") == week)).row(
            0, named=True
        )
        return row["intensity_shrunk"]

    assert _shrunk("A", weeks[0]) is None
    assert _shrunk("B", weeks[0]) is None
    assert _shrunk("A", weeks[1]) == pytest.approx(116 / 7)
    assert _shrunk("B", weeks[1]) == pytest.approx(20 / 3)
    assert _shrunk("A", weeks[2]) == pytest.approx(11.2)
    assert _shrunk("B", weeks[2]) == pytest.approx(10.0)


def test_add_intensity_shrunk_group_mean_excludes_current_week() -> None:
    """Corrupting a style's CURRENT-week raw value must not change ANOTHER style's shrunk value
    for that same week -- proof the group mean used is trailing, not same-week cross-sectional.

    If the group mean were computed same-week (cross-sectionally), changing A's week1 raw value
    would change the group mean feeding into B's week1 shrinkage. Since the design is trailing
    (built only from weeks strictly before), B's week1 intensity_shrunk must be identical whether
    A's week1 raw value is 20.0 or a wildly different number.
    """
    weeks = _weeks(3)

    def _panel(a_week1_raw: float) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "style_key": ["A", "B", "A", "B", "A", "B"],
                "index_group_name": ["G"] * 6,
                "garment_group_name": ["GG"] * 6,
                "week_start": [weeks[0], weeks[0], weeks[1], weeks[1], weeks[2], weeks[2]],
                "n_active_articles": [4, 2, 5, 1, 3, 1],
                "units_per_active_article": [10.0, 6.0, a_week1_raw, 4.0, 12.0, 3.0],
            }
        )

    def _b_week1_shrunk(panel: pl.DataFrame) -> float | None:
        out = add_intensity_shrunk(panel, k=2.0)
        row = out.filter((pl.col("style_key") == "B") & (pl.col("week_start") == weeks[1])).row(
            0, named=True
        )
        return row["intensity_shrunk"]

    baseline = _b_week1_shrunk(_panel(a_week1_raw=20.0))
    corrupted = _b_week1_shrunk(_panel(a_week1_raw=999_999.0))
    assert baseline == corrupted


def test_add_intensity_shrunk_forward_carries_across_whole_group_gap_weeks() -> None:
    """Regression test: a week where the ENTIRE group has zero active styles must forward-carry
    the trailing group mean, not go null.

    A plain equi-join on (group_cols, week_start) only attaches a trailing value to weeks that are
    themselves present in the active-weeks table -- so a week where literally no style in the group
    sold anything (a real, common case for small/niche groups) would incorrectly null out
    `group_mean_trailing` even with abundant earlier history. Single style, single group: active at
    weeks 0-2, a 3-week group-wide gap at weeks 3-5 (n_active_articles == 0, so the GROUP itself has
    no active-week entry either), active again at week 6.
    """
    weeks = _weeks(7)
    panel = pl.DataFrame(
        {
            "style_key": ["A"] * 7,
            "index_group_name": ["G"] * 7,
            "garment_group_name": ["GG"] * 7,
            "week_start": weeks,
            "n_active_articles": [2, 2, 2, 0, 0, 0, 2],
            "units_per_active_article": [10.0, 20.0, 30.0, 0.0, 0.0, 0.0, 40.0],
        }
    )

    out = add_intensity_shrunk(panel, k=2.0).sort("week_start")
    trailing_by_week = dict(
        zip(out["week_start"].to_list(), out["intensity_shrunk"].to_list(), strict=True)
    )

    # Group-mean-trailing sequence: null, 10.0, 15.0, 20.0, 20.0, 20.0, 20.0 (hand-verified: after
    # week2 the cumulative-inclusive group mean is mean(10, 20, 30) = 20.0, and it must carry
    # forward unchanged through every subsequent zero-activity week and into week6, since no new
    # active week occurs in between). At week6 itself (n_active=2, w=2/4=0.5, raw=40.0):
    # shrunk = 0.5*40 + 0.5*20 = 30.0.
    assert trailing_by_week[weeks[0]] is None
    assert trailing_by_week[weeks[3]] == pytest.approx(20.0)  # gap week, w=0 -> pure trailing
    assert trailing_by_week[weeks[4]] == pytest.approx(20.0)
    assert trailing_by_week[weeks[5]] == pytest.approx(20.0)
    assert trailing_by_week[weeks[6]] == pytest.approx(30.0)
