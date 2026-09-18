from __future__ import annotations

import datetime
from pathlib import Path

import polars as pl

from nss.features.style_panel import STYLE_KEY_COLS, build_style_week_panel, filter_by_support

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

    Style A has 2 articles (1, 2) spanning two ISO weeks (2018-09-17 and 2018-09-24), 4 total
    units. Style B has 1 article (3), 1 unit, in the 2018-09-24 week. Manually verified expected
    aggregates are asserted in the tests below.
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
            ],
            "customer_id": ["cust1", "cust2", "cust1", "cust3", "cust1"],
            "article_id": [1, 1, 2, 2, 3],
            "price": [10.0, 10.0, 20.0, 20.0, 5.0],
            "sales_channel_id": [1, 2, 1, 1, 2],
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

    assert panel.height == 3  # style A x 2 weeks + style B x 1 week
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
    assert row["units_online"] == 1  # channel 1, per the documented ASSUMPTION
    assert row["units_store"] == 1  # channel 2, per the documented ASSUMPTION
    assert row["first_week_seen"] == datetime.date(2018, 9, 17)
    assert row["last_week_seen"] == datetime.date(2018, 9, 24)

    style_b = panel.filter(pl.col("index_group_name") == "Menswear")
    row_b = style_b.row(0, named=True)
    assert row_b["units"] == 1
    assert row_b["revenue"] == 5.0
    assert row_b["units_online"] == 0
    assert row_b["units_store"] == 1

    # Total units in the panel must equal total transaction row count (no double counting).
    assert panel["units"].sum() == 5
    assert lifetime["lifetime_units"].sum() == 5


def test_filter_by_support_retention(tmp_path: Path) -> None:
    """Style B (1 article, 1 unit) is dropped; style A (2 articles, 4 units) is kept."""
    transactions_dir, articles_path = _write_fixture(tmp_path)
    panel, lifetime = build_style_week_panel(transactions_dir, articles_path)

    filtered_panel, stats = filter_by_support(panel, lifetime, min_articles=2, min_units=3)

    assert stats["n_styles_before"] == 2
    assert stats["n_styles_after"] == 1
    assert stats["pct_units_retained"] == 80.0  # 4 kept units / 5 total units
    assert set(filtered_panel["index_group_name"]) == {"Ladieswear"}
