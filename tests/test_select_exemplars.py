from __future__ import annotations

import datetime
from pathlib import Path

import polars as pl
import pytest

from nss.data.select_exemplars import (
    N_EXEMPLARS_PER_STYLE,
    build_manifest_rows,
    compute_lookback_cutoff,
    select_control_style,
    select_top_selling_articles,
    style_key_values_from_panel,
)

_STYLE_A = {
    "index_group_name": "Ladieswear",
    "product_type_name": "T-shirt",
    "garment_group_name": "Jersey Basic",
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


def _articles_fixture() -> pl.DataFrame:
    """3 style-A articles (1, 2, 3) + 1 style-B article (4)."""
    return pl.DataFrame(
        [
            {"article_id": 1, **_STYLE_A},
            {"article_id": 2, **_STYLE_A},
            {"article_id": 3, **_STYLE_A},
            {"article_id": 4, **_STYLE_B},
        ]
    )


def _transactions_fixture() -> pl.DataFrame:
    """Style A: article 1 sells 3x, article 2 sells 2x (both in-window), article 3 sells 1x but
    OUTSIDE the window (must be excluded). Style B (article 4) sells 5x, always outside style A's
    selection since it's a different style_key.
    """
    return pl.DataFrame(
        {
            "article_id": [1, 1, 1, 2, 2, 3, 4, 4, 4, 4, 4],
            "t_dat": [
                "2020-04-01",
                "2020-05-01",
                "2020-09-01",
                "2020-04-15",
                "2020-08-01",
                "2020-01-01",  # before the window -- must not count
                "2020-04-01",
                "2020-04-02",
                "2020-04-03",
                "2020-04-04",
                "2020-04-05",
            ],
        }
    ).with_columns(pl.col("t_dat").str.to_date())


def test_compute_lookback_cutoff() -> None:
    """26 weeks before 2020-09-21 (a Monday) is 2020-03-23 (also a Monday)."""
    cutoff = compute_lookback_cutoff(datetime.date(2020, 9, 21), 26)
    assert cutoff == datetime.date(2020, 3, 23)
    assert cutoff.weekday() == 0


def test_select_top_selling_articles_ranks_by_units_in_window() -> None:
    """Article 1 (3 sales) outranks article 2 (2 sales); article 3's out-of-window sale + style
    B's article 4 are both excluded."""
    transactions = _transactions_fixture()
    articles = _articles_fixture()
    window_start = datetime.date(2020, 3, 23)
    window_end = datetime.date(2020, 9, 21)

    ranked = select_top_selling_articles(transactions, articles, _STYLE_A, window_start, window_end)

    assert ranked["article_id"].to_list() == [1, 2]
    assert ranked["units_sold_last_26w"].to_list() == [3, 2]


def test_select_top_selling_articles_truncates_to_n() -> None:
    """`n` caps the number of returned articles even when more had sales in-window."""
    transactions = _transactions_fixture()
    articles = _articles_fixture()
    window_start = datetime.date(2020, 3, 23)
    window_end = datetime.date(2020, 9, 21)

    ranked = select_top_selling_articles(
        transactions, articles, _STYLE_A, window_start, window_end, n=1
    )

    assert ranked.height == 1
    assert ranked["article_id"].to_list() == [1]


def test_select_top_selling_articles_fewer_than_n_not_padded() -> None:
    """A style with fewer than `n` distinct in-window sellers returns exactly that many rows --
    never padded with untraded articles."""
    transactions = _transactions_fixture()
    articles = _articles_fixture()
    window_start = datetime.date(2020, 3, 23)
    window_end = datetime.date(2020, 9, 21)

    ranked = select_top_selling_articles(
        transactions, articles, _STYLE_A, window_start, window_end, n=N_EXEMPLARS_PER_STYLE
    )

    assert ranked.height == 2  # only articles 1 and 2 sold in-window for style A


def test_select_control_style_excludes_and_is_deterministic() -> None:
    """seed=42 always picks the same candidate, and it is never one of the excluded style_keys."""
    panel = pl.DataFrame({"style_key": ["A", "B", "C", "D", "E"]})
    excluded = {"A", "B"}

    chosen_1 = select_control_style(panel, excluded, seed=42)
    chosen_2 = select_control_style(panel, excluded, seed=42)

    assert chosen_1 == chosen_2
    assert chosen_1 not in excluded
    assert chosen_1 in {"C", "D", "E"}


def test_select_control_style_raises_when_no_candidates_remain() -> None:
    """All candidates excluded -> explicit failure, never a silent fallback."""
    panel = pl.DataFrame({"style_key": ["A", "B"]})
    with pytest.raises(ValueError, match="no non-excluded"):
        select_control_style(panel, {"A", "B"}, seed=42)


def test_style_key_values_from_panel_looks_up_constituent_columns() -> None:
    panel = pl.DataFrame(
        [
            {"style_key": "A", **_STYLE_A},
            {"style_key": "B", **_STYLE_B},
        ]
    )
    assert style_key_values_from_panel(panel, "B") == _STYLE_B


def test_build_manifest_rows_records_fetch_outcome_per_article() -> None:
    ranked = pl.DataFrame({"article_id": [1, 2], "units_sold_last_26w": [3, 2]})
    fetch_results = {"1": True, "2": False}

    rows = build_manifest_rows("winner_rank_1", "A", ranked, Path("imgs"), fetch_results)

    assert rows == [
        {
            "role": "winner_rank_1",
            "style_key": "A",
            "article_id": 1,
            "units_sold_last_26w": 3,
            "local_image_path": str(Path("imgs") / "0000000001.jpg"),
            "fetch_success": True,
        },
        {
            "role": "winner_rank_1",
            "style_key": "A",
            "article_id": 2,
            "units_sold_last_26w": 2,
            "local_image_path": str(Path("imgs") / "0000000002.jpg"),
            "fetch_success": False,
        },
    ]
