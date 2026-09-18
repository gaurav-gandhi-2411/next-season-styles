from __future__ import annotations

import datetime
from pathlib import Path

import polars as pl

from nss.viz.channel_check import trough_vs_baseline_summary, weekly_channel_shares


def _write_txn_fixture(tmp_path: Path) -> Path:
    """Write a tiny synthetic partitioned transactions Parquet dataset for channel-share tests.

    Two ISO weeks: 2018-09-17 has 3 channel-1 txns + 1 channel-2 txn (share: 0.75 / 0.25).
    2018-09-24 has 1 channel-1 txn + 3 channel-2 txns (share: 0.25 / 0.75) -- deliberately the
    mirror image so a test can assert both weeks' shares independently sum to 1.0.
    """
    transactions_dir = tmp_path / "transactions"
    transactions_dir.mkdir()
    txns = pl.DataFrame(
        {
            "t_dat": [
                "2018-09-17",
                "2018-09-18",
                "2018-09-19",
                "2018-09-20",
                "2018-09-24",
                "2018-09-25",
                "2018-09-26",
                "2018-09-27",
            ],
            "sales_channel_id": [1, 1, 1, 2, 1, 2, 2, 2],
        }
    ).with_columns(pl.col("t_dat").str.to_date())
    part_dir = transactions_dir / "part"
    part_dir.mkdir()
    txns.write_parquet(part_dir / "data.parquet")
    return transactions_dir


def test_weekly_channel_shares_sums_to_one(tmp_path: Path) -> None:
    """Each week's channel shares sum to 1.0 and match a hand tally."""
    transactions_dir = _write_txn_fixture(tmp_path)
    shares = weekly_channel_shares(transactions_dir)

    week1 = shares.filter(pl.col("week_start") == datetime.date(2018, 9, 17)).sort(
        "sales_channel_id"
    )
    assert week1["n"].to_list() == [3, 1]
    assert week1["share"].to_list() == [0.75, 0.25]

    week2 = shares.filter(pl.col("week_start") == datetime.date(2018, 9, 24)).sort(
        "sales_channel_id"
    )
    assert week2["n"].to_list() == [1, 3]
    assert week2["share"].to_list() == [0.25, 0.75]

    totals = shares.group_by("week_start").agg(s=pl.col("share").sum())
    assert (totals["s"] - 1.0).abs().max() < 1e-12


def test_weekly_channel_shares_includes_zero_txn_weeks_for_absent_channel(tmp_path: Path) -> None:
    """A channel with zero transactions in a week still gets an explicit n=0 / share=0.0 row.

    Regression guard for the exact signature the COVID-trough resolution depends on: a channel
    collapsing to zero transactions must be visible as `n == 0`, not a missing row that a naive
    join could silently drop.
    """
    transactions_dir = tmp_path / "transactions"
    transactions_dir.mkdir()
    txns = pl.DataFrame(
        {"t_dat": ["2018-09-17", "2018-09-18"], "sales_channel_id": [1, 1]}
    ).with_columns(pl.col("t_dat").str.to_date())
    part_dir = transactions_dir / "part"
    part_dir.mkdir()
    txns.write_parquet(part_dir / "data.parquet")

    shares = weekly_channel_shares(transactions_dir)
    channel_2_row = shares.filter(pl.col("sales_channel_id") == 2)
    assert channel_2_row.height == 1
    assert channel_2_row["n"].item() == 0
    assert channel_2_row["share"].item() == 0.0


def test_trough_vs_baseline_summary_labels_periods(tmp_path: Path) -> None:
    """Weeks are correctly bucketed into pre_trough / trough / post_trough by week_start."""
    transactions_dir = _write_txn_fixture(tmp_path)
    shares = weekly_channel_shares(transactions_dir)

    summary = trough_vs_baseline_summary(
        shares,
        trough_start=datetime.date(2018, 9, 24),
        trough_end=datetime.date(2018, 9, 24),
    )

    periods_seen = set(summary["period"].to_list())
    assert periods_seen == {"pre_trough", "trough"}
    trough_ch2 = summary.filter((pl.col("period") == "trough") & (pl.col("sales_channel_id") == 2))
    assert trough_ch2["mean_share"].item() == 0.75
    assert trough_ch2["n_weeks"].item() == 1
