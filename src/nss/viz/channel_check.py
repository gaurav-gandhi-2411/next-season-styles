"""Empirically resolve the `sales_channel_id` online/store mapping from raw transactions.

`nss.features.style_panel` flags `sales_channel_id` in {1, 2} as an unverified assumption -- no
data dictionary ships with the raw Kaggle download to say which value means "online" vs
"in-store". This module resolves it empirically using the COVID-19 lockdown as a natural
experiment: physical non-essential retail across most of Europe (including H&M) was forced to
close mid-March through April 2020, while online ordering continued. The channel whose SHARE of
weekly transactions (not necessarily absolute volume -- everything drops in absolute terms during
a demand shock) rises sharply during that window, relative to its pre-lockdown baseline, is the
online channel; the channel whose share collapses is the store channel.

Supporting (non-deciding) evidence considered alongside the trough signal: a multi-year share
trend across the full 2018-2020 window, and mean transaction price by channel.

This module is the source of the "VALIDATED FINDING" documented in
`nss.features.style_panel`'s module docstring and `SALES_CHANNEL_ONLINE`/`SALES_CHANNEL_STORE`
constants. Re-run `main()` against the raw transactions to reproduce the numbers.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless -- this module only writes PNGs, never shows a window.
import matplotlib.pyplot as plt
import polars as pl

DEFAULT_TRANSACTIONS_DIR = Path("data/interim/transactions_train_parquet")
DEFAULT_FIGURES_DIR = Path("reports/figures")

# COVID-19 lockdown trough: mid-March through end of April 2020, the period of sharpest
# lockdown-driven in-store disruption across most of Europe.
TROUGH_START = datetime.date(2020, 3, 15)
TROUGH_END = datetime.date(2020, 4, 30)

CHANNEL_IDS = [1, 2]


def weekly_channel_shares(transactions_dir: Path = DEFAULT_TRANSACTIONS_DIR) -> pl.DataFrame:
    """Compute each `sales_channel_id`'s share of weekly transaction volume.

    Every (week_start, channel) combination is present in the output -- including weeks where a
    channel had literally zero transactions -- so shares are well-defined (sum to 1.0) every week
    and a channel's near-total absence during a window is visible as `n == 0`, not a missing row.

    Args:
        transactions_dir: Directory of the year-month partitioned transactions Parquet dataset.

    Returns:
        One row per (week_start, sales_channel_id) with `n` (transaction count) and `share`
        (fraction of that week's total transactions), sorted by (week_start, sales_channel_id).
    """
    txns = pl.scan_parquet(str(transactions_dir / "**" / "*.parquet")).select(
        ["t_dat", "sales_channel_id"]
    )
    weekly = (
        txns.with_columns(pl.col("t_dat").dt.truncate("1w").alias("week_start"))
        .group_by(["week_start", "sales_channel_id"])
        .agg(n=pl.len())
        .collect(engine="streaming")
    )

    all_weeks = weekly.select("week_start").unique().sort("week_start")
    grid = all_weeks.join(pl.DataFrame({"sales_channel_id": CHANNEL_IDS}), how="cross")
    weekly_full = grid.join(weekly, on=["week_start", "sales_channel_id"], how="left").with_columns(
        pl.col("n").fill_null(0)
    )
    totals = weekly_full.group_by("week_start").agg(total=pl.col("n").sum())
    shares = weekly_full.join(totals, on="week_start").with_columns(
        (pl.col("n") / pl.col("total")).alias("share")
    )
    return shares.sort(["week_start", "sales_channel_id"])


def trough_vs_baseline_summary(
    shares: pl.DataFrame,
    trough_start: datetime.date = TROUGH_START,
    trough_end: datetime.date = TROUGH_END,
) -> pl.DataFrame:
    """Summarize mean channel share pre-trough, during the trough, and post-trough.

    Args:
        shares: Output of `weekly_channel_shares`.
        trough_start: First `week_start` (inclusive) of the COVID lockdown trough window.
        trough_end: Last `week_start` (inclusive) of the trough window.

    Returns:
        One row per (period, sales_channel_id) with `mean_share`, `n_weeks`, and `total_n`,
        where `period` is one of `"pre_trough"`, `"trough"`, `"post_trough"`.
    """
    labeled = shares.with_columns(
        pl.when(pl.col("week_start") < trough_start)
        .then(pl.lit("pre_trough"))
        .when(pl.col("week_start") <= trough_end)
        .then(pl.lit("trough"))
        .otherwise(pl.lit("post_trough"))
        .alias("period")
    )
    return (
        labeled.group_by(["period", "sales_channel_id"])
        .agg(
            mean_share=pl.col("share").mean(),
            n_weeks=pl.col("week_start").n_unique(),
            total_n=pl.col("n").sum(),
        )
        .sort(["period", "sales_channel_id"])
    )


def channel_mean_price(transactions_dir: Path = DEFAULT_TRANSACTIONS_DIR) -> pl.DataFrame:
    """Compute mean and median transaction price by `sales_channel_id`, across all transactions.

    Supporting (non-deciding) evidence only -- different channels can have different typical
    basket/price compositions for reasons unrelated to online-vs-store.

    Args:
        transactions_dir: Directory of the year-month partitioned transactions Parquet dataset.

    Returns:
        One row per `sales_channel_id` with `mean_price`, `median_price`, `n`.
    """
    txns = pl.scan_parquet(str(transactions_dir / "**" / "*.parquet")).select(
        ["price", "sales_channel_id"]
    )
    return (
        txns.group_by("sales_channel_id")
        .agg(mean_price=pl.col("price").mean(), median_price=pl.col("price").median(), n=pl.len())
        .collect(engine="streaming")
        .sort("sales_channel_id")
    )


def plot_channel_share(
    shares: pl.DataFrame,
    trough_start: datetime.date = TROUGH_START,
    trough_end: datetime.date = TROUGH_END,
    out_path: Path = DEFAULT_FIGURES_DIR / "channel_share.png",
) -> Path:
    """Plot each channel's weekly share over time, with the COVID trough window shaded.

    Args:
        shares: Output of `weekly_channel_shares`.
        trough_start: First `week_start` (inclusive) of the shaded trough window.
        trough_end: Last `week_start` (inclusive) of the shaded trough window.
        out_path: Destination PNG path; parent directories are created if missing.

    Returns:
        `out_path`, for convenience chaining.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    for ch in CHANNEL_IDS:
        series = shares.filter(pl.col("sales_channel_id") == ch).sort("week_start")
        ax.plot(
            series["week_start"], series["share"], label=f"sales_channel_id={ch}", linewidth=1.4
        )

    ax.axvspan(
        trough_start,
        trough_end,
        color="red",
        alpha=0.15,
        label="COVID trough (2020-03-15 to 2020-04-30)",
    )
    ax.set_xlabel("Week")
    ax.set_ylabel("Share of weekly transaction volume")
    ax.set_title("Weekly sales_channel_id share, with COVID-19 lockdown trough shaded")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_ylim(0.0, 1.0)
    fig.autofmt_xdate()
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    """Run the full channel-mapping resolution: shares, trough summary, price, and the plot.

    Prints every number needed to reproduce the "VALIDATED FINDING" documented in
    `nss.features.style_panel` and writes `reports/figures/channel_share.png`.
    """
    shares = weekly_channel_shares()
    summary = trough_vs_baseline_summary(shares)
    print("=== Mean share by period ===")
    print(summary)

    price = channel_mean_price()
    print("\n=== Mean/median price by channel (all transactions) ===")
    print(price)

    plot_path = plot_channel_share(shares)
    print(f"\nWrote {plot_path}")


if __name__ == "__main__":
    main()
