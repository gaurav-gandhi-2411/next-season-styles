"""Build the real per-(style, week) buyer-mix feature table from the raw H&M files.

Reads `customers.csv`, the month-partitioned transactions Parquet and `articles.csv` (all
read-only, under `NSS_DATA_ROOT`, default `data/`), maps every transaction to
`(style, week, customer)`, and calls `nss.features.customer_features.build_customer_features` for
EVERY panel `(style, week)` row.

PRIVACY: `customer_id`, `postal_code` and the per-customer attributes exist only in memory. The
single output is a per-(style, week) aggregate table written to the gitignored scratch folder
`reports/u_scratch/customer_features.parquet`; it holds no customer-level row.

Usage:
    python -m nss.features.customer_features_build
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import polars as pl

from nss.features.customer_features import build_customer_features
from nss.features.style_panel import STYLE_KEY_COLS, STYLE_KEY_SEPARATOR

DATA_ROOT = Path(os.environ.get("NSS_DATA_ROOT", "data"))
PANEL_PATH = DATA_ROOT / "processed" / "style_week_panel.parquet"
ARTICLES_PATH = DATA_ROOT / "raw" / "articles.csv"
CUSTOMERS_PATH = DATA_ROOT / "raw" / "customers.csv"
TRANSACTIONS_DIR = DATA_ROOT / "interim" / "transactions_train_parquet"
SCRATCH_DIR = Path("reports/u_scratch")
OUT_PATH = SCRATCH_DIR / "customer_features.parquet"


def load_purchases(styles: pl.DataFrame, customer_ids: pl.DataFrame) -> pl.DataFrame:
    """All transactions as `(style_key code, week_start, customer_id code)` rows.

    Args:
        styles: `style_key` (str) and `style_code` (UInt32) for the panel's styles.
        customer_ids: `customer_id` (str) and `cid` (UInt32) for every customer.

    Returns:
        One row per transaction line of a panel style's article, with integer codes only.
    """
    articles = (
        pl.read_csv(ARTICLES_PATH, columns=["article_id", *STYLE_KEY_COLS])
        .with_columns(
            pl.concat_str([pl.col(c) for c in STYLE_KEY_COLS], separator=STYLE_KEY_SEPARATOR).alias(
                "style_key"
            )
        )
        .join(styles, on="style_key", how="inner")
        .select("article_id", "style_code")
    )
    parts: list[pl.DataFrame] = []
    for month_dir in sorted(TRANSACTIONS_DIR.iterdir()):
        tx = pl.read_parquet(
            month_dir / "*.parquet", columns=["t_dat", "customer_id", "article_id"]
        )
        parts.append(
            tx.join(customer_ids, on="customer_id", how="inner")
            .join(articles, on="article_id", how="inner")
            .select(
                pl.col("style_code").alias("style_key"),
                pl.col("t_dat").dt.truncate("1w").alias("week_start"),
                pl.col("cid").alias("customer_id"),
            )
        )
    return pl.concat(parts)


def main() -> None:
    """Build and write the scratch table; print row counts and timing (no customer-level output)."""
    started = time.time()
    panel = pl.read_parquet(PANEL_PATH, columns=["style_key", "week_start"])
    styles = (
        panel.select("style_key")
        .unique()
        .sort("style_key")
        .with_row_index("style_code")
        .with_columns(pl.col("style_code").cast(pl.UInt32))
    )
    customers_raw = pl.read_csv(
        CUSTOMERS_PATH,
        schema_overrides={"age": pl.Float64, "postal_code": pl.String},
        columns=[
            "customer_id",
            "age",
            "club_member_status",
            "fashion_news_frequency",
            "postal_code",
        ],
    ).with_row_index("cid")
    customer_ids = customers_raw.select("customer_id", pl.col("cid").cast(pl.UInt32))
    customers = customers_raw.drop("customer_id").rename({"cid": "customer_id"})
    print(f"customers: {customers.height}, styles: {styles.height}")

    purchases = load_purchases(styles, customer_ids)
    print(
        f"purchase lines mapped to panel styles: {purchases.height} ({time.time() - started:.0f}s)"
    )

    panel_coded = panel.join(styles, on="style_key").select(
        pl.col("style_code").alias("style_key"), "week_start"
    )
    features = build_customer_features(
        purchases.with_columns(pl.col("style_key").cast(pl.UInt32)),
        customers.with_columns(pl.col("customer_id").cast(pl.UInt32)),
        panel_coded,
        origin_weeks=None,
    )
    features = (
        features.rename({"style_key": "style_code"})
        .join(styles, on="style_code")
        .drop("style_code")
        .select("style_key", "origin_week", pl.exclude("style_key", "origin_week"))
    )
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    features.write_parquet(OUT_PATH)
    print(f"wrote {OUT_PATH}: {features.height} style-week rows ({time.time() - started:.0f}s)")
    with pl.Config(tbl_rows=30, tbl_cols=6, float_precision=4):
        print(features.select(pl.exclude("style_key", "origin_week")).describe())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # polars prints Unicode; avoid cp1252 crash
    main()
