"""Build the style_key x ISO-week panel from H&M transactions + article metadata.

`style_key` is the composite of 5 article-level categorical columns from `articles.csv`:
`index_group_name`, `product_type_name`, `garment_group_name`, `perceived_colour_master_name`,
`graphical_appearance_name`. The panel groups every transaction by (style_key, week_start), where
`week_start` is the Monday of the ISO week containing `t_dat`.

Column choice: the panel keeps the 5 constituent style columns as the actual group/join keys
(robust, no separator-collision risk) *and* adds a derived `style_key` string column (the 5
values joined with " || ") as a convenience single-column identifier for downstream filtering and
joins. All grouping/aggregation logic operates on the 5 constituent columns, never on the string.

CAUSAL-SAFETY WARNING: `first_week_seen` and `last_week_seen` are lifetime constants computed
across the *entire* dataset and repeated on every row for a given style_key. `last_week_seen` in
particular is NOT available at prediction time for any week before it occurs -- using it (or any
column derived from it) as a predictive feature for a week prior to `last_week_seen` leaks future
information into the past. These two columns exist for panel-level descriptive/filtering use
only. Later phases must never feed `last_week_seen` into a model trained/scored as of a week
where it has not yet occurred.

sales_channel_id ASSUMPTION (NOT independently verified): raw transactions ship with
`sales_channel_id` in {1, 2}. No data dictionary shipped with this Kaggle download (checked
`data/raw/` for one -- none present) authoritatively defines which value means "online" vs
"in-store". This module assumes 1 = online, 2 = store, based on a convention recalled from
general familiarity with public H&M competition kernels/discussion -- this is NOT confirmed
against any authoritative source and confidence in the *direction* of the mapping is LOW. If the
online/store split matters for a downstream decision, verify independently before trusting
`units_online` / `units_store` directionally.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

STYLE_KEY_COLS: list[str] = [
    "index_group_name",
    "product_type_name",
    "garment_group_name",
    "perceived_colour_master_name",
    "graphical_appearance_name",
]

STYLE_KEY_SEPARATOR = " || "

# ASSUMPTION -- see module docstring. NOT independently verified against a data dictionary.
SALES_CHANNEL_ONLINE = 1
SALES_CHANNEL_STORE = 2

MIN_ARTICLES_PER_STYLE = 5
MIN_LIFETIME_UNITS_PER_STYLE = 500
RETENTION_GATE_PCT = 80.0

DEFAULT_TRANSACTIONS_DIR = Path("data/interim/transactions_train_parquet")
DEFAULT_ARTICLES_PATH = Path("data/raw/articles.csv")
DEFAULT_OUT_PATH = Path("data/processed/style_week_panel.parquet")


def _load_joined_transactions(transactions_dir: Path, articles_path: Path) -> pl.LazyFrame:
    """Join transactions with the 5 style columns from `articles.csv` and add `week_start`.

    Args:
        transactions_dir: Directory of the year-month partitioned transactions Parquet dataset.
        articles_path: Path to `articles.csv`.

    Returns:
        A lazy frame with one row per transaction, augmented with the 5 style columns and the
        Monday `week_start` date of the ISO week containing `t_dat`.
    """
    articles = pl.scan_csv(articles_path).select(["article_id", *STYLE_KEY_COLS])
    txns = pl.scan_parquet(str(transactions_dir / "**" / "*.parquet"))
    joined = txns.join(articles, on="article_id", how="inner")
    return joined.with_columns(pl.col("t_dat").dt.truncate("1w").alias("week_start"))


def build_style_week_panel(
    transactions_dir: Path = DEFAULT_TRANSACTIONS_DIR,
    articles_path: Path = DEFAULT_ARTICLES_PATH,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build the unfiltered style_key x ISO-week panel plus a lifetime style summary.

    Args:
        transactions_dir: Directory of the year-month partitioned transactions Parquet dataset.
        articles_path: Path to `articles.csv`.

    Returns:
        A tuple `(panel, lifetime)`:
        - `panel`: one row per (style_key, week_start) with columns `style_key`, the 5 style
          columns, `week_start`, `units`, `revenue`, `n_active_articles`,
          `units_per_active_article`, `mean_price`, `median_price`, `n_customers`,
          `units_online`, `units_store`, `first_week_seen`, `last_week_seen`.
        - `lifetime`: one row per style_key with lifetime aggregates (`lifetime_units`,
          `lifetime_n_articles`) used only for the support filter -- not part of the published
          panel schema.
    """
    joined = _load_joined_transactions(transactions_dir, articles_path)

    weekly = joined.group_by([*STYLE_KEY_COLS, "week_start"]).agg(
        units=pl.len(),
        revenue=pl.col("price").sum(),
        n_active_articles=pl.col("article_id").n_unique(),
        mean_price=pl.col("price").mean(),
        median_price=pl.col("price").median(),
        n_customers=pl.col("customer_id").n_unique(),
        units_online=(pl.col("sales_channel_id") == SALES_CHANNEL_ONLINE).sum(),
        units_store=(pl.col("sales_channel_id") == SALES_CHANNEL_STORE).sum(),
    )
    weekly = weekly.with_columns(
        (pl.col("units") / pl.col("n_active_articles")).alias("units_per_active_article")
    )

    lifetime = joined.group_by(STYLE_KEY_COLS).agg(
        lifetime_units=pl.len(),
        lifetime_n_articles=pl.col("article_id").n_unique(),
        first_week_seen=pl.col("week_start").min(),
        last_week_seen=pl.col("week_start").max(),
    )

    panel = weekly.join(
        lifetime.select([*STYLE_KEY_COLS, "first_week_seen", "last_week_seen"]),
        on=STYLE_KEY_COLS,
        how="left",
    )
    panel = panel.with_columns(
        pl.concat_str([pl.col(c) for c in STYLE_KEY_COLS], separator=STYLE_KEY_SEPARATOR).alias(
            "style_key"
        )
    )

    column_order = [
        "style_key",
        *STYLE_KEY_COLS,
        "week_start",
        "units",
        "revenue",
        "n_active_articles",
        "units_per_active_article",
        "mean_price",
        "median_price",
        "n_customers",
        "units_online",
        "units_store",
        "first_week_seen",
        "last_week_seen",
    ]
    panel = panel.select(column_order)

    panel_df, lifetime_df = pl.collect_all([panel, lifetime], engine="streaming")
    return panel_df, lifetime_df


def filter_by_support(
    panel: pl.DataFrame,
    lifetime: pl.DataFrame,
    min_articles: int = MIN_ARTICLES_PER_STYLE,
    min_units: int = MIN_LIFETIME_UNITS_PER_STYLE,
) -> tuple[pl.DataFrame, dict[str, float]]:
    """Apply the lifetime support filter and report retention statistics.

    Keeps only style_keys with >= `min_articles` distinct lifetime `article_id`s AND
    >= `min_units` lifetime units, both computed across the whole dataset (not per-week).

    Args:
        panel: Unfiltered style-week panel, as returned by `build_style_week_panel`.
        lifetime: Lifetime style summary, as returned by `build_style_week_panel`.
        min_articles: Minimum distinct lifetime articles per style_key to keep.
        min_units: Minimum lifetime units per style_key to keep.

    Returns:
        A tuple `(filtered_panel, stats)` where `stats` has keys `n_styles_before`,
        `n_styles_after`, and `pct_units_retained`.
    """
    total_units = lifetime["lifetime_units"].sum()
    n_before = lifetime.height

    kept = lifetime.filter(
        (pl.col("lifetime_n_articles") >= min_articles) & (pl.col("lifetime_units") >= min_units)
    )
    n_after = kept.height
    kept_units = kept["lifetime_units"].sum()
    pct_retained = 100.0 * kept_units / total_units if total_units else 0.0

    filtered_panel = panel.join(kept.select(STYLE_KEY_COLS), on=STYLE_KEY_COLS, how="inner")

    stats = {
        "n_styles_before": n_before,
        "n_styles_after": n_after,
        "pct_units_retained": pct_retained,
    }
    return filtered_panel, stats


def main() -> None:
    """CLI entry point: build the style-week panel, apply the support filter, write output.

    Refuses to write the filtered panel (exits non-zero) if lifetime-unit retention falls below
    `RETENTION_GATE_PCT` -- the support threshold must not be silently loosened; a human decides.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transactions-dir", type=Path, default=DEFAULT_TRANSACTIONS_DIR)
    parser.add_argument("--articles-path", type=Path, default=DEFAULT_ARTICLES_PATH)
    parser.add_argument("--out-path", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--retention-gate-pct", type=float, default=RETENTION_GATE_PCT)
    args = parser.parse_args()

    print("Building style x ISO-week panel...")
    panel, lifetime = build_style_week_panel(args.transactions_dir, args.articles_path)
    print(f"Unfiltered panel: {panel.height} rows, {panel.width} columns")

    filtered_panel, stats = filter_by_support(panel, lifetime)

    print(f"Style count before filtering: {stats['n_styles_before']}")
    print(f"Style count after filtering:  {stats['n_styles_after']}")
    print(f"% of lifetime units retained: {stats['pct_units_retained']:.2f}%")

    if stats["pct_units_retained"] < args.retention_gate_pct:
        print(
            f"GATE FAILED: retention {stats['pct_units_retained']:.2f}% is below the "
            f"{args.retention_gate_pct:.0f}% gate. NOT writing the filtered panel -- "
            "do not loosen the support threshold without explicit sign-off."
        )
        raise SystemExit(1)

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_panel.write_parquet(args.out_path)
    print(f"Wrote filtered panel to {args.out_path} ({filtered_panel.height} rows)")


if __name__ == "__main__":
    main()
