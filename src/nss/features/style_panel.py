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

sales_channel_id VALIDATED FINDING: raw transactions ship with `sales_channel_id` in {1, 2}. No
data dictionary shipped with this Kaggle download (checked `data/raw/` for one -- none present)
authoritatively defines which value means "online" vs "in-store", so the mapping was resolved
empirically (see `nss.viz.channel_check`, which produced these numbers and
`reports/figures/channel_share.png`) using the COVID-19 lockdown as a natural experiment: physical
non-essential retail across most of Europe (including H&M) was forced to close mid-March through
end of April 2020, while online ordering continued.

Primary (deciding) signal -- weekly share of `sales_channel_id == 1` collapses during the trough
window (2020-03-15 to 2020-04-30) and recovers immediately after, while `sales_channel_id == 2`'s
share does the mirror image:

| period                          | share(channel=1) | share(channel=2) |
|----------------------------------|------------------:|------------------:|
| pre-trough baseline (78 weeks)   | 30.93%            | 69.07%            |
| COVID trough (7 weeks)           | 0.89%             | 99.11%            |
| post-trough (21 weeks)           | 33.63%            | 66.37%            |

Channel 1 has *literally zero* transactions in 4 of the 7 trough weeks (2020-03-23, 03-30, 04-06,
04-13) -- a near-total outage precisely coincident with the lockdown window, consistent only with
a channel that was physically forced to stop operating. Channel 2's share rises to 99-100% over
the same window. This is the signature of "stores closed, online continues": channel 1 = store,
channel 2 = online.

Supporting (non-deciding) evidence:
- Multi-year share trend, computed separately pre-trough (78 weeks) and post-trough (21 weeks) to
  avoid the trough itself distorting a linear fit: channel 1's share drifts slightly UP over both
  sub-periods (29.9% -> 31.3% pre-trough; 33.5% -> 33.8% post-trough) while channel 2's drifts
  slightly down by the same margin. This is a small, secular trend (order of 0.1 percentage
  points/week) and, taken alone, sits in tension with the general e-commerce-growth prior (online
  share was expected to trend up, not down) -- it does NOT corroborate the resolved mapping and is
  noted here for honesty rather than cherry-picked. It is treated as non-decisive: the effect size
  is an order of magnitude smaller than the trough signal, plausible confounds exist (e.g. physical
  store count / footprint changes over 2018-2020 unrelated to online penetration), and the task's
  own framing treats this signal as supporting, not sole-deciding.
- Mean transaction price: channel 1 mean=0.02292, median=0.01863 (n=9,408,462); channel 2
  mean=0.02989, median=0.02541 (n=22,379,862) (prices are the Kaggle dataset's normalized units,
  not currency). Channel 2 (resolved: online) has a ~30% higher mean price. Not diagnostic on its
  own -- different channels can have different typical basket/price compositions for reasons
  unrelated to online-vs-store -- but not contradictory either.

Net: the trough signal is mechanistically clear and an order of magnitude stronger than the mixed
supporting evidence, so the mapping below is treated as resolved. `units_online` / `units_store`
can be trusted directionally.

DENSITY: `build_style_week_panel` + `filter_by_support` alone produce a *sparse* panel -- one row
per (style_key, week_start) with >=1 sale, and nothing for zero-sale weeks. `densify_panel` (run
after filtering, in `main`) expands this to one row per style_key for every ISO week in
[`first_week_seen`, `last_week_seen`] -- that style's own observed lifetime, not the global date
range -- so sparsity stats and week-over-week comparisons are well-defined with no gaps. Fill
convention for the inserted zero-sale weeks, documented (not incidental): `units`, `revenue`,
`n_active_articles`, `n_customers`, `units_online`, `units_store`, `units_per_active_article` are
filled with 0 (verified zero counts). `mean_price` / `median_price` are left null -- no price was
observed that week, and filling with 0 would fabricate a false price signal.
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

# VALIDATED FINDING -- see module docstring for the empirical COVID-trough evidence.
SALES_CHANNEL_ONLINE = 2
SALES_CHANNEL_STORE = 1

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


# Zero-fill columns for weeks with no sales: verified zero counts, not unknowns.
_DENSIFY_INT_ZERO_COLS = [
    "units",
    "n_active_articles",
    "n_customers",
    "units_online",
    "units_store",
]
_DENSIFY_FLOAT_ZERO_COLS = ["revenue", "units_per_active_article"]


def densify_panel(filtered_panel: pl.DataFrame) -> pl.DataFrame:
    """Expand the sparse (sales-only) filtered panel to a dense grid per style's own lifetime.

    For every style_key present in `filtered_panel`, generates one row per ISO week in
    [`first_week_seen`, `last_week_seen`] inclusive -- that style's own observed active lifetime,
    not the global dataset date range -- then left-joins the real weekly aggregates onto the
    grid. See the module docstring's DENSITY section for the null/zero fill convention.

    Args:
        filtered_panel: The support-filtered sparse panel, as returned by joining
            `filter_by_support`'s output (one row per style-week with >=1 sale).

    Returns:
        The dense panel: one row per (style_key, week_start) for every week in
        [first_week_seen, last_week_seen], for every style_key in `filtered_panel`.
    """
    lifetime_bounds = filtered_panel.select(
        [*STYLE_KEY_COLS, "first_week_seen", "last_week_seen"]
    ).unique()

    grid = lifetime_bounds.with_columns(
        pl.date_ranges(
            pl.col("first_week_seen"), pl.col("last_week_seen"), interval="1w", closed="both"
        ).alias("week_start")
    ).explode("week_start", empty_as_null=True)

    dense = grid.join(
        filtered_panel.drop(["style_key", "first_week_seen", "last_week_seen"]),
        on=[*STYLE_KEY_COLS, "week_start"],
        how="left",
    )
    dense = dense.with_columns(
        [pl.col(c).fill_null(0) for c in _DENSIFY_INT_ZERO_COLS]
        + [pl.col(c).fill_null(0.0) for c in _DENSIFY_FLOAT_ZERO_COLS]
    )
    dense = dense.with_columns(
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
    return dense.select(column_order)


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

    dense_panel = densify_panel(filtered_panel)
    n_zero_sale = dense_panel.filter(pl.col("units") == 0).height
    pct_sparsity = 100.0 * n_zero_sale / dense_panel.height if dense_panel.height else 0.0
    print(f"Dense panel: {dense_panel.height} rows ({filtered_panel.height} had sales)")
    print(f"Sparsity: {n_zero_sale} zero-sale rows ({pct_sparsity:.2f}%)")

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    dense_panel.write_parquet(args.out_path)
    print(f"Wrote dense panel to {args.out_path} ({dense_panel.height} rows)")


if __name__ == "__main__":
    main()
