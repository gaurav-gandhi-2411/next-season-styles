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

PRICE_INDEX (`add_price_index`, run after `densify_panel`): `price_index = mean_price / (trailing
`PRICE_INDEX_TRAILING_WEEKS`-week median of mean_price for that style_key, using only weeks
strictly before the current one)`. Two causal-safety points, both load-bearing:
- The trailing window is built via `.shift(1)` before the rolling median, so the current row's own
  `mean_price` never contributes to its own denominator.
- "52 prior weeks" is enforced as a *row-count* gate (`>= PRICE_INDEX_TRAILING_WEEKS` prior rows in
  the style's own dense per-week series), separate from `rolling_median`'s `min_samples`. This
  matters because ~10.7% of style-weeks are zero-sale (`mean_price` null) -- requiring all 52
  trailing rows to be non-null (`min_samples=52`) would fail on almost every style-week (a run of
  52 consecutive non-null weeks has probability roughly 0.89^52 =~ 0.2% under that null rate), which
  is far stricter than "52 prior weeks of history" and would make the column useless. Instead
  `min_samples=1` lets the median use whatever non-null prices were actually observed among the 52
  trailing rows (zero-sale weeks simply contribute no price observation, same non-fabrication
  principle as the panel's own `mean_price` null-fill above), and the separate row-count gate is
  what actually enforces "needs 52 prior calendar weeks of history" per the module's documented
  contract. NULLABLE EARLY ON: any style-week with fewer than `PRICE_INDEX_TRAILING_WEEKS` prior
  rows (i.e., early in a style's observed lifetime) gets `price_index = null` by design -- not
  imputed, not backfilled.

INTENSITY_SHRUNK (`add_intensity_shrunk`, run after `densify_panel`): empirical-Bayes shrinkage of
`units_per_active_article` toward its `(index_group_name, garment_group_name)` group's mean, with
shrinkage strength `w = n_active_articles / (n_active_articles + K_SHRINKAGE)` and
`shrunk = w * raw + (1 - w) * group_mean`. See `K_SHRINKAGE` for the prior-strength constant and
its justification.

CAUSAL-SAFETY DESIGN DECISION -- the group mean is a TRAILING (expanding, current-week-excluded)
statistic, never a same-week cross-sectional average across other styles in the group. A same-week
cross-sectional mean (e.g. "average `units_per_active_article` across all styles in this
index_group/garment_group THIS week") is not actually knowable at prediction time: in a real
rolling-origin forecast made as of week W, other styles' week-W sales have not happened yet either
-- they are exactly as unobserved as the style's own future. Using a same-week group average would
smuggle future information about the whole cohort into a feature computed "for" week W, which is
precisely the kind of leakage this project's causal-safety constraint exists to prevent. The
trailing/expanding formulation instead computes, for each `(index_group_name, garment_group_name,
week_start)`, the mean `units_per_active_article` across that group's ACTIVE (`n_active_articles >
0`) style-weeks -- consistent with `nss.viz.panel_eda`'s judgment call 1 on what "intensity" means
-- then takes an expanding (all-history), current-week-EXCLUDED mean of those weekly group means
over calendar time. Every input to the shrinkage prior for a given week is therefore dated strictly
before that week, attached to every later week -- active or not -- via an as-of ("most recent
group history strictly before this week") join, so a week where the group itself had zero active
styles still correctly carries forward the last known trailing mean rather than going null (an
earlier version of this function used a plain equi-join, which incorrectly nulled every such gap
week, not just the group's genuinely-first week -- fixed via `join_asof`, see the function's
implementation comments). NULLABLE ONLY: style-weeks strictly before the group's very first ever
active week across the whole dataset get `intensity_shrunk = null` (no group history exists yet);
this is rare and mostly confined to the dataset's very first observed week, since index_group/
garment_group pairs are broad categories nearly all present from the start of the dataset.
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

# See module docstring, PRICE_INDEX section, for the full row-count-gate rationale.
PRICE_INDEX_TRAILING_WEEKS = 52

# Empirical-Bayes prior strength for `add_intensity_shrunk`. Set to the MEDIAN `n_active_articles`
# across active (n_active_articles > 0) style-weeks in `data/processed/style_week_panel.parquet`
# (measured: 3.0 -- see `n_active_articles.describe()` over the active subset). This choice makes
# the shrinkage weight interpretable: a style-week with exactly the typical (median) amount of
# evidence gets w = k / (k + k) = 0.5, i.e. equal trust in its own raw signal and the group's
# trailing prior; below-median-evidence weeks (including zero-sale weeks, w = 0) lean more on the
# group prior, above-median weeks lean more on their own observation.
K_SHRINKAGE = 3.0

# Group used for the empirical-Bayes prior in `add_intensity_shrunk`.
INTENSITY_GROUP_COLS: list[str] = ["index_group_name", "garment_group_name"]

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


def add_price_index(dense_panel: pl.DataFrame) -> pl.DataFrame:
    """Add `price_index`: this week's `mean_price` relative to its own trailing median price.

    `price_index = mean_price / (trailing PRICE_INDEX_TRAILING_WEEKS-week median of mean_price for
    that style_key, using only weeks strictly before the current one)`. See the module docstring's
    PRICE_INDEX section for the full causal-safety rationale (why the row-count gate is separate
    from `rolling_median`'s own null-handling). Must be run on the DENSE panel (one row per
    style_key per calendar week within that style's lifetime) -- the trailing window is a row-count
    window, which is only a calendar-week window if the input has no gaps.

    Args:
        dense_panel: The dense style-week panel, as returned by `densify_panel`.

    Returns:
        `dense_panel` with an added `price_index` column (nullable -- see docstring).
    """
    prior_row_count = pl.col("week_start").cum_count().over("style_key", order_by="week_start") - 1
    trailing_median_price = (
        pl.col("mean_price")
        .shift(1)
        .rolling_median(window_size=PRICE_INDEX_TRAILING_WEEKS, min_samples=1)
        .over("style_key", order_by="week_start")
    )
    trailing_median_price = (
        pl.when(prior_row_count >= PRICE_INDEX_TRAILING_WEEKS)
        .then(trailing_median_price)
        .otherwise(None)
    )

    return dense_panel.with_columns(
        (pl.col("mean_price") / trailing_median_price).alias("price_index")
    )


def add_intensity_shrunk(
    dense_panel: pl.DataFrame,
    k: float = K_SHRINKAGE,
    group_cols: list[str] = INTENSITY_GROUP_COLS,
) -> pl.DataFrame:
    """Add `intensity_shrunk`: empirical-Bayes shrinkage of `units_per_active_article`.

    `shrunk = w * raw + (1 - w) * group_mean_trailing`, `w = n_active_articles / (n_active_articles
    + k)`. See the module docstring's INTENSITY_SHRUNK section for the full formula and, critically,
    why `group_mean_trailing` is a TRAILING (expanding, current-week-excluded) statistic rather than
    a same-week cross-sectional group average (the latter is causally unsafe). Must be run on the
    DENSE panel for the same reason as `add_price_index`.

    Args:
        dense_panel: The dense style-week panel, as returned by `densify_panel`.
        k: Empirical-Bayes prior-strength constant. Defaults to `K_SHRINKAGE`.
        group_cols: Columns defining the shrinkage group. Defaults to `INTENSITY_GROUP_COLS`.

    Returns:
        `dense_panel` with an added `intensity_shrunk` column. Nullable only for style-weeks that
        occur strictly before the group's very first ever active week (no trailing group history
        exists yet); every later week -- active or not -- carries forward the most recent trailing
        group mean via an as-of join (see implementation comment below for why a plain equi-join on
        `(group_cols, week_start)` is NOT sufficient here).
    """
    # One row per (group, week) where the group had >=1 active style that week -- weeks where the
    # WHOLE group was zero-sale (can happen for small/niche groups, e.g. a group with only 1-2
    # style_keys) are simply absent from this table, not present with a 0.
    group_week = (
        dense_panel.filter(pl.col("n_active_articles") > 0)
        .group_by([*group_cols, "week_start"])
        .agg(group_week_mean_intensity=pl.col("units_per_active_article").mean())
        .sort([*group_cols, "week_start"])
    )
    # Cumulative mean INCLUSIVE of each active week's own contribution (not shifted) -- this is
    # intentional, see the as-of join below for why inclusive is correct here.
    group_week = group_week.with_columns(
        cum_mean_inclusive=(
            pl.col("group_week_mean_intensity").cum_sum().over(group_cols, order_by="week_start")
            / pl.col("group_week_mean_intensity")
            .cum_count()
            .over(group_cols, order_by="week_start")
        )
    ).select([*group_cols, "week_start", "cum_mean_inclusive"])

    # An equi-join on (group_cols, week_start) would only attach a trailing value to weeks that are
    # THEMSELVES present in `group_week` -- i.e. weeks where the group had active styles -- leaving
    # every gap week (group had zero active styles that week, but plenty of earlier history) null,
    # even though a valid trailing value exists. Fixed with an as-of ("most recent row at or before
    # this point") backward join instead: probing with `week_start - 1 day` (rather than
    # `week_start` itself) makes the match STRICT ("before", not "at or before") -- so a style's own
    # active week can never match onto its own group_week row and leak its own current-week
    # contribution into its own trailing prior. The matched row's `cum_mean_inclusive` already
    # reflects all group history up to and including that (earlier) active week, which is exactly
    # "all active history strictly before the target week" since by construction no active week
    # exists between the match and the target.
    probe = (
        dense_panel.select([*group_cols, "week_start"])
        .with_row_index("_row_id")
        .with_columns((pl.col("week_start") - pl.duration(days=1)).alias("_probe_week"))
        .sort([*group_cols, "_probe_week"])
    )
    trailing = (
        probe.join_asof(
            group_week.sort([*group_cols, "week_start"]),
            left_on="_probe_week",
            right_on="week_start",
            by=group_cols,
            strategy="backward",
        )
        .select(["_row_id", "cum_mean_inclusive"])
        .rename({"cum_mean_inclusive": "group_mean_trailing"})
    )

    joined = (
        dense_panel.with_row_index("_row_id")
        .join(trailing, on="_row_id", how="left", maintain_order="left")
        .drop("_row_id")
    )
    weight = pl.col("n_active_articles") / (pl.col("n_active_articles") + k)
    joined = joined.with_columns(
        (
            weight * pl.col("units_per_active_article")
            + (1 - weight) * pl.col("group_mean_trailing")
        ).alias("intensity_shrunk")
    )
    return joined.drop("group_mean_trailing")


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

    dense_panel = add_price_index(dense_panel)
    dense_panel = add_intensity_shrunk(dense_panel)
    n_price_index_null = dense_panel["price_index"].null_count()
    pct_price_index_null = (
        100.0 * n_price_index_null / dense_panel.height if dense_panel.height else 0.0
    )
    n_intensity_shrunk_null = dense_panel["intensity_shrunk"].null_count()
    print(
        f"price_index: {n_price_index_null} null rows ({pct_price_index_null:.2f}%, "
        f"expected -- early-life styles with < {PRICE_INDEX_TRAILING_WEEKS} prior weeks)"
    )
    print(f"intensity_shrunk: {n_intensity_shrunk_null} null rows (expected: near 0)")

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    dense_panel.write_parquet(args.out_path)
    print(f"Wrote dense panel to {args.out_path} ({dense_panel.height} rows)")


if __name__ == "__main__":
    main()
