"""Colour concentration vs base rate: why is everything black?

Question: do the incumbent/emerging leaderboards over-concentrate on Black relative to how much
Black sells, or do they simply reflect the market? Reported both ways, whichever way it falls:

a) share of total units by `perceived_colour_master_name` over the full period (the modelled
   panel, and, for reference, the raw transactions);
b) the colour mix of the top-10 incumbent and top-10 emerging leaderboards;
c) the ratio leaderboard share / sales share, per colour;
d) whether "Black" is a coarse bucket: a fixed-seed sample of 30 Black articles with their
   `colour_group_name` / `perceived_colour_value_name` / `detail_desc`, and the share of Black
   units whose finer colour group is not "Black" (charcoal / dark grey absorbed into Black).

Writes `reports/tables/colour_base_rate.csv` (long format, one row per section x colour) and
`reports/tables/colour_black_sample.csv` (the 30-article sample).

Usage:
    uv run python -m nss.features.colour_base_rate
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

PANEL = Path("data/processed/style_week_panel.parquet")
ARTICLES = Path("data/raw/articles.csv")
TRANSACTIONS = Path("data/interim/transactions_train_parquet")
T1 = Path("reports/tables/top_styles_incumbent.csv")
T2 = Path("reports/tables/top_styles_emerging.csv")
OUT = Path("reports/tables/colour_base_rate.csv")
SAMPLE_OUT = Path("reports/tables/colour_black_sample.csv")
COL = "perceived_colour_master_name"
TOP_N = 10
SEED = 42


def share_table(df: pl.DataFrame, weight: str | None, section: str) -> pl.DataFrame:
    """Colour shares for `df` (weighted by `weight` column, else row counts), share-sorted."""
    agg = (
        df.group_by(COL).agg((pl.col(weight).sum() if weight else pl.len()).alias("value"))
        if weight
        else df.group_by(COL).agg(pl.len().alias("value"))
    )
    total = agg["value"].sum()
    return (
        agg.with_columns((pl.col("value") / total).alias("share"), pl.lit(section).alias("section"))
        .rename({COL: "colour"})
        .sort("share", descending=True)
    )


def main() -> None:
    """Compute and write the tables; print the headline comparison."""
    panel = pl.read_parquet(PANEL)
    articles = pl.read_csv(ARTICLES)

    panel_units = share_table(panel, "units", "panel_units")
    txn = pl.scan_parquet(str(TRANSACTIONS / "**" / "*.parquet")).group_by("article_id").len()
    raw = (
        txn.collect()
        .join(articles.select("article_id", COL), on="article_id", how="left")
        .group_by(COL)
        .agg(pl.col("len").sum().alias("value"))
    )
    raw_units = (
        raw.with_columns((pl.col("value") / pl.col("value").sum()).alias("share"))
        .rename({COL: "colour"})
        .with_columns(pl.lit("raw_transactions_units").alias("section"))
        .sort("share", descending=True)
    )
    t1 = share_table(pl.read_csv(T1).head(TOP_N), None, "t1_incumbent_top10")
    t2 = share_table(pl.read_csv(T2).head(TOP_N), None, "t2_emerging_top10")
    both = share_table(
        pl.concat(
            [pl.read_csv(T1).head(TOP_N), pl.read_csv(T2).head(TOP_N)], how="diagonal_relaxed"
        ),
        None,
        "t1_t2_top10_combined",
    )
    # ratio = leaderboard share / sales share (panel units), per colour
    base = panel_units.select("colour", pl.col("share").alias("base_share"))
    ratios = (
        pl.concat([t1, t2, both])
        .join(base, on="colour", how="left")
        .with_columns((pl.col("share") / pl.col("base_share")).alias("ratio_to_sales_share"))
    )
    out = pl.concat(
        [
            panel_units.head(10).with_columns(
                pl.lit(None, dtype=pl.Float64).alias("base_share"),
                pl.lit(None, dtype=pl.Float64).alias("ratio_to_sales_share"),
            ),
            raw_units.head(10).with_columns(
                pl.lit(None, dtype=pl.Float64).alias("base_share"),
                pl.lit(None, dtype=pl.Float64).alias("ratio_to_sales_share"),
            ),
            ratios.select(panel_units.columns + ["base_share", "ratio_to_sales_share"]),
        ],
        how="vertical_relaxed",
    )

    # Extra base rates for the incumbent (intensity) leaderboard: sales share is not the only fair
    # base. (i) colour share among ALL guard-passing styles (the pool the incumbent leaderboard
    # draws from), by style count;
    # (ii) colour mix of the REALISED top-10 by trailing-13-week intensity in that same pool -- if
    #      the market's own top-10 looks like the model's, the model reflects the market.
    from nss.models import final_forecast

    origin = final_forecast.FORECAST_ORIGIN
    guard = final_forecast.build_guard_frame(panel, origin)
    price = panel.filter(pl.col("week_start") == origin).select(
        "style_key", pl.col("price_index").alias("price_index_level")
    )
    passing = guard.join(price, on="style_key", how="left").filter(
        (
            pl.col("guard1_n_active_articles_trailing_mean")
            >= final_forecast.GUARD1_MIN_MEAN_N_ACTIVE_ARTICLES
        )
        & pl.col("price_index_level").is_not_null()
        & (pl.col("price_index_level") >= final_forecast.GUARD2_MIN_PRICE_INDEX)
        & (pl.col("guard3_n_weeks_active_trailing") >= final_forecast.GUARD3_MIN_WEEKS_ACTIVE)
    )
    print("guard-passing pool:", passing.height, "styles")
    trailing = (
        panel.filter(pl.col("week_start") > origin - pl.duration(weeks=13))
        .filter(pl.col("week_start") <= origin)
        .group_by("style_key")
        .agg(pl.col("units_per_active_article").mean().alias("realised_13w_intensity"))
    )
    style_colours = panel.select("style_key", COL).unique()
    pool = passing.join(trailing, on="style_key").join(style_colours, on="style_key")
    pool_share = share_table(pool, None, "guard_passing_styles_by_count")
    realised_top = share_table(
        pool.sort("realised_13w_intensity", descending=True).head(TOP_N),
        None,
        "realised_top10_trailing13w_intensity",
    )
    extra = (
        pl.concat([pool_share.head(10), realised_top])
        .with_columns(
            pl.lit(None, dtype=pl.Float64).alias("base_share"),
            pl.lit(None, dtype=pl.Float64).alias("ratio_to_sales_share"),
        )
        .select(out.columns)
    )
    out = pl.concat([out, extra], how="vertical_relaxed")

    # d) is Black a coarse bucket?
    black = articles.filter(pl.col(COL) == "Black")
    groups = (
        black.group_by("colour_group_name", "perceived_colour_value_name")
        .len()
        .sort("len", descending=True)
    )
    units_by_article = (
        pl.scan_parquet(str(TRANSACTIONS / "**" / "*.parquet"))
        .group_by("article_id")
        .len()
        .collect()
    )
    black_units = black.join(units_by_article, on="article_id", how="left").with_columns(
        pl.col("len").fill_null(0)
    )
    by_group = black_units.group_by("colour_group_name").agg(pl.col("len").sum().alias("units"))
    tot = by_group["units"].sum()
    finer = by_group.with_columns((pl.col("units") / tot).alias("share_of_black_units")).sort(
        "units", descending=True
    )
    finer_rows = finer.select(
        pl.lit("black_bucket_colour_group").alias("section"),
        pl.col("colour_group_name").alias("colour"),
        pl.col("units").cast(pl.Float64).alias("value"),
        pl.col("share_of_black_units").alias("share"),
        pl.lit(None, dtype=pl.Float64).alias("base_share"),
        pl.lit(None, dtype=pl.Float64).alias("ratio_to_sales_share"),
    ).select(out.columns)
    out = pl.concat([out, finer_rows], how="vertical_relaxed")
    out.write_csv(OUT)

    sample = black.sample(n=30, seed=SEED).select(
        "article_id",
        "product_type_name",
        "colour_group_name",
        "perceived_colour_value_name",
        "perceived_colour_master_name",
        "detail_desc",
    )
    sample.write_csv(SAMPLE_OUT)

    with pl.Config(tbl_rows=30, tbl_width_chars=200, fmt_str_lengths=60):
        print("PANEL UNIT SHARE (top 10)\n", panel_units.head(10).select("colour", "share"))
        print("RAW UNIT SHARE (top 10)\n", raw_units.head(10).select("colour", "share"))
        print("Incumbent top-10 colours\n", t1.select("colour", "value", "share"))
        print("Emerging top-10 colours\n", t2.select("colour", "value", "share"))
        print(
            "RATIOS\n",
            ratios.select("section", "colour", "share", "base_share", "ratio_to_sales_share"),
        )
        print("GUARD-PASSING POOL (by style count)\n", pool_share.head(6).select("colour", "share"))
        print(
            "REALISED TOP-10 (trailing 13w intensity)\n",
            realised_top.select("colour", "value", "share"),
        )
        from scipy.stats import binomtest

        b = float(panel_units.filter(pl.col("colour") == "Black")["share"][0])
        for name, n_black in (
            ("incumbent", 7),
            ("emerging", 3),
            ("realised", int(realised_top.filter(pl.col("colour") == "Black")["value"].sum())),
        ):
            print(
                name,
                "black",
                n_black,
                f"/10  P(>=k | p={b:.3f}) = "
                f"{binomtest(n_black, 10, b, alternative='greater').pvalue:.4f}",
            )
        print("BLACK BUCKET by colour_group_name (units)\n", finer)
        print("colour_group x value counts (articles)\n", groups.head(12))
        print(
            "SAMPLE\n",
            sample.select(
                "article_id", "colour_group_name", "perceived_colour_value_name", "detail_desc"
            ),
        )


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # polars prints Unicode; avoid cp1252 crash
    main()
