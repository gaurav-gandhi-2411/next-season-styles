"""Widen the real reference base per final style (task N5): candidates for 25 screened references.

WHY: Gate 1/1b limits and the leave-one-out control rest on only 4-8 real articles per style, so
the p90 thresholds were noisy (a p90 over 4-6 references is essentially the closest real pair).
This module selects the best-selling constituent articles of each final style (the same
`select_top_selling_articles` window the original exemplars used: the trailing 26 weeks to the
forecast origin) and fetches their images with the same on-demand Kaggle fetcher. The candidates
are then screened for framing AND pattern conformity by `nss.generate.widen_screen` with the local
VLM judge, and only the survivors (up to 25, best-selling first) become the reference base.

`CANDIDATES_PER_STYLE` is larger than the 25 targeted so screening losses do not leave the base
short; a style with fewer constituent articles than that is reported as exhausted, never padded.

Usage:
    uv run python -m nss.data.widen_references
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import polars as pl

from nss.data.fetch_images import fetch_images, local_path
from nss.data.select_exemplars import (
    STYLE_KEY_COLS,
    compute_lookback_cutoff,
    select_top_selling_articles,
)
from nss.data.select_final_three_exemplars import PANEL_LAST_WEEK, TRANSACTIONS_DIR

FINAL_THREE_PATH = Path("reports/tables/top_styles_final_three.csv")
ARTICLES_PATH = Path("data/raw/articles.csv")
IMAGES_DIR = Path("data/images")
OUT_PATH = Path("reports/tables/reference_pool_widened.csv")
LOOKBACK_WEEKS = 26
CANDIDATES_PER_STYLE = 40


SUMMER_ORIGIN = date(2020, 6, 1)
SUMMER_SELECTION = Path("reports/tables/summer_selection_log.csv")
SUMMER_OUT = Path("reports/tables/reference_pool_widened_summer.csv")


def main(summer: bool = False) -> None:
    """Select and fetch candidate references for every final style; write the pool manifest.

    `summer=True` does the N6 Summer style instead, with the sales window ending at the summer
    forecast origin (the forecast could not have seen later sales).
    """
    final = (
        pl.read_csv(SUMMER_SELECTION).filter(pl.col("excluded").is_null()).head(1)
        if summer
        else pl.read_csv(FINAL_THREE_PATH)
    )
    last_week = SUMMER_ORIGIN if summer else PANEL_LAST_WEEK
    cutoff = compute_lookback_cutoff(last_week, LOOKBACK_WEEKS)
    articles = pl.read_csv(ARTICLES_PATH).select(["article_id", *STYLE_KEY_COLS])
    txn = pl.scan_parquet(str(TRANSACTIONS_DIR / "**" / "*.parquet")).select(
        ["article_id", "t_dat"]
    )
    window_end = SUMMER_ORIGIN if summer else txn.select(pl.col("t_dat").max()).collect().item()
    txn_df = txn.filter(pl.col("t_dat") >= cutoff).collect()
    rows: list[dict[str, object]] = []
    for style in final.iter_rows(named=True):
        values = {c: style[c] for c in STYLE_KEY_COLS}
        top = select_top_selling_articles(
            txn_df, articles, values, cutoff, window_end, n=CANDIDATES_PER_STYLE
        )
        ids = top["article_id"].to_list()
        print(
            f"{style['style_key']}: {len(ids)} candidate articles "
            f"(target pool {CANDIDATES_PER_STYLE})"
        )
        ok = fetch_images(ids, IMAGES_DIR)
        for aid, units in zip(ids, top["units_sold_last_26w"].to_list(), strict=True):
            path = local_path(aid, IMAGES_DIR)
            rows.append(
                {
                    "style_id": style["style_key"],
                    "article_id": aid,
                    "units_sold_last_26w": units,
                    "image_path": str(path),
                    "fetch_success": bool(ok[str(aid)]),
                }
            )
    pl.DataFrame(rows).write_csv(SUMMER_OUT if summer else OUT_PATH)
    print(pl.DataFrame(rows).group_by("style_id").agg(pl.col("fetch_success").sum(), pl.len()))


if __name__ == "__main__":
    main(summer="summer" in sys.argv[1:])
