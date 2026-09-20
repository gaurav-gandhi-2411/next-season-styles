"""Load the cached Google Trends series into the weekly frame `signal_features` consumes."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import polars as pl

from nss.external.trends_fetch import CACHE_DIR, slug

COMMITTED_WEEKLY = Path("reports/tables/external_signal_trends_weekly.csv")


def load_committed(path: Path = COMMITTED_WEEKLY) -> pl.DataFrame:
    """The committed weekly table (same frame `load_weekly` returns), for clones without the cache.

    The raw API cache under `data/external/` is gitignored and Trends' answers drift over time, so
    the compact table written by `signal_diagnostics` is committed and is what a reviewer runs on.
    """
    return pl.read_csv(path, try_parse_dates=True).select(
        pl.col("term"), pl.col("week_start"), pl.col("value").cast(pl.Float64)
    )


def load_weekly(terms: list[str], cache_dir: Path = CACHE_DIR) -> pl.DataFrame:
    """`term`, `week_start`, `value` for every cached term, aligned to the panel's Monday weeks.

    A Trends week is labelled by the Sunday that starts it; it is mapped to the Monday after so
    the signal week ends (Saturday) inside the panel week it is joined to. Terms with no cache
    file are simply absent (reported by coverage, never invented).
    """
    frames = []
    for term in terms:
        path = cache_dir / f"{slug(term)}.csv"
        if not path.exists():
            continue
        df = pl.read_csv(path, try_parse_dates=True)
        if df.height == 0:
            continue
        frames.append(
            df.select(
                pl.lit(term).alias("term"),
                (pl.col("week_start_sunday") + timedelta(days=1)).alias("week_start"),
                pl.col("value").cast(pl.Float64),
            )
        )
    if not frames:
        return pl.DataFrame(schema={"term": pl.String, "week_start": pl.Date, "value": pl.Float64})
    return pl.concat(frames)
