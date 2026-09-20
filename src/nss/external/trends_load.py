"""Load the cached Google Trends series into the weekly frame `signal_features` consumes."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import polars as pl

from nss.external.trends_fetch import CACHE_DIR, slug


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
