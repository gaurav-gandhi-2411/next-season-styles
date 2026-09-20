"""Fetch weekly Google Trends series for every mapped style term, resumably, into a disk cache.

Run (pytrends is NOT a project dependency; it is layered on for this one command only, so
`pyproject.toml` and `uv.lock` are untouched):

    uv run --no-sync --with pytrends python -m nss.external.trends_fetch

One query per term, never batched: Trends normalises a batch jointly to its own maximum, so a
series would depend on which terms shared a request. Worldwide, timeframe 2018-09-01 ..
2020-09-30, which returns 110 weekly points labelled by the Sunday that starts each week.

Each term is cached to `data/external/trends/<slug>.csv` the moment it is fetched, and existing
files are skipped, so the API calls never need repeating. Terms go in descending order of the
number of styles they cover, so a rate-limited run still buys the most coverage first. A failed
term is logged to `data/external/trends/_failures.csv` and skipped; rate limits back off
exponentially.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import time
from pathlib import Path

import polars as pl

from nss.external.term_mapping import map_styles

CACHE_DIR = Path("data/external/trends")
TIMEFRAME = "2018-09-01 2020-09-30"
PANEL_PATH = Path("data/processed/style_week_panel.parquet")
MAX_TRIES = 5
BASE_SLEEP_S = 1.5  # spacing between requests; measured OK at 25/25 with 1.5 s
BACKOFF_S = 60.0


def slug(term: str) -> str:
    """Filesystem-safe cache name for a term."""
    return re.sub(r"[^a-z0-9]+", "_", term.lower()).strip("_")


def ordered_terms(panel_path: Path = PANEL_PATH) -> list[tuple[str, int]]:
    """(term, n_styles) for every mapped term, most-covering first."""
    styles = pl.read_parquet(panel_path).select(
        "style_key", "product_type_name", "perceived_colour_master_name"
    )
    mapped = map_styles(styles).filter(pl.col("term").is_not_null())
    counts = mapped.group_by("term").len().sort(["len", "term"], descending=[True, False])
    return [(t, int(n)) for t, n in counts.iter_rows()]


def fetch_one(py, term: str) -> pl.DataFrame:  # noqa: ANN001 -- pytrends is an optional import
    """One term -> DataFrame(week_start_sunday, value, is_partial). Retries on rate limits."""
    for attempt in range(1, MAX_TRIES + 1):
        try:
            py.build_payload([term], timeframe=TIMEFRAME, geo="")
            df = py.interest_over_time()
            if df.empty:
                return pl.DataFrame(
                    schema={
                        "week_start_sunday": pl.Date,
                        "value": pl.Int64,
                        "is_partial": pl.Boolean,
                    }
                )
            out = df.reset_index()
            return pl.DataFrame(
                {
                    "week_start_sunday": out["date"].dt.date.tolist(),
                    "value": out[term].astype(int).tolist(),
                    "is_partial": out["isPartial"].astype(bool).tolist(),
                }
            )
        except Exception as exc:  # noqa: BLE001 -- pytrends raises assorted request errors
            wait = BACKOFF_S * (2 ** (attempt - 1))
            print(
                f"  attempt {attempt}/{MAX_TRIES} failed: {type(exc).__name__}: {str(exc)[:90]}"
                f" -- sleeping {wait:.0f}s",
                flush=True,
            )
            if attempt == MAX_TRIES:
                raise
            time.sleep(wait)
    raise RuntimeError("unreachable")


def main() -> None:
    """CLI: fetch every uncached term."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--limit", type=int, default=None, help="stop after N new fetches")
    args = parser.parse_args()

    from pytrends.request import TrendReq

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    terms = ordered_terms()
    todo = [(t, n) for t, n in terms if not (CACHE_DIR / f"{slug(t)}.csv").exists()]
    print(f"{len(terms)} terms, {len(terms) - len(todo)} cached, {len(todo)} to fetch", flush=True)
    py = TrendReq(hl="en-US", tz=0, timeout=(10, 25))
    t0 = time.time()
    done = 0
    for term, n_styles in todo:
        if args.limit is not None and done >= args.limit:
            break
        try:
            df = fetch_one(py, term)
        except Exception as exc:  # noqa: BLE001
            with (CACHE_DIR / "_failures.csv").open("a", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerow([term, type(exc).__name__, str(exc)[:200]])
            print(f"FAILED {term!r} ({type(exc).__name__}); logged and skipped", flush=True)
            continue
        df.write_csv(CACHE_DIR / f"{slug(term)}.csv")
        done += 1
        nz = int((df["value"] > 0).sum()) if df.height else 0
        print(
            f"[{done}/{len(todo)}] {term!r} styles={n_styles} rows={df.height} nonzero={nz} "
            f"elapsed={time.time() - t0:.0f}s",
            flush=True,
        )
        time.sleep(BASE_SLEEP_S + random.uniform(0.0, 1.0))  # noqa: S311 -- jitter, not crypto
    print(f"done: {done} fetched in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
