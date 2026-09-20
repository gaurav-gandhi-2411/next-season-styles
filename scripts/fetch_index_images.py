"""Time-budgeted fetch of catalogue photos for the Q2 retrieval index.

Coverage-first plan (chosen BEFORE any retrieval numbers exist, and independent of the 40 validation
photos): walk the forecast table's styles in predicted-intensity rank order and fetch ONE article
photo per style not yet covered by a local image, then a second photo per style, and so on. Any
prefix of that queue is a valid index, so stopping at the wall-clock budget is clean. Articles used
by the 40 validation photos are never fetched into the index. Fetches run through
`nss.data.fetch_images.fetch_one` on a small thread pool (one Kaggle call per image).

Writes only under `data/images_q2/` (new scratch folder) and a manifest CSV there.

Usage:
    python scripts/fetch_index_images.py --budget-seconds 1500 --workers 8
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl

from nss.data.fetch_images import fetch_one
from nss.generate import concept_forecast as cf
from nss.generate.concept_forecast_validation import sampled_article_ids

ARTICLES = Path("data/raw/articles.csv")
OUT_DIR = Path("data/images_q2")
LOCAL_DIRS = (Path("data/images"), OUT_DIR)


def build_queue(max_per_style: int = 3) -> pl.DataFrame:
    """Fetch queue: (round, style rank, article_id, style_key), round-major then rank order."""
    table = cf.load_table().sort("predicted_intensity", descending=True)
    ranks = table.select("style_key").with_row_index("rank", offset=1)
    articles = pl.read_csv(ARTICLES).with_columns(
        pl.concat_str(
            [
                "index_group_name",
                "product_type_name",
                "garment_group_name",
                "perceived_colour_master_name",
                "graphical_appearance_name",
            ],
            separator=" || ",
        ).alias("style_key")
    )
    eval_ids = set(sampled_article_ids())
    local_ids = {int(p.stem) for d in LOCAL_DIRS for p in d.glob("*.jpg")}
    cand = (
        articles.join(ranks, on="style_key")
        .filter(~pl.col("article_id").is_in(list(eval_ids)))
        .filter(~pl.col("article_id").is_in(list(local_ids)))
        .sort(["rank", "article_id"])
        .with_columns(pl.int_range(pl.len()).over("style_key").alias("k"))
        .filter(pl.col("k") < max_per_style)
    )
    have = (
        articles.filter(pl.col("article_id").is_in(list(local_ids - eval_ids)))
        .group_by("style_key")
        .len()
        .rename({"len": "n_have"})
    )
    cand = cand.join(have, on="style_key", how="left").with_columns(pl.col("n_have").fill_null(0))
    # round = how many index photos the style will have once this one lands
    cand = cand.with_columns((pl.col("n_have") + pl.col("k") + 1).alias("round"))
    return (
        cand.filter(pl.col("round") <= max_per_style)
        .sort(["round", "rank"])
        .select("round", "rank", "article_id", "style_key")
    )


def main() -> None:
    """Fetch the queue until the wall-clock budget is spent."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budget-seconds", type=float, default=1500.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-per-style", type=int, default=2)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    queue = build_queue(args.max_per_style)
    print(f"queue: {queue.height} images, {queue['style_key'].n_unique()} styles", flush=True)
    start = time.time()
    rows = []
    rows_iter = list(queue.iter_rows(named=True))

    # Kaggle answers 429 when hammered (observed with 2 x 8 workers); stop instead of retrying
    # blindly once this many consecutive fetches have failed.
    streak = [0]
    max_streak = 20

    def one(r: dict[str, object]) -> dict[str, object]:
        if time.time() - start > args.budget_seconds or streak[0] >= max_streak:
            return {**r, "ok": None, "seconds": 0.0}
        t0 = time.time()
        ok = fetch_one(int(r["article_id"]), OUT_DIR, max_retries=3)
        streak[0] = 0 if ok else streak[0] + 1
        return {**r, "ok": ok, "seconds": time.time() - t0}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(one, r) for r in rows_iter]
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            if i % 25 == 0:
                done = [r for r in rows if r["ok"] is not None]
                n_ok = sum(bool(r["ok"]) for r in done)
                print(f"{i}/{len(futs)} elapsed {time.time() - start:.0f}s ok={n_ok}", flush=True)
    elapsed = time.time() - start
    pl.DataFrame(rows).write_csv(OUT_DIR / "manifest.csv")
    n_ok = sum(bool(r["ok"]) for r in rows)
    n_att = sum(r["ok"] is not None for r in rows)
    print(f"DONE attempted={n_att} ok={n_ok} elapsed={elapsed:.0f}s", flush=True)


if __name__ == "__main__":
    main()
