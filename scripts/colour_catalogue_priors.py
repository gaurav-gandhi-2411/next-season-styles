"""N3: catalogue-wide colour priors by pattern class.

Method and class definition pre-registered in `reports/v3/PREREGISTRATION.md` (section N3,
commit 635ac6d), before this scored anything. Universe: the 1,980 autumn forecast-eligible
styles (`reports/tables/forecast_all_styles.csv`), not the five calibrated ones. For each style
with >= 2 real catalogue photos (`concept_forecast_index.collect_index_images`, cap 8 per style),
the unchanged M2 rembg dominant-colour pipeline gives a raw p90 nearest-sibling CIEDE2000
threshold; the class prior is the median of those over styles in the class (solid = {Solid,
Melange}, patterned = everything else). This does not change any of the five calibrated styles'
own thresholds (M1/M2 stand); it only computes the two class priors for N4.

Parallelised across worker processes (each with its own rembg/onnxruntime session; onnxruntime
here is CPU-only, no CUDA provider). Benchmarked at ~0.78 s/image single-threaded.

    uv run --no-sync python scripts/colour_catalogue_priors.py [--limit N] [--workers N]
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any

import polars as pl

TABLES = "reports/tables"
FORECAST = f"{TABLES}/forecast_all_styles.csv"
MIN_STYLES_FOR_PRIOR = 20  # below this, the class median is unstable; fall back to the pooled one


def _dominant_worker(path_str: str) -> tuple[str, float, float, float, bool, str | None]:
    """Run in a worker process; import here so each process builds its own rembg session."""
    from nss.generate import colour_check as cc

    try:
        dom = cc.dominant_colour(Path(path_str))
        return path_str, *dom.lab, dom.used_fallback, None
    except Exception as exc:  # a handful of catalogue photos are corrupt/unreadable; skip, don't crash
        return path_str, 0.0, 0.0, 0.0, False, f"{type(exc).__name__}: {exc}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="cap total photos processed (pilot runs)")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    from nss.generate import colour_check as cc
    from nss.generate import concept_forecast_index as cfi

    styles_df = pl.read_csv(FORECAST).select("style_key", "graphical_appearance_name")
    style_class = {
        r["style_key"]: ("solid" if r["graphical_appearance_name"] in ("Solid", "Melange") else "patterned")
        for r in styles_df.to_dicts()
    }
    print(f"universe: {len(style_class)} styles")

    t0 = time.time()
    by_style = cfi.collect_index_images(style_class.keys(), max_per_style=8)
    print(f"collected images for {len(by_style)} styles in {time.time() - t0:.1f}s")

    all_paths = sorted({str(p) for paths in by_style.values() for p in paths})
    if args.limit is not None:
        all_paths = all_paths[: args.limit]
        wanted = set(all_paths)
        by_style = {s: [p for p in ps if str(p) in wanted] for s, ps in by_style.items()}
    print(f"{len(all_paths)} distinct photos to process")

    t0 = time.time()
    with mp.Pool(processes=args.workers) as pool:
        results = pool.map(_dominant_worker, all_paths, chunksize=8)
    dt = time.time() - t0
    per_image = dt / max(len(all_paths), 1)
    print(f"dominant-colour extraction: {dt:.1f}s for {len(all_paths)} photos ({per_image:.3f} s/photo)")

    errors = [r for r in results if r[5] is not None]
    if errors:
        print(f"{len(errors)} photos failed and were skipped, e.g. {errors[0][:1]} {errors[0][5]}")
    lab_by_path = {r[0]: (r[1], r[2], r[3]) for r in results if r[5] is None}
    fallback_by_path = {r[0]: r[4] for r in results if r[5] is None}

    rows: list[dict[str, Any]] = []
    for style, paths in by_style.items():
        labs = [lab_by_path[str(p)] for p in paths if str(p) in lab_by_path]
        n = len(labs)
        row: dict[str, Any] = {
            "style_key": style,
            "class": style_class[style],
            "n_photos_requested": len(paths),
            "n_photos_ok": n,
        }
        if n < 2:
            row["raw_threshold"] = None
            row["n_fallback"] = sum(fallback_by_path.get(str(p), False) for p in paths)
        else:
            row["raw_threshold"] = cc.threshold_from(labs)
            row["n_fallback"] = sum(fallback_by_path.get(str(p), False) for p in paths)
        rows.append(row)
    df = pl.DataFrame(rows)
    df.write_csv(f"{TABLES}/v3_catalogue_colour_priors_by_style.csv")

    summary_rows = []
    for cls in ("solid", "patterned"):
        sub = df.filter((pl.col("class") == cls) & pl.col("raw_threshold").is_not_null())
        n_styles = sub.height
        excluded = df.filter((pl.col("class") == cls) & pl.col("raw_threshold").is_null()).height
        used_pooled_fallback = n_styles < MIN_STYLES_FOR_PRIOR
        if used_pooled_fallback:
            all_valid = df.filter(pl.col("raw_threshold").is_not_null())
            median = float(all_valid["raw_threshold"].median()) if all_valid.height else None
        else:
            median = float(sub["raw_threshold"].median())
        summary_rows.append(
            {
                "class": cls,
                "n_styles_with_threshold": n_styles,
                "n_excluded_lt_2_photos": excluded,
                "median_threshold": median,
                "used_pooled_fallback": used_pooled_fallback,
            }
        )
    summary = pl.DataFrame(summary_rows)
    summary.write_csv(f"{TABLES}/v3_catalogue_colour_priors_by_class.csv")
    with pl.Config(tbl_cols=-1, tbl_width_chars=200):
        print(summary)


if __name__ == "__main__":
    main()
