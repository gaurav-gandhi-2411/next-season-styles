"""Validate the closed loop's style retrieval on REAL catalogue photos (tasks N8, Q2).

The 40 evaluation photos are exactly the ones the N8 free-text baseline was scored on:
`sample_images()` (unchanged) draws 40 styles at random (seed 42) from the frozen model's forecast
table (styles with >= 5 articles) and takes each style's lowest-`article_id` photo. All 40 fetched
successfully, so the comparison is like-for-like.

METHOD (`concept_forecast_index`): nearest style by cosine similarity to the mean embedding of that
style's real photos. Headline configuration, fixed before any number was computed: the AVERAGE of
the CLIP ViT-L/14 and DINOv2-base cosine similarities (`avg`); `clip` and `dino` alone are
ablations. No index or method choice was tuned on these photos.

LEAVE-OUT (critical): the 40 evaluation `article_id`s are dropped from every style's index BEFORE
the style means are formed. A style left with no photo is UNCOVERED, can never be retrieved, and
counts as a MISS at every k (reported as `coverage`, and every metric is reported both over all 40
photos and over the covered subset only). The index only holds photos already on disk plus a
time-budgeted fetch (`scripts/q2_fetch_index_images.py`), so coverage of the ~2,000 forecast styles
is partial by construction; top-k accuracy over all 40 is bounded above by coverage.

Reported per configuration: top-1 / top-5 / top-10 style accuracy; product-type-alone and
colour-alone accuracy of the top-1 style; coverage; and, for the headline configuration, accuracy
by confidence label (calibration). Top-5 is the fair headline (styles are near-ties by
construction); top-1 is reported too. Baseline: the N8 free-text route, 12.5% exact style
(SmolVLM), 2.5% (Florence-2) (`reports/tables/concept_forecast_validation.csv`, untouched).

Output: `reports/tables/q2_retrieval_validation.csv` (`row_type` = `photo` | `summary`;
`config` = `clip` | `dino` | `avg`; summary rows have `style_key` = `ALL`).

Usage:
    python -m nss.generate.concept_forecast_validation
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from nss.data.fetch_images import fetch_images, local_path
from nss.generate import concept_forecast as cf
from nss.generate import concept_forecast_index as cfi

ARTICLES = Path("data/raw/articles.csv")
IMAGES_DIR = Path("data/images")
EVAL_FETCH_DIR = Path("data/images_q2_eval")
OUT = Path("reports/tables/q2_retrieval_validation.csv")
N_STYLES = 40
SEED = 42
KS = (1, 5, 10)


def _sample_reps() -> pl.DataFrame:
    """The 40 (style_key, article_id) validation draws: no fetching, nothing written."""
    table = cf.load_table()
    articles = pl.read_csv(ARTICLES).with_columns(
        (
            pl.col("index_group_name")
            + " || "
            + pl.col(cf.TYPE)
            + " || "
            + pl.col("garment_group_name")
            + " || "
            + pl.col(cf.COLOUR)
            + " || "
            + pl.col(cf.PATTERN)
        ).alias("style_key")
    )
    counts = articles.group_by("style_key").len().filter(pl.col("len") >= 5)
    pool = table.join(counts, on="style_key").sample(n=N_STYLES, seed=SEED)
    # deterministic representative article: the lowest article_id of the style (no sales lookup
    # needed for validation; any real photo of the style is a valid positive)
    reps = (
        articles.filter(pl.col("style_key").is_in(pool["style_key"].to_list()))
        .sort("article_id")
        .group_by("style_key", maintain_order=True)
        .first()
        .select("style_key", "article_id")
    )
    return reps


def sampled_article_ids() -> list[int]:
    """All 40 validation article ids (usable or not): every one is kept out of the index."""
    return _sample_reps()["article_id"].to_list()


def sample_images() -> pl.DataFrame:
    """One lowest-article-id photo per randomly drawn eligible style, fetched if missing.

    Photos already in `data/images` are reused as they are; missing ones are fetched into the
    scratch folder `EVAL_FETCH_DIR` (never into the shared `data/images`). Returns the usable
    subset with `image_path`.
    """
    reps = _sample_reps()
    paths: dict[int, Path] = {}
    missing = []
    for a in reps["article_id"].to_list():
        if local_path(a, IMAGES_DIR).exists():
            paths[a] = local_path(a, IMAGES_DIR)
        else:
            missing.append(a)
    ok = fetch_images(missing, EVAL_FETCH_DIR) if missing else {}
    for a in missing:
        if ok[str(a)]:
            paths[a] = local_path(a, EVAL_FETCH_DIR)
    usable = reps.filter(pl.col("article_id").is_in(list(paths)))
    return usable.with_columns(
        pl.Series("image_path", [str(paths[a]) for a in usable["article_id"].to_list()])
    )


def _photo_rows(
    reps: pl.DataFrame,
    table: pl.DataFrame,
    index: cfi.RetrievalIndex,
) -> list[dict[str, object]]:
    """One row per (config, photo): rank of the true style and top-1 attribute agreement."""
    attrs = {r["style_key"]: r for r in table.iter_rows(named=True)}
    candidates = table["style_key"].to_list()
    rows: list[dict[str, object]] = []
    for r in reps.iter_rows(named=True):
        true = r["style_key"]
        covered = true in index.n_images
        result = cfi.retrieve(
            cfi.embed_query(Path(r["image_path"]), index.embedders or None), index, candidates
        )
        for view in cfi.VIEWS:
            order = [s for s, _ in result.ranked[view]]
            rank = order.index(true) + 1 if true in order else None
            top1 = order[0]
            rows.append(
                {
                    "row_type": "photo",
                    "config": view,
                    "style_key": true,
                    "article_id": r["article_id"],
                    "n_index_images": index.n_images.get(true, 0),
                    "covered": covered,
                    "pred_top1": top1,
                    "true_rank": rank,
                    "top1": rank is not None and rank <= 1,
                    "top5": rank is not None and rank <= 5,
                    "top10": rank is not None and rank <= 10,
                    "type_ok": attrs[top1][cf.TYPE] == attrs[true][cf.TYPE],
                    "colour_ok": attrs[top1][cf.COLOUR] == attrs[true][cf.COLOUR],
                    "confidence": result.confidence if view == cfi.HEADLINE_VIEW else None,
                    "margin": result.margin if view == cfi.HEADLINE_VIEW else None,
                    "n": None,
                }
            )
    return rows


def _summary_rows(photos: pl.DataFrame, n_candidates: int) -> list[dict[str, object]]:
    """Per-config means over all photos ('all'), the covered subset, and by confidence label."""
    rows: list[dict[str, object]] = []
    base = {
        "style_key": "ALL",
        "article_id": None,
        "n_index_images": None,
        "pred_top1": None,
        "true_rank": None,
        "margin": None,
    }
    for view in cfi.VIEWS:
        d = photos.filter(pl.col("config") == view)
        subsets = [("all", d), ("covered_only", d.filter(pl.col("covered")))]
        if view == cfi.HEADLINE_VIEW:
            subsets += [
                (f"confidence_{lab}", d.filter(pl.col("confidence") == lab))
                for lab in ("high", "medium", "low")
            ]
        for name, sub in subsets:
            n = sub.height
            rows.append(
                {
                    **base,
                    "row_type": f"summary_{name}",
                    "config": view,
                    "covered": float(d["covered"].mean()) if name == "all" else None,
                    "top1": float(sub["top1"].mean()) if n else None,
                    "top5": float(sub["top5"].mean()) if n else None,
                    "top10": float(sub["top10"].mean()) if n else None,
                    "type_ok": float(sub["type_ok"].mean()) if n else None,
                    "colour_ok": float(sub["colour_ok"].mean()) if n else None,
                    "confidence": None,
                    "n": n,
                }
            )
    rows.append(
        {
            **base,
            "row_type": "summary_chance_top5_of_candidates",
            "config": cfi.HEADLINE_VIEW,
            "covered": None,
            "top1": 1 / n_candidates,
            "top5": 5 / n_candidates,
            "top10": 10 / n_candidates,
            "type_ok": None,
            "colour_ok": None,
            "confidence": None,
            "n": n_candidates,
        }
    )
    return rows


def main() -> None:
    """Build the leave-out index, score the 40 photos under every view, write the table."""
    table = cf.load_table()
    reps = sample_images()
    eval_ids = sampled_article_ids()  # all 40 draws, usable or not
    by_style = cfi.collect_index_images(table["style_key"].to_list(), exclude_articles=eval_ids)
    index = cfi.build_index(by_style)
    assert not (
        set(eval_ids) & {int(p.stem) for ps in by_style.values() for p in ps}
    ), "leave-out violated: an evaluation photo is in the index"
    photos = pl.DataFrame(_photo_rows(reps, table, index))
    n_cand = len(index.styles)
    summary = pl.DataFrame(_summary_rows(photos, n_cand), infer_schema_length=None)
    flags = ["covered", "top1", "top5", "top10", "type_ok", "colour_ok"]
    photos_f = photos.with_columns([pl.col(c).cast(pl.Float64) for c in flags])
    out = pl.concat([photos_f, summary.select(photos_f.columns)], how="vertical_relaxed")
    out.write_csv(OUT)
    n_img = sum(index.n_images.values())
    print(f"index: {n_cand} styles / {n_img} photos; eval photos: {reps.height}")
    with pl.Config(tbl_rows=40, tbl_width_chars=200, float_precision=3):
        print(
            summary.select(
                "row_type",
                "config",
                "covered",
                "top1",
                "top5",
                "top10",
                "type_ok",
                "colour_ok",
                "n",
            )
        )


if __name__ == "__main__":
    main()
