"""Validate the closed loop's style retrieval on REAL catalogue photos.

The 40 evaluation photos are exactly the ones the free-text baseline was scored on: the 40
`style_key`s in the committed per-image file `concept_forecast_validation.csv`, each with its
lowest-`article_id` photo (the baseline's rule). All 40 were usable in the baseline, so the
comparison is like-for-like. The baseline's draw code (`table.join(...).sample(n=40, seed=42)`) is
NOT re-run: re-running it today reproduces only 3 of the 40 styles (the join's row order is not
stable across polars versions / runs), so the file, not the seed, is the source of truth. Photos not
already in `data/images` are fetched into the scratch folder `data/images_q2_eval/`.

METHOD (`concept_forecast_index`): nearest style by cosine similarity to the mean embedding of that
style's real photos. Headline configuration, fixed before any number was computed: the AVERAGE of
the CLIP ViT-L/14 and DINOv2-base cosine similarities (`avg`); `clip` and `dino` alone are
ablations. No index or method choice was tuned on these photos.

LEAVE-OUT (critical): the 40 evaluation `article_id`s are dropped from every style's index BEFORE
the style means are formed. A style left with no photo is UNCOVERED, can never be retrieved, and
counts as a MISS at every k (reported as `coverage`, and every metric is reported both over all 40
photos and over the covered subset only). The index only holds photos already on disk plus a
time-budgeted fetch (`scripts/fetch_index_images.py`), so coverage of the ~2,000 forecast styles
is partial by construction; top-k accuracy over all 40 is bounded above by coverage.

TWO INDEX CONDITIONS (declared before any 40-photo number was computed; a 6-photo debugging run of
the pipeline preceded this and was not used to choose anything). With ~450 of ~1,980 styles indexed,
a random draw of 40 styles has ~1 covered style (measured 1 of 40, before adding any gallery), so
the deployment-coverage condition can only measure coverage, not retrieval quality:
- `deployment_coverage`: the index as it exists on disk, minus the 40 evaluation photos and minus
  the gallery photos below -- the honest picture of what a user gets today (coverage-bound);
- `gallery_covers_eval` (HEADLINE for retrieval quality): the same index plus up to
  `GALLERY_PER_STYLE` OTHER real articles of each evaluation style (never the evaluation photo), the
  standard closed-set protocol (the gallery must contain the classes being queried). The candidate
  set is then every indexed style (~490), not the ~1,980 the free-text baseline could name, so the
  chance level is stated (`summary_chance_top5_of_candidates`) and the comparison with the 12.5%
  baseline is favourable to retrieval in one respect (a smaller label space) and unfavourable in
  another (the baseline had no coverage limit); both caveats are reported with the numbers;
- `loo_deployment_index` (SUPPLEMENTARY, larger n, added when the gallery fetch was rate-limited by
  Kaggle): every index photo of a style with >= 2 index photos is a query against an index whose
  own-style prototype excludes it (leave-one-out); these are real photos of catalogue styles, but
  they are not the 40 baseline photos, and near-duplicate articles within a style make it
  optimistic.

Reported per configuration: top-1 / top-5 / top-10 style accuracy; product-type-alone and
colour-alone accuracy of the top-1 style; coverage; and, for the headline configuration, accuracy
by confidence label (calibration). Top-5 is the fair headline (styles are near-ties by
construction); top-1 is reported too. Baseline: the free-text route, 12.5% exact style
(SmolVLM), 2.5% (Florence-2) (`reports/tables/concept_forecast_validation.csv`, untouched).

Output: `reports/tables/retrieval_validation_partial_index.csv` (`condition` = `deployment_coverage`
| `gallery_covers_eval` | `loo_deployment_index`; `row_type` = `photo` | `summary_*`;
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

N8_VALIDATION = Path("reports/tables/concept_forecast_validation.csv")  # the baseline's 40 styles
IMAGES_DIR = Path("data/images")
EVAL_FETCH_DIR = Path("data/images_q2_eval")
OUT = Path("reports/tables/retrieval_validation_partial_index.csv")
OLD_OUT = OUT  # the 415-style results (kept untouched); the full-catalogue run writes FULL_OUT
FULL_OUT = Path("reports/tables/retrieval_validation.csv")
N_STYLES = 40
GALLERY_PER_STYLE = 2


def _sample_reps() -> pl.DataFrame:
    """The 40 (style_key, article_id) validation photos of the baseline; nothing is fetched.

    Styles come from the committed per-image file; each style's photo is its lowest
    `article_id` (the baseline's deterministic representative-article rule).
    """
    styles = pl.read_csv(N8_VALIDATION)["style_key"].unique().sort().to_list()
    articles = cfi.load_article_styles()
    return (
        articles.filter(pl.col("style_key").is_in(styles))
        .sort("article_id")
        .group_by("style_key", maintain_order=True)
        .first()
        .select("style_key", "article_id")
        .sort("style_key")
    )


def sampled_article_ids() -> list[int]:
    """All 40 validation article ids (usable or not): every one is kept out of the index."""
    return _sample_reps()["article_id"].to_list()


def sample_images() -> pl.DataFrame:
    """The baseline photo (lowest article id) of each of the 40 styles, fetched if missing.

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
    condition: str,
    reps: pl.DataFrame,
    table: pl.DataFrame,
    index: cfi.RetrievalIndex,
    leave_one_out: bool = False,
) -> list[dict[str, object]]:
    """One row per (config, photo): rank of the true style and top-1 attribute agreement.

    `leave_one_out`: the photos ARE index photos (`reps` rows must have >= 2 index photos for
    their style); each is matched against an index whose own style prototype excludes it.
    """
    attrs = {r["style_key"]: r for r in table.iter_rows(named=True)}
    candidates = table["style_key"].to_list()
    rows: list[dict[str, object]] = []
    for r in reps.iter_rows(named=True):
        true = r["style_key"]
        covered = true in index.n_images
        result = cfi.retrieve(
            cfi.embed_query(Path(r["image_path"]), index.embedders or None),
            index,
            candidates,
            leave_out=true if leave_one_out else None,
        )
        for view in cfi.VIEWS:
            order = [s for s, _ in result.ranked[view]]
            rank = order.index(true) + 1 if true in order else None
            top1 = order[0]
            rows.append(
                {
                    "condition": condition,
                    "row_type": "photo",
                    "config": view,
                    "style_key": true,
                    "article_id": r["article_id"],
                    "n_index_images": index.n_images.get(true, 0) - int(leave_one_out),
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


def _summary_rows(
    condition: str, photos: pl.DataFrame, n_candidates: int
) -> list[dict[str, object]]:
    """Per-config means over all photos ('all'), the covered subset, and by confidence label."""
    rows: list[dict[str, object]] = []
    base = {
        "condition": condition,
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


def gallery_article_ids(per_style: int = GALLERY_PER_STYLE) -> list[int]:
    """Up to `per_style` OTHER articles (lowest ids) of each evaluation style: the gallery photos
    (`scripts/fetch_validation_photos.py` fetches them; never an evaluation photo)."""
    reps = _sample_reps()
    art = cfi.load_article_styles()
    out: list[int] = []
    for row in reps.iter_rows(named=True):
        others = (
            art.filter(
                (pl.col("style_key") == row["style_key"])
                & (pl.col("article_id") != row["article_id"])
            )
            .sort("article_id")["article_id"]
            .to_list()
        )
        out.extend(others[:per_style])
    return out


def loo_queries(by_style: dict[str, list[Path]]) -> pl.DataFrame:
    """Every index photo of a style that has >= 2 index photos, as a leave-one-out query."""
    return pl.DataFrame(
        [
            {"style_key": s, "article_id": int(p.stem), "image_path": str(p)}
            for s, ps in sorted(by_style.items())
            if len(ps) >= 2
            for p in ps
        ],
        schema={"style_key": pl.String, "article_id": pl.Int64, "image_path": pl.String},
    )


def run_condition(
    condition: str,
    reps: pl.DataFrame | None,
    table: pl.DataFrame,
    exclude: list[int],
    leave_one_out: bool = False,
) -> tuple[pl.DataFrame, pl.DataFrame, cfi.RetrievalIndex]:
    """Build the index minus `exclude`, score every query photo; `(photo rows, summary rows,
    index)`. With `leave_one_out` the queries are the index photos themselves (`reps` ignored)."""
    by_style = cfi.collect_index_images(table["style_key"].to_list(), exclude_articles=exclude)
    index = cfi.build_index(by_style)
    eval_ids = set(sampled_article_ids())
    assert not (
        eval_ids & {int(p.stem) for ps in by_style.values() for p in ps}
    ), "leave-out violated: an evaluation photo is in the index"
    queries = loo_queries(by_style) if leave_one_out else reps
    assert queries is not None
    photos = pl.DataFrame(_photo_rows(condition, queries, table, index, leave_one_out))
    summary = pl.DataFrame(
        _summary_rows(condition, photos, len(index.styles)), infer_schema_length=None
    )
    return photos, summary, index


def main() -> None:
    """Score the validation photos under every view for all index conditions; write the table."""
    table = cf.load_table()
    reps = sample_images()
    eval_ids = sampled_article_ids()  # all 40 draws, usable or not
    gallery = gallery_article_ids()
    flags = ["covered", "top1", "top5", "top10", "type_ok", "colour_ok"]
    parts = []
    summaries = []
    # (condition, ids kept out of the index, leave-one-out queries?)
    conditions = {
        "deployment_coverage": (eval_ids + gallery, False),  # index as it exists without gallery
        "gallery_covers_eval": (eval_ids, False),  # HEADLINE: evaluated styles have other photos
        "loo_deployment_index": (eval_ids + gallery, True),  # supplementary, larger n
    }
    for condition, (exclude, loo) in conditions.items():
        photos, summary, index = run_condition(condition, reps, table, exclude, loo)
        photos_f = photos.with_columns([pl.col(c).cast(pl.Float64) for c in flags])
        parts.append(photos_f)
        parts.append(summary.select(photos_f.columns))
        summaries.append(summary)
        n_img = sum(index.n_images.values())
        print(
            f"{condition}: index {len(index.styles)} styles / {n_img} photos; "
            f"queries: {summary.filter(pl.col('row_type') == 'summary_all')['n'][0]}"
        )
    pl.concat(parts, how="vertical_relaxed").write_csv(OUT)
    with pl.Config(tbl_rows=60, tbl_width_chars=220, float_precision=3):
        print(
            pl.concat(summaries, how="vertical_relaxed").select(
                "condition",
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


def main_full() -> None:
    """Full-catalogue validation: the whole local image tree, <= `PER_STYLE_CAP` photos per
    style, over the 1,980 forecast styles; writes `FULL_OUT` and leaves the 415-style file alone.

    Conditions (declared before any number was computed):
    - `full_index_40`: the SAME 40 baseline photos, every one dropped from the index before any
      style mean is formed (leave-one-out by exclusion), candidates = all 1,980 forecast styles;
    - `loo_full_index_159`: the SAME 159 photos as the 415-style leave-one-out, now against the
      full index (each photo's own style prototype is recomputed without it);
    - `loo_full_index_1style1photo` (supplementary, larger n): one query per style (its first
      index photo) for every style with >= 2 index photos, same leave-one-out.
    """
    import time

    t0 = time.time()
    table = cf.load_table()
    old = pl.read_csv(OLD_OUT)
    loo_ids = old.filter(
        (pl.col("condition") == "loo_deployment_index")
        & (pl.col("config") == "avg")
        & (pl.col("row_type") == "photo")
    )["article_id"].to_list()
    eval_ids = sampled_article_ids()
    reps = _sample_reps()
    reps = reps.with_columns(
        pl.Series("image_path", [str(cfi.find_photo(a)) for a in reps["article_id"].to_list()])
    )
    flags = ["covered", "top1", "top5", "top10", "type_ok", "colour_ok"]
    parts, summaries = [], []

    def score(condition: str, queries: pl.DataFrame, index: cfi.RetrievalIndex, loo: bool) -> None:
        photos = pl.DataFrame(_photo_rows(condition, queries, table, index, loo))
        summary = pl.DataFrame(
            _summary_rows(condition, photos, len(index.styles)), infer_schema_length=None
        )
        photos_f = photos.with_columns([pl.col(c).cast(pl.Float64) for c in flags])
        parts.extend([photos_f, summary.select(photos_f.columns)])
        summaries.append(summary)
        n_img = sum(index.n_images.values())
        print(
            f"{condition}: index {len(index.styles)} styles / {n_img} photos, "
            f"{queries.height} queries"
        )

    styles = table["style_key"].to_list()
    by_style = cfi.collect_index_images(
        styles, exclude_articles=eval_ids, max_per_style=cfi.PER_STYLE_CAP
    )
    index = cfi.build_index(by_style)
    assert not (eval_ids and {int(p.stem) for ps in by_style.values() for p in ps} & set(eval_ids))
    uncovered = [s for s in styles if s not in by_style]
    print(f"coverage: {len(by_style)}/{len(styles)} styles, uncovered: {len(uncovered)}")
    score("full_index_40", reps, index, False)

    by_style_l = cfi.collect_index_images(
        styles, exclude_articles=eval_ids, max_per_style=cfi.PER_STYLE_CAP, require=loo_ids
    )
    index_l = cfi.build_index(by_style_l)
    id_path = {int(p.stem): p for ps in by_style_l.values() for p in ps}
    art = cfi.load_article_styles()
    style_of = dict(zip(art["article_id"].to_list(), art["style_key"].to_list(), strict=True))
    q159 = pl.DataFrame(
        [
            {"style_key": style_of[a], "article_id": a, "image_path": str(id_path[a])}
            for a in loo_ids
        ],
        schema={"style_key": pl.String, "article_id": pl.Int64, "image_path": pl.String},
    )
    score("loo_full_index_159", q159, index_l, True)
    q_all = loo_queries(by_style_l)
    q_one = q_all.group_by("style_key", maintain_order=True).first()
    score("loo_full_index_1style1photo", q_one, index_l, True)

    pl.concat(parts, how="vertical_relaxed").write_csv(FULL_OUT)
    pl.DataFrame({"style_key": uncovered}, schema={"style_key": pl.String}).write_csv(
        FULL_OUT.with_name("retrieval_uncovered_styles.csv")
    )
    with pl.Config(tbl_rows=60, tbl_width_chars=220, float_precision=3):
        print(
            pl.concat(summaries, how="vertical_relaxed")
            .filter(pl.col("row_type").is_in(["summary_all", "summary_chance_top5_of_candidates"]))
            .select(
                "condition",
                "row_type",
                "config",
                "top1",
                "top5",
                "top10",
                "type_ok",
                "colour_ok",
                "n",
            )
        )
        print(
            pl.concat(summaries, how="vertical_relaxed")
            .filter(pl.col("row_type").str.starts_with("summary_confidence"))
            .select("condition", "row_type", "top1", "top5", "n")
        )
    print(f"validation time {time.time() - t0:.0f}s")


if __name__ == "__main__":
    import sys

    main_full() if "--full" in sys.argv else main()
