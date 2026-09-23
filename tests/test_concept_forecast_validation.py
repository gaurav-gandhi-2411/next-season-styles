from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from nss.generate import concept_forecast_validation as v


def _photos() -> pl.DataFrame:
    """Four photos under the headline view: 2 covered (1 top-1 hit, 1 top-5-only), 2 uncovered."""
    rows = []
    for covered, rank, conf in (
        (True, 1, "high"),
        (True, 4, "medium"),
        (False, None, "low"),
        (False, None, "medium"),
    ):
        rows.append(
            {
                "condition": "c",
                "row_type": "photo",
                "config": "avg",
                "style_key": "S",
                "article_id": 1,
                "n_index_images": 1 if covered else 0,
                "covered": covered,
                "pred_top1": "S",
                "true_rank": rank,
                "top1": rank is not None and rank <= 1,
                "top5": rank is not None and rank <= 5,
                "top10": rank is not None and rank <= 10,
                "type_ok": covered,
                "colour_ok": True,
                "confidence": conf,
                "margin": 0.0,
                "n": None,
            }
        )
    # every view must exist for _summary_rows; reuse the same rows relabelled
    df = pl.DataFrame(rows)
    return pl.concat(
        [df.with_columns(pl.lit(view).alias("config")) for view in ("avg", "clip", "dino")]
    )


def test_summary_counts_uncovered_photos_as_misses() -> None:
    rows = v._summary_rows("c", _photos(), n_candidates=100)
    by = {(r["config"], r["row_type"]): r for r in rows}
    all_avg = by[("avg", "summary_all")]
    assert all_avg["n"] == 4
    assert all_avg["top1"] == pytest.approx(0.25)  # 1 of 4 photos; uncovered are misses
    assert all_avg["top5"] == pytest.approx(0.5)
    assert all_avg["covered"] == pytest.approx(0.5)  # coverage
    cov = by[("avg", "summary_covered_only")]
    assert cov["n"] == 2 and cov["top1"] == pytest.approx(0.5) and cov["top5"] == pytest.approx(1.0)
    assert by[("avg", "summary_confidence_high")]["top1"] == pytest.approx(1.0)
    assert by[("avg", "summary_confidence_medium")]["n"] == 2
    assert by[("avg", "summary_confidence_low")]["n"] == 1
    chance = by[("avg", "summary_chance_top5_of_candidates")]
    assert chance["top5"] == pytest.approx(0.05) and chance["n"] == 100


def test_summary_carries_the_condition() -> None:
    rows = v._summary_rows("gallery_covers_eval", _photos(), n_candidates=10)
    assert {r["condition"] for r in rows} == {"gallery_covers_eval"}


@pytest.mark.skipif(
    not (
        Path("data/raw/articles.csv").exists()
        and Path("reports/tables/forecast_all_styles.csv").exists()
    ),
    reason="needs the local catalogue and forecast table",
)
def test_sample_matches_the_n8_evaluation_set_and_gallery_excludes_eval_photos() -> None:
    reps = v._sample_reps()
    old = pl.read_csv(v.N8_VALIDATION)
    # like-for-like with the 12.5% baseline: exactly the same 40 styles
    assert set(reps["style_key"].to_list()) == set(old["style_key"].to_list())
    assert reps.height == v.N_STYLES
    # the free-text baseline's representative-article rule: the lowest article_id of each style
    art = v.cfi.load_article_styles()
    for row in reps.iter_rows(named=True):
        ids = art.filter(pl.col("style_key") == row["style_key"])["article_id"]
        assert row["article_id"] == ids.min()
    eval_ids = set(reps["article_id"].to_list())
    gallery = v.gallery_article_ids(2)
    assert not (set(gallery) & eval_ids)  # a gallery photo is never an evaluation photo
    assert len(gallery) <= 2 * v.N_STYLES


def test_main_full_print_path_survives_a_cp1252_console() -> None:
    """The bug this guards: polars' default table format uses Unicode box-drawing characters,
    which raise UnicodeEncodeError on a Windows console (cp1252) unless stdout is reconfigured
    to UTF-8 first -- exactly what `__main__` does before calling `main_full()`. Printing a
    polars DataFrame with default (non-ASCII_FULL) formatting is enough to reproduce it; the
    DataFrame's own content need not be non-ASCII.
    """
    import io

    df = pl.DataFrame({"condition": ["full_index_40"], "top1": [0.275], "n": [40]})

    cp1252_stream = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    with pytest.raises(UnicodeEncodeError):
        print(df, file=cp1252_stream)  # reproduces the crash without the fix

    utf8_stream = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    utf8_stream.reconfigure(encoding="utf-8")  # what `if __name__ == "__main__":` does first
    print(df, file=utf8_stream)  # must not raise
