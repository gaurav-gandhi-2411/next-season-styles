from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from nss.generate import concept_forecast as cf
from nss.generate import concept_forecast_index as cfi

# Style -> (clip direction, dino direction) in a 4-dim toy space; photos are direction + jitter.
_DIRS = {
    "A": (np.array([1.0, 0, 0, 0]), np.array([1.0, 0, 0, 0])),
    "B": (np.array([0, 1.0, 0, 0]), np.array([0, 1.0, 0, 0])),
    "C": (np.array([0, 0, 1.0, 0]), np.array([0, 0, 1.0, 0])),
    "D": (np.array([0, 0, 0, 1.0]), np.array([0, 0, 0, 1.0])),
    "E": (np.array([0.7, 0.7, 0, 0]), np.array([0, 0, 0.7, 0.7])),
    "F": (np.array([0, 0.7, 0.7, 0]), np.array([0.7, 0, 0, 0.7])),
}


def _fake_embedders() -> dict[str, cfi.Embedder]:
    """Photo `<style>_<n>.jpg` -> its style's direction (+ tiny jitter by n); query the same."""

    def make(view: int) -> cfi.Embedder:
        def embed(path: Path) -> np.ndarray:
            style, n = path.stem.split("_")
            v = _DIRS[style][view].astype(np.float32).copy()
            v[view] += 0.01 * int(n)
            return v / np.linalg.norm(v)

        return embed

    return {"clip": make(0), "dino": make(1)}


def _by_style(styles: str = "ABCDEF", n: int = 2) -> dict[str, list[Path]]:
    return {s: [Path(f"{s}_{i}.jpg") for i in range(1, n + 1)] for s in styles}


@pytest.fixture(autouse=True)
def _tmp_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cfi, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(cfi, "EXEMPLARS_SCREENED", tmp_path / "missing.csv")


def _table() -> pl.DataFrame:
    keys = list("ABCDEF")
    return pl.DataFrame(
        {
            "style_key": keys,
            "predicted_intensity": [30.0, 25.0, 20.0, 15.0, 10.0, 5.0],
        }
    )


def test_index_prototype_is_unit_norm_mean_and_query_finds_own_style() -> None:
    index = cfi.build_index(_by_style(), _fake_embedders())
    assert index.styles == list("ABCDEF")
    assert np.allclose(np.linalg.norm(index.means["clip"], axis=1), 1.0, atol=1e-5)
    q = cfi.embed_query(Path("C_9.jpg"), index.embedders)
    res = cfi.retrieve(q, index)
    assert res.top("avg")[0][0] == "C"
    assert res.top("clip")[0][0] == "C" and res.top("dino")[0][0] == "C"
    sims = [s for _, s in res.ranked["avg"]]
    assert sims == sorted(sims, reverse=True)


def test_confidence_high_needs_agreement_and_margin() -> None:
    index = cfi.build_index(_by_style(), _fake_embedders())
    res = cfi.retrieve(cfi.embed_query(Path("A_9.jpg"), index.embedders), index)
    assert res.confidence == "high" and res.margin >= cfi.MARGIN_HIGH


def test_confidence_labels_from_synthetic_rankings() -> None:
    def ranking(order: list[str], top: float, rest: float) -> list[tuple[str, float]]:
        return [(s, top if i == 0 else rest) for i, s in enumerate(order)]

    a6 = list("ABCDEF")
    agree_wide = {
        "clip": ranking(a6, 0.9, 0.5),
        "dino": ranking(a6, 0.9, 0.5),
        "avg": ranking(a6, 0.9, 0.5),
    }
    assert cfi.confidence_label(agree_wide)[0] == "high"
    agree_narrow = {**agree_wide, "avg": ranking(a6, 0.51, 0.5)}  # margin 0.01 < MARGIN_HIGH
    assert cfi.confidence_label(agree_narrow)[0] == "medium"
    cross = {  # views disagree on top-1 but each top-1 is in the other's top-5
        "clip": ranking(list("ABCDEF"), 0.9, 0.5),
        "dino": ranking(list("BACDEF"), 0.9, 0.5),
        "avg": ranking(a6, 0.9, 0.5),
    }
    assert cfi.confidence_label(cross)[0] == "medium"
    disjoint = {
        "clip": ranking(list("ABCDEF"), 0.9, 0.5),
        "dino": ranking(list("FEDCBA"), 0.9, 0.5),  # A is 6th in dino, F is 6th in clip
        "avg": ranking(a6, 0.9, 0.5),
    }
    assert cfi.confidence_label(disjoint)[0] == "low"


def test_collect_index_images_leave_out_and_uncovered(tmp_path: Path) -> None:
    for aid in (1, 2, 3, 4):
        (tmp_path / f"{aid:010d}.jpg").write_bytes(b"x")
    (tmp_path / "notes.jpg").write_bytes(b"x")  # non-numeric stem is ignored
    art = pl.DataFrame({"article_id": [1, 2, 3, 4], "style_key": ["S1", "S1", "S2", "S3"]})
    by_style = cfi.collect_index_images(
        ["S1", "S2"], exclude_articles=[3], article_styles=art, image_dirs=[tmp_path]
    )
    # article 3 (the validation photo) is dropped, leaving S2 UNCOVERED; S3 is not requested
    assert set(by_style) == {"S1"}
    assert [p.stem for p in by_style["S1"]] == ["0000000001", "0000000002"]


def test_uncovered_style_is_never_retrieved() -> None:
    index = cfi.build_index(_by_style("ABCDE"), _fake_embedders())  # F has no photos
    res = cfi.retrieve(cfi.embed_query(Path("F_9.jpg"), index.embedders), index)
    assert "F" not in [s for s, _ in res.ranked["avg"]]


def test_retrieve_with_empty_candidate_set_raises() -> None:
    index = cfi.build_index(_by_style("AB"), _fake_embedders())
    q = cfi.embed_query(Path("A_9.jpg"), index.embedders)
    with pytest.raises(ValueError):
        cfi.retrieve(q, index, restrict=["Z"])


def test_forecast_concept_end_to_end_returns_top5_and_lookup() -> None:
    index = cfi.build_index(_by_style(), _fake_embedders())
    r = cf.forecast_concept(Path("B_9.jpg"), _table(), index)
    assert (r.style_key, r.forecast, r.rank, r.n_styles) == ("B", 25.0, 2, 6)
    assert r.match_level == "retrieval" and r.confidence == "high"
    assert len(r.top5) == 5 and r.top5[0]["style_key"] == "B"
    assert r.normalised == {"clip": {"style_key": "B"}, "dino": {"style_key": "B"}}
    assert r.similarity > 0.9 and r.margin >= cfi.MARGIN_HIGH
    assert r.sentence() == (
        "maps to B; forecast 25.0 units/product/week; rank 2 of 6; confidence high"
    )


def test_forecast_concept_restricts_to_table_styles() -> None:
    index = cfi.build_index(_by_style(), _fake_embedders())
    table = _table().filter(pl.col("style_key") != "B")
    r = cf.forecast_concept(Path("B_9.jpg"), table, index)
    assert r.style_key != "B" and "B" not in [t["style_key"] for t in r.top5]


def test_concept_forecast_new_fields_have_defaults() -> None:
    r = cf.ConceptForecast("S", 1.0, 1, 10, "exact", "low", {})
    assert r.top5 == [] and r.similarity == 0.0 and r.margin == 0.0 and r.unavailable == {}
