"""Image-retrieval index of real catalogue photos per style (Q2: the closed loop as retrieval).

The free-text route (VLM caption -> parse -> style_key) asked a small captioner to reproduce H&M's
internal taxonomy from a photo and reached 12.5% exact-style accuracy (SmolVLM) / 2.5% (Florence-2)
on 40 real photos. This module replaces it: embed the photo and find the nearest STYLE by cosine
similarity to the MEAN embedding of that style's real reference photos.

EMBEDDING VIEWS: CLIP ViT-L/14 (`clip_scoring.embed_image`) and DINOv2-base
(`dino_scoring.embed_image`). HEADLINE CONFIGURATION (fixed before any test number was computed):
the AVERAGE of the two cosine similarities, `(cos_clip + cos_dino) / 2` (`AVG`). CLIP-only and
DINOv2-only are reported as ablations. Each style is scored against the L2-normalised mean of its
index photos' L2-normalised embeddings.

CONFIDENCE (defined here, before evaluation; never derived from the forecast itself):
- `high`   = the CLIP view and the DINOv2 view agree on the top-1 style AND the averaged-similarity
             margin between the top-1 and the 6th-ranked style is >= `MARGIN_HIGH`;
- `medium` = not high, and either view's top-1 style is inside the OTHER view's top-5;
- `low`    = otherwise.
`MARGIN_HIGH = 0.02` is a fixed a-priori constant on the averaged-cosine scale (top-1 minus 6th
rank): it was not tuned on the validation photos; the confidence labels' accuracy is REPORTED by
`concept_forecast_validation`, not assumed.

INDEX CONTENT: every real article photo on disk (`data/images`, `data/images_q2`) whose article's
`style_key` is in the target forecast table, grouped by style, plus the screened reference sets
(`n9_generate.load_refs()`: `reference_base_widened*.csv`). Photos flagged non-full-garment in
`exemplar_images_screened.csv` (texture crops) are excluded. Articles listed in `exclude_articles`
(the validation photos) are dropped from every style BEFORE the mean is formed; a style left with no
photo is UNCOVERED and can never be retrieved (it counts as a miss in validation).

Embeddings are cached under `data/q2_cache/` (one `.npz` per model, keyed by file stem) so re-runs
cost nothing; the CPU embedding cost is ~1s/image for CLIP ViT-L/14 (`clip_scoring` docstring).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

ARTICLES = Path("data/raw/articles.csv")
IMAGE_DIRS: tuple[Path, ...] = (Path("data/images"), Path("data/images_q2"))
CACHE_DIR = Path("data/q2_cache")
EXEMPLARS_SCREENED = Path("reports/tables/exemplar_images_screened.csv")
STYLE_COLS = [
    "index_group_name",
    "product_type_name",
    "garment_group_name",
    "perceived_colour_master_name",
    "graphical_appearance_name",
]
MARGIN_HIGH = 0.02  # see module docstring CONFIDENCE
RANK_MARGIN = 6  # margin is top-1 similarity minus the RANK_MARGIN-th ranked style's
VIEWS = ("clip", "dino", "avg")
HEADLINE_VIEW = "avg"

Embedder = Callable[[Path], np.ndarray]


def _embedders() -> dict[str, Embedder]:
    """The two embedding views (imported lazily: transformers/torch are heavy)."""
    from nss.generate import clip_scoring, dino_scoring

    return {"clip": clip_scoring.embed_image, "dino": dino_scoring.embed_image}


def style_key_of(row: dict[str, object]) -> str:
    """`' || '`-joined style key from an articles row."""
    return " || ".join(str(row[c]) for c in STYLE_COLS)


def load_article_styles(path: Path = ARTICLES) -> pl.DataFrame:
    """`article_id`, `style_key` and the five attribute columns for every catalogue article."""
    return (
        pl.read_csv(path)
        .with_columns(pl.concat_str(STYLE_COLS, separator=" || ").alias("style_key"))
        .select("article_id", "style_key", *STYLE_COLS)
    )


def _excluded_crops() -> set[int]:
    """Article ids the earlier screening flagged as not a full-garment photo (texture crops)."""
    if not EXEMPLARS_SCREENED.exists():
        return set()
    df = pl.read_csv(EXEMPLARS_SCREENED)
    return set(df.filter(~pl.col("is_full_garment"))["article_id"].to_list())


def collect_index_images(
    styles: Iterable[str],
    exclude_articles: Iterable[int] = (),
    article_styles: pl.DataFrame | None = None,
    image_dirs: Sequence[Path] = IMAGE_DIRS,
) -> dict[str, list[Path]]:
    """Real photos on disk per style (restricted to `styles`), minus `exclude_articles`.

    Photos are found by scanning `image_dirs` for `<article_id>.jpg` files and joining to the
    catalogue (so the screened reference sets, whose files live in `data/images`, are included);
    articles flagged as texture crops are dropped. The result is deterministic (sorted paths).
    """
    wanted = set(styles)
    skip = set(exclude_articles) | _excluded_crops()
    art = load_article_styles() if article_styles is None else article_styles
    style_of = dict(zip(art["article_id"].to_list(), art["style_key"].to_list(), strict=True))
    out: dict[str, list[Path]] = {}
    seen: set[int] = set()
    for d in image_dirs:
        for p in sorted(d.glob("*.jpg")):
            if not p.stem.isdigit():
                continue
            aid = int(p.stem)
            key = style_of.get(aid)
            if key is None or key not in wanted or aid in skip or aid in seen:
                continue
            seen.add(aid)
            out.setdefault(key, []).append(p)
    return out


def _load_cache(name: str) -> dict[str, np.ndarray]:
    path = CACHE_DIR / f"emb_{name}.npz"
    if not path.exists():
        return {}
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def _save_cache(name: str, cache: dict[str, np.ndarray]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE_DIR / f"emb_{name}.npz", **cache)


def embed_cached(name: str, embed: Embedder, paths: Sequence[Path]) -> np.ndarray:
    """Embeddings of `paths` (rows), cached on disk by file stem for catalogue photos only.

    Only digit-named files (`<article_id>.jpg`) are cached: generated concepts have non-unique
    stems (`s0.35_seed45.png` exists in many folders), so they are always embedded fresh.
    """
    cache = _load_cache(name)
    missing = [p for p in paths if p.stem.isdigit() and p.stem not in cache]
    for i, p in enumerate(missing, 1):
        cache[p.stem] = np.asarray(embed(p), dtype=np.float32)
        if i % 100 == 0:
            _save_cache(name, cache)
    if missing:
        _save_cache(name, cache)
    rows = [
        cache[p.stem] if p.stem.isdigit() else np.asarray(embed(p), dtype=np.float32) for p in paths
    ]
    return np.stack(rows) if rows else np.empty((0, 0), np.float32)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


@dataclass
class RetrievalIndex:
    """Per-style mean embeddings (unit norm) for each view, plus how many photos back each style."""

    styles: list[str]
    means: dict[str, np.ndarray]  # view ("clip"/"dino") -> (n_styles, dim)
    n_images: dict[str, int]
    embedders: dict[str, Embedder] = field(default_factory=dict, repr=False)
    # view -> (n_styles, dim) SUM of each style's unit photo embeddings (for leave-one-out)
    sums: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    def similarities(
        self,
        q: dict[str, np.ndarray],
        restrict: Sequence[str] | None = None,
        leave_out: str | None = None,
    ):
        """Cosine similarity of one query (per-view unit vectors) to every style, per view.

        Returns `(styles, {"clip": s, "dino": s, "avg": s})`, restricted to `restrict` if given.
        `leave_out` names the query's OWN style when the query is itself one of that style's index
        photos: that style's prototype is recomputed without the query photo (leave-one-out), so a
        photo is never matched against a prototype it helped form. It needs >= 2 photos.
        """
        keep = np.arange(len(self.styles))
        if restrict is not None:
            allowed = set(restrict)
            keep = np.array([i for i, s in enumerate(self.styles) if s in allowed], dtype=int)
        sims = {v: self.means[v][keep] @ q[v] for v in ("clip", "dino")}
        if leave_out is not None:
            if self.n_images.get(leave_out, 0) < 2:
                raise ValueError(f"leave-one-out needs >= 2 photos for {leave_out!r}")
            own = int(self.styles.index(leave_out))
            pos = np.flatnonzero(keep == own)
            if pos.size:
                for v in ("clip", "dino"):
                    proto = _unit(self.sums[v][own] - q[v])
                    sims[v][pos[0]] = proto @ q[v]
        sims["avg"] = (sims["clip"] + sims["dino"]) / 2
        return [self.styles[i] for i in keep], sims


def build_index(
    by_style: dict[str, list[Path]], embedders: dict[str, Embedder] | None = None
) -> RetrievalIndex:
    """Embed every index photo (cached) and form each style's mean-embedding prototype."""
    emb = _embedders() if embedders is None else embedders
    styles = sorted(by_style)
    flat = [(s, p) for s in styles for p in by_style[s]]
    means: dict[str, np.ndarray] = {}
    sums: dict[str, np.ndarray] = {}
    for view in ("clip", "dino"):
        vecs = _unit(embed_cached(view, emb[view], [p for _, p in flat]))
        rows = []
        for s in styles:
            idx = [i for i, (st, _) in enumerate(flat) if st == s]
            rows.append(vecs[idx].sum(axis=0))
        sums[view] = np.stack(rows)
        means[view] = _unit(sums[view])  # the unit mean == the unit sum
    return RetrievalIndex(styles, means, {s: len(by_style[s]) for s in styles}, emb, sums)


def embed_query(path: Path, embedders: dict[str, Embedder] | None = None) -> dict[str, np.ndarray]:
    """Unit-norm per-view embedding of one query image (cached by file stem for the CLIP/DINO
    caches, so validation photos already embedded during index scans are not recomputed)."""
    emb = _embedders() if embedders is None else embedders
    return {v: _unit(embed_cached(v, emb[v], [path])[0]) for v in ("clip", "dino")}


@dataclass(frozen=True)
class Retrieval:
    """One query's ranked retrieval under every view, plus the confidence label."""

    ranked: dict[str, list[tuple[str, float]]]  # view -> [(style_key, similarity), ...] best first
    confidence: str
    margin: float

    def top(self, view: str = HEADLINE_VIEW, k: int = 1) -> list[tuple[str, float]]:
        """The `k` best `(style, similarity)` pairs under `view`."""
        return self.ranked[view][:k]


def confidence_label(ranked: dict[str, list[tuple[str, float]]]) -> tuple[str, float]:
    """`(label, margin)` per the module docstring CONFIDENCE definition."""
    avg = ranked["avg"]
    sixth = avg[RANK_MARGIN - 1][1] if len(avg) >= RANK_MARGIN else avg[-1][1]
    margin = float(avg[0][1] - sixth)
    clip_top, dino_top = ranked["clip"][0][0], ranked["dino"][0][0]
    if clip_top == dino_top and margin >= MARGIN_HIGH:
        return "high", margin
    clip5 = {s for s, _ in ranked["clip"][:5]}
    dino5 = {s for s, _ in ranked["dino"][:5]}
    if clip_top in dino5 or dino_top in clip5:
        return "medium", margin
    return "low", margin


def retrieve(
    query: dict[str, np.ndarray],
    index: RetrievalIndex,
    restrict: Sequence[str] | None = None,
    leave_out: str | None = None,
) -> Retrieval:
    """Rank every (allowed) covered style for one query embedding under all three views.

    `leave_out`: the query's own style, when the query is one of that style's index photos (see
    `RetrievalIndex.similarities`).
    """
    styles, sims = index.similarities(query, restrict, leave_out)
    if not styles:
        raise ValueError("retrieval index has no style in the requested candidate set")
    ranked = {
        v: sorted(zip(styles, map(float, s), strict=True), key=lambda t: (-t[1], t[0]))
        for v, s in sims.items()
    }
    label, margin = confidence_label(ranked)
    return Retrieval(ranked=ranked, confidence=label, margin=margin)
