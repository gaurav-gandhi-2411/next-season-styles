"""Compare the 5-column `style_key` against a reduced 4-column key via article-level silhouette.

The earlier `style_validation.py` compared text-embedding clusters against `index_group_name`
(coarse, only 5 values) via ARI/NMI and found weak agreement (ARI~=0.025, NMI~=0.15). That
comparison has since been judged the WRONG test for style_key COHERENCE: `index_group_name` is far
coarser than the 3,076-way style_key partition, so weak agreement with it is close to
uninformative by construction (a cardinality mismatch, not a semantic-coherence measurement) --
see that module's own JUDGMENT CALL note. `style_validation.py` and its outputs
(`reports/tables/style_validation_ari_nmi.csv`, `..._disagreements.csv`) are left untouched; this
module supersedes the coherence QUESTION with a better-targeted test, it does not erase the
record.

NEW PRIMARY TEST: does the style_key partition itself explain more structure in `detail_desc`
sentence-embedding space than a same-shaped random partition would by chance? Silhouette score
(cohesion vs. separation of a partition, measured IN the embedding space) answers this directly,
with no coarser comparison partition needed. A label-permutation null (shuffle style_key labels,
embeddings held fixed, 50 times) calibrates "how much silhouette would ANY partition of this shape
get by chance" -- the z-score is `(observed_silhouette - null_mean) / null_sd`.

GRANULARITY JUDGMENT CALL -- article-level, not style_key-level: silhouette is defined per POINT
as `(b - a) / max(a, b)`, using that point's mean distance to other members of its own cluster (a)
vs. to its nearest other cluster (b). Style_key-level mean-pooling collapses each style_key to ONE
point, which carries no information about whether the ARTICLES within a style_key actually sit
close together in embedding space -- exactly the coherence question this test is meant to answer.
Evaluating on individual article embeddings, labelled by their style_key, is the standard and
defensible use of silhouette; that is what this module does.

COMPUTE JUDGMENT CALL -- fixed random subsample, not the full ~70k-90k article set: silhouette
(and the pairwise distance matrix it needs) is O(n^2). A full n~=90,000 distance matrix is
~65 GB (n^2 float64) and infeasible on this CPU-only setup. Instead, `SILHOUETTE_SAMPLE_SIZE`
articles are drawn once per key (`SILHOUETTE_SUBSAMPLE_SEED`, without replacement), and the SAME
subsample -- same points, same precomputed distance matrix -- is reused for the observed score and
all `N_PERMUTATIONS` null draws, so only the label assignment ever changes across observed vs.
null, never the points or the distances between them.

Distance metric: cosine -- the standard choice for sentence-transformers embeddings, which are
trained and compared via cosine similarity, not Euclidean distance.

ARTICLE EMBEDDING CACHE: embedding is the expensive step (CPU-minutes for tens of thousands of
articles). Embeddings are computed once for the UNION of articles needed by both keys and cached
to `DEFAULT_EMBEDDING_CACHE_PATH` (gitignored, under `data/interim/`, per this repo's existing
data/-directory convention) so a re-run of this script does not silently recompute them.
`style_validation.py` predates this cache and does not read or write it -- this module is purely
additive, reusing `style_validation.embed_articles` (the model-loading/encoding logic) rather than
duplicating it.

REDUCED KEY: drops `index_group_name` from the production `STYLE_KEY_COLS`, keeping
`product_type_name`, `garment_group_name`, `perceived_colour_master_name`, and
`graphical_appearance_name`. Support-filter thresholds and logic are identical to
`style_panel.filter_by_support` (>= `MIN_ARTICLES_PER_STYLE` articles AND
`MIN_LIFETIME_UNITS_PER_STYLE` units, both lifetime), reimplemented here to operate on a
lifetime-aggregate table directly (`compute_lifetime_by_key` skips the weekly grouping the
production panel needs but this comparison does not) -- this is a standalone analysis, it never
writes to or modifies `style_panel.py`, `STYLE_KEY_COLS`, or the production panel parquet.

THIS MODULE IS REPORT-ONLY: even if the reduced key looks better here, switching the production
`STYLE_KEY_COLS` requires explicit sign-off after reviewing this report -- `main()` never writes
anywhere under `data/processed/` or touches `style_panel.py`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import pairwise_distances, silhouette_score

from nss.features.style_panel import (
    DEFAULT_ARTICLES_PATH,
    DEFAULT_TRANSACTIONS_DIR,
    MIN_ARTICLES_PER_STYLE,
    MIN_LIFETIME_UNITS_PER_STYLE,
    STYLE_KEY_COLS,
    STYLE_KEY_SEPARATOR,
)
from nss.features.style_validation import embed_articles

REDUCED_STYLE_KEY_COLS: list[str] = [
    "product_type_name",
    "garment_group_name",
    "perceived_colour_master_name",
    "graphical_appearance_name",
]

N_PERMUTATIONS = 50
# Seeds 42, 43, ..., 91 -- one per permutation, documented scheme.
PERMUTATION_SEED_START = 42
SILHOUETTE_SAMPLE_SIZE = 8000
SILHOUETTE_SUBSAMPLE_SEED = 42
SILHOUETTE_METRIC = "cosine"

DEFAULT_EMBEDDING_CACHE_PATH = Path("data/interim/article_detail_desc_embeddings.parquet")
DEFAULT_COMPARISON_OUT = Path("reports/tables/style_key_silhouette_comparison.csv")


def compute_lifetime_by_key(
    key_cols: list[str],
    transactions_dir: Path = DEFAULT_TRANSACTIONS_DIR,
    articles_path: Path = DEFAULT_ARTICLES_PATH,
) -> pl.DataFrame:
    """Compute lifetime (article count, units) aggregates grouped by an arbitrary key column set.

    Mirrors `style_panel.build_style_week_panel`'s lifetime aggregation but skips the weekly
    grouping entirely -- this module only ever needs lifetime totals (for the support filter) and
    article-level rows for embedding, never the week-level panel itself.

    Args:
        key_cols: Article-level categorical columns from `articles.csv` defining the key.
        transactions_dir: Directory of the year-month partitioned transactions Parquet dataset.
        articles_path: Path to `articles.csv`.

    Returns:
        One row per distinct combination of `key_cols`, with `lifetime_units` (transaction-row
        count) and `lifetime_n_articles` (distinct lifetime `article_id` count).
    """
    articles = pl.scan_csv(articles_path).select(["article_id", *key_cols])
    txns = pl.scan_parquet(str(transactions_dir / "**" / "*.parquet"))
    joined = txns.join(articles, on="article_id", how="inner")
    lifetime = joined.group_by(key_cols).agg(
        lifetime_units=pl.len(),
        lifetime_n_articles=pl.col("article_id").n_unique(),
    )
    return lifetime.collect(engine="streaming")


def filter_lifetime_by_support(
    lifetime: pl.DataFrame,
    min_articles: int = MIN_ARTICLES_PER_STYLE,
    min_units: int = MIN_LIFETIME_UNITS_PER_STYLE,
) -> tuple[pl.DataFrame, dict[str, float]]:
    """Apply the lifetime support filter to a `compute_lifetime_by_key` table.

    Same thresholds/semantics as `style_panel.filter_by_support` (>= `min_articles` distinct
    lifetime articles AND >= `min_units` lifetime units), reimplemented to operate directly on a
    lifetime table with no panel to join against -- see module docstring.

    Args:
        lifetime: Lifetime aggregates, as returned by `compute_lifetime_by_key`.
        min_articles: Minimum distinct lifetime articles per key combination to keep.
        min_units: Minimum lifetime units per key combination to keep.

    Returns:
        A tuple `(kept_lifetime, stats)` where `stats` has keys `n_styles_before`,
        `n_styles_after`, and `pct_units_retained`.
    """
    total_units = lifetime["lifetime_units"].sum()
    n_before = lifetime.height

    kept = lifetime.filter(
        (pl.col("lifetime_n_articles") >= min_articles) & (pl.col("lifetime_units") >= min_units)
    )
    n_after = kept.height
    kept_units = kept["lifetime_units"].sum()
    pct_retained = 100.0 * kept_units / total_units if total_units else 0.0

    stats = {
        "n_styles_before": n_before,
        "n_styles_after": n_after,
        "pct_units_retained": pct_retained,
    }
    return kept, stats


def build_style_key_column(df: pl.DataFrame, key_cols: list[str]) -> pl.DataFrame:
    """Add a `style_key` string column: `key_cols` joined with `STYLE_KEY_SEPARATOR`.

    Args:
        df: Any frame containing all of `key_cols`.
        key_cols: Columns to concatenate, in order.

    Returns:
        `df` with an added `style_key` column.
    """
    return df.with_columns(
        pl.concat_str([pl.col(c) for c in key_cols], separator=STYLE_KEY_SEPARATOR).alias(
            "style_key"
        )
    )


def load_articles_for_key(
    kept_key_combos: pl.DataFrame,
    key_cols: list[str],
    articles_path: Path = DEFAULT_ARTICLES_PATH,
) -> pl.DataFrame:
    """Restrict `articles.csv` to articles whose `key_cols` combination passed the support filter.

    Generic version of `style_validation.load_validation_articles` for an arbitrary key column
    set -- reused for both the production 5-column key and the reduced 4-column key so both go
    through identical article-selection logic.

    Args:
        kept_key_combos: Distinct kept `key_cols` combinations (e.g. from
            `filter_lifetime_by_support`).
        key_cols: Article-level categorical columns from `articles.csv` defining the key.
        articles_path: Path to `articles.csv`.

    Returns:
        One row per matching article, with `article_id`, `detail_desc`, and `key_cols`. Articles
        whose combination did not pass the support filter are excluded (inner join).
    """
    articles = pl.scan_csv(articles_path).select(["article_id", "detail_desc", *key_cols])
    joined = articles.join(kept_key_combos.lazy().select(key_cols), on=key_cols, how="inner")
    return joined.collect()


def load_or_compute_article_embeddings(
    article_ids: set[int],
    articles_path: Path = DEFAULT_ARTICLES_PATH,
    cache_path: Path = DEFAULT_EMBEDDING_CACHE_PATH,
) -> pl.DataFrame:
    """Load cached `detail_desc` article embeddings, computing + caching them if needed.

    Args:
        article_ids: `article_id`s that must be present in the returned frame. Ids with a null
            `detail_desc` are simply absent from the result (same drop convention as
            `style_validation.py`) -- not an error.
        articles_path: Path to `articles.csv`.
        cache_path: Parquet cache path (`article_id`, `embedding` list[f32] columns).

    Returns:
        DataFrame with `article_id` and `embedding` (list[f32]) columns, restricted to
        `article_ids` that have a non-null `detail_desc`.
    """
    if cache_path.exists():
        cached = pl.read_parquet(cache_path)
        if article_ids <= set(cached["article_id"].to_list()):
            print(f"Reusing cached article embeddings from {cache_path}")
            return cached.filter(pl.col("article_id").is_in(list(article_ids)))
        print(f"Cache at {cache_path} does not cover all needed article_ids -- recomputing")

    articles = (
        pl.scan_csv(articles_path)
        .select(["article_id", "detail_desc"])
        .filter(pl.col("article_id").is_in(list(article_ids)) & pl.col("detail_desc").is_not_null())
        .collect()
    )
    print(f"Embedding {articles.height} article descriptions (not cached)...")
    embeddings = embed_articles(articles["detail_desc"].to_list())

    result = articles.select("article_id").with_columns(pl.Series("embedding", embeddings.tolist()))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(cache_path)
    print(f"Cached {result.height} article embeddings to {cache_path}")
    return result


def subsample_rows(
    embeddings: np.ndarray,
    labels: np.ndarray,
    sample_size: int,
    seed: int = SILHOUETTE_SUBSAMPLE_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw a fixed random subsample of rows (without replacement); embeddings/labels stay aligned.

    Args:
        embeddings: Array of shape `(n, dim)`.
        labels: Length-`n` array of cluster labels, row-aligned with `embeddings`.
        sample_size: Number of rows to draw. If `>= n`, all rows are returned unchanged.
        seed: Seed for the draw.

    Returns:
        `(sub_embeddings, sub_labels)`: `sample_size` rows each, or all `n` if `n <= sample_size`.
    """
    n = embeddings.shape[0]
    if sample_size >= n:
        return embeddings, labels
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=sample_size, replace=False)
    return embeddings[idx], labels[idx]


def permutation_null_silhouette(
    embeddings: np.ndarray,
    labels: np.ndarray,
    metric: str = SILHOUETTE_METRIC,
    n_permutations: int = N_PERMUTATIONS,
    seed_start: int = PERMUTATION_SEED_START,
) -> dict[str, float]:
    """Silhouette score of `labels` in `embeddings` space, plus a label-permutation null.

    Builds the pairwise distance matrix ONCE and reuses it for the observed score and every null
    draw -- only the label assignment changes across draws, never the points or their distances.
    See module docstring's COMPUTE JUDGMENT CALL.

    Args:
        embeddings: Array of shape `(n, dim)`.
        labels: Length-`n` array of cluster labels, row-aligned with `embeddings`.
        metric: Distance metric for `sklearn.metrics.pairwise_distances`.
        n_permutations: Number of label-shuffle draws for the null distribution.
        seed_start: First seed; seeds `seed_start, seed_start + 1, ..., seed_start +
            n_permutations - 1` are used, one per permutation.

    Returns:
        Dict with `observed_silhouette`, `null_mean`, `null_sd`, `z_score`, `n_samples`,
        `n_permutations`. `z_score` is `nan` if `null_sd` is 0 (degenerate null, not expected in
        practice with thousands of distinct labels).
    """
    labels = np.asarray(labels)
    distance_matrix = pairwise_distances(embeddings, metric=metric)
    observed = float(silhouette_score(distance_matrix, labels, metric="precomputed"))

    null_scores = np.empty(n_permutations, dtype=np.float64)
    for i, seed in enumerate(range(seed_start, seed_start + n_permutations)):
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(labels)
        null_scores[i] = silhouette_score(distance_matrix, shuffled, metric="precomputed")

    null_mean = float(null_scores.mean())
    null_sd = float(null_scores.std(ddof=1))
    z_score = (observed - null_mean) / null_sd if null_sd > 0 else float("nan")

    return {
        "observed_silhouette": observed,
        "null_mean": null_mean,
        "null_sd": null_sd,
        "z_score": z_score,
        "n_samples": embeddings.shape[0],
        "n_permutations": n_permutations,
    }


def main() -> None:
    """CLI entry point: compare the 5-column and reduced 4-column style keys, write the report.

    Report-only -- never writes to `data/processed/`, `style_panel.py`, or `STYLE_KEY_COLS`.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transactions-dir", type=Path, default=DEFAULT_TRANSACTIONS_DIR)
    parser.add_argument("--articles-path", type=Path, default=DEFAULT_ARTICLES_PATH)
    parser.add_argument("--embedding-cache-path", type=Path, default=DEFAULT_EMBEDDING_CACHE_PATH)
    parser.add_argument("--out-path", type=Path, default=DEFAULT_COMPARISON_OUT)
    args = parser.parse_args()

    keys: list[tuple[str, list[int]]] = [
        ("current_5col", STYLE_KEY_COLS),
        ("reduced_4col", REDUCED_STYLE_KEY_COLS),
    ]

    per_key_articles: dict[str, pl.DataFrame] = {}
    per_key_stats: dict[str, dict[str, float]] = {}
    all_article_ids: set[int] = set()

    for name, key_cols in keys:
        print(f"[{name}] computing lifetime aggregates over {key_cols}...")
        lifetime = compute_lifetime_by_key(key_cols, args.transactions_dir, args.articles_path)
        kept, stats = filter_lifetime_by_support(lifetime)
        print(
            f"[{name}] styles before={stats['n_styles_before']} after={stats['n_styles_after']} "
            f"pct_units_retained={stats['pct_units_retained']:.2f}%"
        )
        articles = load_articles_for_key(kept, key_cols, args.articles_path)
        articles = build_style_key_column(articles, key_cols)
        per_key_articles[name] = articles
        per_key_stats[name] = stats
        all_article_ids |= set(articles["article_id"].to_list())

    print(f"Embedding/loading {len(all_article_ids)} distinct articles needed by either key...")
    embeddings_df = load_or_compute_article_embeddings(
        all_article_ids, args.articles_path, args.embedding_cache_path
    )

    rows: list[dict[str, object]] = []
    for name, key_cols in keys:
        articles = per_key_articles[name].join(embeddings_df, on="article_id", how="inner")
        n_embedded = articles.height
        embeddings = np.asarray(articles["embedding"].to_list(), dtype=np.float64)
        labels = articles["style_key"].to_numpy()

        sub_embeddings, sub_labels = subsample_rows(embeddings, labels, SILHOUETTE_SAMPLE_SIZE)
        print(
            f"[{name}] computing silhouette + permutation null on "
            f"{sub_embeddings.shape[0]} articles..."
        )
        result = permutation_null_silhouette(sub_embeddings, sub_labels)
        print(
            f"[{name}] observed={result['observed_silhouette']:.4f} "
            f"null_mean={result['null_mean']:.4f} null_sd={result['null_sd']:.4f} "
            f"z={result['z_score']:.2f}"
        )

        stats = per_key_stats[name]
        rows.append(
            {
                "comparison_key": name,
                "n_key_cols": len(key_cols),
                "style_count_before": stats["n_styles_before"],
                "style_count_after": stats["n_styles_after"],
                "pct_lifetime_units_retained": stats["pct_units_retained"],
                "n_articles_embedded": n_embedded,
                "n_samples_used_for_silhouette": result["n_samples"],
                "observed_silhouette": result["observed_silhouette"],
                "null_mean": result["null_mean"],
                "null_sd": result["null_sd"],
                "z_score": result["z_score"],
                "n_permutations": result["n_permutations"],
            }
        )

    out = pl.DataFrame(rows)
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(args.out_path)
    print(f"Wrote side-by-side comparison to {args.out_path}")

    current = next(r for r in rows if r["comparison_key"] == "current_5col")
    reduced = next(r for r in rows if r["comparison_key"] == "reduced_4col")
    print(
        "\nRECOMMENDATION CRITERION: reduced key retains >= 80% lifetime units AND "
        "z_score(reduced) >= z_score(current_5col)."
    )
    meets_retention = reduced["pct_lifetime_units_retained"] >= 80.0
    meets_silhouette = reduced["z_score"] >= current["z_score"]
    if meets_retention and meets_silhouette:
        print("RECOMMEND: switch to the reduced 4-column key. Report only -- pipeline unchanged.")
    else:
        print("RECOMMEND: keep the current 5-column key. Report only -- pipeline unchanged.")


if __name__ == "__main__":
    main()
