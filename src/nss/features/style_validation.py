"""Validate `style_key` semantic coherence via sentence-embedding clustering of `detail_desc`.

The `style_key` (see `style_panel.py`) is a composite of 5 categorical attribute columns
from `articles.csv`. This module asks: does an independent, purely text-based notion of product
similarity roughly agree with that categorical partition? Clustering products by the free-text
`detail_desc` field encodes visual/semantic similarity through a completely different signal
(natural language) than the 5 attribute columns. If text clusters align with style_key groupings,
that is evidence the style_key definition is a meaningful, non-arbitrary grouping rather than an
artifact of an arbitrary column choice. This is a read-only validation check -- it does not modify
`style_panel.py`, the panel, or the style_key definition itself.

DEPENDENCY NOTE: this module is the sole reason `sentence-transformers` (and its transitive
`torch` CPU dependency) is a project dependency -- strictly heavier than everything else in this
repo. Scope is deliberately narrow: one offline validation script, CPU-only,
`all-MiniLM-L6-v2` (small, fast, well-established sentence embedding model), no GPU, no online
serving. If this dependency ever becomes a build/deploy liability, this module (and it alone) can
be excised without touching any other code path.

Method:
1. Restrict `articles.csv` to `article_id`s whose style_key survived the panel's support filter (the
   distinct style_key combinations present in `data/processed/style_week_panel.parquet`).
2. Embed each article's `detail_desc` with `all-MiniLM-L6-v2` (CPU). ~0.39% of articles have a
   null `detail_desc` (matches the rate the panel build reported); those articles are dropped from
   the embedding step rather than embedded as an empty string, so a missing description never
   contributes a degenerate/near-zero vector that would silently drag down its style_key's mean.
3. Mean-pool article embeddings per style_key -> one embedding vector per style_key.
4. K-means cluster the style_key-level embeddings; pick k by silhouette score over a small sweep.
   k is NEVER tuned to inflate agreement with the categorical partition -- only the silhouette
   score (an intrinsic, label-free clustering-quality measure) drives the choice.
5. Compare the resulting text-cluster labels against `index_group_name` via Adjusted Rand Index
   and Normalized Mutual Information -- see JUDGMENT CALL below for why this is the primary
   comparison partition. Agreement is reported honestly, weak or strong.

JUDGMENT CALL -- comparison partition: the primary comparison is against `index_group_name`
(3 categories -- Ladieswear/Menswear/Baby-ish groupings) rather than the full 5-column style_key.
Comparing against the full style_key (3,076 near-singleton categories) would show weak ARI/NMI
against any coarser k-means partition (k <= 100) almost by construction -- that measures a
cardinality mismatch, not style_key's semantic coherence. `index_group_name` is the coarsest,
most semantically self-evident grouping built from the *same* 5 columns that make up style_key --
it IS one of the 5 style_key columns -- so if text embeddings can't recover even this coarse
structure, that's strong evidence against style_key coherence; if they can, that's
suggestive-but-not-conclusive positive evidence (it only tests 1 of the 5 columns directly). Both
ARI/NMI are also computed against the full style_key partition and reported alongside for
completeness -- expected, and confirmed, to be much weaker, precisely because of the cardinality
mismatch just described, not because it is a more meaningful negative result.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from nss.features.style_panel import STYLE_KEY_COLS, STYLE_KEY_SEPARATOR

MODEL_NAME = "all-MiniLM-L6-v2"
RANDOM_STATE = 42
K_CANDIDATES: list[int] = [10, 20, 30, 40, 50, 75, 100]
N_DISAGREEMENT_EXAMPLES = 5

DEFAULT_ARTICLES_PATH = Path("data/raw/articles.csv")
DEFAULT_PANEL_PATH = Path("data/processed/style_week_panel.parquet")
DEFAULT_ARI_NMI_OUT = Path("reports/tables/style_validation_ari_nmi.csv")
DEFAULT_DISAGREEMENTS_OUT = Path("reports/tables/style_validation_disagreements.csv")


def load_kept_style_keys(panel_path: Path = DEFAULT_PANEL_PATH) -> pl.DataFrame:
    """Load the distinct kept style_key combinations from the (already-filtered) panel.

    Args:
        panel_path: Path to the support-filtered style-week panel Parquet.

    Returns:
        One row per distinct kept style_key, with columns `style_key` + the 5 `STYLE_KEY_COLS`.
    """
    return pl.scan_parquet(panel_path).select(["style_key", *STYLE_KEY_COLS]).unique().collect()


def load_validation_articles(
    kept_style_keys: pl.DataFrame,
    articles_path: Path = DEFAULT_ARTICLES_PATH,
) -> pl.DataFrame:
    """Restrict `articles.csv` to articles whose style_key survived the panel's support filter.

    Args:
        kept_style_keys: Distinct kept style_key rows, as returned by `load_kept_style_keys`.
        articles_path: Path to `articles.csv`.

    Returns:
        One row per matching article, with `article_id`, `detail_desc`, `style_key`, and the 5
        `STYLE_KEY_COLS`. Articles belonging to a style_key that did not pass the panel's support
        filter are excluded (inner join).
    """
    articles = pl.scan_csv(articles_path).select(["article_id", "detail_desc", *STYLE_KEY_COLS])
    joined = articles.join(kept_style_keys.lazy(), on=STYLE_KEY_COLS, how="inner")
    return joined.collect()


def embed_articles(descriptions: list[str], model_name: str = MODEL_NAME) -> np.ndarray:
    """Embed a list of article `detail_desc` strings with a sentence-transformers model (CPU).

    Args:
        descriptions: Non-null description strings, one per article.
        model_name: sentence-transformers model identifier.

    Returns:
        Array of shape `(len(descriptions), embedding_dim)`.
    """
    from sentence_transformers import SentenceTransformer  # local import: heavy, model-loading

    model = SentenceTransformer(model_name, device="cpu")
    return np.asarray(model.encode(descriptions, show_progress_bar=True))


def mean_pool_by_group(
    embeddings: np.ndarray, group_ids: list[Any]
) -> tuple[np.ndarray, list[Any]]:
    """Mean-pool row vectors of `embeddings` by their corresponding `group_ids`.

    Pure numpy, no I/O and no model dependency -- kept separate from `embed_articles` so it is
    cheap to unit test without loading a sentence-transformers model.

    Args:
        embeddings: Array of shape `(n, dim)`, one row per item.
        group_ids: Length-`n` sequence of group labels, one per row of `embeddings`.

    Returns:
        A tuple `(pooled, unique_groups)`:
        - `pooled`: array of shape `(n_unique_groups, dim)`, the mean embedding per group.
        - `unique_groups`: the group labels in the same order as `pooled`'s rows, sorted for
          determinism.

    Raises:
        ValueError: if `embeddings` and `group_ids` have mismatched lengths, or `group_ids` is
            empty.
    """
    if len(group_ids) != embeddings.shape[0]:
        raise ValueError(
            f"embeddings has {embeddings.shape[0]} rows but group_ids has {len(group_ids)}"
        )
    if len(group_ids) == 0:
        raise ValueError("group_ids must be non-empty")

    unique_groups = sorted(set(group_ids))
    group_index = {g: i for i, g in enumerate(unique_groups)}
    dim = embeddings.shape[1]
    sums = np.zeros((len(unique_groups), dim), dtype=np.float64)
    counts = np.zeros(len(unique_groups), dtype=np.int64)
    for row, group in zip(embeddings, group_ids, strict=True):
        idx = group_index[group]
        sums[idx] += row
        counts[idx] += 1

    pooled = sums / counts[:, None]
    return pooled, unique_groups


def sweep_k(
    embeddings: np.ndarray,
    k_candidates: list[int] = K_CANDIDATES,
    random_state: int = RANDOM_STATE,
) -> tuple[int, dict[int, float], np.ndarray]:
    """Sweep candidate k values, clustering with KMeans and scoring with mean silhouette.

    Args:
        embeddings: Array of shape `(n_samples, dim)` to cluster.
        k_candidates: Candidate values of k to try.
        random_state: Seed for `KMeans`.

    Returns:
        A tuple `(best_k, silhouette_by_k, best_labels)`:
        - `best_k`: the k with the highest mean silhouette score.
        - `silhouette_by_k`: mean silhouette score for every candidate k tried.
        - `best_labels`: cluster label per row of `embeddings` for `best_k`.
    """
    silhouette_by_k: dict[int, float] = {}
    labels_by_k: dict[int, np.ndarray] = {}
    for k in k_candidates:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        labels = kmeans.fit_predict(embeddings)
        silhouette_by_k[k] = float(silhouette_score(embeddings, labels))
        labels_by_k[k] = labels

    best_k = max(silhouette_by_k, key=lambda k: silhouette_by_k[k])
    return best_k, silhouette_by_k, labels_by_k[best_k]


def compute_agreement(
    text_labels: np.ndarray, categorical_labels: list[Any]
) -> tuple[float, float]:
    """Compute Adjusted Rand Index and Normalized Mutual Information between two partitions.

    Args:
        text_labels: Cluster label per item, from the text-embedding clustering.
        categorical_labels: Ground-truth-for-comparison label per item (same order/length).

    Returns:
        A tuple `(ari, nmi)`.
    """
    ari = float(adjusted_rand_score(categorical_labels, text_labels))
    nmi = float(normalized_mutual_info_score(categorical_labels, text_labels))
    return ari, nmi


def find_disagreements(
    style_keys: list[str],
    embeddings: np.ndarray,
    index_group_names: list[str],
    n_examples: int = N_DISAGREEMENT_EXAMPLES,
) -> pl.DataFrame:
    """Find the starkest disagreements between text-embedding similarity and `index_group_name`.

    Two disagreement shapes, both surfaced (judgment call: mix both rather than pick one, since
    each tells a different story -- see module docstring):
    - Type A ("text says similar, categories say different"): pairs of style_keys with very high
      cosine similarity in embedding space but different `index_group_name`.
    - Type B ("categories say similar, text says different"): pairs of style_keys sharing the same
      `index_group_name` but with very low cosine similarity in embedding space.

    Args:
        style_keys: style_key string per row of `embeddings`, in matching order.
        embeddings: style_key-level pooled embeddings, shape `(n_styles, dim)`.
        index_group_names: `index_group_name` per row of `embeddings`, in matching order.
        n_examples: Total number of example pairs to return.

    Returns:
        DataFrame with one row per example pair: `disagreement_type`, `style_key_a`,
        `index_group_a`, `style_key_b`, `index_group_b`, `cosine_similarity`.
    """
    norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    sim = norm @ norm.T
    n = len(style_keys)
    groups = np.asarray(index_group_names)

    iu = np.triu_indices(n, k=1)
    pair_sim = sim[iu]
    pair_i, pair_j = iu
    same_group = groups[pair_i] == groups[pair_j]

    n_type_a = (n_examples + 1) // 2
    n_type_b = n_examples - n_type_a

    diff_group_idx = np.where(~same_group)[0]
    type_a_order = diff_group_idx[np.argsort(-pair_sim[diff_group_idx])][:n_type_a]

    same_group_idx = np.where(same_group)[0]
    type_b_order = same_group_idx[np.argsort(pair_sim[same_group_idx])][:n_type_b]

    rows: list[dict[str, Any]] = []
    for label, order in (
        ("text_similar_categorically_different", type_a_order),
        ("categorically_same_text_dissimilar", type_b_order),
    ):
        for pos in order:
            i, j = int(pair_i[pos]), int(pair_j[pos])
            rows.append(
                {
                    "disagreement_type": label,
                    "style_key_a": style_keys[i],
                    "index_group_a": index_group_names[i],
                    "style_key_b": style_keys[j],
                    "index_group_b": index_group_names[j],
                    "cosine_similarity": float(pair_sim[pos]),
                }
            )
    return pl.DataFrame(rows)


def _sample_detail_descs(articles: pl.DataFrame, style_key: str, n: int = 2) -> list[str]:
    """Return up to `n` example `detail_desc` strings for the given style_key."""
    matches = articles.filter(pl.col("style_key") == style_key)["detail_desc"]
    return [d for d in matches.head(n).to_list() if d is not None]


def main() -> None:
    """CLI entry point: embed, cluster, compare against index_group_name, write result tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--articles-path", type=Path, default=DEFAULT_ARTICLES_PATH)
    parser.add_argument("--panel-path", type=Path, default=DEFAULT_PANEL_PATH)
    parser.add_argument("--ari-nmi-out", type=Path, default=DEFAULT_ARI_NMI_OUT)
    parser.add_argument("--disagreements-out", type=Path, default=DEFAULT_DISAGREEMENTS_OUT)
    args = parser.parse_args()

    print("Loading kept style_keys from the panel...")
    kept_style_keys = load_kept_style_keys(args.panel_path)
    print(f"{kept_style_keys.height} kept style_keys")

    print("Loading articles restricted to kept style_keys...")
    articles = load_validation_articles(kept_style_keys, args.articles_path)
    n_total = articles.height
    articles_with_desc = articles.filter(pl.col("detail_desc").is_not_null())
    n_null = n_total - articles_with_desc.height
    print(
        f"{n_total} articles in kept style_keys; {n_null} ({100 * n_null / n_total:.2f}%) have "
        "a null detail_desc and are dropped before embedding"
    )

    print(f"Embedding {articles_with_desc.height} article descriptions with {MODEL_NAME} ...")
    article_embeddings = embed_articles(articles_with_desc["detail_desc"].to_list())

    print("Mean-pooling article embeddings per style_key...")
    style_embeddings, pooled_style_keys = mean_pool_by_group(
        article_embeddings, articles_with_desc["style_key"].to_list()
    )
    n_dropped_styles = kept_style_keys.height - len(pooled_style_keys)
    if n_dropped_styles:
        print(
            f"WARNING: {n_dropped_styles} kept style_keys had zero articles with a non-null "
            "detail_desc and were excluded from clustering"
        )

    print(f"Sweeping k in {K_CANDIDATES} by mean silhouette score...")
    best_k, silhouette_by_k, text_labels = sweep_k(style_embeddings)
    for k, score in silhouette_by_k.items():
        marker = " <- best" if k == best_k else ""
        print(f"  k={k}: silhouette={score:.4f}{marker}")

    style_meta = kept_style_keys.filter(pl.col("style_key").is_in(pooled_style_keys)).sort(
        "style_key"
    )
    assert style_meta["style_key"].to_list() == sorted(pooled_style_keys)

    index_group_names = style_meta["index_group_name"].to_list()
    ari_index_group, nmi_index_group = compute_agreement(text_labels, index_group_names)

    full_style_key_labels = style_meta["style_key"].to_list()
    ari_full_key, nmi_full_key = compute_agreement(text_labels, full_style_key_labels)

    print(f"ARI vs index_group_name: {ari_index_group:.4f}, NMI: {nmi_index_group:.4f}")
    print(f"ARI vs full style_key:   {ari_full_key:.4f}, NMI: {nmi_full_key:.4f}")

    results = pl.DataFrame(
        [
            {
                "comparison_partition": "index_group_name",
                "best_k": best_k,
                "silhouette_score": silhouette_by_k[best_k],
                "ari": ari_index_group,
                "nmi": nmi_index_group,
                "n_style_keys": len(pooled_style_keys),
            },
            {
                "comparison_partition": "full_style_key",
                "best_k": best_k,
                "silhouette_score": silhouette_by_k[best_k],
                "ari": ari_full_key,
                "nmi": nmi_full_key,
                "n_style_keys": len(pooled_style_keys),
            },
        ]
    )
    args.ari_nmi_out.parent.mkdir(parents=True, exist_ok=True)
    results.write_csv(args.ari_nmi_out)
    print(f"Wrote ARI/NMI results to {args.ari_nmi_out}")

    print("Finding starkest disagreement examples...")
    # pooled_style_keys is already sorted (mean_pool_by_group's contract), matching style_meta's
    # sort order above, so style_embeddings/index_group_names are already aligned to it.
    disagreements = find_disagreements(pooled_style_keys, style_embeddings, index_group_names)

    disagreements = disagreements.with_columns(
        pl.col("style_key_a")
        .map_elements(
            lambda sk: STYLE_KEY_SEPARATOR.join(_sample_detail_descs(articles_with_desc, sk)),
            return_dtype=pl.String,
        )
        .alias("sample_desc_a"),
        pl.col("style_key_b")
        .map_elements(
            lambda sk: STYLE_KEY_SEPARATOR.join(_sample_detail_descs(articles_with_desc, sk)),
            return_dtype=pl.String,
        )
        .alias("sample_desc_b"),
    )
    args.disagreements_out.parent.mkdir(parents=True, exist_ok=True)
    disagreements.write_csv(args.disagreements_out)
    print(f"Wrote disagreement examples to {args.disagreements_out}")


if __name__ == "__main__":
    main()
