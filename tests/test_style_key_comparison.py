from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from sklearn.metrics import pairwise_distances, silhouette_score

from nss.features.style_key_comparison import (
    REDUCED_STYLE_KEY_COLS,
    build_style_key_column,
    compute_lifetime_by_key,
    filter_lifetime_by_support,
    load_articles_for_key,
    permutation_null_silhouette,
    subsample_rows,
)


def _two_cluster_embeddings(
    seed: int = 0, n_per_cluster: int = 15
) -> tuple[np.ndarray, np.ndarray]:
    """Two well-separated, tight-variance 2D Gaussian blobs -- a genuinely structured partition."""
    rng = np.random.default_rng(seed)
    cluster_a = rng.normal(loc=[0.0, 0.0], scale=0.2, size=(n_per_cluster, 2))
    cluster_b = rng.normal(loc=[10.0, 10.0], scale=0.2, size=(n_per_cluster, 2))
    embeddings = np.vstack([cluster_a, cluster_b])
    labels = np.array(["a"] * n_per_cluster + ["b"] * n_per_cluster)
    return embeddings, labels


def test_permutation_null_silhouette_structured_partition_beats_null() -> None:
    """A genuinely well-separated partition scores far above its own permutation null."""
    embeddings, labels = _two_cluster_embeddings()

    result = permutation_null_silhouette(
        embeddings, labels, metric="euclidean", n_permutations=10, seed_start=42
    )

    assert result["observed_silhouette"] > 0.9
    assert result["observed_silhouette"] > result["null_mean"]
    assert result["z_score"] > 3.0  # observed sits many null standard deviations above the null


def test_permutation_null_silhouette_matches_hand_replicated_computation() -> None:
    """The function's observed/null scores exactly match an independent, hand-replicated run.

    Recomputes the same permutation scheme (seeds 42..44) directly against sklearn, and confirms
    `permutation_null_silhouette`'s z-score formula matches a manual mean/std/z computation over
    those independently-derived scores -- a mechanism check, not just an output-shape check.
    """
    embeddings, labels = _two_cluster_embeddings(seed=1)
    metric = "euclidean"
    n_permutations = 3
    seed_start = 42

    distance_matrix = pairwise_distances(embeddings, metric=metric)
    expected_observed = float(silhouette_score(distance_matrix, labels, metric="precomputed"))

    expected_null_scores = []
    for seed in range(seed_start, seed_start + n_permutations):
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(labels)
        expected_null_scores.append(
            float(silhouette_score(distance_matrix, shuffled, metric="precomputed"))
        )
    expected_null_arr = np.array(expected_null_scores)
    expected_null_mean = float(expected_null_arr.mean())
    expected_null_sd = float(expected_null_arr.std(ddof=1))
    expected_z = (expected_observed - expected_null_mean) / expected_null_sd

    result = permutation_null_silhouette(
        embeddings, labels, metric=metric, n_permutations=n_permutations, seed_start=seed_start
    )

    assert result["observed_silhouette"] == pytest.approx(expected_observed)
    assert result["null_mean"] == pytest.approx(expected_null_mean)
    assert result["null_sd"] == pytest.approx(expected_null_sd)
    assert result["z_score"] == pytest.approx(expected_z)


def test_permutation_null_silhouette_no_structure_partition_has_small_z() -> None:
    """Labels assigned independently of embedding position score close to their own null."""
    rng = np.random.default_rng(7)
    embeddings = rng.normal(size=(40, 2))
    labels = rng.choice(np.array(["a", "b"]), size=40)  # label uncorrelated with position

    result = permutation_null_silhouette(
        embeddings, labels, metric="euclidean", n_permutations=10, seed_start=42
    )

    # Not a hard structural guarantee (both draws are random), but the z-score magnitude should
    # be small relative to the strongly-structured case above (z > 3.0 there).
    assert abs(result["z_score"]) < 3.0


def test_subsample_rows_returns_all_when_sample_size_exceeds_n() -> None:
    """No subsampling occurs when sample_size >= n -- all rows are returned, unchanged order."""
    embeddings = np.arange(20).reshape(10, 2).astype(float)
    labels = np.arange(10)

    sub_embeddings, sub_labels = subsample_rows(embeddings, labels, sample_size=100)

    np.testing.assert_array_equal(sub_embeddings, embeddings)
    np.testing.assert_array_equal(sub_labels, labels)


def test_subsample_rows_draws_requested_size_deterministically() -> None:
    """Subsampling returns exactly sample_size rows, deterministic given the same seed, aligned."""
    embeddings = np.arange(40).reshape(20, 2).astype(float)
    labels = np.arange(20)

    sub_1 = subsample_rows(embeddings, labels, sample_size=5, seed=42)
    sub_2 = subsample_rows(embeddings, labels, sample_size=5, seed=42)

    assert sub_1[0].shape == (5, 2)
    assert sub_1[1].shape == (5,)
    np.testing.assert_array_equal(sub_1[0], sub_2[0])
    np.testing.assert_array_equal(sub_1[1], sub_2[1])
    for row, label in zip(sub_1[0], sub_1[1], strict=True):
        assert row[0] == label * 2  # embeddings[i] == [2i, 2i+1] by construction -- stays aligned


def test_filter_lifetime_by_support_matches_manual_threshold_filter() -> None:
    """Filtering keeps only rows meeting BOTH thresholds and reports correct retention stats."""
    lifetime = pl.DataFrame(
        {
            "key": ["kept_both", "fails_articles", "fails_units", "kept_edge"],
            "lifetime_n_articles": [10, 2, 10, 5],
            "lifetime_units": [1000, 1000, 100, 500],
        }
    )

    kept, stats = filter_lifetime_by_support(lifetime, min_articles=5, min_units=500)

    assert set(kept["key"].to_list()) == {"kept_both", "kept_edge"}
    assert stats["n_styles_before"] == 4
    assert stats["n_styles_after"] == 2
    total_units = 1000 + 1000 + 100 + 500
    kept_units = 1000 + 500
    assert stats["pct_units_retained"] == pytest.approx(100.0 * kept_units / total_units)


def test_build_style_key_column_joins_with_separator() -> None:
    """style_key is the key_cols joined with ' || ', matching style_panel's convention."""
    df = pl.DataFrame({"a": ["X"], "b": ["Y"], "c": ["Z"]})

    result = build_style_key_column(df, ["a", "b", "c"])

    assert result["style_key"].to_list() == ["X || Y || Z"]


def _write_articles_and_transactions(tmp_path: Path) -> tuple[Path, Path]:
    """Tiny synthetic articles.csv + partitioned transactions Parquet, reduced-key focused.

    Under REDUCED_STYLE_KEY_COLS, articles 1 and 2 share a reduced key (same product/garment/
    colour/graphic) despite different index_group_name -- this is exactly the merge behaviour the
    reduced key is meant to test. Article 3 is a distinct reduced key on its own.
    """
    articles_path = tmp_path / "articles.csv"
    articles = pl.DataFrame(
        [
            {
                "article_id": 1,
                "detail_desc": "desc 1",
                "index_group_name": "Ladieswear",
                "product_type_name": "Trousers",
                "garment_group_name": "Trousers",
                "perceived_colour_master_name": "Black",
                "graphical_appearance_name": "Solid",
            },
            {
                "article_id": 2,
                "detail_desc": "desc 2",
                "index_group_name": "Menswear",  # differs from article 1 -- merges under reduced
                "product_type_name": "Trousers",
                "garment_group_name": "Trousers",
                "perceived_colour_master_name": "Black",
                "graphical_appearance_name": "Solid",
            },
            {
                "article_id": 3,
                "detail_desc": "desc 3",
                "index_group_name": "Menswear",
                "product_type_name": "Shirt",
                "garment_group_name": "Shirts",
                "perceived_colour_master_name": "White",
                "graphical_appearance_name": "Stripe",
            },
        ]
    )
    articles.write_csv(articles_path)

    transactions_dir = tmp_path / "transactions"
    part_dir = transactions_dir / "part"
    part_dir.mkdir(parents=True)
    txns = pl.DataFrame(
        {
            "article_id": [1, 1, 2, 2, 2, 3],
            "price": [10.0] * 6,
        }
    )
    txns.write_parquet(part_dir / "data.parquet")

    return transactions_dir, articles_path


def test_compute_lifetime_by_key_merges_under_reduced_key(tmp_path: Path) -> None:
    """Dropping index_group_name merges articles 1+2's transactions into a single reduced key."""
    transactions_dir, articles_path = _write_articles_and_transactions(tmp_path)

    lifetime = compute_lifetime_by_key(REDUCED_STYLE_KEY_COLS, transactions_dir, articles_path)

    assert lifetime.height == 2  # (Trousers/Black/Solid) merged, (Shirt/White/Stripe) separate
    merged = lifetime.filter(pl.col("product_type_name") == "Trousers").row(0, named=True)
    assert merged["lifetime_n_articles"] == 2
    assert merged["lifetime_units"] == 5  # 2 txns for article 1 + 3 for article 2


def test_load_articles_for_key_restricts_to_kept_combos(tmp_path: Path) -> None:
    """Only articles whose key_cols combination is present in kept_key_combos are returned."""
    _, articles_path = _write_articles_and_transactions(tmp_path)
    kept = pl.DataFrame(
        [
            {
                "product_type_name": "Trousers",
                "garment_group_name": "Trousers",
                "perceived_colour_master_name": "Black",
                "graphical_appearance_name": "Solid",
            }
        ]
    )

    result = load_articles_for_key(kept, REDUCED_STYLE_KEY_COLS, articles_path)

    assert set(result["article_id"].to_list()) == {1, 2}
