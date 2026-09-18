from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from nss.features.style_validation import find_disagreements, mean_pool_by_group


def test_mean_pool_by_group_averages_correctly() -> None:
    """Two groups of known synthetic vectors mean-pool to the hand-computed average."""
    embeddings = np.array(
        [
            [1.0, 0.0],
            [3.0, 0.0],  # group "a" mean: [2.0, 0.0]
            [0.0, 2.0],
            [0.0, 4.0],
            [0.0, 6.0],  # group "b" mean: [0.0, 4.0]
        ]
    )
    group_ids = ["a", "a", "b", "b", "b"]

    pooled, unique_groups = mean_pool_by_group(embeddings, group_ids)

    assert unique_groups == ["a", "b"]
    np.testing.assert_allclose(pooled[unique_groups.index("a")], [2.0, 0.0])
    np.testing.assert_allclose(pooled[unique_groups.index("b")], [0.0, 4.0])


def test_mean_pool_by_group_single_item_group_is_itself() -> None:
    """A group with exactly one member mean-pools to that member's own vector, unchanged."""
    embeddings = np.array([[5.0, -1.0], [1.0, 1.0]])
    group_ids = ["only", "other"]

    pooled, unique_groups = mean_pool_by_group(embeddings, group_ids)

    np.testing.assert_allclose(pooled[unique_groups.index("only")], [5.0, -1.0])
    np.testing.assert_allclose(pooled[unique_groups.index("other")], [1.0, 1.0])


def test_mean_pool_by_group_raises_on_length_mismatch() -> None:
    """Mismatched embeddings/group_ids lengths raise, rather than silently truncating."""
    embeddings = np.zeros((3, 2))
    group_ids = ["a", "b"]

    with pytest.raises(ValueError, match="rows but group_ids has"):
        mean_pool_by_group(embeddings, group_ids)


def test_mean_pool_by_group_raises_on_empty_input() -> None:
    """Empty group_ids raises rather than returning a silently-empty result."""
    embeddings = np.zeros((0, 2))

    with pytest.raises(ValueError, match="non-empty"):
        mean_pool_by_group(embeddings, [])


def test_find_disagreements_surfaces_both_types() -> None:
    """A hand-built 4-style scenario surfaces one type-A and one type-B disagreement pair.

    Styles "same1"/"same2" are near-identical embeddings but different index_group_name (type A:
    text says similar, categories say different). Styles "diffA"/"diffB" share an index_group_name
    but are embedded far apart (type B: categories say similar, text says different).
    """
    style_keys = ["same1", "same2", "diffA", "diffB"]
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.99, 0.01],  # near-identical to "same1"
            [0.0, 1.0],
            [-1.0, 0.0],  # far from "diffA" despite sharing an index_group
        ]
    )
    index_groups = ["Ladieswear", "Menswear", "Sport", "Sport"]

    result = find_disagreements(style_keys, embeddings, index_groups, n_examples=2)

    types = set(result["disagreement_type"].to_list())
    assert types == {"text_similar_categorically_different", "categorically_same_text_dissimilar"}

    type_a = result.filter(pl.col("disagreement_type") == "text_similar_categorically_different")
    assert {type_a["style_key_a"][0], type_a["style_key_b"][0]} == {"same1", "same2"}

    type_b = result.filter(pl.col("disagreement_type") == "categorically_same_text_dissimilar")
    assert {type_b["style_key_a"][0], type_b["style_key_b"][0]} == {"diffA", "diffB"}
