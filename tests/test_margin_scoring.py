from __future__ import annotations

import numpy as np
import pytest

from nss.generate.margin_scoring import margin


def test_margin_hand_computed_positive() -> None:
    """query=[1,0]; style refs=[1,0],[1,0] (cos=1 each, mean=1); control=[0,1] (cos=0, mean=0).

    margin = 1.0 - 0.0 = 1.0.
    """
    query = np.array([1.0, 0.0])
    style_refs = [np.array([1.0, 0.0]), np.array([1.0, 0.0])]
    control_refs = [np.array([0.0, 1.0])]
    assert margin(query, style_refs, control_refs) == pytest.approx(1.0)


def test_margin_hand_computed_zero_when_equally_similar() -> None:
    """Style refs and control refs are identical vectors -> margin is exactly 0."""
    query = np.array([1.0, 0.0])
    same_refs_a = [np.array([0.6, 0.8])]
    same_refs_b = [np.array([0.6, 0.8])]
    assert margin(query, same_refs_a, same_refs_b) == pytest.approx(0.0)


def test_margin_hand_computed_negative() -> None:
    """query more similar to control than to its claimed style -> negative margin.

    query=[1,0]; style refs=[0,1] (cos=0); control=[1,0] (cos=1). margin = 0 - 1 = -1.0.
    """
    query = np.array([1.0, 0.0])
    style_refs = [np.array([0.0, 1.0])]
    control_refs = [np.array([1.0, 0.0])]
    assert margin(query, style_refs, control_refs) == pytest.approx(-1.0)


def test_margin_averages_across_multiple_references() -> None:
    """Multiple style refs / control refs are mean-reduced before subtracting.

    query=[1,0]; style refs=[1,0] (cos=1), [0,1] (cos=0) -> mean 0.5.
    control refs=[1,0] (cos=1), [1,0] (cos=1) -> mean 1.0. margin = 0.5 - 1.0 = -0.5.
    """
    query = np.array([1.0, 0.0])
    style_refs = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    control_refs = [np.array([1.0, 0.0]), np.array([1.0, 0.0])]
    assert margin(query, style_refs, control_refs) == pytest.approx(-0.5)


def test_margin_empty_style_references_raises() -> None:
    """An empty style-reference set is a caller error, not a silent 0.0/NaN."""
    with pytest.raises(ValueError, match="style_reference_embeddings"):
        margin(np.array([1.0, 0.0]), [], [np.array([0.0, 1.0])])


def test_margin_empty_control_pool_raises() -> None:
    """An empty control pool is a caller error, not a silent 0.0/NaN."""
    with pytest.raises(ValueError, match="control_pool_embeddings"):
        margin(np.array([1.0, 0.0]), [np.array([1.0, 0.0])], [])
