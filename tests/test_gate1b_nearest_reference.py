from __future__ import annotations

import numpy as np
import pytest

from nss.generate.gate1b_nearest_reference import (
    gate1b_pass,
    gate1b_threshold,
    nearest_sibling_similarities,
)

# Cosines: 01=.8, 02=.6, 03=0, 12=.96, 13=.6, 23=.8 -> nearest siblings .8, .96, .96, .8.
REFS = [np.array(v, dtype=float) for v in ((1.0, 0.0), (0.8, 0.6), (0.6, 0.8), (0.0, 1.0))]
EMBS = {"clip": REFS, "dinov2": REFS}


def test_nearest_sibling_similarities_hand_values() -> None:
    assert nearest_sibling_similarities(REFS) == pytest.approx([0.8, 0.96, 0.96, 0.8])


def test_threshold_is_p90_of_real_nearest_sibling_values() -> None:
    # sorted [.8,.8,.96,.96], p90 at index 2.7 -> both neighbours are .96 (mutual-NN duplication)
    assert gate1b_threshold(REFS) == pytest.approx(0.96)


def test_exact_clone_of_a_reference_fails() -> None:
    """The control that decides whether Gate 1b may gate: a copy has max cosine 1.0 > threshold."""
    res = gate1b_pass({"clip": REFS[0], "dinov2": REFS[0]}, EMBS)
    assert res["joint_pass"] is False
    assert res["clip_max_sim"] == pytest.approx(1.0)


def test_concept_no_closer_than_real_siblings_passes() -> None:
    concept = np.array([np.cos(np.radians(20)), np.sin(np.radians(20))])  # max cosine 0.957
    concept = concept / np.linalg.norm(concept)
    assert gate1b_pass({"clip": concept, "dinov2": concept}, EMBS)["joint_pass"] is True


def test_joint_and_one_failing_space_fails() -> None:
    near = REFS[0]
    far = np.array([np.cos(np.radians(20)), np.sin(np.radians(20))])
    res = gate1b_pass({"clip": far, "dinov2": near}, EMBS)
    assert res["clip_pass"] is True
    assert res["dinov2_pass"] is False
    assert res["joint_pass"] is False


def test_needs_two_articles() -> None:
    with pytest.raises(ValueError, match="2 distinct"):
        nearest_sibling_similarities([REFS[0]])
