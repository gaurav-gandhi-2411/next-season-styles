from __future__ import annotations

import math

import pytest

from nss.models.metrics import (
    METRIC_KEYS,
    hit_at_k_in_top_n,
    ndcg_at_k,
    precision_at_k,
    score_predictions,
    spearman_rho,
    wmape,
)


def test_precision_at_k_hand_computed() -> None:
    """4 styles. true=[10,5,8,1], pred=[9,1,7,2].

    Actual top-3 by true (desc): idx0(10), idx2(8), idx1(5) -> {0,1,2}.
    Predicted top-3 by pred (desc): idx0(9), idx2(7), idx3(2) -> {0,2,3}.
    Intersection {0,2} -> 2/3.
    Actual/predicted top-10 (k_eff clipped to n=4): every index is in both -> 4/4 = 1.0.
    """
    y_true = [10.0, 5.0, 8.0, 1.0]
    y_pred = [9.0, 1.0, 7.0, 2.0]

    assert precision_at_k(y_true, y_pred, k=3) == pytest.approx(2 / 3)
    assert precision_at_k(y_true, y_pred, k=10) == pytest.approx(1.0)


def test_precision_at_k_perfect_ranking_is_one() -> None:
    y_true = [4.0, 3.0, 2.0, 1.0]
    y_pred = [40.0, 30.0, 20.0, 10.0]

    assert precision_at_k(y_true, y_pred, k=3) == pytest.approx(1.0)


def test_precision_at_k_empty_eval_set_is_nan() -> None:
    assert math.isnan(precision_at_k([], [], k=3))


def test_ndcg_at_k_hand_computed() -> None:
    """4 styles. true=[3,2,3,0], pred=[1,4,3,2], k=2.

    Predicted order (desc): idx1(4), idx2(3), idx3(2), idx0(1) -> top-2 predicted = idx1, idx2.
    Relevance of top-2 predicted, in predicted order: [true[1], true[2]] = [2, 3].
    DCG = 2/log2(2) + 3/log2(3) = 2/1 + 3/1.5849625 = 2 + 1.8927893... = 3.8927893...

    Ideal order (true desc, ties by original index -- stable sort): idx0(3), idx2(3), idx1(2),
    idx3(0) -> top-2 ideal relevance = [3, 3].
    IDCG = 3/log2(2) + 3/log2(3) = 3 + 1.8927893... = 4.8927893...

    NDCG@2 = DCG/IDCG.
    """
    y_true = [3.0, 2.0, 3.0, 0.0]
    y_pred = [1.0, 4.0, 3.0, 2.0]

    discount_2 = 1.0 / math.log2(3)
    dcg = 2.0 / math.log2(2) + 3.0 * discount_2
    idcg = 3.0 / math.log2(2) + 3.0 * discount_2
    expected = dcg / idcg

    assert ndcg_at_k(y_true, y_pred, k=2) == pytest.approx(expected)


def test_ndcg_at_k_perfect_ranking_is_one() -> None:
    y_true = [5.0, 4.0, 3.0, 2.0, 1.0]
    y_pred = [50.0, 40.0, 30.0, 20.0, 10.0]

    assert ndcg_at_k(y_true, y_pred, k=10) == pytest.approx(1.0)


def test_ndcg_at_k_all_zero_relevance_is_zero() -> None:
    """IDCG == 0 (no style has any relevance) -> defined as 0.0, not a division error."""
    y_true = [0.0, 0.0, 0.0]
    y_pred = [1.0, 2.0, 3.0]

    assert ndcg_at_k(y_true, y_pred, k=3) == 0.0


def test_ndcg_at_k_empty_eval_set_is_nan() -> None:
    assert math.isnan(ndcg_at_k([], [], k=10))


def test_hit_at_k_in_top_n_hand_computed() -> None:
    """10 styles, true=[10,9,8,7,6,5,4,3,2,1] (idx0 highest .. idx9 lowest) -> true top-5 = idx0-4.

    pred chosen so the predicted top-3 (k=3) is {idx0, idx5, idx9}: pred[0]=100 (highest),
    pred[5]=90 (2nd), pred[9]=80 (3rd), everything else=1.
    Intersection with true top-5 {0,1,2,3,4} = {0} -> hit = 1/3.
    """
    y_true = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    y_pred = [100.0, 1.0, 1.0, 1.0, 1.0, 90.0, 1.0, 1.0, 1.0, 80.0]

    assert hit_at_k_in_top_n(y_true, y_pred, k=3, n=5) == pytest.approx(1 / 3)


def test_hit_at_k_in_top_n_all_predicted_top_k_in_true_top_n_is_one() -> None:
    y_true = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    y_pred = [100.0, 90.0, 80.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    assert hit_at_k_in_top_n(y_true, y_pred, k=3, n=5) == pytest.approx(1.0)


def test_hit_at_k_in_top_n_clips_both_k_and_n_independently_to_eval_set_size() -> None:
    """Only 4 styles -- k=3 clipped to k_eff=3 (no change), n=20 clipped to n_eff=4 (the whole eval
    set), so the true top-N is trivially everything and any predicted top-3 hits 3/3."""
    y_true = [4.0, 3.0, 2.0, 1.0]
    y_pred = [1.0, 2.0, 3.0, 4.0]

    assert hit_at_k_in_top_n(y_true, y_pred, k=3, n=20) == pytest.approx(1.0)


def test_hit_at_k_in_top_n_empty_eval_set_is_nan() -> None:
    assert math.isnan(hit_at_k_in_top_n([], [], k=3, n=20))


def test_score_predictions_includes_hit_at_3_metrics() -> None:
    y_true = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    y_pred = [100.0, 90.0, 80.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    result = score_predictions(y_true, y_pred)

    assert result["hit_at_3_in_top20"] == pytest.approx(hit_at_k_in_top_n(y_true, y_pred, 3, 20))
    assert result["hit_at_3_in_top10"] == pytest.approx(hit_at_k_in_top_n(y_true, y_pred, 3, 10))


def test_spearman_rho_hand_computed_perfect_and_inverse() -> None:
    y_true = [1.0, 2.0, 3.0, 4.0]

    assert spearman_rho(y_true, [10.0, 20.0, 30.0, 40.0]) == pytest.approx(1.0)
    assert spearman_rho(y_true, [40.0, 30.0, 20.0, 10.0]) == pytest.approx(-1.0)


def test_spearman_rho_fewer_than_two_is_nan() -> None:
    assert math.isnan(spearman_rho([1.0], [2.0]))
    assert math.isnan(spearman_rho([], []))


def test_wmape_hand_computed() -> None:
    """true=[1,2,3], pred=[1.5,2,2.5], weight=[1,1,2].

    numerator = 1*|1-1.5| + 1*|2-2| + 2*|3-2.5| = 0.5 + 0 + 1.0 = 1.5
    denominator = 1*|1| + 1*|2| + 2*|3| = 1 + 2 + 6 = 9
    wmape = 1.5 / 9 = 0.16666...
    """
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.5, 2.0, 2.5]
    weight = [1.0, 1.0, 2.0]

    assert wmape(y_true, y_pred, weight) == pytest.approx(1.5 / 9)


def test_wmape_perfect_prediction_is_zero() -> None:
    y_true = [1.0, 2.0, 3.0]
    assert wmape(y_true, y_true, [1.0, 1.0, 1.0]) == pytest.approx(0.0)


def test_wmape_zero_weighted_denominator_is_nan() -> None:
    """Every actual value is 0 (or every weight is 0) -> denom 0 -> nan, not a ZeroDivisionError."""
    assert math.isnan(wmape([0.0, 0.0], [1.0, 2.0], [1.0, 1.0]))
    assert math.isnan(wmape([1.0, 2.0], [1.0, 2.0], [0.0, 0.0]))


def test_wmape_empty_eval_set_is_nan() -> None:
    assert math.isnan(wmape([], [], []))


def test_score_predictions_returns_every_metric_key_plus_n_eval() -> None:
    y_true = [3.0, 2.0, 3.0, 0.0, 1.0]
    y_pred = [1.0, 4.0, 3.0, 2.0, 0.5]
    weight = [1.0, 1.0, 1.0, 1.0, 1.0]

    result = score_predictions(y_true, y_pred, weight)

    assert result["n_eval"] == 5.0
    for key in METRIC_KEYS:
        assert key in result


def test_score_predictions_defaults_to_uniform_weight() -> None:
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.0, 2.0, 3.0]

    with_explicit_uniform_weight = score_predictions(y_true, y_pred, [1.0, 1.0, 1.0])
    with_default_weight = score_predictions(y_true, y_pred)

    assert with_explicit_uniform_weight["wmape"] == pytest.approx(with_default_weight["wmape"])
