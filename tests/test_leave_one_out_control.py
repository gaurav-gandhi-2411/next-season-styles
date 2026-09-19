from __future__ import annotations

import numpy as np
import pytest

from nss.generate.judge_repeat import summarise
from nss.generate.leave_one_out_control import clone_rows, loo_rows
from nss.generate.leave_one_out_control import summarise as loo_summarise
from nss.generate.within_style_benchmark import style_benchmark

# Four unit vectors, hand-computable cosines: 01=.8, 02=.6, 03=0, 12=.96, 13=.6, 23=.8.
REFS = [np.array(v, dtype=float) for v in ((1.0, 0.0), (0.8, 0.6), (0.6, 0.8), (0.0, 1.0))]
EMBS = {"clip": REFS, "dinov2": REFS}


def test_benchmark_statistics_hand_values() -> None:
    b = style_benchmark(REFS)
    # sorted pairs: 0, 0.6, 0.6, 0.8, 0.8, 0.96 -> median 0.7, p90 (linear, idx 4.5) 0.88, max 0.96
    assert b["pair_median"] == pytest.approx(0.7)
    assert b["pair_p90"] == pytest.approx(0.88)
    assert b["pair_max"] == pytest.approx(0.96)


def test_median_rule_fails_half_of_real_articles_p90_fails_none() -> None:
    """J1 in miniature: LOO means 0.467/0.787/0.787/0.467 vs median 0.7, p90 0.88."""
    median = loo_rows("s", EMBS, "median")
    p90 = loo_rows("s", EMBS, "p90")
    assert [r["joint_pass_full"] for r in median] == [True, False, False, True]
    assert all(r["joint_pass_full"] for r in p90)


def test_loo_scores_each_article_against_only_the_others() -> None:
    rows = loo_rows("s", EMBS, "median")
    assert rows[0]["clip_loo_sim"] == pytest.approx((0.8 + 0.6 + 0.0) / 3)


def test_heldout_benchmark_excludes_the_scored_article() -> None:
    rows = loo_rows("s", EMBS, "p90")
    # Others of article 0 are refs 1..3: pairs 0.96, 0.6, 0.8 -> p90 (idx 1.8) = 0.8 + 0.8*0.16
    assert rows[0]["clip_threshold_heldout"] == pytest.approx(0.928)


def test_summarise_counts_passes() -> None:
    import polars as pl

    counts = loo_summarise(pl.DataFrame(loo_rows("s", EMBS, "median"))).to_dicts()[0]
    assert counts["n_articles"] == 4
    assert counts["joint_pass_full"] == 2


def test_clone_positive_control_documents_mean_statistic_dilution() -> None:
    """An exact copy of reference 0 passes the mean-based gate (mean 0.6) but its max cosine is
    1.0, above the real nearest-neighbour benchmark (0.88): only the NN statistic flags it."""
    rec = clone_rows("s", EMBS)[0]
    assert rec["gate1_pass_p90"] is True
    assert rec["clip_mean_sim"] == pytest.approx(0.6)
    assert rec["clip_max_sim"] == pytest.approx(1.0)
    assert rec["clip_max_sim"] > rec["clip_nn_median_benchmark"]


def test_judge_repeat_summary_median_and_spread() -> None:
    s = summarise([0.64, 0.43, 0.60])
    assert s["median"] == pytest.approx(0.60)
    assert s["spread"] == pytest.approx(0.21)
    assert summarise([])["median"] is None
