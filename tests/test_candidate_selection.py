from __future__ import annotations

import math
from pathlib import Path

import polars as pl
import pytest

from nss.generate.candidate_selection import (
    GATE_COLUMNS,
    intervals_disjoint,
    n_changes_visible,
    pass_rate_table,
    passes_all_gates,
    select_candidate,
    wilson_interval,
)

SCORED = Path("reports/tables/candidates_scored.csv")


def _row(seed: int, *, answers: str = "YY", smol: float = 0.5, flor: float = 0.3, **gates: bool):
    row = dict.fromkeys(GATE_COLUMNS, True) | gates
    return {
        **row,
        "seed": seed,
        "smolvlm_gate3_answers": answers,
        "smolvlm_fidelity": smol,
        "florence2_fidelity": flor,
    }


def test_a_missing_or_null_gate_is_a_fail() -> None:
    assert passes_all_gates(_row(1))
    assert not passes_all_gates(_row(1, gate3_pass=False))
    assert not passes_all_gates({**_row(1), "gate3_pass": None})
    assert not passes_all_gates({k: v for k, v in _row(1).items() if k != "gate1b_pass"})


def test_changes_visible_counts_the_Ys() -> None:
    assert [n_changes_visible(a) for a in ("YY", "YN", "NN", "", None)] == [2, 1, 0, 0, 0]


def test_rank_order_is_changes_then_smolvlm_then_florence_then_lowest_seed() -> None:
    pool = [
        _row(1, answers="YY", smol=0.6, flor=0.9),
        _row(2, answers="YY", smol=0.85, flor=0.1),  # best SmolVLM among two-change rows
        _row(3, answers="YN", smol=0.99, flor=0.9),  # fewer changes visible outranks fidelity
        _row(4, answers="YY", smol=0.85, flor=0.1),  # exact tie with seed 2 -> lowest seed wins
    ]
    assert select_candidate(pool)["seed"] == 2
    pool[1] = _row(2, answers="YY", smol=0.85, flor=0.05)
    assert select_candidate(pool)["seed"] == 4  # Florence breaks the tie before the seed does


def test_a_failing_candidate_is_never_promoted() -> None:
    pool = [_row(1, gate3_pass=False), _row(2, integrity_floor_pass=False)]
    assert select_candidate(pool) is None


def test_wilson_interval_known_values_and_edges() -> None:
    lo, hi = wilson_interval(0, 8)
    assert lo == 0.0 and hi == pytest.approx(0.3244, abs=1e-3)
    lo, hi = wilson_interval(4, 8)
    assert (lo, hi) == (pytest.approx(0.2152, abs=1e-3), pytest.approx(0.7848, abs=1e-3))
    assert math.isnan(wilson_interval(0, 0)[0])


def test_disjointness() -> None:
    assert intervals_disjoint((0.0, 0.2), (0.3, 0.5))
    assert not intervals_disjoint((0.0, 0.3), (0.3, 0.5))


def test_pass_rate_table_splits_seed_groups() -> None:
    scored = pl.DataFrame(
        [{**_row(s, gate3_pass=(s % 2 == 0)), "style_id": "A"} for s in range(42, 50)]
    )
    out = pass_rate_table(scored, {"first": range(42, 46), "second": range(46, 50)})
    assert out["n"].to_list() == [4, 4] and out["n_pass"].to_list() == [2, 2]


@pytest.mark.skipif(not SCORED.exists(), reason="needs the committed candidates_scored.csv")
def test_rule_reproduces_the_prior_pass_counts_and_picks_seed_47_for_the_white_top() -> None:
    """On the committed 8-seed pool at scale 0.35 the rule finds the prior 2/8 for the white top
    (seeds 47 and 48) and ranks seed 47 first (SmolVLM fidelity 0.85 vs 0.567; both show YY)."""
    scored = pl.read_csv(SCORED).filter(
        (pl.col("scale") == 0.35) & pl.col("style_id").str.contains("White")
    )
    passing = sorted(r["seed"] for r in scored.to_dicts() if passes_all_gates(r))
    assert passing == [47, 48]
    assert select_candidate(scored.to_dicts())["seed"] == 47
