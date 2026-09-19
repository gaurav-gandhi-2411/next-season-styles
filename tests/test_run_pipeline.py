"""Light tests for scripts/run_pipeline.py's pure wiring/scoping logic (task D3).

Only the fast, deterministic, no-I/O functions are tested here (`seeds_for_style`, `_stage_index`,
CLI defaults) -- the real end-to-end run itself is "the test" for the stage functions (see task D3
report), matching `tests/test_agent_demo.py`'s identical rationale for not mocking out a script
whose entire point is demonstrating real, live behaviour.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from run_pipeline import (  # noqa: E402 -- see sys.path.insert above
    DEFAULT_TABLES_DIR,
    STAGE_ORDER,
    _stage_index,
    build_arg_parser,
    seeds_for_style,
)

from nss.generate import final_concepts  # noqa: E402 -- see sys.path.insert above


def test_stage_order_is_the_documented_seven_stages() -> None:
    """`STAGE_ORDER` matches the task D3 chain: panel -> ... -> hero."""
    assert STAGE_ORDER == ("panel", "features", "forecast", "briefs", "generate", "score", "hero")


def test_stage_index_is_monotonic_with_stage_order() -> None:
    """`_stage_index` returns each stage's position, so `--stop-after` comparisons are correct."""
    indices = [_stage_index(stage) for stage in STAGE_ORDER]
    assert indices == list(range(len(STAGE_ORDER)))


def test_seeds_for_style_returns_first_n_seeds_for_a_normal_style() -> None:
    """A non-underwear style gets the first `n_seeds` of `final_concepts.SEEDS`, unmodified."""
    assert seeds_for_style("Ladieswear || T-shirt || Jersey Basic || Black || Solid", 1) == (42,)
    assert seeds_for_style("Ladieswear || T-shirt || Jersey Basic || Black || Solid", 2) == (42, 43)


def test_seeds_for_style_skips_underwear_disqualified_seeds() -> None:
    """The underwear style skips C6's manually-confirmed visual-QC-disqualified seeds (42/43/44),
    so a 1-seed run lands on seed 45 instead of spuriously hitting `select_best_candidate`'s
    "every candidate disqualified" error."""
    underwear = final_concepts.UNDERWEAR_STYLE_KEY
    assert seeds_for_style(underwear, 1) == (45,)


def test_seeds_for_style_raises_when_n_seeds_exceeds_available_for_underwear() -> None:
    """Only 1 non-disqualified seed (45) exists for the underwear style -- requesting 2 raises a
    clear, actionable error rather than silently returning a short tuple."""
    underwear = final_concepts.UNDERWEAR_STYLE_KEY
    with pytest.raises(ValueError, match="exceeds the 1 non-disqualified seeds available"):
        seeds_for_style(underwear, 2)


def test_arg_parser_defaults_write_deterministic_stages_to_the_real_tables_dir() -> None:
    """Default `--tables-out-dir` is the REAL `reports/tables/` location (safe -- see module
    docstring STAGE-OUTPUT ISOLATION: panel/features/forecast/briefs are deterministic)."""
    args = build_arg_parser().parse_args([])
    assert args.tables_out_dir == DEFAULT_TABLES_DIR == Path("reports/tables")
    assert args.stop_after == "hero"
    assert args.n_seeds == 1


def test_arg_parser_rejects_n_seeds_outside_one_or_two() -> None:
    """`--n-seeds` is restricted to {1, 2} -- task D3 scoping-down never re-derives all 4 of C6's
    seeds via this script."""
    with pytest.raises(SystemExit):
        build_arg_parser().parse_args(["--n-seeds", "4"])
