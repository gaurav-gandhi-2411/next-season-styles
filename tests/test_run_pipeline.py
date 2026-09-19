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

from nss.generate import (  # noqa: E402 -- see sys.path.insert above
    final_concepts,
    final_concepts_v2,
)


def test_stage_order_is_the_documented_seven_stages() -> None:
    """`STAGE_ORDER` matches the task D3 chain: panel -> ... -> hero."""
    assert STAGE_ORDER == ("panel", "features", "forecast", "briefs", "generate", "score", "hero")


def test_stage_index_is_monotonic_with_stage_order() -> None:
    """`_stage_index` returns each stage's position, so `--stop-after` comparisons are correct."""
    indices = [_stage_index(stage) for stage in STAGE_ORDER]
    assert indices == list(range(len(STAGE_ORDER)))


def test_seeds_for_style_returns_first_n_seeds_for_a_normal_style() -> None:
    """A fully-qualified style gets the first `n_seeds` of `final_concepts_v2.INITIAL_SEEDS`,
    unmodified."""
    assert seeds_for_style("Ladieswear || T-shirt || Jersey Basic || Black || Solid", 1) == (42,)
    assert seeds_for_style("Ladieswear || T-shirt || Jersey Basic || Black || Solid", 2) == (42, 43)


def test_seeds_for_style_ignores_disqualifications_outside_initial_seeds() -> None:
    """The T-shirt style's `VISUAL_QC_DISQUALIFIED_SEEDS` entry (46/47/48/49, E5's retry-round
    finding) has no overlap with `INITIAL_SEEDS` (42-45) -- a round-0-only run like this script's
    is never affected by it, so the first `n_seeds` of `INITIAL_SEEDS` are returned unchanged."""
    t_shirt = "Ladieswear || T-shirt || Jersey Basic || Black || Solid"
    assert final_concepts_v2.VISUAL_QC_DISQUALIFIED_SEEDS[t_shirt] == frozenset({46, 47, 48, 49})
    assert seeds_for_style(t_shirt, 2) == (42, 43)


def test_seeds_for_style_underwear_is_fully_qualified_under_v2() -> None:
    """Unlike C6/C7 (`final_concepts.UNDERWEAR_VISUAL_QC_DISQUALIFIED_SEEDS`), E5 confirmed every
    underwear candidate is free of a human model -- the style is absent from
    `final_concepts_v2.VISUAL_QC_DISQUALIFIED_SEEDS`, so seed selection is unaffected."""
    underwear = final_concepts.UNDERWEAR_STYLE_KEY
    assert underwear not in final_concepts_v2.VISUAL_QC_DISQUALIFIED_SEEDS
    assert seeds_for_style(underwear, 1) == (42,)


def test_seeds_for_style_raises_when_n_seeds_exceeds_available() -> None:
    """Only 4 seeds exist in `INITIAL_SEEDS` -- requesting 5 raises a clear, actionable error
    rather than silently returning a short tuple."""
    t_shirt = "Ladieswear || T-shirt || Jersey Basic || Black || Solid"
    with pytest.raises(ValueError, match="exceeds the 4 non-disqualified seeds available"):
        seeds_for_style(t_shirt, 5)


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
