from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from nss.data.select_margin_reference_styles import (
    compute_guard_passing_styles,
    select_random_reference_styles,
)

_STYLE_KEY_VALUES = {
    "index_group_name": "Ladieswear",
    "product_type_name": "Trousers",
    "garment_group_name": "GG",
    "perceived_colour_master_name": "Black",
    "graphical_appearance_name": "Solid",
}


def _guard_fixture_panel(n_weeks: int = 30) -> pl.DataFrame:
    """2 styles: PASS clears all 3 guards, FAIL fails guard 1 only (low n_active_articles).

    `price_index_level` is a trivial per-row alias of `price_index` (see
    `nss.features.model_features._add_levels`), so no lookback is needed for guard 2; guard 1/3's
    rolling windows use `min_samples=1`, so a panel shorter than their full window sizes is fine
    (project convention -- partial windows are left partial, never imputed).
    """
    weeks = [date(2020, 1, 6) + timedelta(weeks=i) for i in range(n_weeks)]
    first_seen = weeks[0]

    def _style(style_key: str, n_active: float) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "style_key": [style_key] * n_weeks,
                **{col: [val] * n_weeks for col, val in _STYLE_KEY_VALUES.items()},
                "week_start": weeks,
                "first_week_seen": [first_seen] * n_weeks,
                "units": [5.0] * n_weeks,
                "n_active_articles": [n_active] * n_weeks,
                "price_index": [1.0] * n_weeks,
                "intensity_shrunk": [5.0] * n_weeks,
            }
        )

    return pl.concat([_style("PASS", n_active=20.0), _style("FAIL", n_active=1.0)])


def test_compute_guard_passing_styles_keeps_only_all_three_guards_passing() -> None:
    """PASS clears guard1 (n_active=20>=10), guard2 (price_index=1.0>=0.85), guard3 (30
    active weeks >= 26); FAIL's n_active=1.0 fails guard1 only -- must be excluded."""
    panel = _guard_fixture_panel()
    forecast_origin = panel["week_start"].max()

    passing = compute_guard_passing_styles(panel, forecast_origin)

    assert passing["style_key"].to_list() == ["PASS"]


def test_select_random_reference_styles_excludes_given_keys() -> None:
    """Excluded style_keys never appear in the sampled result."""
    candidates = [f"style_{i}" for i in range(30)]
    excluded = frozenset({"style_0", "style_1"})
    selected = select_random_reference_styles(candidates, excluded, n=20, seed=42)
    assert len(selected) == 20
    assert len(set(selected)) == 20  # no duplicates
    assert not set(selected) & excluded


def test_select_random_reference_styles_deterministic_for_fixed_seed() -> None:
    """Same input + same seed -> identical selection (reproducibility, project convention)."""
    candidates = [f"style_{i}" for i in range(30)]
    excluded: frozenset[str] = frozenset()
    first = select_random_reference_styles(candidates, excluded, n=10, seed=42)
    second = select_random_reference_styles(candidates, excluded, n=10, seed=42)
    assert first == second


def test_select_random_reference_styles_ignores_input_order() -> None:
    """Candidates are sorted before sampling, so input row order doesn't affect the result."""
    candidates = [f"style_{i}" for i in range(30)]
    reversed_candidates = list(reversed(candidates))
    excluded: frozenset[str] = frozenset()
    first = select_random_reference_styles(candidates, excluded, n=10, seed=42)
    second = select_random_reference_styles(reversed_candidates, excluded, n=10, seed=42)
    assert first == second


def test_select_random_reference_styles_dedupes_input() -> None:
    """Duplicate style_keys in the input candidate list count once."""
    candidates = ["style_0", "style_0", "style_1"]
    excluded: frozenset[str] = frozenset()
    selected = select_random_reference_styles(candidates, excluded, n=2, seed=42)
    assert set(selected) == {"style_0", "style_1"}


def test_select_random_reference_styles_raises_when_too_few_candidates() -> None:
    """Fewer eligible candidates than requested is a caller error, not a silent short sample."""
    candidates = ["style_0", "style_1"]
    excluded: frozenset[str] = frozenset()
    with pytest.raises(ValueError, match="only 2"):
        select_random_reference_styles(candidates, excluded, n=20, seed=42)
