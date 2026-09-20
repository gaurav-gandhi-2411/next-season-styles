from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from nss.external.term_mapping import COLOUR_WORD, PRODUCT_WORD, map_styles
from nss.external.trends_fetch import slug
from nss.external.trends_load import load_committed, load_weekly


def _styles(rows: list[tuple[str, str, str]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema=["style_key", "product_type_name", "perceived_colour_master_name"],
        orient="row",
    )


def test_a_mapped_style_gets_colour_then_product_word() -> None:
    out = map_styles(_styles([("a", "Sweater", "Beige")]))
    assert out.row(0, named=True) == {
        "style_key": "a",
        "term": "beige sweater",
        "unmapped_reason": None,
    }


def test_unmappable_attributes_are_left_unmapped_with_a_reason_not_invented() -> None:
    out = map_styles(
        _styles(
            [
                ("p", "Garment Set", "Black"),  # H&M catch-all product type
                ("c", "Dress", "Mole"),  # H&M-specific colour word
                ("b", "Other accessories", "Unknown"),
            ]
        )
    ).sort("style_key")
    by_key = {r["style_key"]: r for r in out.iter_rows(named=True)}
    assert by_key["p"]["term"] is None and by_key["p"]["unmapped_reason"] == "product type unmapped"
    assert by_key["c"]["term"] is None and by_key["c"]["unmapped_reason"] == "colour unmapped"
    assert by_key["b"]["unmapped_reason"] == "product type and colour both unmapped"


def test_pattern_is_not_part_of_the_term() -> None:
    """Styles differing only in pattern share a term (a documented resolution limit)."""
    styles = pl.DataFrame(
        {
            "style_key": ["solid", "stripe"],
            "product_type_name": ["Dress", "Dress"],
            "perceived_colour_master_name": ["Red", "Red"],
            "graphical_appearance_name": ["Solid", "Stripe"],
        }
    )
    terms = map_styles(styles)["term"].to_list()
    assert terms == ["red dress", "red dress"]


def test_every_mapped_word_is_a_nonempty_lowercase_search_word() -> None:
    for word in [*PRODUCT_WORD.values(), *COLOUR_WORD.values()]:
        assert word and word == word.lower()


def test_loader_maps_trends_sunday_to_the_following_monday(tmp_path: Path) -> None:
    """A Trends week starting Sunday D is joined to panel week D + 1 day (Monday)."""
    term = "red dress"
    pl.DataFrame(
        {
            "week_start_sunday": [date(2019, 1, 6), date(2019, 1, 13)],
            "value": [40, 55],
            "is_partial": [False, False],
        }
    ).write_csv(tmp_path / f"{slug(term)}.csv")
    out = load_weekly([term, "no such term"], cache_dir=tmp_path)
    assert out["week_start"].to_list() == [date(2019, 1, 7), date(2019, 1, 14)]
    assert out["term"].unique().to_list() == [term]  # the uncached term is absent, not invented
    assert all(d.weekday() == 0 for d in out["week_start"].to_list())  # Mondays


def test_committed_table_round_trips_to_the_same_frame(tmp_path: Path) -> None:
    term = "red dress"
    pl.DataFrame(
        {"week_start_sunday": [date(2019, 1, 6)], "value": [40], "is_partial": [False]}
    ).write_csv(tmp_path / f"{slug(term)}.csv")
    cached = load_weekly([term], cache_dir=tmp_path)
    path = tmp_path / "committed.csv"
    cached.write_csv(path)
    assert load_committed(path).equals(cached)
