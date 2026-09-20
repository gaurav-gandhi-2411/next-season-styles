from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from nss.generate import concept_forecast as cf


def _table() -> pl.DataFrame:
    rows = [
        ("A", "Sweater", "Knitwear", "Beige", "Melange", 17.8, 20.0),
        ("B", "Sweater", "Knitwear", "Beige", "Solid", 12.0, 30.0),
        ("C", "Sweater", "Basic", "Beige", "Melange", 9.0, 5.0),
        ("D", "Dress", "Dresses Ladies", "Red", "Solid", 11.0, 17.0),
        ("E", "T-shirt", "Jersey Basic", "Black", "Solid", 33.8, 23.0),
    ]
    return pl.DataFrame(
        {
            "style_key": [r[0] for r in rows],
            cf.TYPE: [r[1] for r in rows],
            "garment_group_name": [r[2] for r in rows],
            cf.COLOUR: [r[3] for r in rows],
            cf.PATTERN: [r[4] for r in rows],
            "predicted_intensity": [r[5] for r in rows],
            "guard1_n_active_articles_trailing_mean": [r[6] for r in rows],
        }
    )


def test_normalise_maps_free_text_to_catalogue_vocabulary() -> None:
    n = cf.normalise(
        {
            "product_type": "a V-neck jumper",
            "colour_family": "Sand",
            "graphical_treatment": "marled",
        },
        _table(),
    )
    assert n == {"product_type": "Sweater", "colour": "Beige", "pattern": "Melange"}


def test_normalise_unknown_text_is_none_not_a_guess() -> None:
    n = cf.normalise(
        {"product_type": "spaceship", "colour_family": "", "graphical_treatment": "xyzzy"},
        _table(),
    )
    assert n == {"product_type": None, "colour": None, "pattern": None}


def test_pick_prefers_dominant_variant_and_reports_backoff() -> None:
    table = _table()
    row, level = cf._pick(table, "Sweater", "Beige", "Melange")
    assert (row["style_key"], level) == ("A", "exact")  # A has more active articles than C
    row, level = cf._pick(table, "Sweater", "Beige", "Stripe")
    assert level == "pattern_relaxed" and row["style_key"] == "B"  # most active beige sweater
    row, level = cf._pick(table, "Dress", "Blue", "Solid")
    assert level == "colour_relaxed" and row["style_key"] == "D"
    row, level = cf._pick(table, "Dress", "Blue", "Stripe")
    assert level == "type_only" and row["style_key"] == "D"
    assert cf._pick(table, "Coat", "Red", "Solid") == (None, "no_match")


def test_forecast_concept_confidence_from_judge_agreement() -> None:
    table = _table()
    same = {"product_type": "sweater", "colour_family": "beige", "graphical_treatment": "melange"}
    diff = {"product_type": "sweater", "colour_family": "beige", "graphical_treatment": "solid"}
    high = cf.forecast_concept_freetext(
        Path("x"), {"a": lambda _p: same, "b": lambda _p: same}, table
    )
    assert (high.style_key, high.confidence, high.rank) == ("A", "high", 2)
    medium = cf.forecast_concept_freetext(
        Path("x"), {"a": lambda _p: same, "b": lambda _p: diff}, table
    )
    assert medium.confidence == "medium"
    single = cf.forecast_concept_freetext(Path("x"), {"a": lambda _p: same}, table)
    assert single.confidence == "low"  # one judge cannot show agreement


def test_forecast_concept_no_match_raises() -> None:
    bad = {"product_type": "spaceship", "colour_family": "red", "graphical_treatment": "solid"}
    with pytest.raises(ValueError):
        cf.forecast_concept_freetext(Path("x"), {"a": lambda _p: bad}, _table())


def test_sentence_format() -> None:
    r = cf.ConceptForecast("S", 12.34, 7, 1980, "exact", "high", {})
    assert r.sentence() == (
        "maps to S; forecast 12.3 units/product/week; rank 7 of 1,980; confidence high"
    )
