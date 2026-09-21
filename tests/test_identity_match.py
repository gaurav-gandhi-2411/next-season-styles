from __future__ import annotations

import pytest

from nss.generate import identity_match as im


@pytest.mark.parametrize(
    ("reading", "truth", "ok"),
    [
        ("Dress.", "Dress", True),
        ("Turtleneck.", "Sweater", True),  # neck variant of a sweater
        ("Sweater", "Sweater", True),
        ("Bikini.", "Bikini top", True),  # the head-noun synonym
        ("Blouse", "Top", True),
        ("Sweater.", "Top", False),  # a different garment
        ("Cardigan", "Sweater", False),
        ("Sweater dress", "Dress", False),  # names two families: conflict
        ("Bikini top", "Top", False),  # a longer phrase from another family wins
        ("", "Dress", False),
        ("Dress.", "Skirt", False),  # unknown truth: token containment only
        ("Skirt.", "Skirt", True),
    ],
)
def test_product_type(reading: str, truth: str, ok: bool) -> None:
    assert im.product_type_ok(reading, truth) is ok


@pytest.mark.parametrize(
    ("reading", "truth", "ok"),
    [
        ("Red.", "Red", True),
        ("Crimson", "Red", True),
        ("Orange.", "Red", False),  # the coral dress reading
        ("Green.", "Red", False),  # the emerald dress reading
        ("Coral.", "Red", False),  # names no known colour
        ("Pink", "Red", False),
        ("Red and white", "Red", False),  # conflict
        ("Off-white", "White", True),
        ("White.", "Beige", False),
        ("Beige.", "Beige", True),
    ],
)
def test_colour(reading: str, truth: str, ok: bool) -> None:
    assert im.colour_ok(reading, truth) is ok


def test_no_word_belongs_to_two_families_of_one_vocabulary() -> None:
    for vocab in (im.PRODUCT_SYNONYMS, im.COLOUR_SYNONYMS):
        seen: dict[str, str] = {}
        for family, words in vocab.items():
            for w in words:
                # "top"/"tank top" and "t-shirt"/"tee" are shared on purpose between Top and
                # T-shirt only; everything else must be unique so a family match is unambiguous
                if w in seen and {seen[w], family} != {"top", "t-shirt"}:
                    pytest.fail(f"{w!r} is in both {seen[w]!r} and {family!r}")
                seen[w] = family


def test_a_word_shared_by_two_families_is_not_a_conflict() -> None:
    assert im.product_type_ok("T-shirt", "Top") is True
    assert im.product_type_ok("Tee", "T-shirt") is True
    assert im.product_type_ok("Top", "T-shirt") is False  # "top" is not accepted for a T-shirt
