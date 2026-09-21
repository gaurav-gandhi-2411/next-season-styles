from __future__ import annotations

import pytest

from nss.generate import product_retrieval as pr


def test_product_type_is_the_second_field_of_the_style_key() -> None:
    assert pr.product_type_of("Ladieswear || Dress || Dresses Ladies || Red || Solid") == "Dress"
    assert pr.product_type_of("Ladieswear || Bikini top || Swimwear || Orange || All over") == (
        "Bikini top"
    )


@pytest.mark.parametrize(
    ("retrieved", "style", "ok"),
    [
        ("Dress", "Ladieswear || Dress || Dresses Ladies || Red || Solid", True),
        ("Top", "Ladieswear || Top || Jersey Basic || White || Solid", True),
        ("Sweater", "Ladieswear || Top || Jersey Basic || White || Solid", False),
        # exact strings only, no synonyms: "Swimwear bottom" is not "Underwear bottom"
        (
            "Swimwear bottom",
            "Ladieswear || Underwear bottom || Under-, Nightwear || Red || Solid",
            False,
        ),
    ],
)
def test_check_is_an_exact_string_match_on_the_retrieved_product_type(
    monkeypatch: pytest.MonkeyPatch, retrieved: str, style: str, ok: bool
) -> None:
    monkeypatch.setattr(
        pr,
        "top1",
        lambda _img: {
            "style_key": f"X || {retrieved} || G || C || P",
            "product_type": retrieved,
            "similarity": 0.9,
        },
    )
    out = pr.check("any.png", style)
    assert out["pass"] is ok and out["retrieved"] == retrieved
