from __future__ import annotations

import polars as pl
import pytest

from nss.data.select_final_three_exemplars import (
    carry_forward_rows,
    find_reusable_rows,
    select_final_three_targets,
)

_T1_ROW = {
    "style_key": "Ladieswear || T-shirt || Jersey Basic || Black || Solid",
    "index_group_name": "Ladieswear",
    "product_type_name": "T-shirt",
    "garment_group_name": "Jersey Basic",
    "perceived_colour_master_name": "Black",
    "graphical_appearance_name": "Solid",
    "predicted_intensity": 33.7,
    "growth_ratio": None,
    "source_table": "T1_incumbent",
}
_T2_HIGH_GROWTH_ROW = {
    "style_key": "Ladieswear || Underwear bottom || Under-, Nightwear || Red || Solid",
    "index_group_name": "Ladieswear",
    "product_type_name": "Underwear bottom",
    "garment_group_name": "Under-, Nightwear",
    "perceived_colour_master_name": "Red",
    "graphical_appearance_name": "Solid",
    "predicted_intensity": 16.2,
    "growth_ratio": 3.67,
    "source_table": "T2_emerging",
}
_T2_LOW_GROWTH_ROW = {
    "style_key": "Ladieswear || Sweater || Knitwear || Beige || Melange",
    "index_group_name": "Ladieswear",
    "product_type_name": "Sweater",
    "garment_group_name": "Knitwear",
    "perceived_colour_master_name": "Beige",
    "graphical_appearance_name": "Melange",
    "predicted_intensity": 19.0,
    "growth_ratio": 1.69,
    "source_table": "T2_emerging",
}


def _final_three_fixture(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(rows)


def test_select_final_three_targets_orders_t1_then_t2_by_growth_ratio_desc() -> None:
    """final_rank_1 is the T1 row; final_rank_2/3 are T2 rows in descending growth_ratio order,
    regardless of the input table's row order."""
    # Deliberately out of final order, to prove sorting (not table order) drives the result.
    final_three = _final_three_fixture([_T2_LOW_GROWTH_ROW, _T1_ROW, _T2_HIGH_GROWTH_ROW])

    targets = select_final_three_targets(final_three)

    assert [role for role, _, _ in targets] == ["final_rank_1", "final_rank_2", "final_rank_3"]
    assert targets[0][1] == _T1_ROW["style_key"]
    assert targets[1][1] == _T2_HIGH_GROWTH_ROW["style_key"]
    assert targets[2][1] == _T2_LOW_GROWTH_ROW["style_key"]
    assert targets[0][2] == {
        "index_group_name": "Ladieswear",
        "product_type_name": "T-shirt",
        "garment_group_name": "Jersey Basic",
        "perceived_colour_master_name": "Black",
        "graphical_appearance_name": "Solid",
    }


def test_select_final_three_targets_rejects_wrong_t1_count() -> None:
    final_three = _final_three_fixture([_T2_HIGH_GROWTH_ROW, _T2_LOW_GROWTH_ROW])
    with pytest.raises(ValueError, match="T1_incumbent"):
        select_final_three_targets(final_three)


def test_select_final_three_targets_rejects_wrong_t2_count() -> None:
    final_three = _final_three_fixture([_T1_ROW, _T2_HIGH_GROWTH_ROW])
    with pytest.raises(ValueError, match="T2_emerging"):
        select_final_three_targets(final_three)


def _existing_manifest_fixture() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "role": "winner_rank_1",
                "style_key": "Ladieswear || T-shirt || Jersey Basic || Black || Solid",
                "article_id": 1,
                "units_sold_last_26w": 100,
                "local_image_path": "data/images/0000000001.jpg",
                "fetch_success": True,
            },
            {
                "role": "winner_rank_1",
                "style_key": "Ladieswear || T-shirt || Jersey Basic || Black || Solid",
                "article_id": 2,
                "units_sold_last_26w": 50,
                "local_image_path": "data/images/0000000002.jpg",
                "fetch_success": False,
            },
            {
                "role": "winner_rank_2",
                "style_key": "Ladieswear || Cardigan || Knitwear || Black || Solid",
                "article_id": 3,
                "units_sold_last_26w": 30,
                "local_image_path": "data/images/0000000003.jpg",
                "fetch_success": True,
            },
            {
                "role": "control",
                "style_key": "Menswear || Scarf || Accessories || Grey || Melange",
                "article_id": 4,
                "units_sold_last_26w": 5,
                "local_image_path": "data/images/0000000004.jpg",
                "fetch_success": True,
            },
        ]
    )


def test_find_reusable_rows_matches_identical_winner_style_key() -> None:
    existing = _existing_manifest_fixture()

    reused = find_reusable_rows(existing, "Ladieswear || T-shirt || Jersey Basic || Black || Solid")

    assert reused is not None
    assert reused.height == 2
    assert reused["article_id"].to_list() == [1, 2]


def test_find_reusable_rows_returns_none_for_unseen_style_key() -> None:
    existing = _existing_manifest_fixture()

    reused = find_reusable_rows(
        existing, "Ladieswear || Underwear bottom || Under-, Nightwear || Red || Solid"
    )

    assert reused is None


def test_find_reusable_rows_ignores_control_role() -> None:
    """A style_key that happens to only appear under `control` must not be treated as reusable
    (only Phase 2 `winner_rank_*` rows are equivalent exemplar selections)."""
    existing = _existing_manifest_fixture()

    reused = find_reusable_rows(existing, "Menswear || Scarf || Accessories || Grey || Melange")

    assert reused is None


def test_carry_forward_rows_relabels_role_and_preserves_other_fields() -> None:
    existing = _existing_manifest_fixture()
    reused = find_reusable_rows(existing, "Ladieswear || T-shirt || Jersey Basic || Black || Solid")
    assert reused is not None

    rows = carry_forward_rows(reused, "final_rank_1")

    assert rows == [
        {
            "role": "final_rank_1",
            "style_key": "Ladieswear || T-shirt || Jersey Basic || Black || Solid",
            "article_id": 1,
            "units_sold_last_26w": 100,
            "local_image_path": "data/images/0000000001.jpg",
            "fetch_success": True,
        },
        {
            "role": "final_rank_1",
            "style_key": "Ladieswear || T-shirt || Jersey Basic || Black || Solid",
            "article_id": 2,
            "units_sold_last_26w": 50,
            "local_image_path": "data/images/0000000002.jpg",
            "fetch_success": False,
        },
    ]
