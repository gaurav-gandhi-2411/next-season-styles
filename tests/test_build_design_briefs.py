"""Tests for `nss.generate.build_design_briefs` -- the H&M-specific adapter that maps this
project's persisted CSVs onto the `style-brief` skill's generic schema.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from nss.generate.build_design_briefs import (
    build_all_design_briefs,
    build_style_profile,
    infer_sensitivity_tag,
    load_reference_image_paths,
)

_TOP_STYLES_ROW = {
    "style_key": "Ladieswear || T-shirt || Jersey Basic || Black || Solid",
    "product_type_name": "T-shirt",
    "garment_group_name": "Jersey Basic",
    "perceived_colour_master_name": "Black",
    "graphical_appearance_name": "Solid",
    "predicted_intensity": 33.69,
    "growth_ratio": None,
}
_UNDERWEAR_TOP_STYLES_ROW = {
    "style_key": "Ladieswear || Underwear bottom || Under-, Nightwear || Red || Solid",
    "product_type_name": "Underwear bottom",
    "garment_group_name": "Under-, Nightwear",
    "perceived_colour_master_name": "Red",
    "graphical_appearance_name": "Solid",
    "predicted_intensity": 16.25,
    "growth_ratio": 3.67,
}
_SHAP_VERDICT_ROW = {
    "style_key": "Ladieswear || T-shirt || Jersey Basic || Black || Solid",
    "dominant_mechanism": (
        "persistence (lag_1) dominates: SHAP=0.75 (also the single largest driver overall)"
    ),
    "shap_driver_1_feature": "lag_1",
    "shap_driver_1_value": 0.7451,
    "shap_driver_2_feature": "n_active_articles_level",
    "shap_driver_2_value": 0.2130,
    "shap_driver_3_feature": "perceived_colour_master_name",
    "shap_driver_3_value": 0.1196,
    "shap_driver_4_feature": "fourier_sin_1",
    "shap_driver_4_value": -0.1043,
    "shap_driver_5_feature": "garment_group_name",
    "shap_driver_5_value": 0.1011,
}


def test_infer_sensitivity_tag_flags_underwear_category() -> None:
    assert infer_sensitivity_tag("Underwear bottom", "Under-, Nightwear") == "intimate_apparel"


def test_infer_sensitivity_tag_flags_nightwear_construction_group() -> None:
    assert infer_sensitivity_tag("Pyjama set", "Nightwear") == "intimate_apparel"


def test_infer_sensitivity_tag_none_for_ordinary_category() -> None:
    assert infer_sensitivity_tag("T-shirt", "Jersey Basic") is None


def test_build_style_profile_maps_hm_columns_onto_generic_schema() -> None:
    profile = build_style_profile(
        _TOP_STYLES_ROW, _SHAP_VERDICT_ROW, ["data/images/0800691008.jpg"]
    )

    assert profile["style_id"] == _TOP_STYLES_ROW["style_key"]
    assert profile["attributes"] == {
        "garment_category": "T-shirt",
        "construction_group": "Jersey Basic",
        "colour_name": "Black",
        "pattern_or_finish": "Solid",
    }
    assert profile["performance_signal"] == {"predicted_intensity": 33.69, "growth_ratio": None}
    assert profile["dominant_mechanism"] == _SHAP_VERDICT_ROW["dominant_mechanism"]
    assert profile["top_drivers"][0] == {"feature": "lag_1", "importance": 0.7451}
    assert len(profile["top_drivers"]) == 5
    assert profile["reference_image_paths"] == ["data/images/0800691008.jpg"]
    assert profile["sensitivity_tag"] is None


def test_build_style_profile_underwear_row_gets_sensitivity_tag() -> None:
    shap_row = {**_SHAP_VERDICT_ROW, "style_key": _UNDERWEAR_TOP_STYLES_ROW["style_key"]}

    profile = build_style_profile(_UNDERWEAR_TOP_STYLES_ROW, shap_row, [])

    assert profile["sensitivity_tag"] == "intimate_apparel"
    assert profile["performance_signal"]["growth_ratio"] == 3.67


def test_load_reference_image_paths_excludes_fetch_failures_and_non_final_rank_roles(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "exemplar_images_final_three.csv"
    pl.DataFrame(
        [
            {
                "role": "final_rank_1",
                "style_key": "STYLE_A",
                "article_id": 1,
                "local_image_path": "a1.jpg",
                "fetch_success": True,
            },
            {
                "role": "final_rank_1",
                "style_key": "STYLE_A",
                "article_id": 2,
                "local_image_path": "a2.jpg",
                "fetch_success": False,
            },
            {
                "role": "control",
                "style_key": "STYLE_CONTROL",
                "article_id": 3,
                "local_image_path": "c1.jpg",
                "fetch_success": True,
            },
        ]
    ).write_csv(manifest_path)

    paths = load_reference_image_paths(manifest_path)

    assert paths == {"STYLE_A": ["a1.jpg"]}


def test_build_all_design_briefs_end_to_end_with_tiny_fixtures(tmp_path: Path) -> None:
    """Full read -> adapt -> skill -> write pipeline against small on-disk fixtures."""
    top_styles_path = tmp_path / "top_styles_final_three.csv"
    shap_verdict_path = tmp_path / "final_three_shap_verdict.csv"
    exemplar_images_path = tmp_path / "exemplar_images_final_three.csv"

    pl.DataFrame([_TOP_STYLES_ROW, _UNDERWEAR_TOP_STYLES_ROW]).write_csv(top_styles_path)
    underwear_shap_row = {
        **_SHAP_VERDICT_ROW,
        "style_key": _UNDERWEAR_TOP_STYLES_ROW["style_key"],
    }
    pl.DataFrame([_SHAP_VERDICT_ROW, underwear_shap_row]).write_csv(shap_verdict_path)
    pl.DataFrame(
        [
            {
                "role": "final_rank_1",
                "style_key": _TOP_STYLES_ROW["style_key"],
                "article_id": 1,
                "local_image_path": "a1.jpg",
                "fetch_success": True,
            },
            {
                "role": "final_rank_2",
                "style_key": _UNDERWEAR_TOP_STYLES_ROW["style_key"],
                "article_id": 2,
                "local_image_path": "b1.jpg",
                "fetch_success": True,
            },
        ]
    ).write_csv(exemplar_images_path)

    briefs = build_all_design_briefs(top_styles_path, shap_verdict_path, exemplar_images_path)

    assert len(briefs) == 2
    assert briefs[0]["style_id"] == _TOP_STYLES_ROW["style_key"]
    assert briefs[1]["style_id"] == _UNDERWEAR_TOP_STYLES_ROW["style_key"]
    assert "winning combination as a whole" in " ".join(briefs[1]["preserve"])


def test_build_all_design_briefs_raises_on_missing_shap_verdict_row(tmp_path: Path) -> None:
    top_styles_path = tmp_path / "top_styles_final_three.csv"
    shap_verdict_path = tmp_path / "final_three_shap_verdict.csv"
    exemplar_images_path = tmp_path / "exemplar_images_final_three.csv"

    pl.DataFrame([_TOP_STYLES_ROW]).write_csv(top_styles_path)
    pl.DataFrame([{**_SHAP_VERDICT_ROW, "style_key": "some-other-style-key"}]).write_csv(
        shap_verdict_path
    )
    pl.DataFrame(
        [
            {
                "role": "final_rank_1",
                "style_key": _TOP_STYLES_ROW["style_key"],
                "article_id": 1,
                "local_image_path": "a1.jpg",
                "fetch_success": True,
            }
        ]
    ).write_csv(exemplar_images_path)

    with pytest.raises(ValueError, match="no SHAP verdict found"):
        build_all_design_briefs(top_styles_path, shap_verdict_path, exemplar_images_path)
