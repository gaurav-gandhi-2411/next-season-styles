from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

from nss import mcp_server
from nss.mcp_server import (
    _extract_shap_drivers_row,
    _get_shap_drivers,
    _load_clip_band,
    _load_margin_band,
    _parse_style_key,
    compose_final_sheet,
    forecast_styles,
    generate_concept,
    get_reference_images,
    get_style_profile,
    query_transactions,
    score_concept,
)

KNOWN_STYLE_KEY = "Ladieswear || T-shirt || Jersey Basic || Black || Solid"  # T1 rank 1
UNKNOWN_STYLE_KEY = "Menswear || Bogus Type || Bogus Group || Purple || Dotted"


def test_mcp_server_registers_all_7_tools() -> None:
    """The MCPServer instance ends up with exactly the 7 tools the task brief specifies."""
    tool_names = {t.name for t in mcp_server.mcp._tool_manager.list_tools()}
    assert tool_names == {
        "query_transactions",
        "get_style_profile",
        "forecast_styles",
        "get_reference_images",
        "generate_concept",
        "score_concept",
        "compose_final_sheet",
    }


# --- _parse_style_key ---


def test_parse_style_key_splits_into_5_columns() -> None:
    """A well-formed style_key splits into the 5 STYLE_KEY_COLS, in order."""
    result = _parse_style_key(KNOWN_STYLE_KEY)
    assert result == {
        "index_group_name": "Ladieswear",
        "product_type_name": "T-shirt",
        "garment_group_name": "Jersey Basic",
        "perceived_colour_master_name": "Black",
        "graphical_appearance_name": "Solid",
    }


def test_parse_style_key_wrong_part_count_raises() -> None:
    """A style_key with the wrong number of ' || '-separated parts is a caller error."""
    with pytest.raises(ValueError, match="must have 5"):
        _parse_style_key("too || few || parts")


# --- query_transactions ---


def test_query_transactions_no_style_filter_returns_aggregates() -> None:
    """A narrow, unfiltered date range returns non-negative aggregate stats over all styles."""
    result = query_transactions(None, "2020-09-14", "2020-09-21")
    assert result["style_key"] is None
    assert result["date_from"] == "2020-09-14"
    assert result["date_to"] == "2020-09-21"
    assert result["units"] > 0
    assert result["revenue"] > 0.0
    assert result["n_customers"] > 0


def test_query_transactions_with_style_filter_matches_known_high_volume_style() -> None:
    """Filtering to a known high-volume style over its full lifetime yields a positive count."""
    result = query_transactions(KNOWN_STYLE_KEY, "2018-09-20", "2020-09-21")
    assert result["style_key"] == KNOWN_STYLE_KEY
    assert result["units"] > 0
    assert result["n_customers"] > 0


def test_query_transactions_no_matches_returns_zeroes() -> None:
    """A date range with zero matching transactions returns explicit zeroes, not nulls/errors."""
    result = query_transactions(None, "2015-01-01", "2015-01-02")
    assert result == {
        "style_key": None,
        "date_from": "2015-01-01",
        "date_to": "2015-01-02",
        "units": 0,
        "revenue": 0.0,
        "n_customers": 0,
    }


def test_query_transactions_date_from_after_date_to_raises() -> None:
    """date_from > date_to is a caller error, not a silently empty result."""
    with pytest.raises(ValueError, match="must not be after"):
        query_transactions(None, "2020-09-21", "2020-09-14")


def test_query_transactions_malformed_style_key_raises() -> None:
    """A malformed style_key surfaces the same ValueError as _parse_style_key."""
    with pytest.raises(ValueError, match="must have 5"):
        query_transactions("not-a-style-key", "2020-09-14", "2020-09-21")


# --- get_style_profile / _get_shap_drivers / _extract_shap_drivers_row ---


def test_extract_shap_drivers_row_reads_5_slots() -> None:
    """Pulls all 5 shap_driver_{i}_{feature,value} pairs out of a row dict."""
    row = {f"shap_driver_{i}_feature": f"f{i}" for i in range(1, 6)} | {
        f"shap_driver_{i}_value": float(i) for i in range(1, 6)
    }
    drivers = _extract_shap_drivers_row(row)
    assert drivers == [{"feature": f"f{i}", "value": float(i)} for i in range(1, 6)]


def test_extract_shap_drivers_row_skips_missing_slots() -> None:
    """A row missing some driver slots only returns the slots actually present."""
    row = {"shap_driver_1_feature": "lag_1", "shap_driver_1_value": 0.5}
    assert _extract_shap_drivers_row(row) == [{"feature": "lag_1", "value": 0.5}]


def test_get_shap_drivers_known_style_is_style_specific() -> None:
    """A style present in one of the per-style SHAP tables returns style_specific drivers."""
    result = _get_shap_drivers(KNOWN_STYLE_KEY)
    assert result["style_specific"] is True
    assert result["note"] is None
    assert len(result["drivers"]) > 0
    assert all({"feature", "value"} == set(d) for d in result["drivers"])


def test_get_shap_drivers_unknown_style_falls_back_to_global() -> None:
    """A style absent from every per-style table falls back to global importance, with a note."""
    result = _get_shap_drivers(UNKNOWN_STYLE_KEY)
    assert result["style_specific"] is False
    assert result["note"] is not None
    assert UNKNOWN_STYLE_KEY in result["note"]
    assert Path(result["source"]).name == "shap_global_importance.csv"
    assert len(result["drivers"]) == mcp_server.N_GLOBAL_SHAP_FALLBACK


def test_get_shap_drivers_no_artifacts_at_all(tmp_path: Path) -> None:
    """With every SHAP artifact missing, the fallback returns an explicit empty result."""
    with (
        patch.object(mcp_server, "FINAL_THREE_SHAP_VERDICT_PATH", tmp_path / "nope1.csv"),
        patch.object(mcp_server, "FORECAST_TABLES", {"incumbent": tmp_path / "nope2.csv"}),
        patch.object(mcp_server, "SHAP_GLOBAL_IMPORTANCE_PATH", tmp_path / "nope3.csv"),
    ):
        result = _get_shap_drivers(KNOWN_STYLE_KEY)
    assert result == {
        "source": None,
        "style_specific": False,
        "drivers": [],
        "note": "No SHAP outputs found on disk (neither per-style nor global).",
    }


def test_get_style_profile_known_style() -> None:
    """A known, support-filtered style has attributes, an available trajectory, and SHAP drivers."""
    result = get_style_profile(KNOWN_STYLE_KEY)
    assert result["style_key"] == KNOWN_STYLE_KEY
    assert result["attributes"]["product_type_name"] == "T-shirt"
    assert result["trajectory"]["available"] is True
    assert result["trajectory"]["n_weeks_total"] > 0
    assert len(result["trajectory"]["recent_weeks"]) <= mcp_server.TRAJECTORY_RECENT_WEEKS
    assert result["trajectory"]["recent_weeks"][-1]["week_start"] <= "2020-09-21"
    assert result["shap_drivers"]["style_specific"] is True


def test_get_style_profile_unknown_style_trajectory_unavailable() -> None:
    """A style never observed in the panel gets an explicit unavailable trajectory, not a crash."""
    result = get_style_profile(UNKNOWN_STYLE_KEY)
    assert result["trajectory"]["available"] is False
    assert UNKNOWN_STYLE_KEY in result["trajectory"]["note"]
    # SHAP still falls back gracefully for the same unknown style.
    assert result["shap_drivers"]["style_specific"] is False


def test_get_style_profile_malformed_style_key_raises() -> None:
    """A malformed style_key is rejected before any file I/O."""
    with pytest.raises(ValueError, match="must have 5"):
        get_style_profile("not-a-style-key")


# --- forecast_styles ---


def test_forecast_styles_matching_request_has_no_note() -> None:
    """Requesting exactly the pre-computed origin/horizon returns rows with no mismatch note."""
    rows = forecast_styles(
        mcp_server.PRECOMPUTED_FORECAST_ORIGIN,
        mcp_server.PRECOMPUTED_FORECAST_HORIZON_WEEKS,
        table="incumbent",
        top_n=3,
    )
    assert len(rows) == 3
    assert [r["rank"] for r in rows] == [1, 2, 3]
    assert all("_note" not in r for r in rows)
    assert all(r["_forecast_origin_date"] == mcp_server.PRECOMPUTED_FORECAST_ORIGIN for r in rows)


def test_forecast_styles_mismatched_request_adds_note_but_still_returns_data() -> None:
    """A mismatched origin/horizon still returns the only available forecast, with a note."""
    rows = forecast_styles("2021-01-01", 4, table="emerging", top_n=2)
    assert len(rows) == 2
    assert all("_note" in r for r in rows)
    assert all(r["_requested_origin_date"] == "2021-01-01" for r in rows)
    expected_horizon = mcp_server.PRECOMPUTED_FORECAST_HORIZON_WEEKS
    assert all(r["_forecast_horizon_weeks"] == expected_horizon for r in rows)


def test_forecast_styles_invalid_table_raises() -> None:
    """An unknown table name is rejected, not silently defaulted."""
    with pytest.raises(ValueError, match="table must be one of"):
        forecast_styles(mcp_server.PRECOMPUTED_FORECAST_ORIGIN, 13, table="bogus", top_n=1)


def test_forecast_styles_top_n_less_than_1_raises() -> None:
    """top_n must be a positive count."""
    with pytest.raises(ValueError, match="top_n must be >= 1"):
        forecast_styles(mcp_server.PRECOMPUTED_FORECAST_ORIGIN, 13, table="incumbent", top_n=0)


# --- get_reference_images ---


def test_get_reference_images_known_style_sorted_by_sales_desc() -> None:
    """Reference images for a known style are fetched, real, and sorted by recent sales desc."""
    paths = get_reference_images(KNOWN_STYLE_KEY, 3)
    assert len(paths) == 3
    for p in paths:
        assert Path(p).exists()
    # The manifest's known highest-seller for this style (excluding the failed fetch) is this one.
    assert paths[0] == "data/images/0800691008.jpg"


def test_get_reference_images_unknown_style_returns_empty_list() -> None:
    """A style with no exemplar manifest rows returns an empty list, not an error."""
    assert get_reference_images(UNKNOWN_STYLE_KEY, 5) == []


def test_get_reference_images_n_less_than_1_raises() -> None:
    """n must be a positive count."""
    with pytest.raises(ValueError, match="n must be >= 1"):
        get_reference_images(KNOWN_STYLE_KEY, 0)


# --- generate_concept (mocked backend -- no GPU touched) ---


def test_generate_concept_wraps_backend_and_converts_paths() -> None:
    """generate_concept converts str paths to Path on the way in and back to str on the way out."""
    fake_result = [Path("data/generated/local_sdxl/seed42_00.png")]
    with patch.object(mcp_server.backends, "generate_concept", return_value=fake_result) as mock:
        result = generate_concept(
            prompt="a black solid jersey basic t-shirt concept",
            reference_images=["data/images/0800691008.jpg"],
            backend="local_sdxl",
            ip_adapter_scale=0.6,
            seed=42,
            n=1,
        )
    mock.assert_called_once_with(
        prompt="a black solid jersey basic t-shirt concept",
        reference_images=[Path("data/images/0800691008.jpg")],
        backend="local_sdxl",
        ip_adapter_scale=0.6,
        seed=42,
        n=1,
    )
    assert result == ["data/generated/local_sdxl/seed42_00.png"]


# --- _load_clip_band / _load_margin_band ---


def test_load_clip_band_reads_real_artifact() -> None:
    """The already-computed CLIP band CSV parses into a (lower, upper) float tuple."""
    lower, upper = _load_clip_band()
    assert 0.0 <= lower < upper <= 1.0


def test_load_clip_band_missing_file_raises(tmp_path: Path) -> None:
    """A missing band CSV is a loud RuntimeError, not a silent default."""
    with (
        patch.object(mcp_server, "CLIP_BAND_PATH", tmp_path / "missing.csv"),
        pytest.raises(RuntimeError, match="not found"),
    ):
        _load_clip_band()


def test_load_margin_band_missing_returns_none() -> None:
    """The margin-band artifacts have not been produced yet (Track C, in flight) -- returns None."""
    assert _load_margin_band("clip") is None
    assert _load_margin_band("dinov2") is None


def test_load_margin_band_present_reads_band_bounds(tmp_path: Path) -> None:
    """Once a margin_anchors CSV exists, its band_lower/band_upper columns are read correctly."""
    band_path = tmp_path / "margin_anchors_clip.csv"
    pl.DataFrame(
        {"row_type": ["upper_anchor"], "band_lower": [0.1], "band_upper": [0.4]}
    ).write_csv(band_path)
    with patch.object(mcp_server, "MARGIN_BAND_PATHS", {"clip": band_path}):
        assert _load_margin_band("clip") == (0.1, 0.4)


# --- score_concept (mocked embedders -- no CLIP/DINOv2 model load, matches project convention) ---


def _fake_embed(vectors: dict[Path, object]):
    def _embed(path: Path):
        return vectors[path]

    return _embed


def test_score_concept_missing_concept_path_raises() -> None:
    """A nonexistent concept image path is a loud FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="does not exist"):
        score_concept("data/generated/does_not_exist.png", KNOWN_STYLE_KEY)


def test_score_concept_unknown_style_raises(tmp_path: Path) -> None:
    """A style with zero fetched reference images cannot be scored."""
    concept = tmp_path / "concept.png"
    concept.write_bytes(b"not a real png, existence is all that's checked before this point")
    with pytest.raises(ValueError, match="No fetched reference images"):
        score_concept(str(concept), UNKNOWN_STYLE_KEY)


def test_score_concept_falls_back_to_clip_band_when_margin_bands_absent(tmp_path: Path) -> None:
    """With no margin_anchors_*.csv on disk, pass proxies the legacy in_clip_band signal."""
    concept = tmp_path / "concept.png"
    concept.write_bytes(b"stand-in bytes; clip_similarity/control_similarity are mocked below")

    with (
        patch.object(mcp_server, "clip_similarity", return_value={"mean": 0.92, "max": 0.95}),
        patch.object(mcp_server, "control_similarity", return_value={"mean": 0.5, "max": 0.5}),
        patch.object(mcp_server.clip_scoring, "embed_image", return_value=object()),
        patch.object(mcp_server.dino_scoring, "embed_image", return_value=object()),
        patch.object(mcp_server, "embedding_margin", side_effect=[0.42, 0.77]),
    ):
        result = score_concept(str(concept), KNOWN_STYLE_KEY)

    assert result["in_clip_band"] is True  # 0.92 is within the real [0.8906, 0.9401] band.
    assert result["clip_margin"] == pytest.approx(0.42)
    assert result["dino_margin"] == pytest.approx(0.77)
    assert result["in_clip_margin_band"] is None
    assert result["in_dino_margin_band"] is None
    assert result["pass"] is True
    assert "in_clip_band" in result["note"]


def test_score_concept_uses_margin_bands_when_both_available(tmp_path: Path) -> None:
    """Once both margin_anchors CSVs exist, pass requires both spaces' margin bands to agree."""
    concept = tmp_path / "concept.png"
    concept.write_bytes(b"stand-in bytes; clip_similarity/control_similarity are mocked below")

    clip_band_path = tmp_path / "margin_anchors_clip.csv"
    dino_band_path = tmp_path / "margin_anchors_dinov2.csv"
    pl.DataFrame({"band_lower": [0.1], "band_upper": [0.5]}).write_csv(clip_band_path)
    pl.DataFrame({"band_lower": [0.6], "band_upper": [0.9]}).write_csv(dino_band_path)

    with (
        patch.object(mcp_server, "clip_similarity", return_value={"mean": 0.92, "max": 0.95}),
        patch.object(mcp_server, "control_similarity", return_value={"mean": 0.5, "max": 0.5}),
        patch.object(mcp_server.clip_scoring, "embed_image", return_value=object()),
        patch.object(mcp_server.dino_scoring, "embed_image", return_value=object()),
        # clip_margin=0.3 (in [0.1, 0.5]) but dino_margin=0.1 (NOT in [0.6, 0.9]) -> overall fail.
        patch.object(mcp_server, "embedding_margin", side_effect=[0.3, 0.1]),
        patch.object(
            mcp_server, "MARGIN_BAND_PATHS", {"clip": clip_band_path, "dinov2": dino_band_path}
        ),
    ):
        result = score_concept(str(concept), KNOWN_STYLE_KEY)

    assert result["in_clip_margin_band"] is True
    assert result["in_dino_margin_band"] is False
    assert result["pass"] is False
    assert "in_clip_margin_band AND in_dino_margin_band" in result["note"]


def test_score_concept_no_control_images_leaves_margins_none(tmp_path: Path) -> None:
    """With no fetch_success control-role images, margin/control_similarity fields stay None."""
    concept = tmp_path / "concept.png"
    concept.write_bytes(b"stand-in bytes")

    empty_manifest = pl.DataFrame(
        {
            "role": ["final_rank_1"],
            "style_key": [KNOWN_STYLE_KEY],
            "article_id": [1],
            "units_sold_last_26w": [1],
            "local_image_path": ["data/images/0800691008.jpg"],
            "fetch_success": [True],
        }
    )
    with (
        patch.object(mcp_server, "_load_exemplar_manifest", return_value=empty_manifest),
        patch.object(mcp_server, "clip_similarity", return_value={"mean": 0.92, "max": 0.95}),
    ):
        result = score_concept(str(concept), KNOWN_STYLE_KEY)

    assert result["control_similarity"] is None
    assert result["clip_margin"] is None
    assert result["dino_margin"] is None
    assert result["pass"] is True  # falls back to in_clip_band, which is True here.


# --- compose_final_sheet ---


def test_compose_final_sheet_writes_a_png(tmp_path: Path) -> None:
    """Composes 2 real reference images + captions into one saved PNG under a tmp dir."""
    with patch.object(mcp_server, "CONCEPT_SHEET_DIR", tmp_path):
        out_path = compose_final_sheet(
            ["data/images/0800691008.jpg", "data/images/0554598001.jpg"],
            ["ref 1", "ref 2"],
        )
    assert Path(out_path).exists()
    assert Path(out_path).suffix == ".png"
    assert Path(out_path).parent == tmp_path


def test_compose_final_sheet_single_image(tmp_path: Path) -> None:
    """A single-image sheet doesn't crash on matplotlib's non-array single-Axes return."""
    with patch.object(mcp_server, "CONCEPT_SHEET_DIR", tmp_path):
        out_path = compose_final_sheet(["data/images/0800691008.jpg"], ["only one"])
    assert Path(out_path).exists()


def test_compose_final_sheet_empty_raises() -> None:
    """An empty concept_paths list is a caller error."""
    with pytest.raises(ValueError, match="non-empty"):
        compose_final_sheet([], [])


def test_compose_final_sheet_mismatched_lengths_raises() -> None:
    """concept_paths and captions must be the same length."""
    with pytest.raises(ValueError, match="same length"):
        compose_final_sheet(["a.png", "b.png"], ["only one caption"])
