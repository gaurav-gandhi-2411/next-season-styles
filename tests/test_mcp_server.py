from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

from nss import mcp_server
from nss.mcp_server import (
    _extract_shap_drivers_row,
    _get_shap_drivers,
    _parse_style_key,
    compose_final_sheet,
    forecast_styles,
    generate_concept,
    get_reference_images,
    get_style_profile,
    query_transactions,
    score_concept,
)

KNOWN_STYLE_KEY = "Ladieswear || T-shirt || Jersey Basic || Black || Solid"  # incumbent rank 1
UNKNOWN_STYLE_KEY = "Menswear || Bogus Type || Bogus Group || Purple || Dotted"


def test_mcp_server_registers_all_8_tools() -> None:
    """Exactly the 7 brief tools plus the closed-loop `forecast_concept`."""
    tool_names = {t.name for t in mcp_server.mcp._tool_manager.list_tools()}
    assert tool_names == {
        "query_transactions",
        "get_style_profile",
        "forecast_styles",
        "get_reference_images",
        "generate_concept",
        "score_concept",
        "compose_final_sheet",
        "forecast_concept",
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


@pytest.mark.needs_data("data/interim/transactions_train_parquet")
def test_query_transactions_no_style_filter_returns_aggregates() -> None:
    """A narrow, unfiltered date range returns non-negative aggregate stats over all styles."""
    result = query_transactions(None, "2020-09-14", "2020-09-21")
    assert result["style_key"] is None
    assert result["date_from"] == "2020-09-14"
    assert result["date_to"] == "2020-09-21"
    assert result["units"] > 0
    assert result["revenue"] > 0.0
    assert result["n_customers"] > 0


@pytest.mark.needs_data("data/interim/transactions_train_parquet", "data/raw/articles.csv")
def test_query_transactions_with_style_filter_matches_known_high_volume_style() -> None:
    """Filtering to a known high-volume style over its full lifetime yields a positive count."""
    result = query_transactions(KNOWN_STYLE_KEY, "2018-09-20", "2020-09-21")
    assert result["style_key"] == KNOWN_STYLE_KEY
    assert result["units"] > 0
    assert result["n_customers"] > 0


@pytest.mark.needs_data("data/interim/transactions_train_parquet")
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


@pytest.mark.needs_data("data/processed/style_week_panel.parquet")
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


@pytest.mark.needs_data("data/processed/style_week_panel.parquet")
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
    # `data/images/` is gitignored -- on a checkout without the fetched exemplars this is a
    # missing fixture, not a defect (restore with `python -m nss.data.fetch_images`).
    if not all(Path(p).exists() for p in paths):
        pytest.skip("exemplar images not fetched into data/images/ on this checkout")
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
    fake_result = [Path("data/generated/local_sdxl/nostyle_s0.60_seed42_00.png")]
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
        style_key=None,
    )
    assert result == ["data/generated/local_sdxl/nostyle_s0.60_seed42_00.png"]


# --- score_concept: the shipped gates; qc_gates is mocked -- no model load ---


def test_score_concept_delegates_to_qc_gates_with_flag() -> None:
    """The tool is a thin wrapper over `qc_gates.score_gates` and forwards `include_fidelity`."""
    with patch("nss.generate.qc_gates.score_gates", return_value={"verdict": "ok"}) as m:
        assert score_concept("x.png", KNOWN_STYLE_KEY, include_fidelity=True) == {"verdict": "ok"}
    m.assert_called_once_with("x.png", KNOWN_STYLE_KEY, include_fidelity=True, changes=None)


def test_score_concept_missing_concept_path_raises() -> None:
    """A nonexistent concept image path is a loud FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="does not exist"):
        score_concept("data/generated/does_not_exist.png", KNOWN_STYLE_KEY)


def test_score_concept_unknown_style_raises(tmp_path: Path) -> None:
    """A style with no screened references cannot be gated (never a silent pass)."""
    concept = tmp_path / "concept.png"
    concept.write_bytes(b"stand-in")
    with pytest.raises(ValueError, match="No screened reference images"):
        score_concept(str(concept), UNKNOWN_STYLE_KEY)


def test_score_concept_output_reports_every_shipped_gate_and_human_check() -> None:
    """Against a real final concept: Gates 1/1b present, clone control validated, human check."""
    concept = Path("reports/concepts/black-jersey-basic-tshirt_seed43.png")
    if not concept.exists() or not Path("data/images").exists():
        pytest.skip("needs the committed concept image and the local reference images")
    result = score_concept(str(concept), KNOWN_STYLE_KEY)
    assert set(result) >= {"gate1", "gate1b", "gate2", "human_visual_check", "verdict"}
    assert result["gate1b"]["clone_control_failed_as_required"] is True
    assert result["gate2"]["status"] == "not_run"
    assert result["human_visual_check"]["required"] is True
    assert result["automated_gates_pass"] is not True  # Gate 2 not run and 1b fails for this tee


# --- compose_final_sheet ---


@pytest.mark.needs_data("data/images/0800691008.jpg", "data/images/0554598001.jpg")
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


@pytest.mark.needs_data("data/images/0800691008.jpg")
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


def test_forecast_concept_tool_returns_retrieval_fields(tmp_path: Path) -> None:
    """The retrieval tool keeps the old return keys and adds top5/similarity/margin."""
    from nss.generate.concept_forecast import ConceptForecast

    image = tmp_path / "concept.png"
    image.write_bytes(b"x")
    fake = ConceptForecast(
        "S",
        12.0,
        3,
        1980,
        "retrieval",
        "medium",
        {"clip": {"style_key": "S"}, "dino": {"style_key": "T"}},
        top5=[{"style_key": "S", "similarity": 0.9, "forecast": 12.0, "rank": 3}],
        similarity=0.9,
        margin=0.01,
    )
    with patch("nss.generate.concept_forecast.forecast_concept", return_value=fake):
        out = mcp_server.forecast_concept(str(image))
    assert out["style_key"] == "S" and out["confidence"] == "medium"
    assert out["judges"] == ["clip", "dino"] and out["unavailable_judges"] == {}
    assert out["top5"][0]["style_key"] == "S" and out["margin"] == 0.01
    assert out["sentence"].startswith("maps to S; forecast 12.0")


def test_forecast_concept_tool_missing_file_raises() -> None:
    """A nonexistent concept path is a caller error."""
    with pytest.raises(FileNotFoundError):
        mcp_server.forecast_concept("does/not/exist.png")


def test_forecast_concept_summer_origin_scores_against_the_summer_table(tmp_path: Path) -> None:
    """origin="2020-06-01" looks the concept up in the summer table, not the autumn one."""
    from nss.generate.concept_forecast import ConceptForecast

    image = tmp_path / "concept.png"
    image.write_bytes(b"x")
    fake = ConceptForecast("S", 37.9, 118, 3000, "retrieval", "medium", {})
    with (
        patch("nss.generate.concept_forecast.default_index", return_value="idx") as index,
        patch("nss.generate.concept_forecast.forecast_concept", return_value=fake) as fc,
    ):
        out = mcp_server.forecast_concept(str(image), origin="2020-06-01")
    table = fc.call_args.args[1]
    assert fc.call_args.args[2] == "idx" and index.call_args.args[0] is table
    assert table.equals(pl.read_csv("reports/tables/forecast_all_styles_summer.csv"))
    assert out["n_styles"] == 3000 and out["rank"] == 118


def test_forecast_concept_unknown_origin_is_refused(tmp_path: Path) -> None:
    """An origin with no forecast on disk is a ValueError, never scored against another table."""
    image = tmp_path / "concept.png"
    image.write_bytes(b"x")
    with pytest.raises(ValueError, match="origin must be one of"):
        mcp_server.forecast_concept(str(image), origin="2021-01-01")


# --- generate_concept, production mode (mocked candidate generator -- no GPU) ---

DRESS_KEY = "Ladieswear || Dress || Dresses Ladies || Red || Solid"


def test_generate_concept_production_mode_uses_the_deliverables_generator() -> None:
    """With `style_key`, the tool calls `generate_candidate` (the code that made the submission)."""
    seen: list[tuple] = []

    def fake(style: str, scale: float, seed: int, out_root: Path) -> tuple[Path, float]:
        seen.append((style, scale, seed, out_root))
        return out_root / f"s{scale:.2f}_seed{seed}.png", 1.0

    with patch("nss.generate.concept_generation.generate_candidate", side_effect=fake):
        result = generate_concept(style_key=DRESS_KEY, ip_adapter_scale=0.35, seed=44, n=2)
    root = mcp_server.PRODUCTION_OUTPUT_ROOT
    assert seen == [(DRESS_KEY, 0.35, 44, root), (DRESS_KEY, 0.35, 45, root)]
    assert result == [
        (root / "s0.35_seed44.png").as_posix(),
        (root / "s0.35_seed45.png").as_posix(),
    ]
    assert root != Path("data/generated/n9")  # never writes into the deliverables' candidate dir


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {"style_key": "Nope || X || Y || Z || W", "ip_adapter_scale": 0.35},
            "no production brief",
        ),
        ({"style_key": DRESS_KEY}, "ip_adapter_scale"),
        ({"style_key": DRESS_KEY, "ip_adapter_scale": 0.35, "backend": "gemini"}, "local_sdxl"),
        ({"style_key": DRESS_KEY, "ip_adapter_scale": 0.35, "prompt": "x"}, "leave prompt"),
        (
            {"style_key": DRESS_KEY, "ip_adapter_scale": 0.35, "reference_images": ["a.jpg"]},
            "leave prompt",
        ),
    ],
)
def test_generate_concept_production_mode_fails_loudly_on_bad_input(
    kwargs: dict, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        generate_concept(**kwargs)


def test_model_heavy_calls_are_serialised() -> None:
    """Concurrent generate/score calls all failed; the lock lets one run at a time."""
    import threading
    import time

    active = {"now": 0, "max": 0}
    guard = threading.Lock()

    def slow(style: str, scale: float, seed: int, out_root: Path) -> tuple[Path, float]:
        with guard:
            active["now"] += 1
            active["max"] = max(active["max"], active["now"])
        time.sleep(0.05)
        with guard:
            active["now"] -= 1
        return out_root / f"{seed}.png", 0.05

    with patch("nss.generate.concept_generation.generate_candidate", side_effect=slow):
        threads = [
            threading.Thread(
                target=generate_concept,
                kwargs={"style_key": DRESS_KEY, "ip_adapter_scale": 0.35, "seed": 40 + i},
            )
            for i in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    assert active["max"] == 1


def test_generate_concept_keeps_its_signature_for_the_mcp_schema() -> None:
    import inspect

    params = inspect.signature(generate_concept).parameters
    assert list(params) == [
        "prompt",
        "reference_images",
        "backend",
        "ip_adapter_scale",
        "seed",
        "n",
        "style_key",
    ]


def test_score_and_forecast_concept_share_the_lock_and_keep_their_signatures() -> None:
    import inspect

    for fn in (mcp_server.score_concept, mcp_server.forecast_concept):
        assert hasattr(fn, "__wrapped__")  # decorated with the serialiser
    assert list(inspect.signature(mcp_server.score_concept).parameters) == [
        "concept_path",
        "style_key",
        "include_fidelity",
        "changes",
    ]
