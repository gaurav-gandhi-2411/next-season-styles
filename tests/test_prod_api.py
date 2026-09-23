from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import polars as pl
import pytest
from fastapi.testclient import TestClient

from nss.prod import INTERVAL_SEMANTICS
from nss.prod.api import PANEL_SCHEMA, create_app
from nss.prod.fixture import build_toy_artifact, toy_panel
from nss.prod.inference import Predictor

AS_OF = date(2020, 6, 22)


@pytest.fixture(scope="module")
def predictor(tmp_path_factory: pytest.TempPathFactory) -> Predictor:
    return Predictor.from_dir(str(build_toy_artifact(tmp_path_factory.mktemp("art"))))


@pytest.fixture(scope="module")
def client(predictor: Predictor) -> TestClient:
    return TestClient(create_app(predictor))


def _rows(n_styles: int = 15) -> list[dict]:
    p = toy_panel()
    keep = p.filter(pl.col("week_start") == AS_OF)["style_key"].unique().sort().head(n_styles)
    sub = p.filter(pl.col("style_key").is_in(keep.implode()) & (pl.col("week_start") <= AS_OF))
    return json.loads(sub.select(list(PANEL_SCHEMA)).write_json())


def test_forecast_returns_the_contracted_fields(client: TestClient) -> None:
    r = client.post("/v1/forecast", json={"as_of": str(AS_OF), "rows": _rows()})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_styles"] == 15 and body["model_version"] == "toy"
    ranks = [f["rank"] for f in body["forecasts"]]
    assert ranks == list(range(1, 16))
    for f in body["forecasts"]:
        assert f["interval"]["lower"] <= f["interval"]["upper"]
        assert f["interval"]["nominal_coverage"] == 0.8
        assert f["interval_semantics"] == INTERVAL_SEMANTICS
        assert "long-run 80%" in f["interval_semantics"].lower()
    forecasts = [f["forecast"] for f in body["forecasts"]]
    assert forecasts == sorted(forecasts, reverse=True)


def test_malformed_body_is_422(client: TestClient) -> None:
    rows = _rows(2)
    rows[0]["units"] = "many"
    assert client.post("/v1/forecast", json={"as_of": str(AS_OF), "rows": rows}).status_code == 422
    assert client.post("/v1/forecast", json={"rows": _rows(2)}).status_code == 422
    extra = _rows(2)
    extra[0]["surprise"] = 1
    assert client.post("/v1/forecast", json={"as_of": str(AS_OF), "rows": extra}).status_code == 422


def test_contract_violation_is_400_with_row_report(client: TestClient) -> None:
    rows = _rows(3)
    rows[1]["style_key"] = "does not match its attributes"
    r = client.post("/v1/forecast", json={"as_of": str(AS_OF), "rows": rows})
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert "style_week_panel" in detail["error"]
    assert any(
        v["check"] == "style_key_matches_attributes" and v["index"] == 1
        for v in detail["violations"]
    )


def test_as_of_without_rows_is_400(client: TestClient) -> None:
    r = client.post("/v1/forecast", json={"as_of": "2030-01-07", "rows": _rows(2)})
    assert r.status_code == 400
    assert "no style has a panel row" in r.json()["detail"]["error"]


def test_health_and_model(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok", "model_version": "toy"}
    assert client.get("/v1/model").json()["interval_semantics"] == INTERVAL_SEMANTICS


def test_not_ready_is_503(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NSS_ARTIFACT_DIR", raising=False)
    monkeypatch.delenv("NSS_REGISTRY_ROOT", raising=False)
    with TestClient(create_app()) as c:
        r = c.get("/healthz")
    assert r.status_code == 503 and "NSS_ARTIFACT_DIR" in r.json()["detail"]


def test_unknown_attribute_value_is_encoded_missing(predictor: Predictor) -> None:
    """Training vocabulary is used at inference: an unseen category becomes missing, and the
    codes of known categories do not shift."""
    p = toy_panel().filter(pl.col("week_start") <= AS_OF)
    base = predictor.features(p, AS_OF)
    unseen = p.with_columns(
        pl.when(pl.col("style_key") == base["style_key"][0])
        .then(pl.lit("Never Seen Colour"))
        .otherwise(pl.col("perceived_colour_master_name"))
        .alias("perceived_colour_master_name")
    )
    got = predictor.features(unseen, AS_OF)
    assert got.filter(pl.col("style_key") == base["style_key"][0])[
        "perceived_colour_master_name"
    ].to_list() == [None]
    others = got.filter(pl.col("style_key") != base["style_key"][0]).sort("style_key")
    ref = base.filter(pl.col("style_key") != base["style_key"][0]).sort("style_key")
    assert (
        others["perceived_colour_master_name"].to_physical().to_list()
        == ref["perceived_colour_master_name"].to_physical().to_list()
    )


def test_scoring_log_records_every_prediction(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nss.prod.monitoring import ScoringLog

    monkeypatch.setenv("NSS_SCORING_LOG_DIR", str(tmp_path / "log"))
    client.post("/v1/forecast", json={"as_of": str(AS_OF), "rows": _rows(5)})
    log = ScoringLog(str(tmp_path / "log")).read()
    assert log.height == 5 and log["model_version"].unique().to_list() == ["toy"]
