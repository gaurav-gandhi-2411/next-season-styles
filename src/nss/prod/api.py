"""FastAPI service for the champion forecaster.

    uvicorn nss.prod.api:app --host 0.0.0.0 --port 8080

The model is chosen at startup, from `NSS_ARTIFACT_DIR` (one artifact) or else the champion of the
registry at `NSS_REGISTRY_ROOT`. With neither set, or the load failing, the service still starts
and `/healthz` answers 503 with the reason, so a misconfigured deployment is visible rather than
crashing in a loop. Every forecast is appended to the scoring log at `NSS_SCORING_LOG_DIR` if set.

Endpoints:
    GET  /healthz       200 with the model version when ready, else 503
    GET  /v1/model      model version, backtest summary and interval semantics
    POST /v1/forecast   {"as_of": date, "rows": [style-week panel rows]} -> ranked forecasts

Validation, in order: the JSON body against the pydantic model (422 on type errors); the rows as
a table against the panel contract (400 with a row-level report); scoring preconditions such as
the as-of week being present (400).
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date
from typing import Any

import polars as pl
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from nss.prod import INTERVAL_SEMANTICS
from nss.prod.contracts import ContractViolation
from nss.prod.inference import Predictor, ScoringError
from nss.prod.monitoring import ScoringLog

MAX_ROWS = 500_000
# Explicit dtypes: built from JSON, an all-null column would otherwise get polars' Null dtype and
# fail the contract for the wrong reason.
PANEL_SCHEMA: dict[str, Any] = {
    "style_key": pl.String,
    "index_group_name": pl.String,
    "product_type_name": pl.String,
    "garment_group_name": pl.String,
    "perceived_colour_master_name": pl.String,
    "graphical_appearance_name": pl.String,
    "week_start": pl.Date,
    "units": pl.Int64,
    "n_active_articles": pl.Int64,
    "units_per_active_article": pl.Float64,
    "price_index": pl.Float64,
    "intensity_shrunk": pl.Float64,
    "first_week_seen": pl.Date,
}


class PanelRow(BaseModel):
    """One style-week of history, in the panel contract's columns."""

    model_config = ConfigDict(extra="forbid")

    style_key: str = Field(min_length=1)
    index_group_name: str
    product_type_name: str
    garment_group_name: str
    perceived_colour_master_name: str
    graphical_appearance_name: str
    week_start: date
    units: int = Field(ge=0)
    n_active_articles: int = Field(ge=0)
    units_per_active_article: float = Field(ge=0)
    price_index: float | None = None
    intensity_shrunk: float | None = None
    first_week_seen: date


class ForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    as_of: date
    rows: list[PanelRow] = Field(min_length=1, max_length=MAX_ROWS)


class Interval(BaseModel):
    lower: float
    upper: float
    nominal_coverage: float
    crossed: bool = Field(description="quantile models crossed; bounds served in order")


class Forecast(BaseModel):
    style_key: str
    forecast: float = Field(description="units per active article per week, next 13 weeks")
    interval: Interval
    rank: int
    model_version: str
    interval_semantics: str


class ForecastResponse(BaseModel):
    as_of: date
    model_version: str
    n_styles: int
    forecasts: list[Forecast]


def _load() -> tuple[Predictor | None, str | None]:
    try:
        if os.environ.get("NSS_ARTIFACT_DIR"):
            return Predictor.from_dir(os.environ["NSS_ARTIFACT_DIR"]), None
        if os.environ.get("NSS_REGISTRY_ROOT"):
            return Predictor.from_registry(os.environ["NSS_REGISTRY_ROOT"]), None
        return None, "set NSS_ARTIFACT_DIR or NSS_REGISTRY_ROOT"
    except Exception as e:  # noqa: BLE001 -- surfaced through /healthz, not swallowed
        return None, f"model load failed: {type(e).__name__}: {e}"


def create_app(predictor: Predictor | None = None) -> FastAPI:
    state: dict[str, Any] = {"predictor": predictor, "error": None}

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if state["predictor"] is None:
            state["predictor"], state["error"] = _load()
        yield

    app = FastAPI(title="nss forecaster", version="1", lifespan=lifespan)

    def ready() -> Predictor:
        p = state["predictor"]
        if p is None:
            raise HTTPException(503, detail=state["error"] or "model not loaded")
        return p

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok", "model_version": ready().version}

    @app.get("/v1/model")
    def model() -> dict[str, Any]:
        p = ready()
        return {
            "model_version": p.version,
            "as_of": p.metadata.get("as_of"),
            "backtest": p.metadata.get("backtest"),
            "interval_semantics": INTERVAL_SEMANTICS,
        }

    @app.post("/v1/forecast", response_model=ForecastResponse)
    def forecast(req: ForecastRequest) -> ForecastResponse:
        p = ready()
        panel = pl.DataFrame([r.model_dump() for r in req.rows], schema=PANEL_SCHEMA)
        try:
            scored = p.score(panel, req.as_of)
        except ContractViolation as e:
            raise HTTPException(
                400,
                detail={
                    "error": f"rows violate the '{e.contract}' contract",
                    "violations": e.report.head(100).to_dicts(),
                    "n_violations": e.report.height,
                },
            ) from e
        except ScoringError as e:
            raise HTTPException(400, detail={"error": str(e)}) from e
        if os.environ.get("NSS_SCORING_LOG_DIR"):
            ScoringLog(os.environ["NSS_SCORING_LOG_DIR"]).append(scored)
        return ForecastResponse(
            as_of=req.as_of,
            model_version=p.version,
            n_styles=scored.height,
            forecasts=[
                Forecast(
                    style_key=r["style_key"],
                    forecast=r["forecast"],
                    interval=Interval(
                        lower=r["interval_lower"],
                        upper=r["interval_upper"],
                        nominal_coverage=r["interval_nominal_coverage"],
                        crossed=r["interval_crossed"],
                    ),
                    rank=r["rank"],
                    model_version=r["model_version"],
                    interval_semantics=r["interval_semantics"],
                )
                for r in scored.iter_rows(named=True)
            ],
        )

    return app


app = create_app()
