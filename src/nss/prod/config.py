"""The champion config: one YAML file, validated with pydantic, hashed for provenance."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FeatureConfig(_Strict):
    columns: list[str] = Field(min_length=1)


class ModelConfig(_Strict):
    objective: Literal["regression"]
    seed: int
    num_leaves: int = Field(gt=1)
    learning_rate: float = Field(gt=0)
    n_estimators: int = Field(gt=0)
    min_child_samples: int = Field(gt=0)


class ProtocolConfig(_Strict):
    horizon_weeks: int = Field(gt=0)
    grid_step_weeks: int = Field(gt=0)
    initial_pool_size: int = Field(ge=0)
    last_weekly_origin: date
    n_weekly_origins: int = Field(gt=0)


class CalibrationConfig(_Strict):
    method: Literal["cqr_asymmetric"]
    alpha: float = Field(gt=0, lt=1)
    quantiles: tuple[float, float]
    window_origins: int = Field(gt=0)
    history_start: date


class EvaluationConfig(_Strict):
    tolerance_margin: float = Field(ge=0, lt=1)
    reference_per_origin: str
    reference_coverage: str


class ChampionConfig(_Strict):
    name: str
    panel_path: str
    features: FeatureConfig
    model: ModelConfig
    protocol: ProtocolConfig
    calibration: CalibrationConfig
    evaluation: EvaluationConfig
    registry_root: str

    def config_hash(self) -> str:
        """SHA-256 of the canonical JSON: the same settings hash the same, whatever the YAML."""
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


def load_config(path: str | Path) -> ChampionConfig:
    with open(path, encoding="utf-8") as f:
        return ChampionConfig.model_validate(yaml.safe_load(f))
