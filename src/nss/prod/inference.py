"""Batch and on-demand inference with a trained artifact.

`Predictor.score(panel, as_of)` validates the panel against its contract, builds the features at
`as_of` for every style with a panel row that week, encodes the categorical attributes with the
**training** vocabulary stored in the artifact, and returns per style:

    forecast            point forecast, units per active article per week over the next 13 weeks
    interval_lower/_upper  the calibrated interval (long-run 80%), same units
    interval_crossed    True where q10 - Q_lo exceeded q90 + Q_hi (bounds are then served ordered)
    rank                1 = highest forecast among the styles scored
    model_version, interval_semantics

Encoding with the stored vocabulary matters: `build_features` builds its categories from the panel
it is given, so a panel with a different set of attribute values would otherwise shift every
category code silently. An attribute value never seen in training is encoded as missing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import Any

import fsspec
import lightgbm as lgb
import numpy as np
import polars as pl

from nss.features.model_features import build_features
from nss.models.lightgbm_model import _to_lgb_matrix
from nss.prod import INTERVAL_SEMANTICS
from nss.prod.contracts import validate
from nss.prod.registry import Registry

NOMINAL_COVERAGE = 0.80


class ScoringError(ValueError):
    """The request cannot be scored (e.g. no style has a panel row at the as-of week)."""


@dataclass
class Predictor:
    version: str
    models: dict[str, lgb.Booster]
    calibrator: dict[str, Any]
    feature_spec: dict[str, Any]
    metadata: dict[str, Any]

    @classmethod
    def from_dir(cls, path: str) -> Predictor:
        fs, base = fsspec.core.url_to_fs(path)
        base = base.rstrip("/")

        def text(name: str) -> str:
            with fs.open(f"{base}/{name}", "r", encoding="utf-8") as f:
                return f.read()

        models = {k: lgb.Booster(model_str=text(f"model_{k}.txt")) for k in ("point", "q10", "q90")}
        meta = json.loads(text("metadata.json"))
        return cls(
            version=meta["version"],
            models=models,
            calibrator=json.loads(text("calibrator.json")),
            feature_spec=json.loads(text("feature_spec.json")),
            metadata=meta,
        )

    @classmethod
    def from_registry(cls, root: str) -> Predictor:
        reg = Registry(root)
        champion = reg.champion()
        if champion is None:
            raise ScoringError(f"registry {root} has no champion")
        return cls.from_dir(reg.version_path(champion))

    def features(self, panel: pl.DataFrame, as_of: date) -> pl.DataFrame:
        feats = build_features(panel, [as_of])
        if feats.height == 0:
            raise ScoringError(f"no style has a panel row at {as_of}")
        vocab = self.feature_spec["categorical"]
        return feats.with_columns(
            pl.col(c).cast(pl.String).cast(pl.Enum(vocab[c]), strict=False) for c in vocab
        )

    def score(self, panel: pl.DataFrame, as_of: date) -> pl.DataFrame:
        validate(panel, "style_week_panel")
        feats = self.features(panel, as_of)
        X = _to_lgb_matrix(feats, self.feature_spec["columns"])
        point = self.models["point"].predict(X)
        lo = self.models["q10"].predict(X) - self.calibrator["Q_lo"]
        hi = self.models["q90"].predict(X) + self.calibrator["Q_hi"]
        # log1p scale -> units per active article per week; an interval never goes below zero
        out = pl.DataFrame(
            {
                "style_key": feats["style_key"],
                "as_of": [as_of] * feats.height,
                "forecast": np.expm1(point),
                "interval_lower": np.clip(np.expm1(np.minimum(lo, hi)), 0.0, None),
                "interval_upper": np.clip(np.expm1(np.maximum(lo, hi)), 0.0, None),
                # q10 above q90 after calibration (0.006% of backtest rows): bounds are served in
                # order, but flagged, because the analysis counted such rows as not covered
                "interval_crossed": lo > hi,
            }
        )
        return out.with_columns(
            pl.col("forecast").rank("ordinal", descending=True).cast(pl.Int64).alias("rank"),
            pl.lit(NOMINAL_COVERAGE).alias("interval_nominal_coverage"),
            pl.lit(self.version).alias("model_version"),
            pl.lit(INTERVAL_SEMANTICS).alias("interval_semantics"),
        ).sort("rank")
