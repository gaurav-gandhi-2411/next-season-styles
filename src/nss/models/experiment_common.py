"""Shared plumbing for the buyer-mix/price experiments (embargoed retune, train-and-decide).

Both stages reuse the embargoed protocol of `backtest_embargo_check` unchanged (12 shared test
origins = `origins[INITIAL_POOL_SIZE:]`, training origins `o <= t - 16 weeks`, `train_lightgbm` /
`predict_lightgbm` / `score_predictions`); this module only holds what both stages need: the data
paths (read-only, under `NSS_DATA_ROOT`), the frame builders and the generic embargoed arm loop that
takes an already-built model frame (the control and treatment arms differ only in that frame).
"""

from __future__ import annotations

import polars as pl

from nss.features.customer_features import add_customer_features
from nss.features.customer_features_build import DATA_ROOT
from nss.features.customer_features_build import OUT_PATH as CUSTOMER_FEATURES_PATH
from nss.features.price_features import add_price_features
from nss.models.backtest import WMAPE_WEIGHT_COL, Origin
from nss.models.backtest_embargo_check import embargoed_train_origin_weeks
from nss.models.lightgbm_model import (
    INITIAL_POOL_SIZE,
    LGBMConfig,
    build_model_frame,
    feature_columns,
    predict_lightgbm,
    train_lightgbm,
)
from nss.models.metrics import score_predictions

PANEL_PATH = DATA_ROOT / "processed" / "style_week_panel.parquet"
ARTICLES_PATH = DATA_ROOT / "raw" / "articles.csv"

__all__ = [
    "ARTICLES_PATH",
    "CUSTOMER_FEATURES_PATH",
    "PANEL_PATH",
    "build_control_frame",
    "build_treatment_frame",
    "load_panel",
    "run_arm",
]


def load_panel() -> pl.DataFrame:
    """The dense style-week panel (read-only)."""
    return pl.read_parquet(PANEL_PATH)


def build_control_frame(panel: pl.DataFrame, origin_weeks: list) -> pl.DataFrame:
    """The shipped feature set (`build_model_frame`)."""
    return build_model_frame(panel, origin_weeks)


def build_treatment_frame(control_frame: pl.DataFrame, panel: pl.DataFrame) -> pl.DataFrame:
    """Control features + customer features + price features (additive, order-preserving)."""
    customer = pl.read_parquet(CUSTOMER_FEATURES_PATH)
    return add_price_features(add_customer_features(control_frame, customer), panel)


def run_arm(
    frame: pl.DataFrame, origins: list[Origin], config: LGBMConfig, method: str
) -> pl.DataFrame:
    """Embargoed walk-forward over `origins[INITIAL_POOL_SIZE:]` on an already-built model frame.

    Same loop as `backtest_embargo_check.run_embargoed_walk_forward`; raises if a test origin has
    no embargoed training data (the 12 shared origins all do).
    """
    columns = feature_columns(frame)
    rows: list[dict[str, object]] = []
    for test_index in range(INITIAL_POOL_SIZE, len(origins)):
        test_origin = origins[test_index]
        train_weeks = embargoed_train_origin_weeks(origins, test_index)
        train_frame = frame.filter(pl.col("origin_week").is_in(train_weeks))
        test_frame = frame.filter(pl.col("origin_week") == test_origin.origin_week)
        if not train_weeks or train_frame.height == 0 or test_frame.height == 0:
            raise RuntimeError(f"no embargoed training data for {test_origin.origin_week}")
        model = train_lightgbm(train_frame, config, columns)
        preds = predict_lightgbm(model, test_frame, columns)
        metrics = score_predictions(
            test_frame["y_true"].to_numpy(), preds, test_frame[WMAPE_WEIGHT_COL].to_numpy()
        )
        n_eval = metrics.pop("n_eval")
        rows.append(
            {
                "origin_week": test_origin.origin_week,
                "method": method,
                "has_52w_lag": test_origin.has_52w_lag,
                "is_covid": test_origin.is_covid,
                "n_eval_set": test_frame.height,
                "n_eval": int(n_eval),
                **metrics,
            }
        )
    return pl.DataFrame(rows)
