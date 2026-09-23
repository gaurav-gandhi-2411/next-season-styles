"""Config-driven training of the champion to a versioned artifact directory.

`train(config)` does two things, both driven only by the config:

1. **Backtest and reproduction check.** The embargoed walk-forward over the 48 weekly origins
   (point model, q10, q90, rolling asymmetric CQR) exactly as the pre-registered analysis ran it.
   Its per-origin metrics must equal the committed tables (`reports/tables/phase_a_per_origin.csv`
   for the point model, `phase_s_coverage_per_origin.csv` for the calibrated interval) to 1e-9;
   otherwise training stops. The config is never adjusted to force agreement.
2. **Deployable fit.** The point, q10 and q90 models retrained on every grid origin whose outcome
   window closed by the as-of week, and the calibrator fitted on the most recent closed origins.

Artifact directory `<out>/<version>/`:

    model_point.txt, model_q10.txt, model_q90.txt   LightGBM text models
    calibrator.json          Q_lo, Q_hi, window, alpha
    feature_spec.json        feature columns and the categorical vocabularies used for encoding
    reference.json           training feature / prediction distributions for PSI drift checks
    backtest_per_origin.csv  per-origin metrics on the 48 origins (input to the promotion gate)
    backtest_coverage.csv    per-origin calibrated coverage
    metadata.json            git SHA, config hash, panel hash, metrics, checks, timestamp
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl

from nss.features.model_features import _style_key_enum
from nss.models.backtest import Origin, generate_origin_schedule
from nss.models.lightgbm_model import (
    _categorical_indices,
    _to_lgb_matrix,
    build_model_frame,
    feature_columns,
)
from nss.models.metrics import demand_capture_at_k, score_predictions, tolerance_hit_at_k
from nss.prod import INTERVAL_SEMANTICS
from nss.prod.calibration import calibration_window, fit_asymmetric
from nss.prod.config import ChampionConfig
from nss.prod.contracts import STYLE_KEY_COLS, validate

REPRO_ATOL = 1e-9
PSI_BINS = 10
METRIC_COLS: tuple[str, ...] = (
    "hit_at_3_in_top20",
    "hit_at_3_in_top10",
    "precision_at_3",
    "precision_at_10",
    "ndcg_at_10",
    "spearman_rho",
    "wmape",
    "demand_capture_at_3",
    "demand_capture_at_10",
    "demand_capture_at_20",
    "tolerance_hit_at_3",
)


class ReproductionError(RuntimeError):
    """Training did not reproduce the committed champion tables; the artifact is not written."""


# ---------------------------------------------------------------------------
# protocol
# ---------------------------------------------------------------------------


def origins(
    panel: pl.DataFrame, cfg: ChampionConfig
) -> tuple[list[Origin], list[Origin], list[Origin]]:
    """(grid, calibration-history weekly origins, test weekly origins) as the analysis used them."""
    p = cfg.protocol
    grid = generate_origin_schedule(panel, step_weeks=p.grid_step_weeks)
    first_test = grid[p.initial_pool_size].origin_week
    weekly = generate_origin_schedule(panel, step_weeks=1)
    test = [o for o in weekly if first_test <= o.origin_week <= p.last_weekly_origin]
    if len(test) != p.n_weekly_origins:
        raise ValueError(f"{len(test)} weekly test origins, config says {p.n_weekly_origins}")
    history = [o for o in weekly if cfg.calibration.history_start <= o.origin_week < first_test]
    return grid, history, test


def embargoed_train_weeks(grid: Sequence[Origin], k: int, horizon_weeks: int) -> list[date]:
    """Grid origins before block k whose outcome window closed by grid[k]."""
    cutoff = grid[k].origin_week - timedelta(weeks=horizon_weeks)
    return [o.origin_week for o in grid[:k] if o.origin_week <= cutoff]


def fit(
    train: pl.DataFrame,
    columns: list[str],
    cfg: ChampionConfig,
    objective: str = "regression",
    alpha: float | None = None,
) -> lgb.LGBMRegressor:
    """The champion's LightGBM, deterministic, hyperparameters from the config only."""
    m = cfg.model
    extra = {"alpha": alpha} if alpha is not None else {}
    model = lgb.LGBMRegressor(
        objective=objective,
        random_state=m.seed,
        verbosity=-1,
        num_leaves=m.num_leaves,
        learning_rate=m.learning_rate,
        n_estimators=m.n_estimators,
        min_child_samples=m.min_child_samples,
        deterministic=True,
        force_row_wise=True,
        num_threads=1,
        bagging_seed=m.seed,
        feature_fraction_seed=m.seed,
        data_random_seed=m.seed,
        **extra,
    )
    model.fit(
        _to_lgb_matrix(train, columns),
        train["y_true"].to_numpy().astype(np.float64),
        categorical_feature=_categorical_indices(train, columns),
        feature_name=columns,
    )
    return model


def serve_walk_forward(
    frame: pl.DataFrame,
    columns: list[str],
    grid: Sequence[Origin],
    weeks: Sequence[date],
    cfg: ChampionConfig,
    **fit_kw: Any,
) -> pl.DataFrame:
    """Each weekly origin predicted by its grid block's embargoed model."""
    grid_weeks = [o.origin_week for o in grid]
    models: dict[int, lgb.LGBMRegressor] = {}
    parts = []
    for t in sorted(weeks):
        k = max(i for i, w in enumerate(grid_weeks) if w <= t)
        if k not in models:
            train_weeks = embargoed_train_weeks(grid, k, cfg.protocol.horizon_weeks)
            models[k] = fit(
                frame.filter(pl.col("origin_week").is_in(train_weeks)), columns, cfg, **fit_kw
            )
        test = frame.filter(pl.col("origin_week") == t)
        parts.append(
            test.select("style_key", "origin_week").with_columns(
                pl.Series("y_pred", models[k].predict(_to_lgb_matrix(test, columns)))
            )
        )
    return pl.concat(parts)


# ---------------------------------------------------------------------------
# scoring and checks
# ---------------------------------------------------------------------------


def per_origin_metrics(eval_rows: pl.DataFrame, margin: float) -> pl.DataFrame:
    """Every Phase A metric per origin (y_true, y_pred, n_active_articles_level)."""
    rows = []
    for w in eval_rows["origin_week"].unique().sort().to_list():
        s = eval_rows.filter(pl.col("origin_week") == w)
        y, p = s["y_true"].to_numpy(), s["y_pred"].to_numpy()
        m = score_predictions(y, p, s["n_active_articles_level"].to_numpy())
        m.pop("n_eval")
        rows.append(
            {
                "origin_week": w,
                **m,
                **{f"demand_capture_at_{k}": demand_capture_at_k(y, p, k) for k in (3, 10, 20)},
                "tolerance_hit_at_3": tolerance_hit_at_k(y, p, 3, margin),
            }
        )
    return pl.DataFrame(rows)


def calibrated_coverage(
    quant: pl.DataFrame, test: Sequence[Origin], cfg: ChampionConfig
) -> pl.DataFrame:
    """Per test origin: Q_lo, Q_hi and coverage of the calibrated interval (Section S)."""
    history = quant["origin_week"].unique().sort().to_list()
    rows = []
    for o in test:
        win = calibration_window(
            o.origin_week, history, cfg.calibration.window_origins, cfg.protocol.horizon_weeks
        )
        q = fit_asymmetric(quant, win, cfg.calibration.alpha)
        t = quant.filter(pl.col("origin_week") == o.origin_week)
        lo = t["q10"].to_numpy() - q["Q_lo"]
        hi = t["q90"].to_numpy() + q["Q_hi"]
        y = t["y_true"].to_numpy()
        rows.append(
            {
                "origin_week": o.origin_week,
                "is_covid": o.is_covid,
                "Q_lo": q["Q_lo"],
                "Q_hi": q["Q_hi"],
                "coverage": float(np.mean((lo <= y) & (y <= hi) & (lo <= hi))),
                "n": t.height,
            }
        )
    return pl.DataFrame(rows)


def _max_abs(a: pl.Series, b: pl.Series) -> float:
    x, y = a.to_numpy().astype(float), b.to_numpy().astype(float)
    both_nan = np.isnan(x) & np.isnan(y)
    return float(np.max(np.where(both_nan, 0.0, np.abs(x - y))))


def reproduction_checks(
    metrics: pl.DataFrame, coverage: pl.DataFrame, cfg: ChampionConfig
) -> dict[str, float]:
    """Max |diff| against the committed tables; raises ReproductionError above 1e-9."""
    ref = pl.read_csv(cfg.evaluation.reference_per_origin, try_parse_dates=True).filter(
        pl.col("method") == "lightgbm"
    )
    j = metrics.join(ref, on="origin_week", suffix="_ref", how="inner")
    if j.height != metrics.height or j.height != cfg.protocol.n_weekly_origins:
        raise ReproductionError(f"origin mismatch: {j.height} joined of {metrics.height}")
    out = {f"point.{m}": _max_abs(j[m], j[f"{m}_ref"]) for m in METRIC_COLS}
    cref = pl.read_csv(cfg.evaluation.reference_coverage, try_parse_dates=True)
    jc = coverage.join(cref, on="origin_week", how="inner", suffix="_ref")
    if jc.height != coverage.height:
        raise ReproductionError(f"coverage origin mismatch: {jc.height} of {coverage.height}")
    out["interval.coverage"] = _max_abs(jc["coverage"], jc["cov_cqr_asymmetric"])
    out["interval.Q_lo"] = _max_abs(jc["Q_lo"], jc["Q_lo_ref"])
    out["interval.Q_hi"] = _max_abs(jc["Q_hi"], jc["Q_hi_ref"])
    bad = {k: v for k, v in out.items() if not v <= REPRO_ATOL}
    if bad:
        raise ReproductionError(f"does not reproduce the committed tables (atol 1e-9): {bad}")
    return out


# ---------------------------------------------------------------------------
# artifact helpers
# ---------------------------------------------------------------------------


def git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout

    try:
        return {
            "sha": run("rev-parse", "HEAD").strip(),
            "dirty": bool(run("status", "--porcelain").strip()),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"sha": None, "dirty": None}


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def psi_reference(values: np.ndarray, bins: int = PSI_BINS) -> dict[str, list[float]]:
    """Decile edges of the reference distribution and the share of values in each bin."""
    v = values[~np.isnan(values)]
    edges = np.unique(np.quantile(v, np.linspace(0, 1, bins + 1)))
    counts = np.histogram(np.clip(v, edges[0], edges[-1]), bins=edges)[0]
    return {"edges": edges.tolist(), "proportions": (counts / counts.sum()).tolist()}


def feature_spec(panel: pl.DataFrame, columns: list[str]) -> dict[str, Any]:
    return {
        "columns": columns,
        "categorical": {c: list(_style_key_enum(panel, c).categories) for c in STYLE_KEY_COLS},
    }


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


def train(cfg: ChampionConfig, out_dir: str | Path, as_of: date | None = None) -> Path:
    """Train from the config, check reproduction, write the artifact; return its directory."""
    started = datetime.now(UTC)
    panel = validate(pl.read_parquet(cfg.panel_path), "style_week_panel")
    grid, history, test = origins(panel, cfg)
    test_weeks = [o.origin_week for o in test]
    hist_weeks = [o.origin_week for o in history]
    all_weeks = sorted({*(o.origin_week for o in grid), *hist_weeks, *test_weeks})
    frame = build_model_frame(panel, all_weeks)
    columns = feature_columns(frame)
    if columns != cfg.features.columns:
        raise ValueError(f"built feature columns differ from the config: {columns}")

    # 1. backtest + reproduction
    point = serve_walk_forward(frame, columns, grid, test_weeks, cfg)
    q10 = serve_walk_forward(
        frame,
        columns,
        grid,
        [*hist_weeks, *test_weeks],
        cfg,
        objective="quantile",
        alpha=cfg.calibration.quantiles[0],
    )
    q90 = serve_walk_forward(
        frame,
        columns,
        grid,
        [*hist_weeks, *test_weeks],
        cfg,
        objective="quantile",
        alpha=cfg.calibration.quantiles[1],
    )
    truth = frame.select("style_key", "origin_week", "y_true", "n_active_articles_level")
    point_eval = truth.join(point, on=["style_key", "origin_week"], how="inner")
    metrics = per_origin_metrics(point_eval, cfg.evaluation.tolerance_margin)
    quant = truth.join(q10.rename({"y_pred": "q10"}), on=["style_key", "origin_week"]).join(
        q90.rename({"y_pred": "q90"}), on=["style_key", "origin_week"]
    )
    coverage = calibrated_coverage(quant, test, cfg)
    checks = reproduction_checks(metrics, coverage, cfg)

    # 2. deployable fit
    as_of = as_of or panel["week_start"].max()
    horizon = timedelta(weeks=cfg.protocol.horizon_weeks)
    deploy_weeks = [o.origin_week for o in grid if o.origin_week + horizon <= as_of]
    deploy_train = frame.filter(pl.col("origin_week").is_in(deploy_weeks))
    models = {
        "point": fit(deploy_train, columns, cfg),
        "q10": fit(
            deploy_train, columns, cfg, objective="quantile", alpha=cfg.calibration.quantiles[0]
        ),
        "q90": fit(
            deploy_train, columns, cfg, objective="quantile", alpha=cfg.calibration.quantiles[1]
        ),
    }
    window = calibration_window(
        as_of,
        quant["origin_week"].unique().to_list(),
        cfg.calibration.window_origins,
        cfg.protocol.horizon_weeks,
    )
    if len(window) != cfg.calibration.window_origins:
        raise ValueError(f"only {len(window)} closed calibration origins before {as_of}")
    calibrator = {
        "method": cfg.calibration.method,
        "alpha": cfg.calibration.alpha,
        "window": [str(w) for w in window],
        **fit_asymmetric(quant, window, cfg.calibration.alpha),
    }

    cfg_hash = cfg.config_hash()
    version = f"{started:%Y%m%dT%H%M%SZ}-{cfg_hash[:8]}"
    art = Path(out_dir) / version
    art.mkdir(parents=True, exist_ok=False)
    for name, m in models.items():
        m.booster_.save_model(str(art / f"model_{name}.txt"))
    # the saved text model must predict exactly what the in-memory one does
    X = _to_lgb_matrix(deploy_train, columns)
    for name, m in models.items():
        loaded = lgb.Booster(model_file=str(art / f"model_{name}.txt"))
        diff = float(np.max(np.abs(loaded.predict(X) - m.predict(X))))
        checks[f"saved_model_roundtrip.{name}"] = diff
        if diff != 0.0:
            raise ReproductionError(f"saved {name} model does not round-trip: {diff}")

    spec = feature_spec(panel, columns)
    numeric = [c for c in columns if c not in STYLE_KEY_COLS]
    reference = {
        "features": {
            c: psi_reference(deploy_train[c].cast(pl.Float64).to_numpy()) for c in numeric
        },
        "prediction": psi_reference(point["y_pred"].to_numpy()),
    }
    _write_json(art / "calibrator.json", calibrator)
    _write_json(art / "feature_spec.json", spec)
    _write_json(art / "reference.json", reference)
    metrics.write_csv(art / "backtest_per_origin.csv")
    coverage.write_csv(art / "backtest_coverage.csv")
    summary = {m: float(metrics[m].mean()) for m in METRIC_COLS}
    _write_json(
        art / "metadata.json",
        {
            "version": version,
            "name": cfg.name,
            "created_utc": started.isoformat(),
            "git": git_state(),
            "config_hash": cfg_hash,
            "config": cfg.model_dump(mode="json"),
            "panel_path": cfg.panel_path,
            "panel_sha256": file_sha256(cfg.panel_path),
            "as_of": str(as_of),
            "deploy_training_origins": [str(w) for w in deploy_weeks],
            "backtest": {
                "n_origins": metrics.height,
                "mean_metrics": summary,
                "pooled_coverage": float(
                    (coverage["coverage"] * coverage["n"]).sum() / coverage["n"].sum()
                ),
            },
            "checks": checks,
            "interval_semantics": INTERVAL_SEMANTICS,
            "library_versions": {
                "lightgbm": lgb.__version__,
                "polars": pl.__version__,
                "numpy": np.__version__,
            },
        },
    )
    return art
