"""2c: selective prediction from the model's own quantile spread.

Rules are pre-registered in `reports/v3/PREREGISTRATION.md` (2c), committed before any spread was
computed. Uncertainty = `q90 - q10` (log1p scale) from two extra LightGBM quantile models trained
exactly like the point model (same features, locked config, same embargoed training origins); the
point model, and therefore the picks, are unchanged. A pick is confident iff its spread is at or
below the median spread of all evaluated styles at that origin. Statistic: hit rate of confident
picks minus hit rate of all picks, pooled over origins, with a moving-block bootstrap over whole
origins and a within-origin permutation floor (random abstention).

    uv run python -m nss.models.selective_prediction
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import lightgbm as lgb
import numpy as np
import polars as pl
from scipy.stats import spearmanr

from nss.models import final_forecast, growth_backtest
from nss.models.backtest import BOOTSTRAP_N_RESAMPLES, BOOTSTRAP_SEED, generate_origin_schedule
from nss.models.backtest_embargo_check import embargoed_train_origin_weeks
from nss.models.eval_power import ess_ac
from nss.models.lightgbm_model import (
    INITIAL_POOL_SIZE,
    LGBM_DETERMINISM_PARAMS,
    RANDOM_SEED,
    _categorical_indices,
    _to_lgb_matrix,
    build_model_frame,
    feature_columns,
)

ALPHAS = (0.1, 0.9)
TOP_K = 3
HIT_N = 20
COVERAGE_MIN = 0.30
TOL = 1e-9
OUT = "reports/tables"


def _train_quantile(train: pl.DataFrame, columns: list[str], alpha: float) -> lgb.LGBMRegressor:
    cfg = final_forecast.FINAL_MODEL_CONFIG
    model = lgb.LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        random_state=RANDOM_SEED,
        verbosity=-1,
        num_leaves=int(cfg["num_leaves"]),
        learning_rate=float(cfg["learning_rate"]),
        n_estimators=int(cfg["n_estimators"]),
        min_child_samples=int(cfg["min_child_samples"]),
        **LGBM_DETERMINISM_PARAMS,
    )
    model.fit(
        _to_lgb_matrix(train, columns),
        train["y_true"].to_numpy().astype(np.float64),
        categorical_feature=_categorical_indices(train, columns),
        feature_name=columns,
    )
    return model


def quantile_frame(
    panel: pl.DataFrame, grid: Sequence, test_weeks: Sequence[date], initial_pool_size: int
) -> pl.DataFrame:
    """`style_key`, `origin_week`, `q10`, `q90` per test origin (block model as in 2b)."""
    grid_weeks = [o.origin_week for o in grid]
    frame = build_model_frame(panel, sorted({*grid_weeks, *test_weeks}))
    columns = feature_columns(frame)
    models: dict[int, tuple[lgb.LGBMRegressor, lgb.LGBMRegressor]] = {}
    parts = []
    for t in sorted(test_weeks):
        k = max(i for i, w in enumerate(grid_weeks) if w <= t)
        if k not in models:
            weeks = embargoed_train_origin_weeks(list(grid), k)
            train = frame.filter(pl.col("origin_week").is_in(weeks))
            models[k] = tuple(_train_quantile(train, columns, a) for a in ALPHAS)  # type: ignore[assignment]
        test = frame.filter(pl.col("origin_week") == t)
        x = _to_lgb_matrix(test, columns)
        parts.append(
            test.select("style_key", "origin_week").with_columns(
                pl.Series("q10", models[k][0].predict(x)),
                pl.Series("q90", models[k][1].predict(x)),
            )
        )
    return pl.concat(parts)


@dataclass(frozen=True)
class OriginPicks:
    """One origin's 3 picks: whether each is a hit and whether each is confident."""

    origin_week: date
    hit: np.ndarray
    confident: np.ndarray
    n_styles: int


def build_picks(frame: pl.DataFrame) -> list[OriginPicks]:
    """Top-3 picks per origin by point prediction, hits vs the realised top 20, confidence flags."""
    out = []
    for t in sorted(frame["origin_week"].unique().to_list()):
        sub = frame.filter(pl.col("origin_week") == t)
        y = sub["y_true"].to_numpy()
        pred = sub["y_pred_lightgbm"].to_numpy()
        w = (sub["q90"] - sub["q10"]).to_numpy()
        top_pred = np.argsort(-pred, kind="stable")[:TOP_K]
        true_top = set(np.argsort(-y, kind="stable")[: min(HIT_N, y.size)].tolist())
        out.append(
            OriginPicks(
                origin_week=t,
                hit=np.array([int(i in true_top) for i in top_pred]),
                confident=(w[top_pred] <= np.median(w) + TOL).astype(int),
                n_styles=y.size,
            )
        )
    return out


def statistic(picks: Sequence[OriginPicks]) -> tuple[float, float, float, float]:
    """(hit rate of confident picks, hit rate of all picks, difference, coverage)."""
    hit = np.concatenate([p.hit for p in picks])
    conf = np.concatenate([p.confident for p in picks]).astype(bool)
    if hit.size == 0:
        return (np.nan,) * 4
    rate_all = float(hit.mean())
    coverage = float(conf.mean())
    if conf.sum() == 0:
        return (np.nan, rate_all, np.nan, coverage)
    rate_conf = float(hit[conf].mean())
    return (rate_conf, rate_all, rate_conf - rate_all, coverage)


def block_bootstrap_difference(
    picks: Sequence[OriginPicks],
    block: int,
    n_resamples: int = BOOTSTRAP_N_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """95% moving-block bootstrap interval of the difference, resampling whole origins."""
    n = len(picks)
    rng = np.random.default_rng(seed)
    blocks_needed = -(-n // block)
    max_start = max(n - block, 0)
    diffs = []
    for _ in range(n_resamples):
        starts = rng.integers(0, max_start + 1, size=blocks_needed)
        idx = np.concatenate([np.arange(s, min(s + block, n)) for s in starts])[:n]
        d = statistic([picks[i] for i in idx])[2]
        if not np.isnan(d):
            diffs.append(d)
    if not diffs:  # no confident pick in any resample: the difference is undefined, not zero
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)


def permutation_floor(
    picks: Sequence[OriginPicks],
    n_perm: int = BOOTSTRAP_N_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float]:
    """Random abstention: permute confidence labels within each origin's 3 picks."""
    rng = np.random.default_rng(seed)
    observed = statistic(picks)[2]
    null = []
    for _ in range(n_perm):
        shuffled = [
            OriginPicks(p.origin_week, p.hit, rng.permutation(p.confident), p.n_styles)
            for p in picks
        ]
        d = statistic(shuffled)[2]
        if not np.isnan(d):
            null.append(d)
    arr = np.asarray(null)
    if arr.size == 0:
        nan = float("nan")
        return {"null_mean": nan, "null_lo": nan, "null_hi": nan, "p_one_sided": nan}
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return {
        "null_mean": float(arr.mean()),
        "null_lo": float(lo),
        "null_hi": float(hi),
        "p_one_sided": float((np.sum(arr >= observed - TOL) + 1) / (arr.size + 1)),
    }


def verdict(diff_ci_lo: float, coverage: float) -> str:
    """HELPS iff the interval's lower bound is above 0 (tolerance 1e-9) and coverage >= 0.30."""
    return "HELPS" if diff_ci_lo > TOL and coverage >= COVERAGE_MIN else "NOT_DEMONSTRATED"


def spread_error_correlation(frame: pl.DataFrame) -> tuple[float, int]:
    """Mean over origins of Spearman(spread, |point error|) across all evaluated styles."""
    rhos = []
    for t in frame["origin_week"].unique().to_list():
        sub = frame.filter(pl.col("origin_week") == t)
        w = (sub["q90"] - sub["q10"]).to_numpy()
        err = np.abs(sub["y_true"].to_numpy() - sub["y_pred_lightgbm"].to_numpy())
        rhos.append(float(spearmanr(w, err).statistic))
    return float(np.mean(rhos)), len(rhos)


def evaluate(
    frame: pl.DataFrame, label: str, block: int, quantiles: tuple[float, ...] = (0.5,)
) -> list[dict]:
    """Summary rows for one evaluation set at one or more spread quantiles."""
    rows = []
    for q in quantiles:
        picks = build_picks_at(frame, q)
        rate_conf, rate_all, diff, coverage = statistic(picks)
        lo, hi = block_bootstrap_difference(picks, block)
        floor = permutation_floor(picks)
        rows.append(
            {
                "eval_set": label,
                "spread_quantile": q,
                "decision_bearing": q == 0.5 and label == "grid_4w",
                "n_origins": len(picks),
                "n_picks": sum(p.hit.size for p in picks),
                "coverage": coverage,
                "hit_rate_confident": rate_conf,
                "hit_rate_all": rate_all,
                "difference": diff,
                "ci_lo": lo,
                "ci_hi": hi,
                "block": block,
                "ess_ac_of_per_origin_difference": ess_ac(per_origin_difference(picks)),
                **{f"floor_{k}": v for k, v in floor.items()},
                "verdict": (
                    verdict(lo, coverage) if (q == 0.5 and label == "grid_4w") else "exploratory"
                ),
            }
        )
    return rows


def build_picks_at(frame: pl.DataFrame, quantile: float) -> list[OriginPicks]:
    """`build_picks` with the confidence threshold at `quantile` of the origin's spreads."""
    picks = build_picks(frame)
    if quantile == 0.5:
        return picks
    out = []
    for p, t in zip(picks, sorted(frame["origin_week"].unique().to_list()), strict=True):
        sub = frame.filter(pl.col("origin_week") == t)
        w = (sub["q90"] - sub["q10"]).to_numpy()
        pred = sub["y_pred_lightgbm"].to_numpy()
        top = np.argsort(-pred, kind="stable")[:TOP_K]
        out.append(
            OriginPicks(
                p.origin_week,
                p.hit,
                (w[top] <= np.quantile(w, quantile) + TOL).astype(int),
                p.n_styles,
            )
        )
    return out


def per_origin_difference(picks: Sequence[OriginPicks]) -> list[float]:
    """Per-origin (confident hit rate - all hit rate), for the ESS diagnostic only."""
    out = []
    for p in picks:
        if p.confident.sum() > 0:
            out.append(float(p.hit[p.confident.astype(bool)].mean() - p.hit.mean()))
    return out


def main() -> None:
    """Grid (primary) and weekly (secondary) selective-prediction evaluation."""
    from nss.models.backtest_v2 import identify_lightgbm_origins
    from nss.models.eval_power import LAST_WEEKLY_ORIGIN

    panel = pl.read_parquet("data/processed/style_week_panel.parquet")
    grid = generate_origin_schedule(panel)
    grid_weeks = [o.origin_week for o in identify_lightgbm_origins(panel)]
    weekly_weeks = [
        o.origin_week
        for o in generate_origin_schedule(panel, step_weeks=1)
        if grid_weeks[0] <= o.origin_week <= LAST_WEEKLY_ORIGIN
    ]
    point = growth_backtest.predictions_for_origins(panel, grid, weekly_weeks, INITIAL_POOL_SIZE)
    quant = quantile_frame(panel, grid, weekly_weeks, INITIAL_POOL_SIZE)
    frame = point.join(quant, on=["style_key", "origin_week"], how="left").select(
        "style_key", "origin_week", "y_true", "y_pred_lightgbm", "q10", "q90"
    )
    assert frame["q10"].null_count() == 0
    frame.write_parquet("data/generated/v3_selective_frame.parquet")
    crossed = int((frame["q10"] > frame["q90"]).sum())
    print(f"quantile crossing (q10 > q90) in {crossed} of {frame.height} style-origins")

    on_grid = frame.filter(pl.col("origin_week").is_in(grid_weeks))
    # Gate: the picks are the reported ones, so their pooled hit rate must be the 0.528 headline.
    headline = statistic(build_picks(on_grid))[1]
    if abs(headline - 19 / 36) > 1e-9:
        raise SystemExit(f"picks do not reproduce the 0.528 headline (got {headline})")
    print(f"gate OK: all-picks hit rate at the 12 grid origins = {headline:.6f}")
    rows = evaluate(on_grid, "grid_4w", 4, (0.5, 0.25, 0.75))
    rows += evaluate(frame, "weekly_1w", 13, (0.5, 0.25, 0.75))
    table = pl.DataFrame(rows)
    table.write_csv(f"{OUT}/v3_selective_summary.csv")
    rho, n = spread_error_correlation(on_grid)
    rho_w, n_w = spread_error_correlation(frame)
    pl.DataFrame(
        [
            {"eval_set": "grid_4w", "n_origins": n, "mean_spearman_spread_vs_abs_error": rho},
            {"eval_set": "weekly_1w", "n_origins": n_w, "mean_spearman_spread_vs_abs_error": rho_w},
        ]
    ).write_csv(f"{OUT}/v3_selective_spread_error.csv")
    picks_rows = []
    for p in build_picks(on_grid):
        picks_rows.append(
            {
                "origin_week": p.origin_week,
                "hits": int(p.hit.sum()),
                "confident_picks": int(p.confident.sum()),
                "confident_hits": int((p.hit * p.confident).sum()),
            }
        )
    pl.DataFrame(picks_rows).write_csv(f"{OUT}/v3_selective_picks_grid.csv")
    with pl.Config(
        tbl_rows=30, tbl_width_chars=230, tbl_formatting="ASCII_FULL", float_precision=3
    ):
        print(table.drop("ess_ac_of_per_origin_difference"))
        print(pl.read_csv(f"{OUT}/v3_selective_spread_error.csv"))
    primary = table.filter(pl.col("decision_bearing"))
    print("PRIMARY VERDICT:", primary["verdict"][0])


if __name__ == "__main__":
    main()
