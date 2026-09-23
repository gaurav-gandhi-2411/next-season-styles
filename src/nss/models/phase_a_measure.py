"""Phase A (SPEC.md Section 7): demand capture@k, tolerance hit@3, power table, full re-score.

Rules are pre-registered in `reports/v3/PREREGISTRATION.md` (Phase A), committed before any new
metric was computed on any model. Two entry points, run in this order:

    uv run python -m nss.models.phase_a_measure --near-tie   # labels only, feeds the pre-reg
    uv run python -m nss.models.phase_a_measure              # model + baselines + floor

`--near-tie` reads only realised targets (no predictions of any method) and writes the per-origin
gap between the true #3 and #4 styles in raw intensity, from which the tolerance margin is derived.

The default run regenerates the embargoed model's 48 weekly-origin predictions with the locked
`FINAL_MODEL_CONFIG` (no cached frame survived the 2026-09-23 data loss) and refuses to continue
unless every existing metric, for every method at every origin, equals the committed
`v3_power_per_origin_weekly.csv` to 1e-9 -- i.e. the predictions are the ones already reported,
not a new model. It then scores every method on every metric and writes:

- `reports/tables/phase_a_per_origin.csv`: one row per (origin, method), all metrics;
- `reports/tables/phase_a_summary.csv`:    per method and metric, mean and 95% block CI;
- `reports/tables/phase_a_paired.csv`:     model minus each comparator, paired, block 13;
- `reports/tables/phase_a_power.csv`:      CI half-width, ESS, MDE at 80% power vs seasonal-naive;
- `reports/tables/phase_a_primary.csv`:    the pre-registered selection of the Phase B primary.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from datetime import date

import numpy as np
import polars as pl
from scipy.stats import norm

from nss.models import growth_backtest
from nss.models.backtest import (
    BASELINE_METHODS,
    BOOTSTRAP_N_RESAMPLES,
    BOOTSTRAP_SEED,
    Origin,
    build_predictions_frame,
    generate_origin_schedule,
    run_backtest,
)
from nss.models.eval_power import LAST_WEEKLY_ORIGIN, ess_ac, ess_boot, model_rows
from nss.models.lightgbm_model import INITIAL_POOL_SIZE
from nss.models.metrics import METRIC_KEYS, demand_capture_at_k, tolerance_hit_at_k
from nss.models.random_floor import RANDOM_FLOOR_METHOD, RANDOM_FLOOR_SEEDS, permute_predictions

MODEL = growth_backtest.MODEL
COMPARATOR = "seasonal_naive"
BLOCK = 13  # SPEC Section 4 rule 3: block length 13, the forecast horizon in weekly origins
CAPTURE_KS: tuple[int, ...] = (3, 10, 20)
TOL_K = 3
# Pre-registered derivation (PREREGISTRATION.md, Phase A): the margin is the 90th percentile, over
# the 48 weekly origins, of (I3 - I4) / I3 in raw intensity, numpy's default linear interpolation.
# The constant is the value that derivation produced from labels alone, before any model scoring;
# `main` recomputes it and stops if the two ever differ.
MARGIN_QUANTILE = 0.90
MARGIN_PREREGISTERED: float | None = 0.05791057991470009
NEW_METRICS: tuple[str, ...] = (
    *(f"demand_capture_at_{k}" for k in CAPTURE_KS),
    f"tolerance_hit_at_{TOL_K}",
)
# Candidates for the Phase B primary metric, in tie-break order (smaller k first).
CANDIDATES: tuple[str, ...] = (
    f"tolerance_hit_at_{TOL_K}",
    "demand_capture_at_3",
    "hit_at_3_in_top20",
    "demand_capture_at_10",
    "demand_capture_at_20",
)
FLOOR_MAX = 0.5  # relevance check R2: the random floor must sit at most halfway to the oracle's 1.0
MIN_PAIRED_ORIGINS = 8  # relevance check R3, same threshold as backtest_v2's directional-only rule
POWER_Z = float(norm.ppf(0.975) + norm.ppf(0.80))  # two-sided alpha 0.05, 80% power
REPRO_PATH = "reports/tables/v3_power_per_origin_weekly.csv"
NEAR_TIE_OUT = "reports/tables/phase_a_near_tie.csv"
OUT = "reports/tables"


def weekly_origins(panel: pl.DataFrame) -> tuple[list[Origin], list[Origin]]:
    """The 4-week grid and the 48 weekly test origins, exactly as `eval_power` builds them."""
    grid = generate_origin_schedule(panel)
    first = grid[INITIAL_POOL_SIZE].origin_week
    weekly = [
        o
        for o in generate_origin_schedule(panel, step_weeks=1)
        if first <= o.origin_week <= LAST_WEEKLY_ORIGIN
    ]
    assert len(weekly) == 48, len(weekly)
    return grid, weekly


def near_tie_table(eval_frame: pl.DataFrame, origins: Sequence[Origin]) -> pl.DataFrame:
    """Per-origin gap between the true #3 and #4 styles, labels only, log1p and raw intensity."""
    rows = []
    for o in origins:
        y = np.sort(eval_frame.filter(pl.col("origin_week") == o.origin_week)["y_true"].to_numpy())
        y = y[::-1]
        i3, i4 = float(np.expm1(y[2])), float(np.expm1(y[3]))
        rows.append(
            {
                "origin_week": o.origin_week,
                "n_eval_set": y.size,
                "log1p_3": float(y[2]),
                "log1p_4": float(y[3]),
                "gap_log1p_pct": (y[2] - y[3]) / y[2] * 100.0,
                "intensity_3": i3,
                "intensity_4": i4,
                "gap_intensity_frac": (i3 - i4) / i3,
            }
        )
    return pl.DataFrame(rows)


def derive_margin(near_tie: pl.DataFrame) -> float:
    """The pre-registered margin: the MARGIN_QUANTILE of the raw-intensity #3-vs-#4 gap."""
    return float(np.quantile(near_tie["gap_intensity_frac"].to_numpy(), MARGIN_QUANTILE))


def new_metric_values(y_true: np.ndarray, y_pred: np.ndarray, margin: float) -> dict[str, float]:
    """The Phase A metrics for one origin and one method."""
    out = {f"demand_capture_at_{k}": demand_capture_at_k(y_true, y_pred, k) for k in CAPTURE_KS}
    out[f"tolerance_hit_at_{TOL_K}"] = tolerance_hit_at_k(y_true, y_pred, TOL_K, margin)
    return out


def score_new_metrics(
    preds: pl.DataFrame, model_frame: pl.DataFrame, origins: Sequence[Origin], margin: float
) -> pl.DataFrame:
    """New metrics per (origin, method), on each method's own non-null population.

    Same population convention as `run_backtest`: a baseline is scored on the styles where its
    prediction is non-null (seasonal-naive drops styles without 52 weeks of history), the model on
    its full eval set, the random floor on the full eval set averaged over RANDOM_FLOOR_SEEDS.
    """
    rows = []
    for o in origins:
        sub = preds.filter(pl.col("origin_week") == o.origin_week)
        for method in BASELINE_METHODS:
            s = sub.filter(pl.col(f"y_pred_{method}").is_not_null())
            vals = (
                new_metric_values(s["y_true"].to_numpy(), s[f"y_pred_{method}"].to_numpy(), margin)
                if s.height
                else dict.fromkeys(NEW_METRICS, float("nan"))
            )
            rows.append({"origin_week": o.origin_week, "method": method, **vals})
        m = model_frame.filter(pl.col("origin_week") == o.origin_week)
        rows.append(
            {
                "origin_week": o.origin_week,
                "method": MODEL,
                **new_metric_values(
                    m["y_true"].to_numpy(), m[f"y_pred_{MODEL}"].to_numpy(), margin
                ),
            }
        )
        y = sub["y_true"].to_numpy()
        per_seed = [
            new_metric_values(y, permute_predictions(y, s), margin) for s in RANDOM_FLOOR_SEEDS
        ]
        rows.append(
            {
                "origin_week": o.origin_week,
                "method": RANDOM_FLOOR_METHOD,
                **{k: float(np.mean([d[k] for d in per_seed])) for k in NEW_METRICS},
            }
        )
    return pl.DataFrame(rows)


def block_means(
    values: Sequence[float],
    block: int = BLOCK,
    n_resamples: int = BOOTSTRAP_N_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> np.ndarray:
    """Moving-block bootstrap means, the same RNG draws as `backtest.block_bootstrap_ci`."""
    x = np.asarray([v for v in values if not np.isnan(v)], dtype=float)
    n = x.size
    rng = np.random.default_rng(seed)
    blocks_needed = -(-n // block)
    max_start = max(n - block, 0)
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        starts = rng.integers(0, max_start + 1, size=blocks_needed)
        means[i] = np.concatenate([x[s : s + block] for s in starts])[:n].mean()
    return means


def paired_rows(combined: pl.DataFrame, metrics: Sequence[str]) -> pl.DataFrame:
    """Model minus each comparator, per metric: mean, 95% block CI, half-width, ESS, MDE@80%."""
    a = combined.filter(pl.col("method") == MODEL).sort("origin_week")
    rows = []
    for comp in (*BASELINE_METHODS, RANDOM_FLOOR_METHOD):
        b = combined.filter(pl.col("method") == comp).sort("origin_week")
        j = a.join(b, on="origin_week", suffix="_b")
        for m in metrics:
            d = (j[m] - j[f"{m}_b"]).fill_nan(None).drop_nulls().to_numpy()
            if d.size < 2:
                rows.append({"comparator": comp, "metric": m, "n_paired": int(d.size)})
                continue
            means = block_means(d)
            lo, hi = np.percentile(means, [2.5, 97.5])
            se = float(np.std(means, ddof=1))
            rows.append(
                {
                    "comparator": comp,
                    "metric": m,
                    "n_paired": int(d.size),
                    "mean_diff": float(d.mean()),
                    "ci_lo": float(lo),
                    "ci_hi": float(hi),
                    "ci_half_width": float(hi - lo) / 2.0,
                    "se_block": se,
                    "ess_ac": ess_ac(d.tolist()),
                    "ess_boot": ess_boot(d.tolist(), BLOCK),
                    "mde_80": POWER_Z * se,
                    "excludes_zero": bool(lo > 0.0 or hi < 0.0),
                }
            )
    return pl.DataFrame(rows)


def summary_rows(combined: pl.DataFrame, metrics: Sequence[str]) -> pl.DataFrame:
    """Per method and metric: n origins scored, mean, 95% block CI (block 13)."""
    rows = []
    for method in combined["method"].unique(maintain_order=True).to_list():
        sub = combined.filter(pl.col("method") == method).sort("origin_week")
        for m in metrics:
            v = sub[m].fill_nan(None).drop_nulls().to_numpy()
            if v.size == 0:
                rows.append({"method": method, "metric": m, "n_origins": 0})
                continue
            lo, hi = np.percentile(block_means(v), [2.5, 97.5])
            rows.append(
                {
                    "method": method,
                    "metric": m,
                    "n_origins": int(v.size),
                    "mean": float(v.mean()),
                    "ci_lo": float(lo),
                    "ci_hi": float(hi),
                }
            )
    return pl.DataFrame(rows)


def select_primary(power: pl.DataFrame, summary: pl.DataFrame) -> pl.DataFrame:
    """Apply the pre-registered rule: among candidates passing R1-R3, smallest normalised MDE.

    R1 (decision alignment) holds by construction for every entry of CANDIDATES. The normalised MDE
    is MDE / (oracle - floor) with oracle = 1 for every candidate, so metrics on different scales
    compete on the fraction of their achievable range. Only dispersion enters: the point estimate
    of the model's lead plays no part in the choice.
    """
    rows = []
    for rank, m in enumerate(CANDIDATES):
        p = power.filter(pl.col("metric") == m).row(0, named=True)
        floor = summary.filter((pl.col("method") == RANDOM_FLOOR_METHOD) & (pl.col("metric") == m))[
            "mean"
        ][0]
        r2 = floor <= FLOOR_MAX
        r3 = p["n_paired"] >= MIN_PAIRED_ORIGINS and (p.get("se_block") or 0.0) > 0.0
        rows.append(
            {
                "metric": m,
                "tie_break_rank": rank,
                "floor_mean": floor,
                "r2_floor_le_half": r2,
                "r3_paired_ok": r3,
                "passes": r2 and r3,
                "mde_80": p.get("mde_80"),
                "mde_80_normalised": p["mde_80"] / (1.0 - floor) if p.get("mde_80") else None,
            }
        )
    table = pl.DataFrame(rows)
    passing = table.filter(pl.col("passes")).with_columns(
        pl.col("mde_80_normalised").round(3).alias("_key")
    )
    chosen = passing.sort(["_key", "tie_break_rank"])["metric"][0]
    return table.with_columns((pl.col("metric") == chosen).alias("chosen"))


def check_reproduction(combined: pl.DataFrame) -> int:
    """Every existing metric of every method at every origin must equal the committed table."""
    ref = pl.read_csv(REPRO_PATH, try_parse_dates=True)
    j = combined.join(ref, on=["origin_week", "method"], suffix="_ref")
    assert j.height == ref.height == combined.height, (j.height, ref.height, combined.height)
    for m in METRIC_KEYS:
        a, b = j[m].to_numpy(), j[f"{m}_ref"].to_numpy()
        both_nan = np.isnan(a) & np.isnan(b)
        if not (both_nan | np.isclose(a, b, rtol=0, atol=1e-9)).all():
            raise SystemExit(f"reproduction failed on {m}")
    return j.height


def run_near_tie() -> None:
    """Labels only: write the per-origin #3-vs-#4 gaps and print the derived margin."""
    panel = pl.read_parquet("data/processed/style_week_panel.parquet")
    _, weekly = weekly_origins(panel)
    frame = build_predictions_frame(panel, [o.origin_week for o in weekly])
    table = near_tie_table(frame.select("style_key", "origin_week", "y_true"), weekly)
    table.write_csv(NEAR_TIE_OUT)
    g, lg = table["gap_intensity_frac"].to_numpy(), table["gap_log1p_pct"].to_numpy()
    print(f"origins={table.height}")
    print(f"log1p gap %: mean={lg.mean():.4f} median={np.median(lg):.4f}")
    for q in (0.5, 0.75, 0.9, 0.95):
        print(f"intensity gap q{q:.2f} = {np.quantile(g, q):.6f}")
    print(f"intensity gap mean={g.mean():.6f} max={g.max():.6f}")
    print(f"margin (q{MARGIN_QUANTILE}) = {derive_margin(table):.6f}")


def main() -> None:
    """Re-score model, baselines and floor on every metric; power table; primary selection."""
    panel = pl.read_parquet("data/processed/style_week_panel.parquet")
    grid, weekly = weekly_origins(panel)
    weeks = [o.origin_week for o in weekly]

    near_tie = near_tie_table(build_predictions_frame(panel, weeks), weekly)
    margin = derive_margin(near_tie)
    if MARGIN_PREREGISTERED is None or not np.isclose(margin, MARGIN_PREREGISTERED, atol=1e-12):
        raise SystemExit(f"margin {margin} != pre-registered {MARGIN_PREREGISTERED}")

    model_frame = growth_backtest.predictions_for_origins(panel, grid, weeks, INITIAL_POOL_SIZE)
    model_frame.write_parquet(f"data/generated/phase_a_model_predictions_{_stamp(weeks)}.parquet")
    preds = build_predictions_frame(panel, weeks)
    from nss.models.random_floor import (
        aggregate_random_floor_over_seeds,
        score_random_floor_per_origin,
    )

    old = pl.concat(
        [
            run_backtest(panel, weekly),
            model_rows(model_frame, weekly),
            aggregate_random_floor_over_seeds(score_random_floor_per_origin(preds, weekly)),
        ],
        how="diagonal",
    ).select(
        "origin_week", "method", "has_52w_lag", "is_covid", "n_eval_set", "n_eval", *METRIC_KEYS
    )
    print(f"reproduction OK: {check_reproduction(old)} (origin, method) rows at 1e-9")

    new = score_new_metrics(preds, model_frame, weekly, margin)
    combined = old.join(new, on=["origin_week", "method"], how="left").sort("method", "origin_week")
    metrics = (*METRIC_KEYS, *NEW_METRICS)
    combined.write_csv(f"{OUT}/phase_a_per_origin.csv")
    summary = summary_rows(combined, metrics)
    summary.write_csv(f"{OUT}/phase_a_summary.csv")
    paired = paired_rows(combined, metrics)
    paired.write_csv(f"{OUT}/phase_a_paired.csv")
    power = paired.filter(pl.col("comparator") == COMPARATOR)
    power.write_csv(f"{OUT}/phase_a_power.csv")
    primary = select_primary(power, summary)
    primary.write_csv(f"{OUT}/phase_a_primary.csv")

    with pl.Config(tbl_rows=200, tbl_cols=20, tbl_width_chars=220, float_precision=4):
        print(f"margin = {margin:.6f}")
        print(power.drop("comparator"))
        print(summary.pivot(on="method", index="metric", values="mean"))
        print(primary)


def _stamp(weeks: Sequence[date]) -> str:
    """Filename tag: origin span and count, e.g. `w48_2019-07-29_2020-06-22`."""
    return f"w{len(weeks)}_{weeks[0]}_{weeks[-1]}"


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # polars prints Unicode tables
    run_near_tie() if "--near-tie" in sys.argv else main()
