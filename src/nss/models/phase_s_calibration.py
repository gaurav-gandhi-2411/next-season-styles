"""Section S: rolling conformalised quantile regression on the B.5 quantile models.

Pre-registered in `reports/v3/PREREGISTRATION.md` Section S (`a76adae`) before any calibrated
interval was computed.

    uv run python -m nss.models.phase_s_calibration run        # q10/q90 incl. calibration history
    uv run python -m nss.models.phase_s_calibration evaluate   # coverage tables + figure

`run` serves the unchanged B.5 q10 and q90 models at the 48 test origins plus 16 calibration-only
weekly origins before them (2019-04-08 .. 2019-07-22), through `phase_b.serve` (the embargoed
grid-block rule), and stops unless the test-origin predictions equal the B.5 files exactly.
`evaluate` calibrates each test origin on the 4 most recent weekly origins whose outcomes closed
13 weeks before it, for a symmetric and an asymmetric CQR, and reports coverage and width.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from nss.features.targets import HORIZON_WEEKS
from nss.models.backtest import Origin, generate_origin_schedule
from nss.models.circular_bootstrap import circular_ci
from nss.models.lightgbm_model import feature_columns
from nss.models.phase_a_measure import _stamp, weekly_origins
from nss.models.phase_b import GEN, OUT, PANEL, base_frame, lever_path, serve

ALPHA = 0.20  # nominal miscoverage of the q10-q90 interval
WINDOW = 4  # calibration origins per test origin (Section S2)
CAL_START = date(2019, 4, 8)
TARGET = (0.75, 0.85)
METHODS: tuple[str, ...] = ("uncalibrated", "cqr_symmetric", "cqr_asymmetric")
FIGURE = "reports/figures/phase_s_coverage_over_time.png"


def calibration_origins(panel: pl.DataFrame, first_test: date) -> list[Origin]:
    return [
        o
        for o in generate_origin_schedule(panel, step_weeks=1)
        if CAL_START <= o.origin_week < first_test
    ]


def quantile_path(q: int, weeks: Sequence[date]) -> str:
    return str(GEN / f"phase_s_b5_q{q}_seed42_{_stamp(weeks)}.parquet")


def run() -> None:
    panel = pl.read_parquet(PANEL)
    grid, weekly = weekly_origins(panel)
    test_weeks = [o.origin_week for o in weekly]
    cal = calibration_origins(panel, test_weeks[0])
    assert len(cal) == 16, len(cal)
    weeks = [*(o.origin_week for o in cal), *test_weeks]
    frame = base_frame(panel, grid, weeks)
    columns = feature_columns(frame)
    for q in (10, 90):
        preds = serve(frame, columns, grid, weeks, objective="quantile", alpha=q / 100)
        preds = preds.join(
            frame.select("style_key", "origin_week", "y_true"), on=["style_key", "origin_week"]
        )
        ref = pl.read_parquet(lever_path(f"b5_q{q}", test_weeks))
        j = ref.join(preds, on=["style_key", "origin_week"], suffix="_s")
        diff = float(np.max(np.abs(j["y_pred"].to_numpy() - j["y_pred_s"].to_numpy())))
        if j.height != ref.height or diff != 0.0:
            raise SystemExit(f"q{q}: test-origin predictions differ from B.5 (max {diff})")
        preds.write_parquet(quantile_path(q, weeks))
        print(f"q{q}: {preds.height} rows, test origins reproduce B.5 exactly")


def conformal_q(scores: np.ndarray, level: float) -> float:
    """The k-th smallest score, k = ceil(level * (n + 1)), capped at the maximum (k > n).

    This is the Section S1 formula. The first run used `np.quantile(scores, k / n,
    method="higher")`, which the pre-registration also named, but numpy indexes over `n - 1`, so
    that call returns the (k+1)-th smallest score: one order statistic too high. Corrected here;
    the old and new results are both reported in `PHASE_S_T_calibration_and_exploratory.md`.
    """
    n = scores.size
    k = min(n, int(np.ceil(level * (n + 1))))
    return float(np.sort(scores)[k - 1])


def window_for(t: date, history: Sequence[date]) -> list[date]:
    """The WINDOW most recent origins whose 13-week outcome closed by `t`."""
    eligible = [c for c in history if c + timedelta(weeks=HORIZON_WEEKS) <= t]
    return eligible[-WINDOW:]


def calibrate(q: pl.DataFrame, test: Sequence[Origin]) -> pl.DataFrame:
    """Per test row: the three intervals and whether each covers y_true."""
    history = q["origin_week"].unique().sort().to_list()
    parts = []
    for o in test:
        win = window_for(o.origin_week, history)
        assert len(win) == WINDOW, (o.origin_week, win)
        c = q.filter(pl.col("origin_week").is_in(win))
        y, lo, hi = (c[k].to_numpy() for k in ("y_true", "q10", "q90"))
        q_sym = conformal_q(np.maximum(lo - y, y - hi), 1 - ALPHA)
        q_lo = conformal_q(lo - y, 1 - ALPHA / 2)
        q_hi = conformal_q(y - hi, 1 - ALPHA / 2)
        t = q.filter(pl.col("origin_week") == o.origin_week).with_columns(
            pl.lit(o.is_covid).alias("is_covid"),
            pl.lit(q_sym).alias("Q_sym"),
            pl.lit(q_lo).alias("Q_lo"),
            pl.lit(q_hi).alias("Q_hi"),
            pl.lit(str(win[0])).alias("cal_first"),
            pl.lit(str(win[-1])).alias("cal_last"),
        )
        parts.append(t)
    rows = pl.concat(parts)
    bounds = {
        "uncalibrated": (pl.col("q10"), pl.col("q90")),
        "cqr_symmetric": (pl.col("q10") - pl.col("Q_sym"), pl.col("q90") + pl.col("Q_sym")),
        "cqr_asymmetric": (pl.col("q10") - pl.col("Q_lo"), pl.col("q90") + pl.col("Q_hi")),
    }
    exprs = []
    for m, (lo_e, hi_e) in bounds.items():
        exprs += [
            ((lo_e <= pl.col("y_true")) & (pl.col("y_true") <= hi_e) & (lo_e <= hi_e)).alias(
                f"cov_{m}"
            ),
            (hi_e - lo_e).alias(f"wlog_{m}"),
            (hi_e.exp() - lo_e.exp()).alias(f"wraw_{m}"),  # expm1(hi) - expm1(lo)
        ]
    return rows.with_columns(exprs)


def evaluate() -> None:
    panel = pl.read_parquet(PANEL)
    _, weekly = weekly_origins(panel)
    test_weeks = [o.origin_week for o in weekly]
    cal = calibration_origins(panel, test_weeks[0])
    weeks = [*(o.origin_week for o in cal), *test_weeks]
    lo = pl.read_parquet(quantile_path(10, weeks)).rename({"y_pred": "q10"})
    hi = pl.read_parquet(quantile_path(90, weeks)).select(
        "style_key", "origin_week", pl.col("y_pred").alias("q90")
    )
    rows = calibrate(lo.join(hi, on=["style_key", "origin_week"]), weekly)

    per_origin = (
        rows.group_by("origin_week", "is_covid", "cal_first", "cal_last", "Q_sym", "Q_lo", "Q_hi")
        .agg(
            pl.len().alias("n"),
            *[pl.col(f"cov_{m}").mean() for m in METHODS],
            *[pl.col(f"wlog_{m}").mean() for m in METHODS],
            *[pl.col(f"wraw_{m}").median() for m in METHODS],
        )
        .sort("origin_week")
    )
    per_origin.write_csv(f"{OUT}/phase_s_coverage_per_origin.csv")

    summary = []
    for m in METHODS:
        s = {"method": m}
        for regime, sub in (
            ("all", rows),
            ("covid", rows.filter(pl.col("is_covid"))),
            ("non_covid", rows.filter(~pl.col("is_covid"))),
        ):
            s[f"pooled_{regime}"] = float(sub[f"cov_{m}"].mean())
        ci = circular_ci(per_origin[f"cov_{m}"].to_list())
        s |= {
            "per_origin_mean": ci["mean"],
            "per_origin_ci_lo": ci["ci_lo"],
            "per_origin_ci_hi": ci["ci_hi"],
            "per_origin_min": float(per_origin[f"cov_{m}"].min()),
            "per_origin_max": float(per_origin[f"cov_{m}"].max()),
            "mean_width_log1p": float(rows[f"wlog_{m}"].mean()),
            "median_width_raw": float(rows[f"wraw_{m}"].median()),
            "meets_target": TARGET[0] <= float(rows[f"cov_{m}"].mean()) <= TARGET[1],
        }
        summary.append(s)
    table = pl.DataFrame(summary)
    passing = table.filter(pl.col("meets_target") & (pl.col("method") != "uncalibrated"))
    chosen = passing.sort("mean_width_log1p")["method"][0] if passing.height else None
    table = table.with_columns((pl.col("method") == (chosen or "")).alias("chosen"))
    table.write_csv(f"{OUT}/phase_s_summary.csv")
    Path(OUT, "phase_s_choice.json").write_text(
        json.dumps({"chosen": chosen, "n_test_origins": per_origin.height}, indent=2)
    )
    plot(per_origin)
    with pl.Config(tbl_rows=60, tbl_cols=20, tbl_width_chars=240, float_precision=3):
        print(table)
        print(
            per_origin.select(
                "origin_week", "is_covid", *[f"cov_{m}" for m in METHODS], "Q_sym", "Q_lo", "Q_hi"
            )
        )
    print(f"chosen: {chosen}")


def plot(per_origin: pl.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = per_origin["origin_week"].to_list()
    fig, ax = plt.subplots(figsize=(9, 3.6), dpi=150)
    covid = [d for d, c in zip(x, per_origin["is_covid"].to_list(), strict=True) if c]
    if covid:
        ax.axvspan(covid[0], covid[-1], color="#e8e8e8", lw=0, label="COVID horizon")
    ax.axhspan(*TARGET, color="#cfe3f5", lw=0, label="target 0.75-0.85")
    ax.axhline(0.80, color="#7a9cc0", lw=0.8, ls="--")
    styles = {
        "uncalibrated": ("#6f6f6f", "raw q10-q90 (reference)"),
        "cqr_symmetric": ("#1f5f99", "CQR symmetric"),
        "cqr_asymmetric": ("#c2571a", "CQR asymmetric"),
    }
    for m, (col, lab) in styles.items():
        ls = "--" if m == "uncalibrated" else "-"  # neutral reference line, dashed not coloured
        ax.plot(x, per_origin[f"cov_{m}"].to_list(), color=col, lw=1.6, ls=ls, label=lab)
    ax.set_ylim(0.3, 1.0)
    ax.set_ylabel("coverage per origin")
    ax.set_title("q10-q90 coverage by weekly origin (rolling, embargoed calibration)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, frameon=False, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.32))
    fig.tight_layout()
    fig.savefig(FIGURE)
    plt.close(fig)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    {"run": run, "evaluate": evaluate}[sys.argv[1]]()
