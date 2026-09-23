"""B.0: re-baseline Phase A under the circular block bootstrap (PREREGISTRATION.md Section Q).

Reads the per-origin metrics already in `reports/tables/phase_a_per_origin.csv` (nothing is
re-predicted) and recomputes every interval with `nss.models.circular_bootstrap`. Writes:

- `reports/tables/phase_b0_summary.csv`: per method and metric, old vs circular interval;
- `reports/tables/phase_b0_paired.csv`:  model minus each comparator, old vs circular interval,
  circular SE, ESS and MDE@80%.

It then applies the Q3 stop condition and exits non-zero if it fires.

    uv run python -m nss.models.phase_b0_rebaseline
"""

from __future__ import annotations

import sys

import numpy as np
import polars as pl

from nss.models.circular_bootstrap import circular_block_means, circular_ci
from nss.models.eval_power import ess_ac
from nss.models.metrics import METRIC_KEYS
from nss.models.phase_a_measure import MODEL, NEW_METRICS

PER_ORIGIN = "reports/tables/phase_a_per_origin.csv"
OLD_SUMMARY = "reports/tables/phase_a_summary.csv"
OLD_PAIRED = "reports/tables/phase_a_paired.csv"
OUT = "reports/tables"
METRICS: tuple[str, ...] = (*METRIC_KEYS, *NEW_METRICS)
PRIMARY = "demand_capture_at_20"
COMPARATOR = "seasonal_naive"


def ess_circular(diff: np.ndarray) -> float:
    """n * Var_iid(mean) / Var_circular(mean), floored at 1, capped at n."""
    n = diff.size
    if n < 3 or np.var(diff) == 0.0:
        return float(n)
    var_c = float(np.var(circular_block_means(diff), ddof=1))
    var_iid = float(np.var(diff, ddof=1)) / n
    return float(min(n, max(1.0, n * var_iid / var_c))) if var_c > 0 else float(n)


def summary(per_origin: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for method in per_origin["method"].unique(maintain_order=True).to_list():
        sub = per_origin.filter(pl.col("method") == method).sort("origin_week")
        for m in METRICS:
            c = circular_ci(sub[m].fill_nan(None).drop_nulls().to_list())
            if c["n"]:
                rows.append({"method": method, "metric": m, **c})
    new = pl.DataFrame(rows).select(
        "method",
        "metric",
        "n",
        "mean",
        pl.col("ci_lo").alias("circ_lo"),
        pl.col("ci_hi").alias("circ_hi"),
    )
    old = pl.read_csv(OLD_SUMMARY).select(
        "method", "metric", pl.col("ci_lo").alias("old_lo"), pl.col("ci_hi").alias("old_hi")
    )
    return new.join(old, on=["method", "metric"], how="left").select(
        "method", "metric", "n", "mean", "old_lo", "old_hi", "circ_lo", "circ_hi"
    )


def paired(per_origin: pl.DataFrame) -> pl.DataFrame:
    a = per_origin.filter(pl.col("method") == MODEL).sort("origin_week")
    rows = []
    for comp in per_origin["method"].unique(maintain_order=True).to_list():
        if comp == MODEL:
            continue
        j = a.join(per_origin.filter(pl.col("method") == comp), on="origin_week", suffix="_b")
        for m in METRICS:
            d = (j[m] - j[f"{m}_b"]).fill_nan(None).drop_nulls().to_numpy()
            if d.size < 2:
                continue
            c = circular_ci(d)
            rows.append(
                {
                    "comparator": comp,
                    "metric": m,
                    "n_paired": c["n"],
                    "mean_diff": c["mean"],
                    "circ_lo": c["ci_lo"],
                    "circ_hi": c["ci_hi"],
                    "circ_se": c["se"],
                    "ess_ac": ess_ac(d.tolist()),
                    "ess_circ": ess_circular(d),
                    "mde_80": c["mde_80"],
                    "excludes_zero": bool(c["ci_lo"] > 0 or c["ci_hi"] < 0),
                }
            )
    old = pl.read_csv(OLD_PAIRED).select(
        "comparator", "metric", pl.col("ci_lo").alias("old_lo"), pl.col("ci_hi").alias("old_hi")
    )
    return pl.DataFrame(rows).join(old, on=["comparator", "metric"], how="left")


def main() -> None:
    per_origin = pl.read_csv(PER_ORIGIN, try_parse_dates=True)
    s = summary(per_origin)
    s.write_csv(f"{OUT}/phase_b0_summary.csv")
    p = paired(per_origin)
    p.write_csv(f"{OUT}/phase_b0_paired.csv")
    with pl.Config(tbl_rows=200, tbl_width_chars=220, float_precision=3):
        print(s.filter(pl.col("method") == MODEL))
        print(
            p.filter(pl.col("comparator") == COMPARATOR).select(
                "metric",
                "n_paired",
                "mean_diff",
                "old_lo",
                "old_hi",
                "circ_lo",
                "circ_hi",
                "ess_circ",
                "mde_80",
            )
        )
    row = p.filter((pl.col("comparator") == COMPARATOR) & (pl.col("metric") == PRIMARY)).row(
        0, named=True
    )
    verdict = "PROCEED" if row["circ_lo"] > 0 else "STOP"
    print(
        f"Q3 stop condition: {PRIMARY} lead over {COMPARATOR} = {row['mean_diff']:.4f} "
        f"[{row['circ_lo']:.4f}, {row['circ_hi']:.4f}] -> {verdict}"
    )
    if verdict == "STOP":
        raise SystemExit(1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
