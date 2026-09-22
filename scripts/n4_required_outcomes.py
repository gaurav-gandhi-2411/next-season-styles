"""N4: score the pre-registered required outcomes (PREREGISTRATION.md section N4) once
W_BAND_PATTERNED has been measured (colour_histogram_noise.py) and the N3 patterned prior exists.

1. The M1 required outcomes must still hold under the mixed pipeline (four solid-class styles on
   the unchanged M1/M2 dominant-colour path, the bikini on N4's histogram path): FAIL the green
   dress and both coral dresses, PASS the submitted dress, submitted sweater and both red dresses.
2. The bikini's real-article nested leave-one-out pass rate under the histogram statistic, shrunk
   toward the N3 patterned-class prior, must be >= 0.80.
3. The noise-band figure is reported separately by colour_histogram_noise.py (already run: p95
   36.962, NOT well below the old dominant-colour p95 of 28.9 -- reported plainly, not adjusted).

No tuning if unmet -- this script only measures and reports.

    uv run --no-sync python scripts/n4_required_outcomes.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate2_measured_identity import EXTRA_REQUIRED, REQUIRED_FAIL, _paths  # noqa: E402

from nss.generate import colour_check as cc  # noqa: E402

TABLES = Path("reports/tables")
BIKINI = "Ladieswear || Bikini top || Swimwear || Orange || All over pattern"
LOO_THRESHOLD = 0.80
OLD_NOISE_P95 = 28.931445  # M3, dominant-colour statistic (v3_colour_mask_noise.csv)


def main() -> None:
    paths = _paths()
    print("=== M1 required outcomes under the mixed (N4) pipeline ===")
    rows: list[dict[str, Any]] = []
    all_ok = True
    for name in REQUIRED_FAIL:
        style, path = paths[name]
        r = cc.check(path, style)
        ok = r["pass"] is False
        all_ok &= ok
        rows.append({"name": name, "required": "FAIL", "got_pass": r["pass"], "met": ok,
                      "method": r["method"], "distance": r["nearest_delta_e"], "threshold": r["threshold"]})
        print(f"{'OK ' if ok else 'FAIL'} {name}: expected FAIL, got pass={r['pass']} "
              f"method={r['method']} d={r['nearest_delta_e']:.3f} T={r['threshold']:.3f}")
    for name, (style, path) in EXTRA_REQUIRED.items():
        r = cc.check(path, style)
        ok = r["pass"] is True
        all_ok &= ok
        rows.append({"name": name, "required": "PASS", "got_pass": r["pass"], "met": ok,
                      "method": r["method"], "distance": r["nearest_delta_e"], "threshold": r["threshold"]})
        print(f"{'OK ' if ok else 'FAIL'} {name}: expected PASS, got pass={r['pass']} "
              f"method={r['method']} d={r['nearest_delta_e']:.3f} T={r['threshold']:.3f}")
    print(f"\nAll M1 required outcomes hold: {bool(all_ok)}")
    pl.DataFrame(rows).write_csv(f"{TABLES}/v3_n4_m1_regression_check.csv")

    print("\n=== Bikini nested LOO, histogram statistic, N3 patterned prior ===")
    loo = cc.loo_pass_rate_hist(BIKINI)
    print(loo)
    loo_met = loo["pass_rate"] >= LOO_THRESHOLD
    print(f"\nRequired >= {LOO_THRESHOLD}: {'MET' if loo_met else 'NOT MET'}")

    print("\n=== For comparison: old dominant-colour LOO on the bikini (M1, unchanged) ===")
    loo_old = cc.loo_pass_rate(BIKINI)
    print(loo_old)

    band_met = cc.W_BAND_PATTERNED is not None and cc.W_BAND_PATTERNED < OLD_NOISE_P95
    summary = pl.DataFrame(
        [
            {
                "outcome": "M1 regression (7 cases)",
                "required": "all hold",
                "met": bool(all_ok),
            },
            {
                "outcome": "bikini LOO (histogram statistic)",
                "required": f">= {LOO_THRESHOLD}",
                "value": loo["pass_rate"],
                "met": bool(loo_met),
            },
            {
                "outcome": "bikini noise band p95 (histogram statistic)",
                "required": f"well below {OLD_NOISE_P95}",
                "value": cc.W_BAND_PATTERNED,
                "met": bool(band_met),
            },
        ]
    )
    summary.write_csv(f"{TABLES}/v3_n4_required_outcomes_summary.csv")
    print("\n=== Summary ===")
    print(summary)


if __name__ == "__main__":
    main()
