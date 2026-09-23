"""M1/M2: rembg masks and shrunk thresholds, nested leave-one-out, old versus new thresholds.

Rules pre-registered in `reports/v3/PREREGISTRATION.md` (section M, commit 3b63aa9). Compares with
the L1 results in `v3_colour_loo.csv` (border-sampled mask, unshrunk). No concept is scored here.

    uv run --no-sync python scripts/colour_loo_m.py
"""

from __future__ import annotations

import sys

import polars as pl

from nss.generate import colour_check as cc
from nss.generate import qc_gates

TABLES = "reports/tables"
MATERIAL_REL, MATERIAL_ABS = 0.25, 2.0


def main() -> None:
    """Write `v3_colour_loo_m.csv` and the per-reference colours; print the comparison."""
    old = {r["style"]: r for r in pl.read_csv(f"{TABLES}/v3_colour_loo.csv").to_dicts()}
    rows, refs = [], []
    for style in cc.calibrated_styles():
        prof = cc.profile(style)
        raw_loo = cc.loo_pass_rate(style, shrunk=False)
        shr_loo = cc.loo_pass_rate(style, shrunk=True)
        n = prof["n_references"]
        o = old[style]
        rel = abs(prof["raw_threshold"] - o["threshold"]) / o["threshold"]
        rows.append(
            {
                "style": style.split(" || ")[1],
                "n": n,
                "old_threshold_L1": o["threshold"],
                "raw_threshold_rembg": prof["raw_threshold"],
                "shrunk_threshold": prof["threshold"],
                "weight_w": n / (n + cc.K_SHRINKAGE),
                "global_median": cc.global_threshold(),
                "loo_L1_border_unshrunk": o["pass_rate"],
                "loo_rembg_unshrunk": raw_loo["pass_rate"],
                "loo_rembg_shrunk": shr_loo["pass_rate"],
                "fallbacks_L1": o["fallbacks"],
                "fallbacks_rembg": raw_loo["fallbacks"],
                "raw_moved_materially": rel > MATERIAL_REL
                or abs(prof["raw_threshold"] - o["threshold"]) > MATERIAL_ABS,
            }
        )
        for _path, (name, dom) in zip(
            qc_gates.reference_paths_for_style(style), cc._reference_colours(style), strict=True
        ):
            refs.append(
                {
                    "style": style.split(" || ")[1],
                    "image": name,
                    "L": dom.lab[0],
                    "a": dom.lab[1],
                    "b": dom.lab[2],
                    "mask_fraction": dom.mask_fraction,
                    "fallback": dom.used_fallback,
                }
            )
    df = pl.DataFrame(rows)
    df.write_csv(f"{TABLES}/v3_colour_loo_m.csv")
    pl.DataFrame(refs).write_csv(f"{TABLES}/v3_colour_reference_colours_rembg.csv")
    with pl.Config(tbl_cols=-1, tbl_width_chars=250, float_precision=3):
        print(df)
    final = df.filter(pl.col("style") != "Underwear bottom")
    for col in ("loo_L1_border_unshrunk", "loo_rembg_unshrunk", "loo_rembg_shrunk"):
        tot = (final[col] * final["n"]).sum() / final["n"].sum()
        print(f"overall four final styles {col}: {tot:.3f}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # polars prints Unicode; avoid cp1252 crash
    main()
