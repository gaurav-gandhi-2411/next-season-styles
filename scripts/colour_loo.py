"""L1: nested leave-one-out validation of the measured colour check on real reference articles.

Rule and stop rule pre-registered in `reports/v3/PREREGISTRATION.md` (section L1). Writes
`reports/tables/v3_colour_loo.csv` and `v3_colour_reference_colours.csv`; prints the pass rates.
No concept is scored here.

    uv run --no-sync python scripts/colour_loo.py
"""

from __future__ import annotations

import polars as pl

from nss.generate import colour_check, qc_gates
from nss.generate.final_selection_figures import ALL_SELECTED

UNDERWEAR = "Ladieswear || Underwear bottom || Under-, Nightwear || Red || Solid"
STYLES = [*ALL_SELECTED, UNDERWEAR]  # the four final styles, then underwear (reported apart)
STOP_OVERALL, STOP_STYLE = 0.80, 0.75


def main() -> None:
    """Report per-style and overall real-article pass rates and apply the stop rule."""
    rows, refs = [], []
    for style in STYLES:
        r = colour_check.loo_pass_rate(style)
        r["threshold"] = colour_check.profile(style)["threshold"]
        rows.append(r)
        for _path, (name, dom) in zip(
            qc_gates.reference_paths_for_style(style),
            colour_check._reference_colours(style),
            strict=True,
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
    df.write_csv("reports/tables/v3_colour_loo.csv")
    pl.DataFrame(refs).write_csv("reports/tables/v3_colour_reference_colours.csv")
    final = df.filter(pl.col("style") != UNDERWEAR)
    overall = final["passed"].sum() / final["n"].sum()
    for r in rows:
        print(
            f"{r['style'].split(' || ')[1]:18s} n={r['n']:3d} pass={r['passed']:3d} "
            f"rate={r['pass_rate']:.3f} "
            f"threshold={r['threshold']:.2f} fallbacks={r['fallbacks']}"
        )
    print(f"overall (four final styles) {overall:.3f}")
    stop = overall < STOP_OVERALL or bool((final["pass_rate"] < STOP_STYLE).any())
    print("STOP RULE TRIGGERED" if stop else "stop rule not triggered")


if __name__ == "__main__":
    main()
