"""Section T: EXPLORATORY market-momentum check. Not confirmatory; nothing here is adopted.

The hypothesis comes from B.1's random-neighbour control, measured on the same 48 origins these
arms use, so every result is exploratory by construction (PREREGISTRATION.md Section T,
`a76adae`, which fixes the arms and reading rules but confers no confirmatory status). Every
output file is named `phase_t_exploratory_*` and carries `status = EXPLORATORY`.

    uv run python -m nss.models.phase_t_exploratory run <arm>   # a, a_shuffle, a_negctl, b, c
    uv run python -m nss.models.phase_t_exploratory evaluate
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

from nss.models.lightgbm_model import feature_columns
from nss.models.phase_a_measure import _stamp, weekly_origins
from nss.models.phase_b import (
    GEN,
    GUARDRAILS,
    OUT,
    PANEL,
    PRIMARY,
    add_vis_feature,
    base_frame,
    lever_path,
    load_champion,
    paired,
    score_frame,
    serve,
    vis_momentum,
)

SEED = 42
STATUS = "EXPLORATORY"
ARMS: tuple[str, ...] = ("a", "a_shuffle", "a_negctl", "b", "c")
MKT_COLS = ["mkt_slope_4w", "mkt_slope_13w"]


def market_momentum(panel: pl.DataFrame) -> pl.DataFrame:
    """Per week: 2-point trailing slopes of the mean raw intensity over styles active that week."""
    m = (
        panel.group_by("week_start")
        .agg(pl.col("units_per_active_article").mean().alias("M"))
        .sort("week_start")
    )
    return m.select(
        pl.col("week_start").alias("origin_week"),
        ((pl.col("M") - pl.col("M").shift(4)) / 4).alias("mkt_slope_4w"),
        ((pl.col("M") - pl.col("M").shift(13)) / 13).alias("mkt_slope_13w"),
    )


def arm_path(arm: str, weeks: list) -> Path:
    return GEN / f"phase_t_exploratory_{arm}_seed{SEED}_{_stamp(weeks)}.parquet"


def run(arm: str) -> None:
    panel = pl.read_parquet(PANEL)
    grid, weekly = weekly_origins(panel)
    weeks = [o.origin_week for o in weekly]
    frame = base_frame(panel, grid, weeks)
    columns = feature_columns(frame)
    rng = np.random.default_rng(SEED)
    origins = frame["origin_week"].unique().sort().to_list()

    if arm.startswith("a"):
        mkt = market_momentum(panel).filter(pl.col("origin_week").is_in(origins))
        if arm == "a_shuffle":  # whole origin-level vectors moved to other origins
            perm = rng.permutation(mkt.height)
            mkt = mkt.select("origin_week").hstack(mkt.select(MKT_COLS)[perm.tolist()])
        elif arm == "a_negctl":  # origin-level Gaussian noise, matched mean and SD per column
            mkt = mkt.select("origin_week").with_columns(
                [
                    pl.Series(c, rng.normal(mkt[c].mean(), mkt[c].std(), size=mkt.height))
                    for c in MKT_COLS
                ]
            )
        f2 = frame.join(mkt, on="origin_week", how="left", maintain_order="left")
        preds = serve(f2, [*columns, *MKT_COLS], grid, weeks)
    elif arm in ("b", "c"):
        feat = vis_momentum(panel, origins, "random")
        f2 = add_vis_feature(frame, feat, shuffle=False)
        if arm == "b":  # global permutation: no style- or origin-level alignment survives
            f2 = f2.with_columns(
                pl.Series("vis_nbr_momentum", rng.permutation(f2["vis_nbr_momentum"].to_numpy()))
            )
        preds = serve(f2, [*columns, "vis_nbr_momentum"], grid, weeks)
    else:
        raise SystemExit(f"unknown arm {arm}")
    preds.write_parquet(arm_path(arm, weeks))
    print(f"[{STATUS}] arm {arm}: {preds.height} rows -> {arm_path(arm, weeks)}")


def evaluate() -> None:
    panel = pl.read_parquet(PANEL)
    _, weekly = weekly_origins(panel)
    weeks = [o.origin_week for o in weekly]
    champ = load_champion()
    eval_set = champ.select("style_key", "origin_week", "y_true", "n_active_articles_level")

    old_c = pl.read_parquet(lever_path("b1_random", weeks))
    new_c = pl.read_parquet(arm_path("c", weeks))
    jc = old_c.join(new_c, on=["style_key", "origin_week"], suffix="_new")
    c_diff = float(np.max(np.abs(jc["y_pred"].to_numpy() - jc["y_pred_new"].to_numpy())))
    reproduces = jc.height == old_c.height and c_diff == 0.0

    per = [score_frame(eval_set, champ.select("style_key", "origin_week", "y_pred"), "champion")]
    for arm in ARMS:
        per.append(score_frame(eval_set, pl.read_parquet(arm_path(arm, weeks)), arm))
    per_origin = pl.concat(per).with_columns(pl.lit(STATUS).alias("status"))
    per_origin.write_csv(f"{OUT}/phase_t_exploratory_per_origin.csv")
    pairs = pl.DataFrame([r for a in ARMS for r in paired(per_origin, a)]).with_columns(
        pl.lit(STATUS).alias("status")
    )
    pairs.write_csv(f"{OUT}/phase_t_exploratory_paired.csv")

    prim = pairs.filter(pl.col("metric") == PRIMARY)

    def helps(a: str) -> bool:
        return bool(prim.filter(pl.col("challenger") == a)["ci_lo"][0] > 0)

    h = {a: helps(a) for a in ARMS}
    if h["a"] and not h["b"] and not h["a_shuffle"] and not h["a_negctl"]:
        reading = "market signal (exploratory evidence)"
    elif h["a"] and h["b"]:
        reading = "regularisation artefact"
    elif not h["a"] and not h["b"]:
        reading = "B.1 result does not reproduce as a market effect"
    else:
        reading = "inconclusive"
    verdict = {
        "status": STATUS,
        "arm_c_reproduces_b1_random": reproduces,
        "arm_c_max_abs_diff": c_diff,
        "helps": h,
        "reading": reading,
    }
    Path(OUT, "phase_t_exploratory_verdict.json").write_text(json.dumps(verdict, indent=2))
    with pl.Config(tbl_rows=60, tbl_width_chars=200, float_precision=4):
        print(
            pairs.filter(pl.col("metric").is_in([PRIMARY, *GUARDRAILS])).select(
                "challenger", "metric", "mean_diff", "ci_lo", "ci_hi", "p_two_sided"
            )
        )
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if sys.argv[1] == "run":
        run(sys.argv[2])
    else:
        evaluate()
