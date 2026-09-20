"""Run the real embargoed rolling-origin backtest on a small synthetic panel (CPU, no downloads).

`python -m nss.demo_backtest` needs neither the H&M dataset nor a GPU. It reads a committed toy
sparse style-week panel (`src/nss/demo_data/toy_sparse_panel.parquet`) and pushes it through the
same code the reported results come from:

    densify_panel -> add_price_index -> add_intensity_shrunk    (nss.features.style_panel)
    generate_origin_schedule, run_backtest (4 baselines)         (nss.models.backtest)
    run_embargoed_walk_forward (LightGBM, 13-week embargo)       (nss.models.backtest_embargo_check)
    score_random_floor_per_origin (random-permutation floor)     (nss.models.random_floor)
    summarize_backtest, paired_diff_table (bootstrap CIs)        (nss.models.backtest, backtest_v2)

Nothing here reimplements the harness. This module only builds the toy input and prints the output.

THE NUMBERS ARE FROM SYNTHETIC DATA. They show that the machinery runs and behaves sensibly (the
random floor is beaten, the embargo holds); they are not the reported results and say nothing about
H&M. The reported results are in `reports/tables/backtest_embargo_*.csv` and `reports/WRITEUP.md`.

The toy generator (`make_toy_sparse_panel`, seed 42) mimics the real panel's shape: 106 weeks
starting 2018-09-17 (so the origin schedule matches the real one: 20 origins, 12 test origins),
styles that launch and retire at different times, annual seasonality with a per-product-type phase,
slow trends and persistent demand shocks. `--regenerate` rewrites the committed fixture from it.
"""

from __future__ import annotations

import argparse
import time
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from nss.features.style_panel import (
    STYLE_KEY_COLS,
    STYLE_KEY_SEPARATOR,
    add_intensity_shrunk,
    add_price_index,
    densify_panel,
)
from nss.features.targets import HORIZON_WEEKS
from nss.models.backtest import (
    BASELINE_METHODS,
    build_predictions_frame,
    generate_origin_schedule,
    run_backtest,
    summarize_backtest,
)
from nss.models.backtest_embargo_check import (
    embargoed_train_origin_weeks,
    run_embargoed_walk_forward,
)
from nss.models.backtest_v2 import paired_diff_table
from nss.models.lightgbm_model import HYPERPARAM_GRID, INITIAL_POOL_SIZE, METHOD_NAME
from nss.models.random_floor import (
    RANDOM_FLOOR_METHOD,
    aggregate_random_floor_over_seeds,
    score_random_floor_per_origin,
)

SEED = 42
FIXTURE_PATH = Path(__file__).parent / "demo_data" / "toy_sparse_panel.parquet"

N_STYLES = 200
# 2018-09-17 .. 2020-09-21 is the real panel's extent, so the schedule (step 4, burn-in 13, horizon
# 13) gives the same 20 origins and 12 walk-forward test origins as the reported backtest.
PANEL_START = date(2018, 9, 17)
N_WEEKS = 106

# Config #3 of the grid: the one the embargoed re-tune selected on the real data
# (reports/tables/embargoed_retune_selected.csv). Fixed here so the demo does no tuning.
DEMO_LGBM_CONFIG = HYPERPARAM_GRID[3]

# Attribute vocabulary for the toy styles. Real values are H&M's; these are a small stand-in with
# the same five-column structure as `STYLE_KEY_COLS`.
_VOCAB: dict[str, list[str]] = {
    "index_group_name": ["Ladieswear", "Menswear", "Divided", "Baby/Children"],
    "product_type_name": ["Sweater", "T-shirt", "Dress", "Trousers", "Shorts", "Coat", "Top"],
    "garment_group_name": [
        "Knitwear",
        "Jersey Basic",
        "Dresses",
        "Trousers",
        "Outdoor",
        "Jersey Fancy",
    ],
    "perceived_colour_master_name": ["Black", "White", "Beige", "Red", "Blue", "Green", "Pink"],
    "graphical_appearance_name": ["Solid", "Stripe", "All over pattern", "Melange"],
}


def make_toy_sparse_panel(seed: int = SEED) -> pl.DataFrame:
    """Generate the toy sparse (sales-only) style-week panel, same columns as the real one.

    One row per (style, week) with at least one unit sold, matching what `filter_by_support`
    hands to `densify_panel` on the real data. Deterministic given `seed`.
    """
    rng = np.random.default_rng(seed)
    weeks = [PANEL_START + timedelta(weeks=i) for i in range(N_WEEKS)]

    # Seasonal phase and strength are shared within a product type (sweaters peak in winter,
    # shorts in summer): structure a model can learn and a seasonal-naive baseline can exploit.
    product_types = _VOCAB["product_type_name"]
    phase = {p: rng.uniform(0, 2 * np.pi) for p in product_types}
    amplitude = {p: rng.uniform(0.2, 0.7) for p in product_types}

    seen: set[tuple[str, ...]] = set()
    rows: list[dict[str, object]] = []
    while len(seen) < N_STYLES:
        attrs = tuple(str(rng.choice(_VOCAB[c])) for c in STYLE_KEY_COLS)
        if attrs in seen:
            continue
        seen.add(attrs)
        ptype = attrs[STYLE_KEY_COLS.index("product_type_name")]

        launch = 0 if rng.random() < 0.6 else int(rng.integers(1, 60))
        end = N_WEEKS - 1 if rng.random() < 0.85 else int(rng.integers(launch + 30, N_WEEKS))
        level = float(np.exp(rng.normal(1.3, 0.35)))  # baseline units per active article per week
        trend = float(rng.normal(0.0, 0.01))  # log-scale drift per week
        base_price = float(rng.uniform(0.01, 0.06))
        n_art = 1 + int(rng.poisson(3))
        shock = 0.0  # AR(1) log-demand shock: persistence is what makes recent history informative

        for t in range(launch, end + 1):
            shock = 0.85 * shock + float(rng.normal(0.0, 0.3))
            if rng.random() < 0.05:
                n_art = max(1, n_art + int(rng.choice([-1, 1])))
            season = amplitude[ptype] * np.sin(2 * np.pi * t / 52.18 + phase[ptype])
            per_article = level * np.exp(season + trend * t + shock)
            units = int(rng.poisson(n_art * per_article))
            if units == 0:
                continue
            markdown = 1.0 - 0.25 * max(0.0, np.sin(2 * np.pi * (t - 20) / 52.18))
            price = base_price * markdown * float(rng.uniform(0.95, 1.05))
            online = int(rng.binomial(units, 0.3))
            rows.append(
                {
                    **dict(zip(STYLE_KEY_COLS, attrs, strict=True)),
                    "week_start": weeks[t],
                    "units": units,
                    "revenue": units * price,
                    "n_active_articles": n_art,
                    "units_per_active_article": units / n_art,
                    "mean_price": price,
                    "median_price": price * 0.98,
                    "n_customers": max(1, int(rng.binomial(units, 0.9))),
                    "units_online": online,
                    "units_store": units - online,
                }
            )

    sparse = pl.DataFrame(rows).with_columns(
        pl.col("units", "n_active_articles", "n_customers", "units_online", "units_store").cast(
            pl.Int64
        )
    )
    lifetime = sparse.group_by(STYLE_KEY_COLS).agg(
        first_week_seen=pl.col("week_start").min(), last_week_seen=pl.col("week_start").max()
    )
    style_key = pl.concat_str([pl.col(c) for c in STYLE_KEY_COLS], separator=STYLE_KEY_SEPARATOR)
    return (
        sparse.join(lifetime, on=STYLE_KEY_COLS)
        .with_columns(style_key.alias("style_key"))
        .select("style_key", *[c for c in sparse.columns], "first_week_seen", "last_week_seen")
        .sort([*STYLE_KEY_COLS, "week_start"])
    )


def build_dense_panel(sparse: pl.DataFrame) -> pl.DataFrame:
    """The real panel construction chain, applied to the toy sparse panel."""
    return add_intensity_shrunk(add_price_index(densify_panel(sparse)))


def _print_results(combined: pl.DataFrame, paired: pl.DataFrame) -> None:
    summary = summarize_backtest(combined).filter(pl.col("split") == "pooled")
    order = [*BASELINE_METHODS, METHOD_NAME, RANDOM_FLOOR_METHOD]
    display = (
        summary.with_columns(
            pl.col("method").replace_strict(order, list(range(len(order)))).alias("_o")
        )
        .sort("_o")
        .select(
            "method",
            pl.format(
                "{} [{}, {}]",
                pl.col("hit_at_3_in_top20_mean").round(3),
                pl.col("hit_at_3_in_top20_ci_low").round(3),
                pl.col("hit_at_3_in_top20_ci_high").round(3),
            ).alias("hit@3-in-top20 [95% CI]"),
            pl.col("ndcg_at_10_mean").round(3).alias("ndcg@10"),
            pl.col("spearman_rho_mean").round(3).alias("spearman"),
            pl.col("wmape_mean").round(3).alias("wmape (lower=better)"),
        )
    )
    cfg = pl.Config(tbl_rows=20, tbl_width_chars=160, tbl_formatting="ASCII_FULL_CONDENSED")
    with cfg:
        print("\nPooled over the 12 walk-forward test origins (block-bootstrap 95% CI):")
        print(display)

        keep = ["hit_at_3_in_top20", "ndcg_at_10", "spearman_rho", "wmape"]
        diffs = (
            paired.filter((pl.col("split") == "pooled") & pl.col("metric").is_in(keep))
            .select(
                pl.col("method_b").alias("lightgbm minus ..."),
                "metric",
                pl.format(
                    "{} [{}, {}]",
                    pl.col("mean_diff").round(3),
                    pl.col("diff_ci_low").round(3),
                    pl.col("diff_ci_high").round(3),
                ).alias("paired diff [95% CI]"),
                pl.col("n_origins"),
            )
            .sort("metric", "lightgbm minus ...")
        )
        print("\nPaired per-origin differences, embargoed LightGBM minus each comparator:")
        print(diffs)


def main() -> None:
    """Run the demo: toy panel -> panel chain -> baselines + embargoed LightGBM + random floor."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="rewrite the committed fixture from make_toy_sparse_panel (seed 42) and exit",
    )
    args = parser.parse_args()

    if args.regenerate:
        FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
        make_toy_sparse_panel().write_parquet(FIXTURE_PATH, compression="zstd")
        print(f"Wrote {FIXTURE_PATH}")
        return

    # Two warnings the real harness also emits, both benign: polars cannot verify sortedness of an
    # as-of join key when `by` groups are given (inputs are sorted explicitly before the join), and
    # scipy flags a constant prediction vector, for which Spearman is genuinely undefined
    # (global_mean predicts one value per origin).
    warnings.filterwarnings("ignore", message="Sortedness of columns cannot be checked")
    warnings.filterwarnings("ignore", message="An input array is constant")

    t0 = time.perf_counter()
    print("=" * 100)
    print("SYNTHETIC DATA. These numbers come from a generated toy panel, not from H&M.")
    print("They demonstrate that the evaluation code runs; they are NOT the reported results.")
    print("Reported results: reports/tables/backtest_embargo_*.csv, reports/WRITEUP.md")
    print("=" * 100)

    sparse = pl.read_parquet(FIXTURE_PATH)
    panel = build_dense_panel(sparse)
    print(
        f"\nToy panel: {sparse.select(STYLE_KEY_COLS).unique().height}"
        f" styles, {sparse.height} sales rows -> {panel.height} dense style-week rows, "
        f"{panel['week_start'].min()} .. {panel['week_start'].max()}"
    )

    origins = generate_origin_schedule(panel)
    test_origins = origins[INITIAL_POOL_SIZE:]
    print(
        f"Origins: {len(origins)} (step 4 weeks), first {INITIAL_POOL_SIZE} form the initial pool, "
        f"{len(test_origins)} are walk-forward test origins "
        f"{test_origins[0].origin_week} .. {test_origins[-1].origin_week}"
    )

    # The embargo, made visible: for every test origin, the latest training origin's 13-week
    # label window must close on or before the test origin. Checked, not just documented.
    for i, origin in enumerate(origins[INITIAL_POOL_SIZE:], start=INITIAL_POOL_SIZE):
        train_weeks = embargoed_train_origin_weeks(origins, i)
        latest_label_end = max(train_weeks) + timedelta(weeks=HORIZON_WEEKS)
        assert latest_label_end <= origin.origin_week, f"embargo violated at {origin.origin_week}"
    print(
        f"Embargo check: every training label window closes on or before its test origin "
        f"({HORIZON_WEEKS}-week gap) -- OK for all {len(test_origins)} test origins"
    )

    baseline = run_backtest(panel, test_origins)
    embargoed, skipped = run_embargoed_walk_forward(panel, origins, DEMO_LGBM_CONFIG)
    assert not skipped, f"embargoed arm skipped origins {skipped}"
    predictions = build_predictions_frame(panel, [o.origin_week for o in test_origins])
    floor = aggregate_random_floor_over_seeds(
        score_random_floor_per_origin(predictions, test_origins)
    )

    combined = pl.concat([baseline, embargoed.select(baseline.columns), floor], how="vertical")
    paired = paired_diff_table(combined, baseline_methods=BASELINE_METHODS)
    _print_results(combined, paired)

    print(
        "\nNotes: seasonal_naive is undefined at the 2 test origins without a 52-week lag, so its "
        "rows use 10 origins.\nSpearman is undefined (NaN) for global_mean, which predicts one "
        "value for every style."
    )
    print(f"\nDone in {time.perf_counter() - t0:.1f}s (synthetic data, seed {SEED}).")


if __name__ == "__main__":
    main()
