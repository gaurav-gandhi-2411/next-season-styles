"""Select 20 random guard-passing styles as an expanded margin-scoring reference base.

The earlier `derive_similarity_band.py` derived its band from only ~6 distinct styles (3 winners + 2
emerging + 1 control) -- too small a sample to trust, and (separately, see
`nss.generate.margin_scoring`'s docstring) built on the wrong absolute-cosine construction anyway.
This module expands the real-image reference base for the margin-based replacement
(`nss.generate.derive_margin_band`) to 20 NEW randomly-selected guard-passing styles, on top of
the 3 final-three styles + 1 control already fetched by `select_exemplars.py` /
`select_final_three_exemplars.py`.

GUARD-PASSING STYLE SET, COMPUTED READ-ONLY (no retraining, no model load): "guard-passing" means
passing all 3 of `nss.models.final_forecast`'s guards (commercial scale, markdown exclusion,
52-week viability). Only `predicted_intensity` -- unused by this module -- depends on the trained
LightGBM model; all 3 guard booleans are pure panel/feature-engineering computations
(`build_guard_frame` for guards 1+3, `build_features`'s `price_index_level` column for guard 2),
so `compute_guard_passing_styles` below recomputes the guard-passing set directly from those
functions and the project's own guard threshold constants, WITHOUT calling `train_final_model` or
loading any saved model artifact. This necessarily duplicates the small amount of boolean-glue
code in `final_forecast.build_ranking_frame` (that function can't be called without a model), but
reuses every actual threshold value and the windowing logic itself from `final_forecast.py` --
sanity-checked against the real `top_styles.csv` (its 10 guard-passing rows are confirmed a subset
of this function's output on the real panel before this module was written).

Excludes the 3 final-three styles and the existing control style (all already have real reference
images fetched) so the 20 new styles are a genuinely additional, non-overlapping reference base.

Selection: `seed=42`, single seeded `random.Random.sample` call over the sorted candidate list
(same determinism convention as `select_exemplars.select_control_style`). For each selected style,
its top-8 best-selling constituent articles in the trailing 26 weeks (reusing
`select_exemplars.select_top_selling_articles` verbatim) are fetched via
`nss.data.fetch_images.fetch_images` (retry+cache, no bulk-download fallback -- same hard
constraint as every other image-fetching step in this project).
"""

from __future__ import annotations

import argparse
import random
from datetime import date
from pathlib import Path

import polars as pl

from nss.data.fetch_images import fetch_images
from nss.data.select_exemplars import (
    ARTICLES_PATH,
    LOOKBACK_WEEKS,
    N_EXEMPLARS_PER_STYLE,
    PANEL_LAST_WEEK,
    TRANSACTIONS_DIR,
    build_manifest_rows,
    compute_lookback_cutoff,
    select_top_selling_articles,
)
from nss.features.model_features import build_features
from nss.features.style_panel import STYLE_KEY_COLS
from nss.models.final_forecast import (
    FORECAST_ORIGIN,
    GUARD1_MIN_MEAN_N_ACTIVE_ARTICLES,
    GUARD2_MIN_PRICE_INDEX,
    GUARD3_MIN_WEEKS_ACTIVE,
    build_guard_frame,
    verify_forecast_origin,
)

PANEL_PATH = Path("data/processed/style_week_panel.parquet")
IMAGES_DIR = Path("data/images")
MANIFEST_OUT_PATH = Path("reports/tables/exemplar_images_margin_reference.csv")

N_REFERENCE_STYLES = 20
RANDOM_SEED = 42
ROLE_TEMPLATE = "margin_ref_{i:02d}"

# Already have real reference images fetched (final-three winners + existing control) -- exclude
# so the 20 new styles are a genuinely additional reference base, not a re-fetch.
EXCLUDED_STYLE_KEYS: frozenset[str] = frozenset(
    {
        "Ladieswear || T-shirt || Jersey Basic || Black || Solid",
        "Ladieswear || Underwear bottom || Under-, Nightwear || Red || Solid",
        "Ladieswear || Sweater || Knitwear || Beige || Melange",
        "Menswear || Scarf || Accessories || Grey || Melange",  # existing control
    }
)


def compute_guard_passing_styles(
    panel: pl.DataFrame, forecast_origin: date = FORECAST_ORIGIN
) -> pl.DataFrame:
    """Recompute `nss.models.final_forecast`'s guard-passing style set, read-only, no model.

    See module docstring for why this is safe under the modelling-freeze constraint: none of the
    3 guard booleans depend on the trained model's predictions.

    Args:
        panel: The dense style-week panel.
        forecast_origin: The forecast origin week (must match the panel's actual last observed
            week -- verified via `verify_forecast_origin`, never assumed).

    Returns:
        One row per style eligible at `forecast_origin` with `guard1_pass`, `guard2_pass`,
        `guard3_pass` all `True`, columns `style_key` + the 5 `STYLE_KEY_COLS`.
    """
    verify_forecast_origin(panel, forecast_origin)
    forecast_frame = build_features(panel, [forecast_origin]).select(
        "style_key", *STYLE_KEY_COLS, "price_index_level"
    )
    guard_frame = build_guard_frame(panel, forecast_origin)
    joined = forecast_frame.join(guard_frame, on="style_key", how="left")
    joined = joined.with_columns(
        (
            pl.col("guard1_n_active_articles_trailing_mean") >= GUARD1_MIN_MEAN_N_ACTIVE_ARTICLES
        ).alias("guard1_pass"),
        # null price_index (no 52w trailing price history yet) -> can't verify -> fail. Mirrors
        # `final_forecast.build_ranking_frame`'s identical guard2 null-handling verbatim.
        pl.when(pl.col("price_index_level").is_not_null())
        .then(pl.col("price_index_level") >= GUARD2_MIN_PRICE_INDEX)
        .otherwise(False)
        .alias("guard2_pass"),
        (pl.col("guard3_n_weeks_active_trailing") >= GUARD3_MIN_WEEKS_ACTIVE).alias("guard3_pass"),
    )
    return joined.filter(pl.col("guard1_pass") & pl.col("guard2_pass") & pl.col("guard3_pass"))


def select_random_reference_styles(
    guard_passing_style_keys: list[str],
    excluded_style_keys: frozenset[str] = EXCLUDED_STYLE_KEYS,
    n: int = N_REFERENCE_STYLES,
    seed: int = RANDOM_SEED,
) -> list[str]:
    """Deterministically sample `n` guard-passing styles, excluding `excluded_style_keys`.

    Candidates are sorted before the single seeded `random.Random.sample` call, so the result is
    reproducible independent of input row order (same convention as
    `select_exemplars.select_control_style`).

    Args:
        guard_passing_style_keys: Every guard-passing style_key (may contain duplicates; will be
            de-duplicated via `set`).
        excluded_style_keys: style_keys that must not be selected.
        n: Number of styles to sample.
        seed: Random seed (project convention: `seed=42`).

    Returns:
        `n` distinct style_keys, in the order `random.Random.sample` returned them.

    Raises:
        ValueError: if fewer than `n` eligible candidates remain after exclusion.
    """
    candidates = sorted(set(guard_passing_style_keys) - excluded_style_keys)
    if len(candidates) < n:
        raise ValueError(
            f"only {len(candidates)} guard-passing candidate styles available "
            f"(after excluding {len(excluded_style_keys)}), need {n}"
        )
    rng = random.Random(seed)
    return rng.sample(candidates, n)


def main() -> None:
    """CLI entry point: select 20 random guard-passing styles, fetch their top-8-selling images.

    Writes `reports/tables/exemplar_images_margin_reference.csv` (same schema as
    `exemplar_images.csv`, `role` values `margin_ref_01`..`margin_ref_20`) and prints a per-style
    summary plus the overall fetch success count.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-path", type=Path, default=PANEL_PATH)
    parser.add_argument("--articles-path", type=Path, default=ARTICLES_PATH)
    parser.add_argument("--transactions-dir", type=Path, default=TRANSACTIONS_DIR)
    parser.add_argument("--images-dir", type=Path, default=IMAGES_DIR)
    parser.add_argument("--manifest-out-path", type=Path, default=MANIFEST_OUT_PATH)
    parser.add_argument("--lookback-weeks", type=int, default=LOOKBACK_WEEKS)
    parser.add_argument("--n-styles", type=int, default=N_REFERENCE_STYLES)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    panel = pl.read_parquet(args.panel_path)
    guard_passing = compute_guard_passing_styles(panel)
    print(f"Guard-passing styles (read-only, no model): {guard_passing.height}")

    selected_style_keys = select_random_reference_styles(
        guard_passing["style_key"].to_list(), n=args.n_styles, seed=args.seed
    )
    print(f"Selected {len(selected_style_keys)} styles (seed={args.seed}):")
    style_values_by_key = {
        row["style_key"]: {col: row[col] for col in STYLE_KEY_COLS}
        for row in guard_passing.filter(pl.col("style_key").is_in(selected_style_keys)).iter_rows(
            named=True
        )
    }
    for style_key in selected_style_keys:
        print(f"  {style_key}")

    cutoff = compute_lookback_cutoff(PANEL_LAST_WEEK, args.lookback_weeks)
    articles = pl.read_csv(args.articles_path).select(["article_id", *STYLE_KEY_COLS])
    txn_lazy = pl.scan_parquet(str(args.transactions_dir / "**" / "*.parquet")).select(
        ["article_id", "t_dat"]
    )
    window_end = txn_lazy.select(pl.col("t_dat").max()).collect().item()
    print(f"Lookback window: {cutoff} .. {window_end} ({args.lookback_weeks} weeks)")
    transactions = txn_lazy.filter(pl.col("t_dat") >= cutoff).collect(engine="streaming")

    ranked_by_role: dict[str, pl.DataFrame] = {}
    all_article_ids: list[int] = []
    roles: list[tuple[str, str]] = []
    for i, style_key in enumerate(selected_style_keys, start=1):
        role = ROLE_TEMPLATE.format(i=i)
        roles.append((role, style_key))
        ranked = select_top_selling_articles(
            transactions, articles, style_values_by_key[style_key], cutoff, window_end
        )
        ranked_by_role[role] = ranked
        n_found = ranked.height
        if n_found < N_EXEMPLARS_PER_STYLE:
            print(
                f"WARNING: {role} ({style_key}) has only {n_found} distinct constituent "
                f"articles with sales in the lookback window (< {N_EXEMPLARS_PER_STYLE})."
            )
        all_article_ids.extend(int(x) for x in ranked["article_id"].to_list())

    fetch_results = fetch_images(all_article_ids, args.images_dir)
    n_ok = sum(fetch_results.values())
    print(f"Fetched {n_ok}/{len(fetch_results)} images -> {args.images_dir}")
    failed = [aid for aid, ok in fetch_results.items() if not ok]
    if failed:
        print(f"Failed article_ids: {failed}")

    all_manifest_rows: list[dict[str, object]] = []
    for role, style_key in roles:
        ranked = ranked_by_role[role]
        all_manifest_rows.extend(
            build_manifest_rows(role, style_key, ranked, args.images_dir, fetch_results)
        )

    manifest = pl.DataFrame(all_manifest_rows)
    args.manifest_out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_csv(args.manifest_out_path)
    print(f"Wrote manifest ({manifest.height} rows) to {args.manifest_out_path}")


if __name__ == "__main__":
    main()
