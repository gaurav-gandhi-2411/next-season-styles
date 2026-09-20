"""Select best-selling exemplar article images for the top-3 winning styles + one control style.

The later novelty-scoring stage (out of scope here) needs a small, curated set of real product
images: 8 best-selling constituent articles for each of the top-3 winning styles from
`reports/tables/top_styles.csv`, plus 8 for one randomly-chosen non-winning "control" style (not
even ranks 4-10). "Best-selling" is units sold (transaction row count, same convention as
`nss.features.style_panel`'s `units` column) within the most recent `LOOKBACK_WEEKS` of the panel's
observed period, ending at the panel's last real week (`PANEL_LAST_WEEK`, 2020-09-21).

Images are fetched via `nss.data.fetch_images.fetch_images` (the on-demand per-article fetcher) --
no bulk download, ever. See that module's docstring for why.
"""

from __future__ import annotations

import argparse
import random
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from nss.data.fetch_images import fetch_images, local_path
from nss.features.style_panel import STYLE_KEY_COLS

TOP_STYLES_PATH = Path("reports/tables/top_styles.csv")
PANEL_PATH = Path("data/processed/style_week_panel.parquet")
ARTICLES_PATH = Path("data/raw/articles.csv")
TRANSACTIONS_DIR = Path("data/interim/transactions_train_parquet")
IMAGES_DIR = Path("data/images")
MANIFEST_OUT_PATH = Path("reports/tables/exemplar_images.csv")

N_WINNERS = 3
N_EXEMPLARS_PER_STYLE = 8
LOOKBACK_WEEKS = 26
# The panel's last observed week_start (see `nss.features.style_panel`, `last_week_seen` max).
PANEL_LAST_WEEK = date(2020, 9, 21)
RANDOM_SEED = 42

ROLE_WINNER_TEMPLATE = "winner_rank_{rank}"
ROLE_CONTROL = "control"


def compute_lookback_cutoff(last_week: date, lookback_weeks: int) -> date:
    """Compute the inclusive start date of the trailing lookback window.

    Args:
        last_week: The panel's last observed week_start (a Monday).
        lookback_weeks: Number of ISO weeks to look back.

    Returns:
        `last_week - lookback_weeks` (e.g. 2020-09-21 - 26 weeks = 2020-03-23).
    """
    return last_week - timedelta(weeks=lookback_weeks)


def select_top_selling_articles(
    transactions: pl.DataFrame,
    articles: pl.DataFrame,
    style_key_values: dict[str, str],
    window_start: date,
    window_end: date,
    n: int = N_EXEMPLARS_PER_STYLE,
) -> pl.DataFrame:
    """Rank a style's constituent articles by units sold in `[window_start, window_end]`.

    "Units sold" is transaction row count (one row = one unit, same convention as
    `nss.features.style_panel`'s `units` aggregate) -- there is no separate quantity column.

    Args:
        transactions: Transaction rows with at least `article_id`, `t_dat` columns.
        articles: Article metadata with `article_id` and the 5 `STYLE_KEY_COLS` columns.
        style_key_values: Mapping of each of the 5 `STYLE_KEY_COLS` to the target style's value.
        window_start: Inclusive start of the sales window.
        window_end: Inclusive end of the sales window.
        n: Maximum number of top-selling articles to return.

    Returns:
        A DataFrame with columns `article_id`, `units_sold_last_26w`, sorted by units sold
        descending (ties broken by `article_id` ascending for determinism), truncated to `n` rows.
        May have fewer than `n` rows if fewer than `n` distinct constituent articles sold anything
        in the window -- callers must not pad with untraded articles.
    """
    style_articles = articles.filter(
        pl.all_horizontal([pl.col(col) == val for col, val in style_key_values.items()])
    ).select("article_id")

    windowed = transactions.filter(
        (pl.col("t_dat") >= window_start) & (pl.col("t_dat") <= window_end)
    )

    sales = (
        windowed.join(style_articles, on="article_id", how="inner")
        .group_by("article_id")
        .agg(units_sold_last_26w=pl.len())
        .sort(["units_sold_last_26w", "article_id"], descending=[True, False])
    )
    return sales.head(n)


def select_control_style(
    panel: pl.DataFrame, excluded_style_keys: set[str], seed: int = RANDOM_SEED
) -> str:
    """Randomly select one style_key from the panel's kept styles, excluding a given set.

    Deterministic: candidates are sorted before the single seeded `random.Random.choice` call, so
    the result is reproducible independent of the panel's row order.

    Args:
        panel: The dense style-week panel (or any frame with a `style_key` column covering the
            full set of kept styles).
        excluded_style_keys: style_keys that must not be chosen (e.g. the full top-10 table, not
            just the top-3 winners).
        seed: Random seed (project convention: `seed=42`).

    Returns:
        The chosen style_key.
    """
    candidates = sorted(set(panel["style_key"].unique().to_list()) - excluded_style_keys)
    if not candidates:
        raise ValueError("no non-excluded style_key candidates available")
    rng = random.Random(seed)
    return rng.choice(candidates)


def style_key_values_from_panel(panel: pl.DataFrame, style_key: str) -> dict[str, str]:
    """Look up a style_key's 5 constituent attribute values from the panel.

    Args:
        panel: A frame containing `style_key` and the 5 `STYLE_KEY_COLS` columns.
        style_key: The composite style_key to look up.

    Returns:
        Mapping of each `STYLE_KEY_COLS` column to its value for this style_key.
    """
    row = panel.filter(pl.col("style_key") == style_key).row(0, named=True)
    return {col: row[col] for col in STYLE_KEY_COLS}


def build_manifest_rows(
    role: str,
    style_key: str,
    ranked_articles: pl.DataFrame,
    images_dir: Path,
    fetch_results: dict[str, bool],
) -> list[dict[str, object]]:
    """Build manifest rows for one style's selected articles, after images have been fetched.

    Args:
        role: One of `winner_rank_1`/`winner_rank_2`/`winner_rank_3`/`control`.
        style_key: The style's composite style_key string.
        ranked_articles: Output of `select_top_selling_articles` for this style.
        images_dir: Local image cache directory (matches `fetch_images`' `out_dir`).
        fetch_results: Mapping of `str(article_id) -> success`, as returned by `fetch_images`.

    Returns:
        One dict per selected article, ready to build the manifest DataFrame/CSV.
    """
    rows: list[dict[str, object]] = []
    for row in ranked_articles.iter_rows(named=True):
        article_id = int(row["article_id"])
        rows.append(
            {
                "role": role,
                "style_key": style_key,
                "article_id": article_id,
                "units_sold_last_26w": int(row["units_sold_last_26w"]),
                "local_image_path": str(local_path(article_id, images_dir)),
                "fetch_success": bool(fetch_results.get(str(article_id), False)),
            }
        )
    return rows


def main() -> None:
    """CLI entry point: select exemplar articles for the top-3 winners + 1 control, fetch images.

    Writes `reports/tables/exemplar_images.csv` (see module docstring for schema) and prints a
    per-style summary plus a final fetch success count.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-styles-path", type=Path, default=TOP_STYLES_PATH)
    parser.add_argument("--panel-path", type=Path, default=PANEL_PATH)
    parser.add_argument("--articles-path", type=Path, default=ARTICLES_PATH)
    parser.add_argument("--transactions-dir", type=Path, default=TRANSACTIONS_DIR)
    parser.add_argument("--images-dir", type=Path, default=IMAGES_DIR)
    parser.add_argument("--manifest-out-path", type=Path, default=MANIFEST_OUT_PATH)
    parser.add_argument("--lookback-weeks", type=int, default=LOOKBACK_WEEKS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    top_styles = pl.read_csv(args.top_styles_path)
    winners = top_styles.sort("rank").head(N_WINNERS)
    excluded_style_keys = set(top_styles["style_key"].to_list())

    panel = pl.read_parquet(args.panel_path, columns=["style_key", *STYLE_KEY_COLS])
    print(f"Panel has {panel['style_key'].n_unique()} unique kept style_keys.")

    control_style_key = select_control_style(panel, excluded_style_keys, seed=args.seed)
    control_values = style_key_values_from_panel(panel, control_style_key)
    print(f"Control style_key (seed={args.seed}): {control_style_key}")

    cutoff = compute_lookback_cutoff(PANEL_LAST_WEEK, args.lookback_weeks)
    articles = pl.read_csv(args.articles_path).select(["article_id", *STYLE_KEY_COLS])
    txn_lazy = pl.scan_parquet(str(args.transactions_dir / "**" / "*.parquet")).select(
        ["article_id", "t_dat"]
    )
    window_end = txn_lazy.select(pl.col("t_dat").max()).collect().item()
    print(f"Lookback window: {cutoff} .. {window_end} ({args.lookback_weeks} weeks)")
    transactions = txn_lazy.filter(pl.col("t_dat") >= cutoff).collect(engine="streaming")

    targets: list[tuple[str, str, dict[str, str]]] = [
        (
            ROLE_WINNER_TEMPLATE.format(rank=row["rank"]),
            row["style_key"],
            {col: row[col] for col in STYLE_KEY_COLS},
        )
        for row in winners.iter_rows(named=True)
    ]
    targets.append((ROLE_CONTROL, control_style_key, control_values))

    all_manifest_rows: list[dict[str, object]] = []
    all_article_ids: list[int] = []
    ranked_by_role: dict[str, pl.DataFrame] = {}
    for role, style_key, style_values in targets:
        ranked = select_top_selling_articles(
            transactions, articles, style_values, cutoff, window_end
        )
        ranked_by_role[role] = ranked
        n_found = ranked.height
        if n_found < N_EXEMPLARS_PER_STYLE:
            print(
                f"WARNING: {role} ({style_key}) has only {n_found} distinct constituent "
                f"articles with sales in the lookback window (< {N_EXEMPLARS_PER_STYLE})."
            )
        print(f"{role}: {style_key}")
        for row in ranked.iter_rows(named=True):
            units = row["units_sold_last_26w"]
            print(f"  article_id={row['article_id']} units_sold_last_26w={units}")
        all_article_ids.extend(int(x) for x in ranked["article_id"].to_list())

    fetch_results = fetch_images(all_article_ids, args.images_dir)
    n_ok = sum(fetch_results.values())
    print(f"Fetched {n_ok}/{len(fetch_results)} images -> {args.images_dir}")
    failed = [aid for aid, ok in fetch_results.items() if not ok]
    if failed:
        print(f"Failed article_ids: {failed}")

    for role, style_key, _ in targets:
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
