"""Build the exemplar-image manifest for the diversity-constrained final-three styles.

`reports/tables/top_styles_final_three.csv` (the diversity-constrained reselection's output)
replaces the original top-3 winners with: incumbent rank-1 (by absolute `predicted_intensity`) +
emerging ranks 1-2 (emerging winners, by `growth_ratio`). Where a final-three style_key is
IDENTICAL to a style_key already exemplar'd in
`reports/tables/exemplar_images.csv` (the earlier `winner_rank_*` rows), the selection (top-8
best-selling constituent articles in the trailing 26 weeks) and fetched images are necessarily
identical too -- same style_key, same transaction history, same fetcher -- so those rows are
carried forward rather than re-selected/re-fetched. Any final-three style not already present is
selected and fetched fresh, using the exact same logic as `nss.data.select_exemplars`.

The earlier control-style rows are carried forward unchanged (no new control is chosen).

Writes `reports/tables/exemplar_images_final_three.csv` with the same schema as
`exemplar_images.csv`, but `role` values are `final_rank_1`/`final_rank_2`/`final_rank_3`
(instead of `winner_rank_*`).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from nss.data.fetch_images import fetch_images
from nss.data.select_exemplars import (
    ARTICLES_PATH,
    LOOKBACK_WEEKS,
    N_EXEMPLARS_PER_STYLE,
    PANEL_LAST_WEEK,
    ROLE_CONTROL,
    TRANSACTIONS_DIR,
    build_manifest_rows,
    compute_lookback_cutoff,
    select_top_selling_articles,
)
from nss.features.style_panel import STYLE_KEY_COLS

FINAL_THREE_PATH = Path("reports/tables/top_styles_final_three.csv")
EXISTING_MANIFEST_PATH = Path("reports/tables/exemplar_images.csv")
IMAGES_DIR = Path("data/images")
MANIFEST_OUT_PATH = Path("reports/tables/exemplar_images_final_three.csv")

MANIFEST_SCHEMA_COLS = [
    "role",
    "style_key",
    "article_id",
    "units_sold_last_26w",
    "local_image_path",
    "fetch_success",
]

FINAL_ROLE_TEMPLATE = "final_rank_{rank}"
EXISTING_WINNER_ROLE_PREFIX = "winner_rank_"


def select_final_three_targets(
    final_three: pl.DataFrame,
) -> list[tuple[str, str, dict[str, str]]]:
    """Determine the ordered (new_role, style_key, style_key_values) targets from the
    reselection table.

    Ordering matches the stated selection: `final_rank_1` is the sole `T1_incumbent` row
    (incumbent rank-1 by absolute predicted intensity); `final_rank_2`/`final_rank_3` are the two
    `T2_emerging` rows in descending `growth_ratio` order (emerging ranks 1-2, emerging winners).

    Args:
        final_three: `top_styles_final_three.csv`, loaded with `source_table`, `style_key`,
            `predicted_intensity`, `growth_ratio`, and the 5 `STYLE_KEY_COLS` columns.

    Returns:
        Exactly 3 `(role, style_key, style_key_values)` tuples in final-rank order.

    Raises:
        ValueError: if the table doesn't contain exactly 1 T1_incumbent + 2 T2_emerging rows.
    """
    t1 = final_three.filter(pl.col("source_table") == "T1_incumbent").sort(
        "predicted_intensity", descending=True
    )
    if t1.height != 1:
        raise ValueError(f"expected exactly 1 T1_incumbent row, got {t1.height}")

    t2 = final_three.filter(pl.col("source_table") == "T2_emerging").sort(
        "growth_ratio", descending=True
    )
    if t2.height != 2:
        raise ValueError(f"expected exactly 2 T2_emerging rows, got {t2.height}")

    ordered = pl.concat([t1, t2])
    return [
        (
            FINAL_ROLE_TEMPLATE.format(rank=rank),
            row["style_key"],
            {col: row[col] for col in STYLE_KEY_COLS},
        )
        for rank, row in enumerate(ordered.iter_rows(named=True), start=1)
    ]


def find_reusable_rows(existing_manifest: pl.DataFrame, style_key: str) -> pl.DataFrame | None:
    """Find already-fetched `winner_rank_*` rows for an identical style_key, if any.

    Args:
        existing_manifest: The earlier `exemplar_images.csv`, loaded.
        style_key: The final-three style's composite style_key to look up.

    Returns:
        The matching rows (role, style_key, article_id, units_sold_last_26w, local_image_path,
        fetch_success), or `None` if this style_key was not already fetched as a winner.
    """
    matches = existing_manifest.filter(
        (pl.col("style_key") == style_key)
        & pl.col("role").str.starts_with(EXISTING_WINNER_ROLE_PREFIX)
    )
    return matches if matches.height > 0 else None


def carry_forward_rows(reused: pl.DataFrame, new_role: str) -> list[dict[str, object]]:
    """Re-role a reused winner's rows for the final-three manifest, values unchanged.

    Args:
        reused: The matching rows returned by `find_reusable_rows`.
        new_role: The `final_rank_*` role to relabel these rows with.

    Returns:
        One dict per row, schema-identical to `build_manifest_rows`' output, with `role`
        replaced and every other field carried forward verbatim.
    """
    rows = []
    for row in reused.iter_rows(named=True):
        new_row = dict(row)
        new_row["role"] = new_role
        rows.append(new_row)
    return rows


def main() -> None:
    """CLI entry point: build `exemplar_images_final_three.csv` from the final-three table.

    Reuses the earlier exemplar rows for any final-three style_key that's identical to an
    already-fetched winner; selects + fetches fresh for the rest; carries the earlier control
    style forward unchanged.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-three-path", type=Path, default=FINAL_THREE_PATH)
    parser.add_argument("--existing-manifest-path", type=Path, default=EXISTING_MANIFEST_PATH)
    parser.add_argument("--articles-path", type=Path, default=ARTICLES_PATH)
    parser.add_argument("--transactions-dir", type=Path, default=TRANSACTIONS_DIR)
    parser.add_argument("--images-dir", type=Path, default=IMAGES_DIR)
    parser.add_argument("--manifest-out-path", type=Path, default=MANIFEST_OUT_PATH)
    parser.add_argument("--lookback-weeks", type=int, default=LOOKBACK_WEEKS)
    args = parser.parse_args()

    final_three = pl.read_csv(args.final_three_path)
    existing_manifest = pl.read_csv(args.existing_manifest_path)
    targets = select_final_three_targets(final_three)

    all_manifest_rows: list[dict[str, object]] = []
    fresh_targets: list[tuple[str, str, dict[str, str]]] = []
    for new_role, style_key, style_values in targets:
        reused = find_reusable_rows(existing_manifest, style_key)
        if reused is not None:
            print(f"{new_role}: {style_key} -- REUSED from earlier ({reused.height} rows)")
            all_manifest_rows.extend(carry_forward_rows(reused, new_role))
        else:
            print(f"{new_role}: {style_key} -- fresh selection + fetch")
            fresh_targets.append((new_role, style_key, style_values))

    if fresh_targets:
        cutoff = compute_lookback_cutoff(PANEL_LAST_WEEK, args.lookback_weeks)
        articles = pl.read_csv(args.articles_path).select(["article_id", *STYLE_KEY_COLS])
        txn_lazy = pl.scan_parquet(str(args.transactions_dir / "**" / "*.parquet")).select(
            ["article_id", "t_dat"]
        )
        window_end = txn_lazy.select(pl.col("t_dat").max()).collect().item()
        print(f"Lookback window: {cutoff} .. {window_end} ({args.lookback_weeks} weeks)")
        transactions = txn_lazy.filter(pl.col("t_dat") >= cutoff).collect(engine="streaming")

        ranked_by_role: dict[str, pl.DataFrame] = {}
        fresh_article_ids: list[int] = []
        for new_role, style_key, style_values in fresh_targets:
            ranked = select_top_selling_articles(
                transactions, articles, style_values, cutoff, window_end
            )
            ranked_by_role[new_role] = ranked
            n_found = ranked.height
            if n_found < N_EXEMPLARS_PER_STYLE:
                print(
                    f"WARNING: {new_role} ({style_key}) has only {n_found} distinct constituent "
                    f"articles with sales in the lookback window (< {N_EXEMPLARS_PER_STYLE})."
                )
            print(f"{new_role}: {style_key}")
            for row in ranked.iter_rows(named=True):
                units = row["units_sold_last_26w"]
                print(f"  article_id={row['article_id']} units_sold_last_26w={units}")
            fresh_article_ids.extend(int(x) for x in ranked["article_id"].to_list())

        fetch_results = fetch_images(fresh_article_ids, args.images_dir)
        n_ok = sum(fetch_results.values())
        print(f"Fetched {n_ok}/{len(fetch_results)} images -> {args.images_dir}")
        failed = [aid for aid, ok in fetch_results.items() if not ok]
        if failed:
            print(f"Failed article_ids: {failed}")

        for new_role, style_key, _ in fresh_targets:
            ranked = ranked_by_role[new_role]
            all_manifest_rows.extend(
                build_manifest_rows(new_role, style_key, ranked, args.images_dir, fetch_results)
            )

    control_rows = existing_manifest.filter(pl.col("role") == ROLE_CONTROL)
    all_manifest_rows.extend(control_rows.iter_rows(named=True))

    manifest = pl.DataFrame(all_manifest_rows).select(MANIFEST_SCHEMA_COLS)
    args.manifest_out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_csv(args.manifest_out_path)
    print(f"Wrote manifest ({manifest.height} rows) to {args.manifest_out_path}")


if __name__ == "__main__":
    main()
