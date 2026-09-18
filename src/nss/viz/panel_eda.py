"""Descriptive EDA over the style_key x ISO-week panel (`data/processed/style_week_panel.parquet`).

Covers: basic panel/sparsity stats, top-20-by-lifetime-units vs. top-20-by-mean-intensity overlap,
a seasonality plot for the 10 largest styles, and stockout-signature detection. All numbers here
are descriptive (measured facts about the panel), not predictive features -- the causal-safety
warning in `nss.features.style_panel` about `first_week_seen`/`last_week_seen` does not apply to
this module's outputs.

JUDGMENT CALLS (interpretive choices, documented here rather than buried in code):

1. `mean_units_per_active_article` (used for the intensity ranking) is computed ONLY over weeks
   where `n_active_articles > 0`. Zero-sale weeks have `n_active_articles == 0` by construction
   (no inventory sold that week), so `units_per_active_article` is 0 there not because intensity
   was genuinely low but because "per active article" is undefined when there are no active
   articles. Including those weeks would mechanically drag the mean toward 0 for any style with
   long dormant stretches, conflating "how intensely does it sell when stocked" with "how often is
   it stocked" -- two different questions. Restricting to active weeks measures what it claims to.

2. Stockout-signature "drop": `prev_units > 0 AND units_t < 0.3 * prev_units` -- at least a 70%
   decline from a NONZERO prior week. A transition from a 0-unit prior week is excluded by
   requiring `prev_units > 0`: 0 -> anything isn't a "drop," it's restocking.

3. Stockout-signature "recovery": `next_units > 1.7 * units_t` -- the week after the drop sells at
   least 70% more units than the dropped week ITSELF (not relative to the pre-drop level). If
   `units_t == 0` (a full stockout week), this is trivially satisfied by any `next_units > 0`
   since `1.7 * 0 == 0` -- treated as intentional: "0 units this week, any sales resuming next
   week" is exactly the drop-then-recovery shape this detector exists to catch.

4. Eligible style-weeks for the stockout-rate denominator are those with both a defined prior and
   next week WITHIN THE SAME STYLE -- i.e. excluding each style's first and last panel row. Since
   the panel is dense and continuous per style, `shift(1)`/`shift(-1)` within `style_key` (ordered
   by `week_start`) are well-defined (non-null) for every other row.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless -- this module only writes PNGs, never shows a window.
import matplotlib.pyplot as plt
import polars as pl

from nss.features.style_panel import STYLE_KEY_COLS, STYLE_KEY_SEPARATOR

DEFAULT_PANEL_PATH = Path("data/processed/style_week_panel.parquet")
DEFAULT_FIGURES_DIR = Path("reports/figures")
DEFAULT_TABLES_DIR = Path("reports/tables")

TOP_N = 20
TOP_N_SEASONALITY = 10

# Stockout-signature thresholds -- see module docstring items 2-3 for the exact semantics.
DROP_THRESHOLD = 0.3
RECOVER_THRESHOLD = 1.7


def load_panel(path: Path = DEFAULT_PANEL_PATH) -> pl.DataFrame:
    """Read the dense style_key x ISO-week panel from Parquet.

    Args:
        path: Path to the panel Parquet file.

    Returns:
        The panel as a polars DataFrame.
    """
    return pl.read_parquet(path)


def basic_stats(panel: pl.DataFrame) -> dict[str, float]:
    """Compute style count, row count, and zero-sale sparsity for the panel.

    Args:
        panel: The dense style-week panel.

    Returns:
        Dict with keys `n_styles`, `n_rows`, `n_zero_sale`, `sparsity_pct`.
    """
    n_styles = panel["style_key"].n_unique()
    n_rows = panel.height
    n_zero_sale = panel.filter(pl.col("units") == 0).height
    sparsity_pct = 100.0 * n_zero_sale / n_rows if n_rows else 0.0
    return {
        "n_styles": n_styles,
        "n_rows": n_rows,
        "n_zero_sale": n_zero_sale,
        "sparsity_pct": sparsity_pct,
    }


def lifetime_units_by_style(panel: pl.DataFrame) -> pl.DataFrame:
    """Sum `units` across each style's full panel window (its own observed lifetime).

    Args:
        panel: The dense style-week panel.

    Returns:
        One row per style_key with `lifetime_units`, sorted descending.
    """
    return (
        panel.group_by("style_key")
        .agg(lifetime_units=pl.col("units").sum())
        .sort("lifetime_units", descending=True)
    )


def mean_intensity_by_style(panel: pl.DataFrame) -> pl.DataFrame:
    """Mean `units_per_active_article` per style, over weeks with `n_active_articles > 0` only.

    See module docstring, judgment call 1, for why zero-sale weeks are excluded from this mean.

    Args:
        panel: The dense style-week panel.

    Returns:
        One row per style_key with `mean_units_per_active_article`, sorted descending. Styles
        with zero active weeks (should not occur in a panel built from real sales) are absent.
    """
    return (
        panel.filter(pl.col("n_active_articles") > 0)
        .group_by("style_key")
        .agg(mean_units_per_active_article=pl.col("units_per_active_article").mean())
        .sort("mean_units_per_active_article", descending=True)
    )


def _style_lookup(panel: pl.DataFrame) -> pl.DataFrame:
    """One row per style_key with its 5 constituent style columns, for annotating tables."""
    return panel.select(["style_key", *STYLE_KEY_COLS]).unique()


def build_top20_tables(panel: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build the top-20-by-lifetime-units and top-20-by-mean-intensity tables.

    Args:
        panel: The dense style-week panel.

    Returns:
        A tuple `(top20_lifetime, top20_intensity)`, each ranked descending, joined with the 5
        constituent style columns for readability, with an explicit `rank` column (1-20).
    """
    lookup = _style_lookup(panel)

    top20_lifetime = (
        lifetime_units_by_style(panel)
        .head(TOP_N)
        .with_row_index("rank", offset=1)
        .join(lookup, on="style_key", how="left", maintain_order="left")
    )
    top20_intensity = (
        mean_intensity_by_style(panel)
        .head(TOP_N)
        .with_row_index("rank", offset=1)
        .join(lookup, on="style_key", how="left", maintain_order="left")
    )

    col_order = ["rank", "style_key", *STYLE_KEY_COLS]
    top20_lifetime = top20_lifetime.select([*col_order, "lifetime_units"])
    top20_intensity = top20_intensity.select([*col_order, "mean_units_per_active_article"])
    return top20_lifetime, top20_intensity


def overlap_report(top20_lifetime: pl.DataFrame, top20_intensity: pl.DataFrame) -> dict[str, Any]:
    """Quantify overlap between the two top-20 lists.

    Args:
        top20_lifetime: Top-20-by-lifetime-units table (from `build_top20_tables`).
        top20_intensity: Top-20-by-mean-intensity table (from `build_top20_tables`).

    Returns:
        Dict with `n_overlap` (count of style_keys in both lists) and sorted lists
        `only_lifetime` / `only_intensity` of style_keys unique to each ranking.
    """
    a = set(top20_lifetime["style_key"].to_list())
    b = set(top20_intensity["style_key"].to_list())
    return {
        "n_overlap": len(a & b),
        "only_lifetime": sorted(a - b),
        "only_intensity": sorted(b - a),
    }


def _abbreviate_style_key(style_key: str, max_len: int = 45) -> str:
    """Shorten a `||`-separated style_key into a compact `/`-separated legend label."""
    short = style_key.replace(STYLE_KEY_SEPARATOR, "/")
    return short if len(short) <= max_len else short[: max_len - 1] + "…"


def plot_seasonality_top10(
    panel: pl.DataFrame,
    top10_style_keys: list[str],
    out_path: Path = DEFAULT_FIGURES_DIR / "seasonality_top10.png",
) -> Path:
    """Plot `units_per_active_article` over the full observed period for 10 given styles.

    Args:
        panel: The dense style-week panel.
        top10_style_keys: The 10 style_keys to plot, in the order they should be ranked/labeled
            (rank 1 = first element).
        out_path: Destination PNG path; parent directories are created if missing.

    Returns:
        `out_path`, for convenience chaining.
    """
    subset = panel.filter(pl.col("style_key").is_in(top10_style_keys))
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(14, 7))
    for i, style_key in enumerate(top10_style_keys):
        series = subset.filter(pl.col("style_key") == style_key).sort("week_start")
        label = f"#{i + 1} {_abbreviate_style_key(style_key)}"
        ax.plot(
            series["week_start"],
            series["units_per_active_article"],
            label=label,
            color=cmap(i % 10),
            linewidth=1.3,
        )

    ax.set_xlabel("Week")
    ax.set_ylabel("units_per_active_article")
    ax.set_title("Seasonality: units_per_active_article, top 10 styles by lifetime units")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7.5, borderaxespad=0.0)
    fig.autofmt_xdate()
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def detect_stockout_signature(panel: pl.DataFrame) -> pl.DataFrame:
    """Flag style-weeks matching a drop-then-recovery stockout signature.

    See module docstring, judgment calls 2-4, for the exact numeric definitions of "drop,"
    "recovery," and "eligible."

    Args:
        panel: A frame with at least `style_key`, `week_start`, `units` columns (the dense panel,
            or any subset/synthetic frame with those columns for testing).

    Returns:
        `panel` sorted by (style_key, week_start) with added columns `prev_units`, `next_units`,
        `eligible` (bool: both neighbor weeks exist for this style), and `is_stockout_signature`
        (bool: eligible AND drop AND recovery).
    """
    ordered = panel.sort(["style_key", "week_start"])
    flagged = ordered.with_columns(
        prev_units=pl.col("units").shift(1).over("style_key", order_by="week_start"),
        next_units=pl.col("units").shift(-1).over("style_key", order_by="week_start"),
    )
    flagged = flagged.with_columns(
        eligible=pl.col("prev_units").is_not_null() & pl.col("next_units").is_not_null()
    )
    flagged = flagged.with_columns(
        is_drop=pl.col("eligible")
        & (pl.col("prev_units") > 0)
        & (pl.col("units") < DROP_THRESHOLD * pl.col("prev_units"))
    )
    flagged = flagged.with_columns(
        is_stockout_signature=pl.col("is_drop")
        & (pl.col("next_units") > RECOVER_THRESHOLD * pl.col("units"))
    )
    return flagged.drop("is_drop")


def stockout_signature_summary(flagged: pl.DataFrame) -> dict[str, float]:
    """Summarize the stockout-signature detection: numerator, denominator, rate.

    Args:
        flagged: Output of `detect_stockout_signature`.

    Returns:
        Dict with `n_matched`, `n_eligible`, `rate_pct`.
    """
    n_eligible = int(flagged["eligible"].sum())
    n_matched = int(flagged["is_stockout_signature"].sum())
    rate_pct = 100.0 * n_matched / n_eligible if n_eligible else 0.0
    return {"n_matched": n_matched, "n_eligible": n_eligible, "rate_pct": rate_pct}


def main() -> None:
    """Run the full panel EDA: basic stats, top-20 overlap, seasonality plot, stockout signature.

    Writes `reports/tables/top20_lifetime_units.csv`, `reports/tables/top20_intensity.csv`, and
    `reports/figures/seasonality_top10.png`; prints all measured numbers to stdout.
    """
    panel = load_panel()

    stats = basic_stats(panel)
    print(
        f"Styles: {stats['n_styles']} | Rows: {stats['n_rows']} | "
        f"Zero-sale rows: {stats['n_zero_sale']} ({stats['sparsity_pct']:.2f}%)"
    )

    top20_lifetime, top20_intensity = build_top20_tables(panel)
    DEFAULT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    top20_lifetime.write_csv(DEFAULT_TABLES_DIR / "top20_lifetime_units.csv")
    top20_intensity.write_csv(DEFAULT_TABLES_DIR / "top20_intensity.csv")
    print(f"Wrote {DEFAULT_TABLES_DIR / 'top20_lifetime_units.csv'}")
    print(f"Wrote {DEFAULT_TABLES_DIR / 'top20_intensity.csv'}")

    overlap = overlap_report(top20_lifetime, top20_intensity)
    print(f"Top-20 overlap: {overlap['n_overlap']}/20")
    print(f"  Only in lifetime-units top 20:  {overlap['only_lifetime']}")
    print(f"  Only in intensity top 20:       {overlap['only_intensity']}")

    top10_style_keys = top20_lifetime.sort("rank")["style_key"].head(TOP_N_SEASONALITY).to_list()
    plot_path = plot_seasonality_top10(panel, top10_style_keys)
    print(f"Wrote {plot_path}")

    flagged = detect_stockout_signature(panel)
    summary = stockout_signature_summary(flagged)
    print(
        f"Stockout signature: {summary['n_matched']} / {summary['n_eligible']} eligible "
        f"style-weeks ({summary['rate_pct']:.3f}%)"
    )


if __name__ == "__main__":
    main()
