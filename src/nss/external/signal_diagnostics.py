"""F1 coverage and F2 lead-lag diagnostics for the two candidate external signals.

Writes (all under `reports/tables/`, all regenerated from the on-disk caches, no network):

- `external_signal_trends_weekly.csv`     the cached Google Trends series (term, Monday week,
                                          value), committed because the API's answers drift;
- `external_signal_wikipedia_weekly.csv`  the cached Wikipedia series, summed to Monday weeks;
- `external_signal_style_terms.csv`       style_key -> Trends term / unmapped reason;
- `external_signal_coverage.csv`          how many styles each source can serve, and why not;
- `external_signal_lead_lag.csv`          does the signal move BEFORE sales does? (see below).

LEAD-LAG DEFINITION. For lag L in {0, 2, 4, 8} weeks and each week t, the cross-sectional Spearman
correlation across styles between

  X = the style's 4-week signal slope as of week t - L      (`sg_slope_4w`, shifted L weeks), and
  Y = either (a) the style's own 4-week SALES slope at week t (`concurrent`), or
             (b) the realised 13-week-ahead log growth at origin t,
                 y_true - log1p(ewma_halflife_13w)                                (`forward`).

L = 0 is the contemporaneous control. If search LEADS sales, rho should be larger at L > 0 than at
L = 0. As a benchmark the same table is computed with X replaced by the style's own lagged SALES
slope (`sales_own`): a signal only "carries information sales history does not" if it is
competitive with, or adds to, that benchmark. Weekly rhos are averaged with a moving-block
bootstrap CI (block 4, seed 42, the project standard).

TWO WINDOWS. `pre_test` uses only information available before the first walk-forward test origin
(2019-07-29; for `forward`, origins whose 13-week outcome window closes by then). It is the window
on which a lead length could legitimately be chosen. `all` includes the test period; it is
DESCRIPTIVE ONLY and nothing in the model is selected from it. All lead lengths enter the model
regardless.

    uv run --no-sync python -m nss.external.signal_diagnostics
"""

from __future__ import annotations

import warnings
from datetime import date, timedelta
from pathlib import Path

import polars as pl
from scipy.stats import spearmanr

from nss.external import wikipedia_fetch
from nss.external.term_mapping import COLOUR_WORD, PRODUCT_WORD, map_styles
from nss.external.trends_fetch import CACHE_DIR as TRENDS_DIR
from nss.external.trends_fetch import ordered_terms
from nss.external.trends_load import load_weekly
from nss.features.signal_features import (
    MIN_NONZERO_SHARE,
    build_signal_features,
    sales_slopes,
    usable_terms,
)
from nss.models.backtest import block_bootstrap_ci, generate_origin_schedule
from nss.models.lightgbm_model import INITIAL_POOL_SIZE, build_model_frame

PANEL_PATH = Path("data/processed/style_week_panel.parquet")
OUT_DIR = Path("reports/tables")
LAGS = (0, 2, 4, 8)
MIN_STYLES_PER_WEEK = 50
HORIZON_WEEKS = 13


def load_panel() -> pl.DataFrame:
    """The dense style-week panel."""
    return pl.read_parquet(PANEL_PATH)


def wikipedia_weekly() -> pl.DataFrame:
    """Weekly (Mon..Sun) views per article: `term` (article title), `week_start`, `value`."""
    frames = []
    for path in sorted(wikipedia_fetch.CACHE_DIR.glob("*.csv")):
        if path.name.startswith("_"):
            continue
        df = pl.read_csv(path, try_parse_dates=True)
        frames.append(
            df.with_columns(
                (pl.col("day") - pl.duration(days=pl.col("day").dt.weekday() - 1)).alias(
                    "week_start"
                )
            )
            .group_by("week_start")
            .agg(pl.col("views").sum().alias("value"), pl.len().alias("_n_days"))
            .filter(pl.col("_n_days") == 7)  # only complete Mon..Sun weeks
            .select(pl.lit(_title_from_slug(path.stem)).alias("term"), "week_start", "value")
        )
    return pl.concat(frames).sort("term", "week_start")


def _title_from_slug(stem: str) -> str:
    """Recover the title for a cached file by matching slugs against the known titles."""
    titles = {*wikipedia_fetch.PRODUCT_ARTICLE.values(), *wikipedia_fetch.COLOUR_ARTICLE.values()}
    for title in titles:
        if wikipedia_fetch.slug(title) == stem:
            return title
    raise KeyError(stem)


def wikipedia_style_terms(styles: pl.DataFrame) -> pl.DataFrame:
    """style_key -> product-type article (the source has no colour+product resolution)."""
    rows = [
        {"style_key": k, "term": wikipedia_fetch.PRODUCT_ARTICLE.get(p)}
        for k, p in styles.unique("style_key").select("style_key", "product_type_name").iter_rows()
    ]
    return pl.DataFrame(rows, schema={"style_key": pl.String, "term": pl.String})


def coverage_table(
    styles: pl.DataFrame,
    trends_terms: pl.DataFrame,
    trends_weekly: pl.DataFrame,
    wiki_terms: pl.DataFrame,
    wiki_weekly: pl.DataFrame,
) -> pl.DataFrame:
    """One row per source x tier saying how many of the styles can be served, and why not."""
    n = styles["style_key"].n_unique()
    rows: list[dict[str, object]] = []

    def add(source: str, tier: str, count: int, note: str) -> None:
        rows.append(
            {"source": source, "tier": tier, "n_styles": count, "share": count / n, "note": note}
        )

    add("all", "styles in panel", n, "")
    mapped = trends_terms.filter(pl.col("term").is_not_null())
    reasons = trends_terms.filter(pl.col("term").is_null())["unmapped_reason"].value_counts()
    add("google_trends", "mapped to a query term", mapped.height, "colour word + product word")
    for reason, count in reasons.iter_rows():
        add("google_trends", f"unmapped: {reason}", int(count), "no term invented")
    all_terms = ordered_terms()
    fetched = trends_weekly["term"].n_unique() if trends_weekly.height else 0
    add(
        "google_trends",
        "term fetched (cache present)",
        mapped.filter(pl.col("term").is_in(trends_weekly["term"].unique())).height,
        f"{fetched}/{len(all_terms)} terms",
    )
    use = usable_terms(trends_weekly)
    usable_set = use.filter(pl.col("usable"))["term"]
    add(
        "google_trends",
        "usable series (>= 90% non-zero weeks)",
        mapped.filter(pl.col("term").is_in(usable_set)).height,
        f"{usable_set.len()}/{use.height} fetched terms clear the threshold",
    )
    wm = wiki_terms.filter(pl.col("term").is_not_null())
    add("wikipedia", "product type has an article", wm.height, "product-type level only")
    wiki_usable = usable_terms(wiki_weekly).filter(pl.col("usable"))["term"]
    add(
        "wikipedia",
        "usable series",
        wm.filter(pl.col("term").is_in(wiki_usable)).height,
        f"{wiki_usable.len()} articles; {wm['term'].n_unique()} distinct series serve all styles",
    )
    return pl.DataFrame(rows)


def _weekly_rhos(frame: pl.DataFrame, x: str, y: str, time_col: str) -> pl.DataFrame:
    """Cross-sectional Spearman of x vs y for each value of `time_col` with enough styles."""
    out = []
    for key, g in frame.select(time_col, x, y).drop_nulls().group_by(time_col, maintain_order=True):
        if g.height < MIN_STYLES_PER_WEEK or g[x].n_unique() < 2 or g[y].n_unique() < 2:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rho = float(spearmanr(g[x].to_numpy(), g[y].to_numpy()).statistic)
        out.append({time_col: key[0], "rho": rho, "n": g.height})
    return pl.DataFrame(out, schema={time_col: pl.Date, "rho": pl.Float64, "n": pl.Int64})


def lead_lag_rows(
    source: str,
    panel: pl.DataFrame,
    style_terms: pl.DataFrame,
    weekly: pl.DataFrame,
    forward: pl.DataFrame,
    first_test_origin: date,
) -> list[dict[str, object]]:
    """The lead-lag table rows for one source. See module docstring."""
    all_weeks = panel["week_start"].unique().sort().to_list()
    sig = build_signal_features(panel, style_terms, weekly, all_weeks)
    sig = sig.rename({"origin_week": "week_start"}).select("style_key", "week_start", "sg_slope_4w")
    slopes = sales_slopes(panel).select("style_key", "week_start", "_ss4")
    base = sig.join(slopes, on=["style_key", "week_start"], how="left").sort(
        "style_key", "week_start"
    )
    rows: list[dict[str, object]] = []
    fwd_cut = first_test_origin - timedelta(weeks=HORIZON_WEEKS)
    for lag in LAGS:
        lagged = base.with_columns(
            pl.col("sg_slope_4w").shift(lag).over("style_key", order_by="week_start").alias("_sig"),
            pl.col("_ss4").shift(lag).over("style_key", order_by="week_start").alias("_own"),
        )
        fwd = lagged.join(
            forward.rename({"origin_week": "week_start"}),
            on=["style_key", "week_start"],
            how="inner",
        )
        for analysis, frame, ycol in (
            ("concurrent", lagged, "_ss4"),
            ("forward", fwd, "_growth"),
        ):
            for xname, xcol in (("signal", "_sig"), ("sales_own", "_own")):
                rhos = _weekly_rhos(frame, xcol, ycol, "week_start")
                for window in ("pre_test", "all"):
                    if window == "pre_test":
                        limit = fwd_cut if analysis == "forward" else first_test_origin
                        sub = rhos.filter(pl.col("week_start") <= limit)
                    else:
                        sub = rhos
                    mean, lo, hi = block_bootstrap_ci(sub["rho"].to_list())
                    rows.append(
                        {
                            "source": source,
                            "analysis": analysis,
                            "window": window,
                            "x": xname,
                            "lag_weeks": lag,
                            "n_weeks": sub.height,
                            "median_styles_per_week": float(sub["n"].median())
                            if sub.height
                            else None,
                            "mean_rho": mean,
                            "ci_lo": lo,
                            "ci_hi": hi,
                        }
                    )
    return rows


def main() -> None:
    """Write the weekly-signal, mapping, coverage and lead-lag tables."""
    panel = load_panel()
    styles = panel.select("style_key", "product_type_name", "perceived_colour_master_name").unique(
        "style_key"
    )
    trends_terms = map_styles(styles)
    trends_weekly = load_weekly([t for t, _ in ordered_terms()], TRENDS_DIR)
    wiki_weekly = wikipedia_weekly()
    wiki_terms = wikipedia_style_terms(styles)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trends_weekly.write_csv(OUT_DIR / "external_signal_trends_weekly.csv")
    wiki_weekly.write_csv(OUT_DIR / "external_signal_wikipedia_weekly.csv")
    trends_terms.write_csv(OUT_DIR / "external_signal_style_terms.csv")
    coverage = coverage_table(styles, trends_terms, trends_weekly, wiki_terms, wiki_weekly)
    coverage.write_csv(OUT_DIR / "external_signal_coverage.csv")
    with pl.Config(tbl_rows=30, tbl_width_chars=200, fmt_str_lengths=70):
        print(coverage)

    origins = generate_origin_schedule(panel)
    first_test = origins[INITIAL_POOL_SIZE].origin_week
    frame = build_model_frame(panel, [o.origin_week for o in origins])
    forward = frame.select(
        "style_key",
        "origin_week",
        (pl.col("y_true") - pl.col("ewma_halflife_13w").log1p()).alias("_growth"),
    )
    rows = lead_lag_rows("google_trends", panel, trends_terms, trends_weekly, forward, first_test)
    rows += lead_lag_rows("wikipedia", panel, wiki_terms, wiki_weekly, forward, first_test)
    table = pl.DataFrame(rows)
    table.write_csv(OUT_DIR / "external_signal_lead_lag.csv")
    with pl.Config(tbl_rows=80, tbl_width_chars=220, float_precision=3):
        print(table.filter(pl.col("window") == "pre_test"))
    print(
        f"MIN_NONZERO_SHARE={MIN_NONZERO_SHARE}; product words {len(PRODUCT_WORD)}, colours "
        f"{len(COLOUR_WORD)}"
    )


if __name__ == "__main__":
    main()
