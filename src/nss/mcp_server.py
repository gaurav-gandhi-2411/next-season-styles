"""MCP server exposing next-season-styles' read-only forecasting + generation tools (task D1).

Every tool here reads artifacts already on disk (raw/interim/processed data, `reports/tables/*`,
`data/images/*`, `data/generated/*`) -- **modelling is frozen**: nothing in this module retrains a
model, re-runs a backtest, or recomputes a forecast for an origin/horizon other than the one
already on disk. `forecast_styles` in particular returns the closest available pre-computed
forecast and says so explicitly, rather than recomputing.

`generate_concept` is the one exception that touches the GPU -- it is a thin wrapper around
`nss.generate.backends.generate_concept` (Track C, untouched here). That is expected: a client
invoking image generation through this tool is a live generation request, not retraining.

`score_concept` runs the shipped quality gates (`nss.generate.qc_gates`): Gate 1 (within-style
p90), Gate 1b (nearest-reference p90, clone-validated) and, on request, Gate 2 (VLM fidelity),
and always reports the human visual check as required. It supersedes the earlier margin-band
scoring (task L1).

SDK note: this module targets the installed `mcp` package (`mcp==2.2.0` at the time this was
written). In `mcp>=2`, the high-level "define tools with a decorator, run over stdio" API that
earlier `mcp` releases (and most public documentation) call `FastMCP` was renamed to
`MCPServer` (`mcp.server.mcpserver.MCPServer`) -- same role, same `@server.tool()` /
`server.run(transport="stdio")` shape, just a new class name. See
https://py.sdk.modelcontextprotocol.io/v2/migration/#fastmcp-renamed-to-mcpserver.

Run standalone (stdio transport, the default):
    uv run python -m nss.mcp_server

See the README's "MCP server" section for a ready-to-paste `mcpServers` client config.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import polars as pl
from mcp.server.mcpserver import MCPServer

from nss.features.style_panel import STYLE_KEY_COLS, STYLE_KEY_SEPARATOR
from nss.generate import backends

# --- Artifact paths -- every tool reads one or more of these, never writes to data/raw or
# data/processed (compose_final_sheet and generate_concept write new files under data/generated,
# which is the project's existing convention for generation outputs). ---
TRANSACTIONS_DIR = Path("data/interim/transactions_train_parquet")
ARTICLES_PATH = Path("data/raw/articles.csv")
STYLE_WEEK_PANEL_PATH = Path("data/processed/style_week_panel.parquet")

# Per-style SHAP drivers: checked in this order. `final_three_shap_verdict.csv` landed on disk
# (via a concurrent Track C task, `nss.models.final_three_shap_verdict`) partway through this
# module's own development; the two forecast tables below are kept as the next fallback since
# they also carry per-style shap_driver_{1..5}_{feature,value} columns for the styles they rank
# (the "broader SHAP outputs" fallback the task brief anticipates for styles absent from the
# verdict table). `shap_global_importance.csv` (dataset-wide, not per-style) is the last resort.
FINAL_THREE_SHAP_VERDICT_PATH = Path("reports/tables/final_three_shap_verdict.csv")
SHAP_GLOBAL_IMPORTANCE_PATH = Path("reports/tables/shap_global_importance.csv")
N_GLOBAL_SHAP_FALLBACK = 5

FORECAST_TABLES: dict[str, Path] = {
    "incumbent": Path("reports/tables/top_styles_t1_incumbent.csv"),
    "emerging": Path("reports/tables/top_styles_t2_emerging.csv"),
}
# The only forecast actually computed on disk (nss.models.final_forecast.FORECAST_ORIGIN + the
# project's fixed 13-week guard/target window). See module docstring -- modelling is frozen, so
# forecast_styles never recomputes for a different origin/horizon; it reports the mismatch instead.
PRECOMPUTED_FORECAST_ORIGIN = "2020-09-21"
PRECOMPUTED_FORECAST_HORIZON_WEEKS = 13

EXEMPLAR_MANIFEST_PATHS = (
    Path("reports/tables/exemplar_images.csv"),
    Path("reports/tables/exemplar_images_final_three.csv"),
)
CONTROL_ROLE = "control"

CONCEPT_SHEET_DIR = Path("data/generated/concept_sheets")

TRAJECTORY_RECENT_WEEKS = 13  # matches the project's own 13-week guard/target window convention.

mcp = MCPServer(
    name="next-season-styles",
    instructions=(
        "Read-only access to H&M fashion trend forecasts, style trajectories, and the SDXL/"
        "Gemini concept-generation pipeline for next-season-styles. All forecasting/analytics "
        "tools read pre-computed artifacts only -- modelling is frozen, nothing here retrains a "
        "model or re-runs a backtest. generate_concept is the one tool that performs live GPU "
        "work (image generation), on request."
    ),
)


def _parse_style_key(style_key: str) -> dict[str, str]:
    """Split a `style_key` string into its 5 constituent `STYLE_KEY_COLS` values.

    Args:
        style_key: `" || "`-joined string, e.g.
            `"Ladieswear || T-shirt || Jersey Basic || Black || Solid"`.

    Returns:
        Mapping of `STYLE_KEY_COLS` column name -> value, in column order.

    Raises:
        ValueError: if `style_key` does not split into exactly `len(STYLE_KEY_COLS)` parts.
    """
    parts = style_key.split(STYLE_KEY_SEPARATOR)
    if len(parts) != len(STYLE_KEY_COLS):
        raise ValueError(
            f"style_key must have {len(STYLE_KEY_COLS)} {STYLE_KEY_SEPARATOR!r}-separated parts "
            f"{STYLE_KEY_COLS}, got {len(parts)} in {style_key!r}"
        )
    return dict(zip(STYLE_KEY_COLS, parts, strict=True))


def _load_exemplar_manifest() -> pl.DataFrame:
    """Load + de-duplicate the exemplar-image manifests (`role`, `style_key`, image path, etc).

    Combines `EXEMPLAR_MANIFEST_PATHS` (skipping any that don't exist) and de-duplicates by
    `article_id`, keeping the first occurrence -- same convention as
    `nss.generate.derive_similarity_band.load_exemplar_manifest`, extended here to keep the
    `role` and `units_sold_last_26w` columns that function drops (needed for `score_concept`'s
    control-style lookup and for ranking reference images by recent sales volume).

    Returns:
        De-duplicated frame with columns `role`, `style_key`, `article_id`,
        `units_sold_last_26w`, `local_image_path`, `fetch_success`. Empty (correctly-typed) frame
        if none of `EXEMPLAR_MANIFEST_PATHS` exist.
    """
    frames = [pl.read_csv(p) for p in EXEMPLAR_MANIFEST_PATHS if p.exists()]
    if not frames:
        return pl.DataFrame(
            schema={
                "role": pl.String,
                "style_key": pl.String,
                "article_id": pl.Int64,
                "units_sold_last_26w": pl.Int64,
                "local_image_path": pl.String,
                "fetch_success": pl.Boolean,
            }
        )
    combined = pl.concat(frames)
    return combined.unique(subset=["article_id"], keep="first").sort(["style_key", "role"])


def query_transactions(style_key: str | None, date_from: str, date_to: str) -> dict[str, Any]:
    """Aggregate transaction stats (units, revenue, n_customers) over a date range.

    Reads `data/interim/transactions_train_parquet/` (predicate-pushed-down filter on `t_dat`),
    optionally restricted to articles matching `style_key` via an inner join against
    `data/raw/articles.csv`'s 5 style columns.

    Args:
        style_key: `" || "`-joined style key to filter to, or `None` for all styles.
        date_from: Inclusive start date, `"YYYY-MM-DD"`.
        date_to: Inclusive end date, `"YYYY-MM-DD"`.

    Returns:
        `{"style_key", "date_from", "date_to", "units", "revenue", "n_customers"}`. `units` is a
        transaction-row count (one row = one purchased item in this dataset). `revenue` is the
        sum of Kaggle's normalized `price` units, not real currency. All-zero if nothing matches.

    Raises:
        ValueError: `date_from`/`date_to` aren't valid ISO dates, `date_from > date_to`, or
            `style_key` is given but malformed (see `_parse_style_key`).
    """
    d_from = date.fromisoformat(date_from)
    d_to = date.fromisoformat(date_to)
    if d_from > d_to:
        raise ValueError(f"date_from ({date_from}) must not be after date_to ({date_to})")

    lf = pl.scan_parquet(str(TRANSACTIONS_DIR / "**" / "*.parquet"))
    lf = lf.filter((pl.col("t_dat") >= d_from) & (pl.col("t_dat") <= d_to))

    if style_key is not None:
        parts = _parse_style_key(style_key)
        articles = pl.scan_csv(ARTICLES_PATH).select(["article_id", *STYLE_KEY_COLS])
        for col, value in parts.items():
            articles = articles.filter(pl.col(col) == value)
        lf = lf.join(articles.select("article_id"), on="article_id", how="inner")

    row = (
        lf.select(
            units=pl.len(),
            revenue=pl.col("price").sum(),
            n_customers=pl.col("customer_id").n_unique(),
        )
        .collect()
        .row(0, named=True)
    )

    return {
        "style_key": style_key,
        "date_from": date_from,
        "date_to": date_to,
        "units": int(row["units"]),
        "revenue": float(row["revenue"]),
        "n_customers": int(row["n_customers"]),
    }


def _extract_shap_drivers_row(row: dict[str, Any], n: int = 5) -> list[dict[str, Any]]:
    """Pull `shap_driver_{i}_feature`/`shap_driver_{i}_value` pairs out of one table row.

    Args:
        row: A `dict`-shaped row (e.g. from `pl.DataFrame.row(0, named=True)`) that may contain
            `shap_driver_1_feature` .. `shap_driver_{n}_value` columns.
        n: Number of driver slots to look for.

    Returns:
        List of `{"feature": str, "value": float}`, in driver-rank order, skipping any slot whose
        columns aren't present in `row`.
    """
    drivers = []
    for i in range(1, n + 1):
        feature_key, value_key = f"shap_driver_{i}_feature", f"shap_driver_{i}_value"
        if feature_key in row and value_key in row and row[feature_key] is not None:
            drivers.append({"feature": row[feature_key], "value": float(row[value_key])})
    return drivers


def _get_shap_drivers(style_key: str) -> dict[str, Any]:
    """Look up per-style SHAP drivers for `style_key`, falling back to global importance.

    Check order: `FINAL_THREE_SHAP_VERDICT_PATH` (the name given in the task brief; not present
    on disk as of writing) -> each of `FORECAST_TABLES` (both carry per-style
    `shap_driver_{1..5}_*` columns for the styles they rank) -> `SHAP_GLOBAL_IMPORTANCE_PATH`
    (dataset-wide feature importance, not style-specific) as a last resort.

    Args:
        style_key: `" || "`-joined style key.

    Returns:
        `{"source": str | None, "style_specific": bool, "drivers": list[dict], "note": str |
        None}`. `note` explains a fallback (or total absence) explicitly rather than silently
        returning global numbers as if they were style-specific.
    """
    candidate_paths = [FINAL_THREE_SHAP_VERDICT_PATH, *FORECAST_TABLES.values()]
    for path in candidate_paths:
        if not path.exists():
            continue
        df = pl.read_csv(path)
        match = df.filter(pl.col("style_key") == style_key)
        if match.height > 0:
            return {
                "source": str(path),
                "style_specific": True,
                "drivers": _extract_shap_drivers_row(match.row(0, named=True)),
                "note": None,
            }

    if SHAP_GLOBAL_IMPORTANCE_PATH.exists():
        global_df = pl.read_csv(SHAP_GLOBAL_IMPORTANCE_PATH).head(N_GLOBAL_SHAP_FALLBACK)
        drivers = [
            {"feature": r["feature"], "value": float(r["mean_abs_shap"])}
            for r in global_df.iter_rows(named=True)
        ]
        return {
            "source": str(SHAP_GLOBAL_IMPORTANCE_PATH),
            "style_specific": False,
            "drivers": drivers,
            "note": (
                f"No style-specific SHAP values found for {style_key!r} in "
                f"{[str(p) for p in candidate_paths]}; returning dataset-wide feature "
                "importance instead."
            ),
        }

    return {
        "source": None,
        "style_specific": False,
        "drivers": [],
        "note": "No SHAP outputs found on disk (neither per-style nor global).",
    }


def get_style_profile(style_key: str) -> dict[str, Any]:
    """Attributes, weekly trajectory summary, and SHAP drivers for one style.

    Args:
        style_key: `" || "`-joined style key (the 5 `STYLE_KEY_COLS` values).

    Returns:
        `{"style_key", "attributes", "trajectory", "shap_drivers"}`. `attributes` is the 5
        constituent style columns. `trajectory` summarizes the trailing `TRAJECTORY_RECENT_WEEKS`
        weeks from `data/processed/style_week_panel.parquet` if the style is present there
        (`{"available": False, "note": ...}` otherwise -- e.g. it was filtered out by the
        project's support gate, or never observed). `shap_drivers` is `_get_shap_drivers`'s
        output -- explicitly notes when no style-specific SHAP values exist for this style.

    Raises:
        ValueError: `style_key` is malformed (see `_parse_style_key`).
    """
    attributes = _parse_style_key(style_key)

    panel_df = (
        pl.scan_parquet(STYLE_WEEK_PANEL_PATH)
        .filter(pl.col("style_key") == style_key)
        .sort("week_start")
        .collect()
    )
    if panel_df.height == 0:
        trajectory: dict[str, Any] = {
            "available": False,
            "note": (
                f"style_key {style_key!r} has no rows in {STYLE_WEEK_PANEL_PATH} -- either "
                "filtered out by the project's support gate or never observed."
            ),
        }
    else:
        recent = panel_df.tail(TRAJECTORY_RECENT_WEEKS)
        recent_weeks = [
            {
                "week_start": r["week_start"].isoformat(),
                "units": int(r["units"]),
                "revenue": float(r["revenue"]),
                "price_index": r["price_index"],
                "intensity_shrunk": r["intensity_shrunk"],
            }
            for r in recent.iter_rows(named=True)
        ]
        trajectory = {
            "available": True,
            "first_week_seen": panel_df["first_week_seen"][0].isoformat(),
            "last_week_seen": panel_df["last_week_seen"][0].isoformat(),
            "n_weeks_total": panel_df.height,
            "recent_weeks": recent_weeks,
            "recent_mean_units": float(recent["units"].mean()),
            "recent_mean_revenue": float(recent["revenue"].mean()),
        }

    return {
        "style_key": style_key,
        "attributes": attributes,
        "trajectory": trajectory,
        "shap_drivers": _get_shap_drivers(style_key),
    }


def forecast_styles(
    origin_date: str, horizon_weeks: int, table: str, top_n: int
) -> list[dict[str, Any]]:
    """Top `top_n` styles from an already-computed forecast table.

    Never retrains or recomputes -- reads `reports/tables/top_styles_t1_incumbent.csv`
    (`table="incumbent"`) or `reports/tables/top_styles_t2_emerging.csv`
    (`table="emerging"`). If `origin_date`/`horizon_weeks` don't match the only forecast actually
    computed (`PRECOMPUTED_FORECAST_ORIGIN`, `PRECOMPUTED_FORECAST_HORIZON_WEEKS`), the closest
    (only) available data is returned anyway, with an explicit `_note` field on every row.

    Args:
        origin_date: Requested forecast origin, `"YYYY-MM-DD"`.
        horizon_weeks: Requested forecast horizon in weeks.
        table: `"incumbent"` or `"emerging"`.
        top_n: Number of top-ranked rows to return.

    Returns:
        Up to `top_n` row dicts (rank-ascending) from the requested table, each carrying
        `_forecast_origin_date`/`_forecast_horizon_weeks` (what was actually computed),
        `_requested_origin_date`/`_requested_horizon_weeks` (what was asked for), and `_note`
        (only present when the request didn't match what's on disk).

    Raises:
        ValueError: `table` isn't a key of `FORECAST_TABLES`, or `top_n < 1`.
    """
    if table not in FORECAST_TABLES:
        raise ValueError(f"table must be one of {sorted(FORECAST_TABLES)}, got {table!r}")
    if top_n < 1:
        raise ValueError(f"top_n must be >= 1, got {top_n}")

    path = FORECAST_TABLES[table]
    rows = pl.read_csv(path).sort("rank").head(top_n).to_dicts()

    matches_precomputed = (
        origin_date == PRECOMPUTED_FORECAST_ORIGIN
        and horizon_weeks == PRECOMPUTED_FORECAST_HORIZON_WEEKS
    )
    note = None
    if not matches_precomputed:
        note = (
            f"Requested origin_date={origin_date!r}/horizon_weeks={horizon_weeks} does not "
            f"match the only pre-computed forecast on disk (origin={PRECOMPUTED_FORECAST_ORIGIN}"
            f", horizon_weeks={PRECOMPUTED_FORECAST_HORIZON_WEEKS}). Modelling is frozen for this "
            "tool -- returning that forecast unchanged rather than recomputing."
        )

    for row in rows:
        row["_forecast_origin_date"] = PRECOMPUTED_FORECAST_ORIGIN
        row["_forecast_horizon_weeks"] = PRECOMPUTED_FORECAST_HORIZON_WEEKS
        row["_requested_origin_date"] = origin_date
        row["_requested_horizon_weeks"] = horizon_weeks
        if note is not None:
            row["_note"] = note
    return rows


def get_reference_images(style_key: str, n: int) -> list[str]:
    """Up to `n` local reference image paths for a style, ranked by recent sales volume.

    Args:
        style_key: `" || "`-joined style key.
        n: Maximum number of image paths to return.

    Returns:
        Up to `n` `local_image_path` values (POSIX-style, e.g. `"data/images/0610776002.jpg"`)
        from the exemplar manifests, restricted to `fetch_success == True` rows and sorted by
        `units_sold_last_26w` descending. Empty list if `style_key` has no fetched exemplars.

    Raises:
        ValueError: `n < 1`.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    manifest = _load_exemplar_manifest()
    matches = manifest.filter((pl.col("style_key") == style_key) & pl.col("fetch_success")).sort(
        "units_sold_last_26w", descending=True
    )
    return [Path(p).as_posix() for p in matches["local_image_path"].head(n).to_list()]


def generate_concept(
    prompt: str,
    reference_images: list[str],
    backend: str,
    ip_adapter_scale: float | None,
    seed: int,
    n: int,
) -> list[str]:
    """Generate `n` concept images -- thin wrapper around `nss.generate.backends.generate_concept`.

    This is the one tool in this module that performs live GPU (or paid-API) work when actually
    invoked: `backend="local_sdxl"` runs SDXL + IP-Adapter locally; `backend="gemini"` calls the
    Gemini 2.5 Flash Image API. No logic is duplicated here -- see `nss.generate.backends` for
    the full generation implementation, output paths, and metadata sidecar format.

    Args:
        prompt: Text prompt describing the fashion concept.
        reference_images: Local image paths used as style/content references (non-empty).
        backend: `"local_sdxl"` or `"gemini"`.
        ip_adapter_scale: IP-Adapter conditioning strength (`"local_sdxl"` only; must be `None`
            for `"gemini"`).
        seed: Random seed.
        n: Number of images to generate.

    Returns:
        POSIX-style paths to the `n` saved images.

    Raises:
        ValueError: see `nss.generate.backends.generate_concept`.
        RuntimeError: missing `GEMINI_API_KEY`, or no CUDA GPU visible for `"local_sdxl"`.
    """
    ref_paths = [Path(p) for p in reference_images]
    result_paths = backends.generate_concept(
        prompt=prompt,
        reference_images=ref_paths,
        backend=backend,
        ip_adapter_scale=ip_adapter_scale,
        seed=seed,
        n=n,
    )
    return [p.as_posix() for p in result_paths]


def score_concept(
    concept_path: str, style_key: str, include_fidelity: bool = False
) -> dict[str, Any]:
    """Score a generated concept image against the project's SHIPPED quality gates.

    Runs `nss.generate.qc_gates.score_gates` (the same functions that scored the final concepts):

    * **Gate 1** -- mean similarity to the style's screened reference photos must be at or below
      the p90 of similarity between distinct REAL articles of that style (CLIP and DINOv2).
    * **Gate 1b** -- the closest single reference must be at or below the p90 of the real
      nearest-sibling similarity; validated live by an exact-clone control that must fail.
    * **Gate 2** -- blind VLM attribute fidelity vs the style's visible attributes, against the
      judge's own calibrated threshold; reported with and without excluded non-visual attributes.
      Only run when `include_fidelity=True` (one Groq call; a single reading varies by ~+/-0.21).
    * **Human visual check** -- always `required`, never automated: the automatic gates have
      passed visibly malformed garments.

    Args:
        concept_path: Path to the generated concept image.
        style_key: `" || "`-joined style key the concept was generated for.
        include_fidelity: Also make one blind judge call for Gate 2 (network).

    Returns:
        `{"gate1", "gate1b", "gate2", "human_visual_check", "automated_gates_pass", "verdict",
        ...}` -- see `nss.generate.qc_gates.score_gates`.

    Raises:
        FileNotFoundError: `concept_path` does not exist.
        ValueError: no screened reference images exist for `style_key`.
    """
    from nss.generate import qc_gates

    return qc_gates.score_gates(concept_path, style_key, include_fidelity=include_fidelity)


def forecast_concept(concept_path: str, include_api_judges: bool = True) -> dict[str, Any]:
    """Score a generated concept THROUGH THE SAME FORECASTER (the closed loop). PROTOTYPE.

    Image retrieval: embed the concept (CLIP ViT-L/14 + DINOv2) -> nearest catalogue STYLE by the
    average cosine similarity to the mean embedding of that style's real photos -> look up the
    frozen model's forecast for that style. The result is about the ARCHETYPE the image reads as,
    not a demand forecast for the new design. Styles are near-ties by construction: read `top5`,
    not only the top-1.

    Args:
        concept_path: Path to the concept image.
        include_api_judges: Deprecated and ignored. Retrieval uses no VLM judges; the parameter is
            kept so existing callers do not break.

    Returns:
        `{"sentence", "style_key", "forecast_units_per_product_per_week", "rank", "n_styles",
        "match_level", "confidence", "judges", "unavailable_judges", "top5", "similarity",
        "margin"}`. `confidence` (`high`/`medium`/`low`) comes from CLIP/DINOv2 agreement and the
        top-1 margin over the 6th-ranked style, never from the forecast itself; `judges` lists the
        two embedding views.

    Raises:
        FileNotFoundError: `concept_path` does not exist.
    """
    from nss.generate import concept_forecast

    path = Path(concept_path)
    if not path.exists():
        raise FileNotFoundError(f"concept image not found: {concept_path}")
    result = concept_forecast.forecast_concept(path)
    return {
        "sentence": result.sentence(),
        "style_key": result.style_key,
        "forecast_units_per_product_per_week": result.forecast,
        "rank": result.rank,
        "n_styles": result.n_styles,
        "match_level": result.match_level,
        "confidence": result.confidence,
        "judges": sorted(result.normalised),
        "unavailable_judges": result.unavailable,
        "top5": result.top5,
        "similarity": result.similarity,
        "margin": result.margin,
    }


def compose_final_sheet(concept_paths: list[str], captions: list[str]) -> str:
    """Compose a labeled, single-row multi-panel image from generated concepts + captions.

    Reuses the headless-matplotlib image-grid pattern already established by
    `nss.generate.scale_sweep.plot_image_strip`.

    Args:
        concept_paths: Paths to the concept images to compose, left to right.
        captions: Caption text for each image, same length/order as `concept_paths`.

    Returns:
        POSIX-style path to the saved PNG under `data/generated/concept_sheets/`.

    Raises:
        ValueError: `concept_paths` is empty, or its length doesn't match `captions`.
    """
    if not concept_paths:
        raise ValueError("concept_paths must be non-empty")
    if len(concept_paths) != len(captions):
        raise ValueError(
            f"concept_paths ({len(concept_paths)}) and captions ({len(captions)}) must be the "
            "same length"
        )

    import matplotlib

    matplotlib.use("Agg")  # headless -- this function only writes a PNG, never shows a window.
    import matplotlib.pyplot as plt
    from PIL import Image

    n = len(concept_paths)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.5))
    axes_flat = [axes] if n == 1 else list(axes)
    for ax, path, caption in zip(axes_flat, concept_paths, captions, strict=True):
        image = Image.open(path).convert("RGB")
        ax.imshow(image)
        ax.set_title(caption, fontsize=10)
        ax.axis("off")
    fig.tight_layout()

    CONCEPT_SHEET_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
    out_path = CONCEPT_SHEET_DIR / f"sheet_{timestamp}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path.as_posix()


# Register the 7 tools with the MCPServer instance. Kept as plain module-level functions (rather
# than defined inline under `@mcp.tool()`) so they stay directly unit-testable without going
# through the MCP protocol -- `mcp.tool()` returns the original function unchanged (verified
# against the installed `mcp==2.2.0` source), so this registration step doesn't affect either
# call path.
mcp.tool()(query_transactions)
mcp.tool()(get_style_profile)
mcp.tool()(forecast_styles)
mcp.tool()(get_reference_images)
mcp.tool()(generate_concept)
mcp.tool()(score_concept)
mcp.tool()(compose_final_sheet)
mcp.tool()(forecast_concept)


def main() -> None:
    """CLI entry point: run this MCP server over stdio."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
