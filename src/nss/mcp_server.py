"""MCP server exposing next-season-styles' read-only forecasting + generation tools (task D1).

Every tool here reads artifacts already on disk (raw/interim/processed data, `reports/tables/*`,
`data/images/*`, `data/generated/*`) -- **modelling is frozen**: nothing in this module retrains a
model, re-runs a backtest, or recomputes a forecast for an origin/horizon other than the one
already on disk. `forecast_styles` in particular returns the closest available pre-computed
forecast and says so explicitly, rather than recomputing.

`generate_concept` is the one exception that touches the GPU -- it is a thin wrapper around
`nss.generate.backends.generate_concept` (Track C, untouched here). That is expected: a client
invoking image generation through this tool is a live generation request, not retraining.

`score_concept` wraps `nss.generate.clip_scoring`, `nss.generate.dino_scoring`, and
`nss.generate.margin_scoring` (all Track C, untouched here -- CPU-only, consumed as-is). Track C
was landing these concurrently with this module's own development; `margin_scoring.py` and
`dino_scoring.py` existed by the time this was finished, but `nss.generate.derive_margin_band`'s
own output (`reports/tables/margin_anchors_{clip,dinov2}.csv`) did not -- `score_concept`
degrades gracefully (documented per-call in its `note` field) rather than computing those bands
itself, which is Track C's judgment call to make, not this module's.

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
from nss.generate import backends, clip_scoring, dino_scoring
from nss.generate.clip_scoring import clip_similarity, control_similarity
from nss.generate.derive_margin_band import OUT_PATHS as MARGIN_BAND_PATHS
from nss.generate.margin_scoring import margin as embedding_margin

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
CLIP_BAND_PATH = Path("reports/tables/clip_similarity_band.csv")
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


def _load_clip_band() -> tuple[float, float]:
    """Read the empirically-derived CLIP in-band `(lower, upper)` thresholds from disk.

    Returns:
        `(lower, upper)` from `reports/tables/clip_similarity_band.csv`.

    Raises:
        RuntimeError: the band CSV doesn't exist -- `nss.generate.derive_similarity_band` must
            have been run at least once (it has, at the time of writing).
    """
    if not CLIP_BAND_PATH.exists():
        raise RuntimeError(
            f"{CLIP_BAND_PATH} not found -- run nss.generate.derive_similarity_band first."
        )
    row = pl.read_csv(CLIP_BAND_PATH).row(0, named=True)
    return float(row["lower"]), float(row["upper"])


def _load_margin_band(space: str) -> tuple[float, float] | None:
    """Read the `[band_lower, band_upper]` margin thresholds `derive_margin_band` would write.

    Args:
        space: `"clip"` or `"dinov2"` (a key of `MARGIN_BAND_PATHS`).

    Returns:
        `(lower, upper)`, or `None` if `nss.generate.derive_margin_band` has not been run for
        `space` yet -- its output CSV (`MARGIN_BAND_PATHS[space]`) does not exist on disk. This
        script also depends on `reports/tables/exemplar_images_margin_reference.csv`
        (`nss.data.select_margin_reference_styles`'s output), which does not exist yet either as
        of this writing -- confirmed by direct file-existence check, not assumed.
    """
    path = MARGIN_BAND_PATHS[space]
    if not path.exists():
        return None
    row = pl.read_csv(path).row(0, named=True)
    return float(row["band_lower"]), float(row["band_upper"])


def score_concept(concept_path: str, style_key: str) -> dict[str, Any]:
    """Score a generated concept image's CLIP/DINOv2 similarity + margin against its target style.

    Combines two generations of the project's novelty scoring:

    1. **CLIP absolute-similarity band** (`nss.generate.clip_scoring` /
       `nss.generate.derive_similarity_band`, task B3) -- always available, already on disk.
    2. **Margin-based scoring in both embedding spaces** (`nss.generate.margin_scoring` +
       `nss.generate.dino_scoring`, task C2) -- computes `clip_margin`/`dino_margin` live via
       `margin_scoring.margin()` (a stable, already-landed pure function; not duplicated here),
       and looks up each space's derived `[lower, upper]` band from
       `reports/tables/margin_anchors_{clip,dinov2}.csv` if `nss.generate.derive_margin_band` has
       been run. That script's own output did not exist on disk at the time this was written
       (verified: no such CSVs under `reports/tables/`, and its own input --
       `exemplar_images_margin_reference.csv` -- doesn't exist either), so
       `in_clip_margin_band`/`in_dino_margin_band` are `None` until it lands; `pass` falls back to
       the B3 CLIP band alone in that case, and upgrades to requiring both margin bands to pass
       once available (see `note`, which states which policy was actually used for a given call).

    Args:
        concept_path: Path to the generated concept image.
        style_key: `" || "`-joined style key the concept was generated for.

    Returns:
        `{"concept_path", "style_key", "clip_similarity", "control_similarity", "clip_band",
        "in_clip_band", "clip_margin", "dino_margin", "margin_bands", "in_clip_margin_band",
        "in_dino_margin_band", "pass", "note"}`. Margin fields are `None` when no `role ==
        "control"` exemplar images exist on disk (margin is undefined without a control pool).

    Raises:
        FileNotFoundError: `concept_path` does not exist.
        ValueError: no fetched reference images exist for `style_key`.
    """
    concept = Path(concept_path)
    if not concept.exists():
        raise FileNotFoundError(f"concept_path {concept_path!r} does not exist")

    manifest = _load_exemplar_manifest()
    ref_rows = manifest.filter((pl.col("style_key") == style_key) & pl.col("fetch_success"))
    if ref_rows.height == 0:
        raise ValueError(f"No fetched reference images found for style_key {style_key!r}")
    reference_imgs = [Path(p) for p in ref_rows["local_image_path"].to_list()]

    control_rows = manifest.filter((pl.col("role") == CONTROL_ROLE) & pl.col("fetch_success"))
    control_imgs = [Path(p) for p in control_rows["local_image_path"].to_list()]

    own_sim = clip_similarity(concept, reference_imgs)
    clip_lower, clip_upper = _load_clip_band()
    in_clip_band = clip_lower <= own_sim["mean"] <= clip_upper

    control_sim: dict[str, float] | None = None
    clip_margin_value: float | None = None
    dino_margin_value: float | None = None
    if control_imgs:
        control_sim = control_similarity(concept, control_imgs)

        clip_ref_embs = [clip_scoring.embed_image(p) for p in reference_imgs]
        clip_control_embs = [clip_scoring.embed_image(p) for p in control_imgs]
        clip_margin_value = embedding_margin(
            clip_scoring.embed_image(concept), clip_ref_embs, clip_control_embs
        )

        dino_ref_embs = [dino_scoring.embed_image(p) for p in reference_imgs]
        dino_control_embs = [dino_scoring.embed_image(p) for p in control_imgs]
        dino_margin_value = embedding_margin(
            dino_scoring.embed_image(concept), dino_ref_embs, dino_control_embs
        )

    clip_margin_band = _load_margin_band("clip")
    dino_margin_band = _load_margin_band("dinov2")
    in_clip_margin_band = (
        clip_margin_band[0] <= clip_margin_value <= clip_margin_band[1]
        if clip_margin_band is not None and clip_margin_value is not None
        else None
    )
    in_dino_margin_band = (
        dino_margin_band[0] <= dino_margin_value <= dino_margin_band[1]
        if dino_margin_band is not None and dino_margin_value is not None
        else None
    )

    if in_clip_margin_band is not None and in_dino_margin_band is not None:
        overall_pass = in_clip_margin_band and in_dino_margin_band
        note = (
            "pass = in_clip_margin_band AND in_dino_margin_band (nss.generate.derive_margin_band "
            "has been run for both embedding spaces; this tool's own conservative combination "
            "policy -- requiring both to agree -- not an asserted project-wide convention)."
        )
    else:
        overall_pass = in_clip_band
        note = (
            "pass = in_clip_band (legacy B3 absolute-cosine band) -- "
            "nss.generate.derive_margin_band has not produced margin_anchors_{clip,dinov2}.csv "
            "yet, so the margin-based bands are unavailable. clip_margin/dino_margin are still "
            "reported (computed live via nss.generate.margin_scoring.margin) for visibility."
        )

    return {
        "concept_path": str(concept),
        "style_key": style_key,
        "clip_similarity": own_sim,
        "control_similarity": control_sim,
        "clip_band": {"lower": clip_lower, "upper": clip_upper},
        "in_clip_band": in_clip_band,
        "clip_margin": clip_margin_value,
        "dino_margin": dino_margin_value,
        "margin_bands": {
            "clip": None if clip_margin_band is None else list(clip_margin_band),
            "dinov2": None if dino_margin_band is None else list(dino_margin_band),
        },
        "in_clip_margin_band": in_clip_margin_band,
        "in_dino_margin_band": in_dino_margin_band,
        "pass": overall_pass,
        "note": note,
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


def main() -> None:
    """CLI entry point: run this MCP server over stdio."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
