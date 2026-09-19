"""Single, non-interactive, end-to-end reproduction script (task D3): panel -> features ->
forecast -> top-3 -> briefs -> generate -> score -> hero image.

Runs the full chain with NO agent runtime required, from one command:

    uv run --no-sync python scripts/run_pipeline.py

Every step below calls an ALREADY-EXISTING, already-tested function from this project's modules,
in sequence, with the SAME frozen hyperparameters/config those functions already use everywhere
else in the codebase -- this script is pure orchestration/wiring, it reimplements no modelling,
selection, or generation logic.

HARD CONSTRAINT ("modelling is frozen"): no hyperparameter, feature, or target definition is
touched here. This project has no persisted model artifact -- every prediction happens via a
fresh `.fit()` call with the SAME frozen hyperparameters/data/code, confirmed bit-identical across
processes by task D3a (`tests/test_determinism_cross_process.py`). Calling
`nss.models.final_forecast.train_final_model` from this script is therefore normal pipeline
execution, not retraining in the forbidden sense.

STAGE-OUTPUT ISOLATION (load-bearing, read before changing default paths): the `panel`,
`features`, `forecast`, and `briefs` stages are fully DETERMINISTIC given the frozen model/config
(see D3a) -- they default to writing straight into the REAL `reports/tables/` location, which
safely just reproduces the already-committed artifacts bit-for-bit. The `generate`, `score`, and
`hero` stages are NOT wire-compatible with the real E5/E8 deliverables, because this script
deliberately generates only 1-2 seeds per style, round-0 only, no retry rounds (see SCOPING-DOWN
below) instead of E5's full adaptive up-to-8-seed retry sweep -- a genuinely different
(lower-fidelity) selection could result. Those three stages therefore ALWAYS write to an isolated
`--pipeline-out-dir` (default `reports/pipeline_run/`, gitignored) and NEVER touch
`reports/tables/final_concepts_v2.csv`, or `reports/figures/FINAL_concepts.png` /
`evidence_chain.png` -- those remain the authoritative, unmodified E5/E8 deliverables (the same
holds for the older, superseded `reports/tables/final_concepts.csv` /
`reports/tables/concept_qc_results.csv` -- this script has never written to those and still
doesn't).

RECONNECTED TO THE CURRENT PIPELINE (task, post-E8): the `generate`/`score` stages call
`nss.generate.final_concepts_v2` -- E5's adaptive per-style retry-loop module with the corrected
E2 Gate-1 copy-check, `ip_adapter_scale=0.45`, and the fixed (<=77-token, truncation-safe)
prompts -- reusing its own `score_candidate`/`select_final_candidate`/`write_results_table`/
`apply_visual_qc_and_rewrite` primitives verbatim, NOT the superseded C6/C7
`nss.generate.final_concepts.select_best_candidate` (old two-sided real-space band) /
`nss.generate.concept_qc_pipeline.run_qc_with_retries` (old QC gate) path this script used before.
The `hero` stage calls `nss.generate.final_deliverables.main(final_concepts_v2_path=..., ...)`,
E8's current signature. `generate`/`score` also load reference images via
`nss.generate.screen_references.load_screened_references` (task F3), not
`final_concepts.load_final_three_references` -- see that module's docstring for why.

SCOPING-DOWN (documented, not silently done): task D3's purpose is proving the WIRING between
already-tested stages works end-to-end from one command, not re-deriving E5's full adaptive
retry-sweep selection. `--n-seeds` (default 1, max 2) controls how many of
`final_concepts_v2.INITIAL_SEEDS` (E5's own round-0 seed set, unchanged in value from C6's
`final_concepts.SEEDS`) are generated per style, round-0 only -- this script never runs a retry
round, since 1-2 seeds is enough to prove the wiring (see `run_score_stage`). The operating point
(`ip_adapter_scale=0.45`) and the full-gate selection rule itself are reused verbatim from
`nss.generate.final_concepts_v2`, unchanged. Any seed `final_concepts_v2.VISUAL_QC_DISQUALIFIED_
SEEDS` records for a style is still skipped when choosing which seed(s) to generate (see
`seeds_for_style`) -- currently a no-op at `n_seeds<=2` since every presently-disqualified seed
(the T-shirt style's 46-49) was only ever generated in a retry round this script never reaches,
but applied unconditionally for correctness.

VLM judges (Gemini/Groq) in the `score` stage are exercised when reachable but are optional: a
`GROQ_API_KEY` availability check that raises an unexpected error (not the two documented,
already-handled cases inside `nss.generate.concept_qc_pipeline`/`skills/concept-qc/run_qc.py`)
degrades to "Groq unavailable" with a printed note rather than failing the whole script -- the
margin-band scoring (CLIP + DINOv2, no external API) always runs.

`--tables-out-dir` exists ONLY for the task D3 step-3 cross-process forecast-determinism re-check
(two separate process runs, each pointed at its own scratch directory, diffed against each other)
-- combining it with `--stop-after` beyond `forecast` is refused at startup, because the
`briefs`/`generate`/`score`/`hero` stages call functions that read `design_briefs.json` /
exemplar-reference paths from their REAL `reports/tables/` defaults regardless of this flag.

BRIEFS-STAGE GUARD (load-bearing, see `is_curated_design_briefs`/`run_briefs_stage`): the real,
committed `design_briefs.json` has been hand-refined (tasks C5/E5) with an `applied_changes` field
per style and SDXL-77-CLIP-token-safe prompts -- neither of which the naive
`build_design_briefs.build_all_design_briefs()` rebuild reproduces. So the `briefs` stage reuses an
existing curated file untouched by default instead of silently overwriting it with a worse rebuild
(which would also later crash the `hero` stage with a `KeyError` on the missing `applied_changes`
key). Pass `--force-briefs` to genuinely rebuild from scratch anyway.

Usage:
    uv run --no-sync python scripts/run_pipeline.py                       # full pipeline
    uv run --no-sync python scripts/run_pipeline.py --stop-after forecast # panel+features+
                                                                           # forecast+top-3 only
    uv run --no-sync python scripts/run_pipeline.py --stop-after briefs   # skip GPU generation
    uv run --no-sync python scripts/run_pipeline.py --n-seeds 2           # 2 seeds/style
    uv run --no-sync python scripts/run_pipeline.py --force-briefs        # rebuild briefs (loses
                                                                           # applied_changes/token
                                                                           # -safety curation)
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from nss.data import select_final_three_exemplars
from nss.data.fetch_images import fetch_images
from nss.features import model_features, style_panel
from nss.features.style_panel import STYLE_KEY_COLS
from nss.generate import build_design_briefs as build_design_briefs_mod
from nss.generate import (
    concept_qc_pipeline,
    final_concepts,
    final_concepts_v2,
    final_deliverables,
    screen_references,
    vlm_judges,
)
from nss.generate.derive_margin_band import CONTROL_MANIFEST_PATH, load_control_pool
from nss.generate.scale_sweep import free_sdxl_pipeline
from nss.models import diversity_forecast, final_forecast, final_three_shap_verdict

STAGE_ORDER: tuple[str, ...] = (
    "panel",
    "features",
    "forecast",
    "briefs",
    "generate",
    "score",
    "hero",
)

DEFAULT_TABLES_DIR = Path("reports/tables")
DEFAULT_IMAGES_DIR = Path("data/images")
DEFAULT_PIPELINE_OUT_DIR = Path("reports/pipeline_run")
DEFAULT_GENERATED_IMAGES_DIR = Path("data/generated/pipeline_run")


def _stage_index(stage: str) -> int:
    """Position of `stage` in `STAGE_ORDER` (for `--stop-after` gating)."""
    return STAGE_ORDER.index(stage)


# --- Stage 1: panel ----------------------------------------------------------------------------


def run_panel_stage(
    rebuild: bool,
    transactions_dir: Path,
    articles_path: Path,
    panel_path: Path,
) -> pl.DataFrame:
    """Load the dense style-week panel from `panel_path`, or build it if missing/`rebuild=True`.

    Mirrors `nss.features.style_panel.main`'s body exactly (same functions, same order, same
    retention gate) -- the only difference is this returns the built `DataFrame` in-memory for the
    rest of this script's stages, instead of only writing it to disk and exiting.

    Args:
        rebuild: Force a rebuild from raw/interim data even if `panel_path` already exists.
        transactions_dir: Directory of the year-month partitioned transactions Parquet dataset.
        articles_path: Path to `articles.csv`.
        panel_path: Path to read/write the dense panel parquet.

    Returns:
        The dense style-week panel.

    Raises:
        SystemExit: if a freshly-built panel fails the retention gate (same gate as
            `nss.features.style_panel.main`; never loosened silently).
    """
    if panel_path.exists() and not rebuild:
        panel = pl.read_parquet(panel_path)
        print(f"[panel] loaded existing panel from {panel_path} ({panel.height} rows)")
        return panel

    print(f"[panel] building panel from {transactions_dir} + {articles_path} (rebuild={rebuild})")
    raw_panel, lifetime = style_panel.build_style_week_panel(transactions_dir, articles_path)
    filtered_panel, stats = style_panel.filter_by_support(raw_panel, lifetime)
    print(
        f"[panel] style count before/after filtering: "
        f"{stats['n_styles_before']}/{stats['n_styles_after']}"
    )
    print(f"[panel] % lifetime units retained: {stats['pct_units_retained']:.2f}%")
    if stats["pct_units_retained"] < style_panel.RETENTION_GATE_PCT:
        raise SystemExit(
            f"[panel] GATE FAILED: retention {stats['pct_units_retained']:.2f}% is below the "
            f"{style_panel.RETENTION_GATE_PCT:.0f}% gate -- not writing the panel. Same gate as "
            "nss.features.style_panel.main; do not loosen it without explicit sign-off."
        )
    dense_panel = style_panel.densify_panel(filtered_panel)
    dense_panel = style_panel.add_price_index(dense_panel)
    dense_panel = style_panel.add_intensity_shrunk(dense_panel)
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    dense_panel.write_parquet(panel_path)
    print(f"[panel] wrote dense panel to {panel_path} ({dense_panel.height} rows)")
    return dense_panel


# --- Stage 2: features --------------------------------------------------------------------------


def run_features_stage(panel: pl.DataFrame, forecast_origin: date) -> pl.DataFrame:
    """Confirm `build_features` runs cleanly against the loaded panel at the forecast origin.

    `final_forecast.train_final_model` / `build_forecast_frame` (called by the `forecast` stage)
    independently call `build_features` again with their own, wider, multi-origin origin set --
    this call is a cheap (single-origin), non-consumed sanity check that the feature pipeline is
    wired correctly BEFORE the much more expensive training call, not a required input to later
    stages (see module docstring).

    Args:
        panel: The dense style-week panel.
        forecast_origin: The forecast origin week to confirm.

    Returns:
        The single-origin feature frame (unused by later stages; returned for inspection/testing).

    Raises:
        ValueError: if `forecast_origin` does not match the panel's actual last observed week
            (via `final_forecast.verify_forecast_origin`).
    """
    final_forecast.verify_forecast_origin(panel, forecast_origin)
    frame = model_features.build_features(panel, [forecast_origin])
    print(f"[features] {frame.height} eligible (style_key, origin_week) rows at {forecast_origin}")
    return frame


# --- Stage 3+4: forecast (T1/T2 diversity tables) + top-3 (final three) ------------------------


@dataclass(frozen=True)
class ForecastStageResult:
    """Paths the later stages need from the forecast stage."""

    t1_path: Path
    t2_path: Path
    final_three_path: Path


def run_forecast_stage(panel: pl.DataFrame, tables_out_dir: Path) -> ForecastStageResult:
    """Train the frozen final model, then run the diversity-constrained T1/T2 + final-three
    selection, writing all three CSVs to `tables_out_dir`.

    Reuses `nss.models.final_forecast.train_final_model` / `build_ranking_frame` /
    `build_forecast_frame` / `compute_local_shap_drivers` and
    `nss.models.diversity_forecast.select_t1_incumbent` / `select_t2_emerging` /
    `build_final_three` verbatim, in the same order `diversity_forecast.main` itself uses -- no
    selection/ranking/SHAP logic is reimplemented here.

    Args:
        panel: The dense style-week panel.
        tables_out_dir: Directory to write `top_styles_t1_incumbent.csv`,
            `top_styles_t2_emerging.csv`, and `top_styles_final_three.csv` into.

    Returns:
        Paths to the three written CSVs.
    """
    model, _model_frame, columns = final_forecast.train_final_model(panel)
    ranking = final_forecast.build_ranking_frame(panel, model, columns)
    forecast_frame = final_forecast.build_forecast_frame(panel)
    print(
        f"[forecast] forecast-eligible styles at {final_forecast.FORECAST_ORIGIN}: {ranking.height}"
    )

    t1 = diversity_forecast.select_t1_incumbent(ranking)
    t2 = diversity_forecast.select_t2_emerging(panel, ranking)
    if t1.height < final_forecast.TOP_N:
        print(f"[forecast] T1: diversity-constrained pool exhausted at {t1.height} styles")
    if t2.height < final_forecast.TOP_N:
        print(f"[forecast] T2: diversity-constrained pool exhausted at {t2.height} styles")

    t1_shap = final_forecast.compute_local_shap_drivers(
        model, t1["style_key"].to_list(), forecast_frame, columns
    )
    t1_out = diversity_forecast._t1_output_frame(t1.join(t1_shap, on="style_key", how="left"))
    t2_shap = final_forecast.compute_local_shap_drivers(
        model, t2["style_key"].to_list(), forecast_frame, columns
    )
    t2_out = diversity_forecast._t2_output_frame(t2.join(t2_shap, on="style_key", how="left"))
    final_three = diversity_forecast.build_final_three(t1, t2, model, forecast_frame, columns)

    tables_out_dir.mkdir(parents=True, exist_ok=True)
    t1_path = tables_out_dir / diversity_forecast.DEFAULT_T1_OUT_PATH.name
    t2_path = tables_out_dir / diversity_forecast.DEFAULT_T2_OUT_PATH.name
    final_three_path = tables_out_dir / diversity_forecast.DEFAULT_FINAL_THREE_OUT_PATH.name
    t1_out.write_csv(t1_path)
    t2_out.write_csv(t2_path)
    final_three.write_csv(final_three_path)
    print(
        f"[forecast] wrote {t1_path} ({t1_out.height} rows), {t2_path} ({t2_out.height} rows), "
        f"{final_three_path} ({final_three.height} rows)"
    )
    return ForecastStageResult(t1_path=t1_path, t2_path=t2_path, final_three_path=final_three_path)


# --- Stage 5: briefs (SHAP verdict + exemplars + design briefs) --------------------------------


def run_exemplars_stage(
    final_three_path: Path,
    manifest_out_path: Path,
    images_dir: Path,
) -> Path:
    """Build the final-three exemplar-image manifest for THIS run's `final_three_path`.

    Mirrors `nss.data.select_final_three_exemplars.main`'s body exactly, calling that module's own
    already-tested helper functions (`select_final_three_targets`, `find_reusable_rows`,
    `carry_forward_rows`, plus `nss.data.select_exemplars`'s `compute_lookback_cutoff` /
    `select_top_selling_articles`) in the same order -- no selection/reuse/fetch logic is
    reimplemented here, only the output path is parameterized (so this can point at either the
    real `reports/tables/` location or a scratch determinism-check directory).

    Args:
        final_three_path: Path to THIS run's `top_styles_final_three.csv`.
        manifest_out_path: Destination for the exemplar-image manifest.
        images_dir: Local image cache directory (shared with Phase 2 -- real, on-disk images are
            content-addressed by article_id, safe to reuse across runs).

    Returns:
        `manifest_out_path`.
    """
    sfte = select_final_three_exemplars
    final_three = pl.read_csv(final_three_path)
    existing_manifest = pl.read_csv(sfte.EXISTING_MANIFEST_PATH)
    targets = sfte.select_final_three_targets(final_three)

    all_rows: list[dict[str, object]] = []
    fresh_targets: list[tuple[str, str, dict[str, str]]] = []
    for new_role, style_key, style_values in targets:
        reused = sfte.find_reusable_rows(existing_manifest, style_key)
        if reused is not None:
            print(
                f"[exemplars] {new_role}: {style_key} -- REUSED from Phase 2 ({reused.height} rows)"
            )
            all_rows.extend(sfte.carry_forward_rows(reused, new_role))
        else:
            print(f"[exemplars] {new_role}: {style_key} -- fresh selection + fetch")
            fresh_targets.append((new_role, style_key, style_values))

    if fresh_targets:
        cutoff = sfte.compute_lookback_cutoff(sfte.PANEL_LAST_WEEK, sfte.LOOKBACK_WEEKS)
        articles = pl.read_csv(sfte.ARTICLES_PATH).select(["article_id", *STYLE_KEY_COLS])
        txn_lazy = pl.scan_parquet(str(sfte.TRANSACTIONS_DIR / "**" / "*.parquet")).select(
            ["article_id", "t_dat"]
        )
        window_end = txn_lazy.select(pl.col("t_dat").max()).collect().item()
        transactions = txn_lazy.filter(pl.col("t_dat") >= cutoff).collect(engine="streaming")

        ranked_by_role: dict[str, pl.DataFrame] = {}
        fresh_article_ids: list[int] = []
        for new_role, _style_key, style_values in fresh_targets:
            ranked = sfte.select_top_selling_articles(
                transactions, articles, style_values, cutoff, window_end
            )
            ranked_by_role[new_role] = ranked
            fresh_article_ids.extend(int(x) for x in ranked["article_id"].to_list())

        fetch_results = fetch_images(fresh_article_ids, images_dir)
        n_ok = sum(fetch_results.values())
        print(f"[exemplars] fetched {n_ok}/{len(fetch_results)} images -> {images_dir}")

        for new_role, style_key, _ in fresh_targets:
            ranked = ranked_by_role[new_role]
            all_rows.extend(
                sfte.build_manifest_rows(new_role, style_key, ranked, images_dir, fetch_results)
            )

    control_rows = existing_manifest.filter(pl.col("role") == sfte.ROLE_CONTROL)
    all_rows.extend(control_rows.iter_rows(named=True))

    manifest = pl.DataFrame(all_rows).select(sfte.MANIFEST_SCHEMA_COLS)
    manifest_out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_csv(manifest_out_path)
    print(f"[exemplars] wrote manifest ({manifest.height} rows) to {manifest_out_path}")
    return manifest_out_path


EXPECTED_N_DESIGN_BRIEFS = 3


def is_curated_design_briefs(path: Path) -> bool:
    """Whether `path` holds a real, hand-refined `design_briefs.json` (tasks C5/E5), not a stub.

    "Real, refined" means: valid JSON, a list of exactly `EXPECTED_N_DESIGN_BRIEFS` (3) entries,
    each with a non-empty `style_id` AND a non-empty `applied_changes` list -- the field E5 added
    by hand that the naive `build_all_design_briefs()` rebuild does NOT reproduce (see module
    docstring). A missing, empty, malformed, or stub file returns `False`, which is the correct
    "genuinely clean state" signal for the from-scratch rebuild fallback in `run_briefs_stage`.

    Args:
        path: Path to `design_briefs.json`.

    Returns:
        `True` if `path` looks like a genuinely curated file; `False` otherwise (never raises on a
        malformed file -- a parse error degrades to "not curated", not a crash).
    """
    if not path.exists():
        return False
    try:
        briefs = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    if not isinstance(briefs, list) or len(briefs) != EXPECTED_N_DESIGN_BRIEFS:
        return False
    return all(
        isinstance(brief, dict) and brief.get("style_id") and brief.get("applied_changes")
        for brief in briefs
    )


def run_briefs_stage(
    final_three_path: Path, tables_out_dir: Path, images_dir: Path, force_briefs: bool = False
) -> Path:
    """SHAP-verdict report + exemplar-image manifest + design briefs, built from THIS run's
    `final_three_path`.

    Reuses `nss.models.final_three_shap_verdict.build_verdict_table` (pure function),
    `run_exemplars_stage` (above), and `nss.generate.build_design_briefs.build_all_design_briefs`
    (pure function) -- no brief-generation/classification logic is reimplemented here.

    GUARD (do not remove without re-reading the module docstring's design_briefs.json note): the
    real, committed `design_briefs.json` has been hand-refined across tasks C5/E5 -- it carries an
    `applied_changes` field (used by the hero/evidence-chain figures; its absence is a `hero`-stage
    `KeyError`) and SDXL-77-CLIP-token-safe prompts, NEITHER of which the naive
    `build_all_design_briefs()` rebuild reproduces. So unless `force_briefs=True`, an existing
    curated file (see `is_curated_design_briefs`) is left untouched and reused as-is instead of
    being silently overwritten by a worse rebuild -- only a genuinely missing/stub file (a truly
    clean checkout with `reports/tables/` wiped) falls through to the from-scratch build.

    Args:
        final_three_path: Path to THIS run's `top_styles_final_three.csv`.
        tables_out_dir: Directory to write `final_three_shap_verdict.csv`,
            `exemplar_images_final_three.csv`, and (if regenerated) `design_briefs.json` into.
        images_dir: Local image cache directory (see `run_exemplars_stage`).
        force_briefs: Regenerate `design_briefs.json` from scratch even if an existing curated
            file is present (accepts the known limitation that the naive rebuild will NOT
            reproduce `applied_changes` or the truncation-safe prompt fixes -- see
            `--force-briefs`'s CLI help text).

    Returns:
        Path to `design_briefs.json` (existing curated file if reused, freshly written otherwise).
    """
    final_three = pl.read_csv(final_three_path)
    verdict = final_three_shap_verdict.build_verdict_table(final_three)
    shap_verdict_path = tables_out_dir / final_three_shap_verdict.DEFAULT_OUT_PATH.name
    verdict.write_csv(shap_verdict_path)
    print(f"[briefs] wrote {shap_verdict_path} ({verdict.height} rows)")

    exemplar_manifest_path = tables_out_dir / select_final_three_exemplars.MANIFEST_OUT_PATH.name
    run_exemplars_stage(final_three_path, exemplar_manifest_path, images_dir)

    design_briefs_path = tables_out_dir / build_design_briefs_mod.OUT_PATH.name
    if not force_briefs and is_curated_design_briefs(design_briefs_path):
        print(
            f"[briefs] using existing curated {design_briefs_path} "
            f"({EXPECTED_N_DESIGN_BRIEFS} styles, each with applied_changes) -- not "
            "regenerating. Pass --force-briefs to override."
        )
        return design_briefs_path

    briefs = build_design_briefs_mod.build_all_design_briefs(
        top_styles_path=final_three_path,
        shap_verdict_path=shap_verdict_path,
        exemplar_images_path=exemplar_manifest_path,
    )
    design_briefs_path.parent.mkdir(parents=True, exist_ok=True)
    design_briefs_path.write_text(json.dumps(briefs, indent=2), encoding="utf-8")
    print(f"[briefs] wrote {design_briefs_path} ({len(briefs)} design briefs)")
    return design_briefs_path


# --- Stage 6: generate ---------------------------------------------------------------------------


def seeds_for_style(style_id: str, n_seeds: int) -> tuple[int, ...]:
    """The `n_seeds` seeds to generate for one style's round-0 batch (scoping-down -- see module
    docstring).

    Reuses `nss.generate.final_concepts_v2.INITIAL_SEEDS` (E5's own round-0 seed set, unchanged in
    value from C6's `final_concepts.SEEDS`) rather than inventing new seeds, so any candidate this
    script generates is directly comparable to E5's own round-0 results. Any seed
    `final_concepts_v2.VISUAL_QC_DISQUALIFIED_SEEDS` records for `style_id` (E5's own
    manually-confirmed visual-QC veto list) is skipped -- at `n_seeds<=2` this is currently a
    no-op (every presently-disqualified seed, e.g. the T-shirt style's 46-49, was only ever
    generated in a RETRY round this script never reaches -- see module docstring SCOPING-DOWN),
    but is applied unconditionally so this stays correct if `n_seeds` or the veto list change.

    Args:
        style_id: The style's `design_briefs.json` `style_id`.
        n_seeds: How many seeds to generate for this style.

    Returns:
        The first `n_seeds` non-disqualified entries of `final_concepts_v2.INITIAL_SEEDS`, in
        order.

    Raises:
        ValueError: if `n_seeds` exceeds the number of non-disqualified seeds available.
    """
    disqualified = final_concepts_v2.VISUAL_QC_DISQUALIFIED_SEEDS.get(style_id, frozenset())
    available = [s for s in final_concepts_v2.INITIAL_SEEDS if s not in disqualified]
    if n_seeds > len(available):
        raise ValueError(
            f"n_seeds={n_seeds} exceeds the {len(available)} non-disqualified seeds available "
            f"for style_id={style_id!r} (INITIAL_SEEDS={final_concepts_v2.INITIAL_SEEDS}, "
            f"disqualified={sorted(disqualified)})"
        )
    return tuple(available[:n_seeds])


def run_generate_stage(
    design_briefs: dict[str, dict[str, Any]],
    style_references: dict[str, list[Path]],
    n_seeds: int,
    generated_images_dir: Path,
) -> dict[str, list[final_concepts.Candidate]]:
    """Generate `n_seeds` `local_sdxl` candidate(s) per style, round-0 only (see module docstring
    SCOPING-DOWN) -- this script never runs a retry round, so scoring/selection (which needs the
    full E2 gate, including a network VLM judge call) happens entirely in `run_score_stage`.

    Reuses `nss.generate.final_concepts.generate_candidates_for_style` verbatim -- the SAME
    primitive `nss.generate.final_concepts_v2`'s own `generate_fn` closure calls (see that
    module's `main`) -- at E5's `final_concepts_v2.IP_ADAPTER_SCALE=0.45` operating point (C6's
    superseded 0.2 is never used here). Only the SEED COUNT differs from E5's own round-0 batch
    (see `seeds_for_style`), never the scale or the generation primitive itself. Also passes
    `final_concepts.build_prompt_2(style_id)` (task F4) as `prompt_2`/`negative_prompt_2`, the same
    second-text-encoder wiring `final_concepts_v2.main`'s own `generate_fn` closure now applies.

    Args:
        design_briefs: Output of `nss.generate.final_concepts.load_design_briefs`.
        style_references: Output of `nss.generate.final_concepts.load_final_three_references`.
        n_seeds: Seeds per style (see `seeds_for_style`).
        generated_images_dir: Directory to write generated candidate images into.

    Returns:
        Mapping of `style_id -> list[Candidate]` (round-0 only), for the `score` stage.
    """
    all_candidates: dict[str, list[final_concepts.Candidate]] = {}
    for style_id, brief in design_briefs.items():
        prompt, negative_prompt = final_concepts.build_generation_spec(style_id, brief)
        prompt_2 = final_concepts.build_prompt_2(style_id)  # task F4 -- SDXL's second text encoder
        seeds = seeds_for_style(style_id, n_seeds)
        references = style_references[style_id]
        print(f"[generate] style={style_id!r} seeds={seeds} references={len(references)}")
        all_candidates[style_id] = final_concepts.generate_candidates_for_style(
            style_id,
            prompt,
            negative_prompt,
            references,
            seeds=seeds,
            ip_adapter_scale=final_concepts_v2.IP_ADAPTER_SCALE,
            output_dir=generated_images_dir,
            prompt_2=prompt_2,
            negative_prompt_2=negative_prompt,
        )

    vram_before, vram_after = free_sdxl_pipeline()
    print(f"[generate] VRAM before free: {vram_before:.3f} GB, after free: {vram_after:.3f} GB")
    return all_candidates


# --- Stage 7: score -------------------------------------------------------------------------------


def run_score_stage(
    design_briefs: dict[str, dict[str, Any]],
    style_references: dict[str, list[Path]],
    control_images: list[Path],
    candidates_by_style: dict[str, list[final_concepts.Candidate]],
    out_table_path: Path,
) -> pl.DataFrame:
    """Score every round-0 candidate against the FULL E2 gate and select the final concept per
    style.

    Reuses `nss.generate.final_concepts_v2`'s own scoring/selection/write primitives verbatim
    (`score_candidate`, `select_final_candidate`, `write_results_table`,
    `apply_visual_qc_and_rewrite`) -- the SAME two-step "write, then apply the manual visual-QC
    veto list as a separate rewrite" flow `final_concepts_v2.main` itself uses, NOT the superseded
    `nss.generate.concept_qc_pipeline.run_qc_with_retries` path this script used before. No retry
    round is ever triggered here (this stage only scores the ALREADY-generated
    `candidates_by_style` -- see module docstring SCOPING-DOWN), so every row's `retry_round` is 0
    and `n_retry_rounds_used` is always 0.

    Writes ISOLATED to `out_table_path` (never `reports/tables/final_concepts_v2.csv` -- see
    module docstring STAGE-OUTPUT ISOLATION). The one-time Groq reachability check is wrapped so
    an unexpected network/API error degrades to "Groq unavailable" (printed, not raised) rather
    than failing the whole script -- Gemini's OWN call failures are already caught internally by
    `final_concepts_v2.score_candidate`'s call to `run_judge_panel`, so no extra wrapping is
    needed there.

    Args:
        design_briefs: Output of `nss.generate.final_concepts.load_design_briefs`.
        style_references: Output of `nss.generate.final_concepts.load_final_three_references`.
        control_images: Output of `nss.generate.derive_margin_band.load_control_pool`.
        candidates_by_style: Output of `run_generate_stage`.
        out_table_path: Destination CSV for the full per-style scored history + selection (same
            shape as `nss.generate.final_concepts_v2.OUTPUT_TABLE_PATH`).

    Returns:
        The written results `DataFrame`, post visual-QC rewrite (output of
        `final_concepts_v2.apply_visual_qc_and_rewrite`).
    """
    from nss.generate import clip_scoring, dino_scoring

    try:
        groq_available, groq_detail = vlm_judges.check_groq_availability()
    except Exception as exc:  # noqa: BLE001 -- deliberate: any unexpected error must degrade, not crash.
        groq_available, groq_detail = (
            False,
            f"availability check raised an unexpected error: {exc!r}",
        )
    print(f"[score] groq_available={groq_available} ({groq_detail})")

    copy_anchors = concept_qc_pipeline.load_copy_anchors_gen()
    clip_control = [clip_scoring.embed_image(p) for p in control_images]
    dino_control = [dino_scoring.embed_image(p) for p in control_images]

    results: dict[str, dict[str, Any]] = {}
    for style_id in design_briefs:
        candidates = candidates_by_style[style_id]
        references = style_references[style_id]
        ground_truth = concept_qc_pipeline.parse_style_attributes(style_id)
        clip_refs = [clip_scoring.embed_image(p) for p in references]
        dino_refs = [dino_scoring.embed_image(p) for p in references]

        all_scored = [
            final_concepts_v2.score_candidate(
                candidate,
                clip_refs,
                clip_control,
                dino_refs,
                dino_control,
                copy_anchors[style_id]["clip"],
                copy_anchors[style_id]["dinov2"],
                ground_truth,
                groq_available,
                groq_detail,
                retry_round=0,
            )
            for candidate in candidates
        ]
        # No disqualification applied yet here -- mirrors `final_concepts_v2.main`'s own flow,
        # which writes an initial "fully qualified" selection, then applies the manual visual-QC
        # veto list as a SEPARATE rewrite step below (`apply_visual_qc_and_rewrite`).
        selection = final_concepts_v2.select_final_candidate(all_scored)
        results[style_id] = {
            "style_id": style_id,
            "all_scored": all_scored,
            "selection": selection,
            "n_retry_rounds_used": 0,
            "seeds_tried": [c.seed for c in candidates],
        }
        selected = selection["selected"]
        print(
            f"[score] {style_id}: seed={selected['seed']} "
            f"overall_pass={selected['overall_pass']} "
            f"(copy_check_pass={selected['copy_check_pass']}, "
            f"fidelity_pass={selected['fidelity_pass']}, "
            f"mean_attribute_fidelity={selected['mean_attribute_fidelity']:.3f}) "
            f"selection_mode={selection['selection_mode']}"
        )

    final_concepts_v2.write_results_table(results, path=out_table_path)
    final_df = final_concepts_v2.apply_visual_qc_and_rewrite(
        path=out_table_path,
        disqualified_seeds_by_style=final_concepts_v2.VISUAL_QC_DISQUALIFIED_SEEDS,
    )
    print(f"[score] wrote {out_table_path}")
    return final_df


# --- CLI / orchestration -------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser (see module docstring for flag semantics)."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--rebuild-panel",
        action="store_true",
        help="Force-rebuild the panel from raw/interim data even if the parquet already exists "
        "(default: load the existing parquet if present).",
    )
    parser.add_argument(
        "--transactions-dir", type=Path, default=style_panel.DEFAULT_TRANSACTIONS_DIR
    )
    parser.add_argument("--articles-path", type=Path, default=style_panel.DEFAULT_ARTICLES_PATH)
    parser.add_argument("--panel-path", type=Path, default=style_panel.DEFAULT_OUT_PATH)
    parser.add_argument(
        "--tables-out-dir",
        type=Path,
        default=DEFAULT_TABLES_DIR,
        help="Where panel/features/forecast/briefs stages write their CSV/JSON artifacts. Only "
        "override this together with --stop-after forecast (see module docstring).",
    )
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument(
        "--pipeline-out-dir",
        type=Path,
        default=DEFAULT_PIPELINE_OUT_DIR,
        help="Where the generate/score/hero stages write their scoped-down artifacts, isolated "
        "from the real reports/tables|figures deliverables (see module docstring).",
    )
    parser.add_argument("--generated-images-dir", type=Path, default=DEFAULT_GENERATED_IMAGES_DIR)
    parser.add_argument(
        "--force-briefs",
        action="store_true",
        help="Regenerate design_briefs.json from scratch via build_all_design_briefs(), even if "
        "an existing curated file (3 styles, each with an applied_changes field -- see "
        "is_curated_design_briefs) is already present. Default: reuse the existing curated file "
        "untouched, since it has been hand-refined (tasks C5/E5) with applied_changes entries "
        "and SDXL-77-CLIP-token-safe prompts that the naive rebuild does NOT reproduce -- passing "
        "this flag accepts that known regression (missing applied_changes will KeyError the hero "
        "stage; unrefined prompts may exceed the CLIP token limit and get silently truncated) in "
        "exchange for a genuinely from-scratch briefs rebuild.",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=1,
        choices=[1, 2],
        help="Seeds per style for the generate stage (see module docstring SCOPING-DOWN).",
    )
    parser.add_argument("--stop-after", choices=STAGE_ORDER, default=STAGE_ORDER[-1])
    return parser


def _print_summary(timings: dict[str, float], t_start: float) -> None:
    """Print a per-stage + total wall-clock timing summary."""
    total = time.perf_counter() - t_start
    print("\n=== Stage timings ===")
    for name, secs in timings.items():
        print(f"  {name}: {secs:.2f}s")
    if "generate" in timings:
        print(f"  (generate step specifically: {timings['generate']:.2f}s)")
    print(f"  TOTAL: {total:.2f}s")


def main() -> None:
    """Run the pipeline through `--stop-after` (default: the full chain, ending at `hero`)."""
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.tables_out_dir != DEFAULT_TABLES_DIR and _stage_index(args.stop_after) > _stage_index(
        "forecast"
    ):
        parser.error(
            "--tables-out-dir may only be combined with --stop-after panel/features/forecast -- "
            "briefs/generate/score/hero read design_briefs.json and exemplar references from the "
            "REAL reports/tables/ location regardless of this flag (see module docstring)."
        )

    t_start = time.perf_counter()
    timings: dict[str, float] = {}

    def _run_stage(name: str, fn: Callable[[], Any]) -> Any:
        t0 = time.perf_counter()
        result = fn()
        timings[name] = time.perf_counter() - t0
        print(f"[timing] {name} stage: {timings[name]:.2f}s")
        return result

    panel = _run_stage(
        "panel",
        lambda: run_panel_stage(
            args.rebuild_panel, args.transactions_dir, args.articles_path, args.panel_path
        ),
    )
    if _stage_index(args.stop_after) < _stage_index("features"):
        _print_summary(timings, t_start)
        return

    _run_stage("features", lambda: run_features_stage(panel, final_forecast.FORECAST_ORIGIN))
    if _stage_index(args.stop_after) < _stage_index("forecast"):
        _print_summary(timings, t_start)
        return

    forecast_result: ForecastStageResult = _run_stage(
        "forecast", lambda: run_forecast_stage(panel, args.tables_out_dir)
    )
    if _stage_index(args.stop_after) < _stage_index("briefs"):
        _print_summary(timings, t_start)
        return

    _run_stage(
        "briefs",
        lambda: run_briefs_stage(
            forecast_result.final_three_path,
            args.tables_out_dir,
            args.images_dir,
            args.force_briefs,
        ),
    )
    if _stage_index(args.stop_after) < _stage_index("generate"):
        _print_summary(timings, t_start)
        return

    design_briefs = final_concepts.load_design_briefs()
    # F3: screened (full-garment only, best-selling-first) references, not the unscreened,
    # arbitrarily-alphabetically-ordered `final_concepts.load_final_three_references` -- see
    # `nss.generate.screen_references` module docstring for the mechanism this fixes.
    style_references = screen_references.load_screened_references()
    control_images = load_control_pool(CONTROL_MANIFEST_PATH)

    candidates_by_style = _run_stage(
        "generate",
        lambda: run_generate_stage(
            design_briefs,
            style_references,
            args.n_seeds,
            args.generated_images_dir,
        ),
    )
    if _stage_index(args.stop_after) < _stage_index("score"):
        _print_summary(timings, t_start)
        return

    final_concepts_v2_path = args.pipeline_out_dir / "tables" / "final_concepts_v2.csv"
    _run_stage(
        "score",
        lambda: run_score_stage(
            design_briefs,
            style_references,
            control_images,
            candidates_by_style,
            final_concepts_v2_path,
        ),
    )
    if _stage_index(args.stop_after) < _stage_index("hero"):
        _print_summary(timings, t_start)
        return

    hero_path, evidence_path = _run_stage(
        "hero",
        lambda: final_deliverables.main(
            final_concepts_v3_path=final_concepts_v2_path,
            hero_out_path=args.pipeline_out_dir / "figures" / "FINAL_concepts.png",
            evidence_out_path=args.pipeline_out_dir / "figures" / "evidence_chain.png",
        ),
    )
    print(
        f"\n=== Pipeline complete: hero image at {hero_path} (evidence chain: {evidence_path}) ==="
    )
    _print_summary(timings, t_start)


if __name__ == "__main__":
    main()
