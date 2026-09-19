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
`hero` stages are NOT wire-compatible with the real C6-C8 deliverables, because this script
deliberately generates only 1-2 seeds per style (see SCOPING-DOWN below) instead of C6's full
4-seed sweep -- a genuinely different (lower-fidelity) selection could result. Those three stages
therefore ALWAYS write to an isolated `--pipeline-out-dir` (default `reports/pipeline_run/`,
gitignored) and NEVER touch `reports/tables/final_concepts.csv`,
`reports/tables/concept_qc_results.csv`, or `reports/figures/FINAL_concepts.png` /
`evidence_chain.png` -- those remain the authoritative, unmodified C6/C7/C8 deliverables.

SCOPING-DOWN (documented, not silently done): task D3's purpose is proving the WIRING between
already-tested stages works end-to-end from one command, not re-deriving C6's exact 4-seed
selection. `--n-seeds` (default 1, max 2) controls how many of C6's own `final_concepts.SEEDS`
are generated per style; the operating point (`ip_adapter_scale=0.2`) and the selection rule
itself are reused verbatim from `nss.generate.final_concepts`, unchanged. For the underwear style
specifically, C6's own manually-confirmed `UNDERWEAR_VISUAL_QC_DISQUALIFIED_SEEDS` are skipped
when choosing which seed(s) to generate (see `seeds_for_style`), for the same documented reason
C6 itself excludes them.

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

Usage:
    uv run --no-sync python scripts/run_pipeline.py                       # full pipeline
    uv run --no-sync python scripts/run_pipeline.py --stop-after forecast # panel+features+
                                                                           # forecast+top-3 only
    uv run --no-sync python scripts/run_pipeline.py --stop-after briefs   # skip GPU generation
    uv run --no-sync python scripts/run_pipeline.py --n-seeds 2           # 2 seeds/style
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
from nss.generate import (
    backends,
    concept_qc_pipeline,
    final_concepts,
    final_deliverables,
    vlm_judges,
)
from nss.generate import build_design_briefs as build_design_briefs_mod
from nss.generate.derive_margin_band import CONTROL_MANIFEST_PATH, load_control_pool
from nss.generate.scale_sweep import (
    CLIP_BAND_PATH,
    DINO_BAND_PATH,
    free_sdxl_pipeline,
    load_margin_band,
)
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


def run_briefs_stage(final_three_path: Path, tables_out_dir: Path, images_dir: Path) -> Path:
    """SHAP-verdict report + exemplar-image manifest + design briefs, built from THIS run's
    `final_three_path`.

    Reuses `nss.models.final_three_shap_verdict.build_verdict_table` (pure function),
    `run_exemplars_stage` (above), and `nss.generate.build_design_briefs.build_all_design_briefs`
    (pure function) -- no brief-generation/classification logic is reimplemented here.

    Args:
        final_three_path: Path to THIS run's `top_styles_final_three.csv`.
        tables_out_dir: Directory to write `final_three_shap_verdict.csv`,
            `exemplar_images_final_three.csv`, and `design_briefs.json` into.
        images_dir: Local image cache directory (see `run_exemplars_stage`).

    Returns:
        Path to the written `design_briefs.json`.
    """
    final_three = pl.read_csv(final_three_path)
    verdict = final_three_shap_verdict.build_verdict_table(final_three)
    shap_verdict_path = tables_out_dir / final_three_shap_verdict.DEFAULT_OUT_PATH.name
    verdict.write_csv(shap_verdict_path)
    print(f"[briefs] wrote {shap_verdict_path} ({verdict.height} rows)")

    exemplar_manifest_path = tables_out_dir / select_final_three_exemplars.MANIFEST_OUT_PATH.name
    run_exemplars_stage(final_three_path, exemplar_manifest_path, images_dir)

    briefs = build_design_briefs_mod.build_all_design_briefs(
        top_styles_path=final_three_path,
        shap_verdict_path=shap_verdict_path,
        exemplar_images_path=exemplar_manifest_path,
    )
    design_briefs_path = tables_out_dir / build_design_briefs_mod.OUT_PATH.name
    design_briefs_path.parent.mkdir(parents=True, exist_ok=True)
    design_briefs_path.write_text(json.dumps(briefs, indent=2), encoding="utf-8")
    print(f"[briefs] wrote {design_briefs_path} ({len(briefs)} design briefs)")
    return design_briefs_path


# --- Stage 6: generate ---------------------------------------------------------------------------


def seeds_for_style(style_id: str, n_seeds: int) -> tuple[int, ...]:
    """The `n_seeds` seeds to generate for one style (scoping-down -- see module docstring).

    Reuses `nss.generate.final_concepts.SEEDS` (C6's own 4-seed set) rather than inventing new
    seeds, so any candidate this script generates is directly comparable to C6's own per-seed
    results. For the underwear style specifically, C6's own manually-confirmed
    `UNDERWEAR_VISUAL_QC_DISQUALIFIED_SEEDS` (seeds 42/43/44 each showed a human model despite the
    strengthened negative prompt) are skipped, so a 1-seed integration run doesn't spuriously hit
    `select_best_candidate`'s "every candidate disqualified" `ValueError` for a reason already
    documented and unrelated to this script's wiring.

    Args:
        style_id: The style's `design_briefs.json` `style_id`.
        n_seeds: How many seeds to generate for this style.

    Returns:
        The first `n_seeds` non-disqualified entries of `final_concepts.SEEDS`, in order.

    Raises:
        ValueError: if `n_seeds` exceeds the number of non-disqualified seeds available.
    """
    disqualified = (
        final_concepts.UNDERWEAR_VISUAL_QC_DISQUALIFIED_SEEDS
        if style_id == final_concepts.UNDERWEAR_STYLE_KEY
        else frozenset()
    )
    available = [s for s in final_concepts.SEEDS if s not in disqualified]
    if n_seeds > len(available):
        raise ValueError(
            f"n_seeds={n_seeds} exceeds the {len(available)} non-disqualified seeds available "
            f"for style_id={style_id!r} (SEEDS={final_concepts.SEEDS}, "
            f"disqualified={sorted(disqualified)})"
        )
    return tuple(available[:n_seeds])


def run_generate_stage(
    design_briefs: dict[str, dict[str, Any]],
    style_references: dict[str, list[Path]],
    control_images: list[Path],
    clip_band: tuple[float, float],
    dino_band: tuple[float, float],
    n_seeds: int,
    generated_images_dir: Path,
    out_table_path: Path,
) -> dict[str, dict[str, Any]]:
    """Generate `n_seeds` `local_sdxl` candidate(s) per style and select the best.

    Reuses `nss.generate.final_concepts`'s own generation/scoring/selection primitives verbatim
    (`generate_candidates_for_style`, `score_candidates`, `select_best_candidate`) -- only the
    SEED COUNT differs from C6's own 4-seed run (see `seeds_for_style`), never the operating point
    (`final_concepts.IP_ADAPTER_SCALE=0.2`) or the selection rule itself.

    Writes ISOLATED to `out_table_path` (never `reports/tables/final_concepts.csv` -- see module
    docstring STAGE-OUTPUT ISOLATION).

    Args:
        design_briefs: Output of `nss.generate.final_concepts.load_design_briefs`.
        style_references: Output of `nss.generate.final_concepts.load_final_three_references`.
        control_images: Output of `nss.generate.derive_margin_band.load_control_pool`.
        clip_band: `(lower, upper)` CLIP margin band.
        dino_band: `(lower, upper)` DINOv2 margin band.
        n_seeds: Seeds per style (see `seeds_for_style`).
        generated_images_dir: Directory to write generated candidate images into.
        out_table_path: Destination CSV for the per-style selections.

    Returns:
        Mapping of `style_id -> selected candidate dict` (the `"selected"` entry of
        `select_best_candidate`'s return value), for use by the `score` stage.
    """
    all_candidates: dict[str, list[final_concepts.Candidate]] = {}
    for style_id, brief in design_briefs.items():
        prompt, negative_prompt = final_concepts.build_generation_spec(style_id, brief)
        seeds = seeds_for_style(style_id, n_seeds)
        references = style_references[style_id]
        print(f"[generate] style={style_id!r} seeds={seeds} references={len(references)}")
        all_candidates[style_id] = final_concepts.generate_candidates_for_style(
            style_id,
            prompt,
            negative_prompt,
            references,
            seeds=seeds,
            output_dir=generated_images_dir,
        )

    vram_before, vram_after = free_sdxl_pipeline()
    print(f"[generate] VRAM before free: {vram_before:.3f} GB, after free: {vram_after:.3f} GB")

    rows: list[dict[str, Any]] = []
    selections: dict[str, dict[str, Any]] = {}
    for style_id, candidates in all_candidates.items():
        scored = final_concepts.score_candidates(
            candidates, style_references[style_id], control_images
        )
        disqualified = (
            final_concepts.UNDERWEAR_VISUAL_QC_DISQUALIFIED_SEEDS
            if style_id == final_concepts.UNDERWEAR_STYLE_KEY
            else frozenset()
        )
        result = final_concepts.select_best_candidate(
            scored, clip_band, dino_band, disqualified_seeds=disqualified
        )
        selected = result["selected"]
        selections[style_id] = selected
        print(
            f"[generate] {style_id}: chosen seed={selected['seed']} "
            f"clip_margin={selected['clip_margin']:.4f} (in_band={selected['clip_in_band']}) "
            f"dino_margin={selected['dino_margin']:.4f} (in_band={selected['dino_in_band']})"
        )
        rows.append(
            {
                "style_id": style_id,
                "chosen_seed": selected["seed"],
                "local_path": str(selected["image_path"]),
                "clip_margin": selected["clip_margin"],
                "dino_margin": selected["dino_margin"],
                "clip_in_band": selected["clip_in_band"],
                "dino_in_band": selected["dino_in_band"],
                "n_seeds_tried": len(candidates),
            }
        )

    out_table_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(out_table_path)
    print(f"[generate] wrote {out_table_path}")
    return selections


# --- Stage 7: score -------------------------------------------------------------------------------


def run_score_stage(
    design_briefs: dict[str, dict[str, Any]],
    style_references: dict[str, list[Path]],
    control_images: list[Path],
    selections: dict[str, dict[str, Any]],
    clip_band: tuple[float, float],
    dino_band: tuple[float, float],
    out_path: Path,
) -> list[dict[str, Any]]:
    """Score each selected candidate: margin-band always, VLM judges when reachable.

    Reuses `nss.generate.concept_qc_pipeline.run_qc_with_retries` with `max_retries=0` -- i.e.
    exactly ONE scoring pass per style (this script's `generate` stage already did the seed
    selection; this stage never triggers a retry-generation, since `generate_fn` is never called
    when `max_retries=0`), reusing the SAME `qc_verdict`/margin/judge-panel logic C7 uses, just
    without its retry loop. Margin-band scoring (CLIP + DINOv2) needs no external API and always
    runs; the one-time Groq reachability check is wrapped so an unexpected network/API error
    degrades to "Groq unavailable" (printed, not raised) rather than failing the whole script --
    Gemini's OWN call failures are already caught internally by `skills/concept-qc/run_qc.py`'s
    `run_judge` (see that function's docstring), so no extra wrapping is needed there.

    Args:
        design_briefs: Output of `nss.generate.final_concepts.load_design_briefs`.
        style_references: Output of `nss.generate.final_concepts.load_final_three_references`.
        control_images: Output of `nss.generate.derive_margin_band.load_control_pool`.
        selections: Output of `run_generate_stage`.
        clip_band: `(lower, upper)` CLIP margin band.
        dino_band: `(lower, upper)` DINOv2 margin band.
        out_path: Destination CSV for the full per-style QC results.

    Returns:
        One dict per style (a single-attempt `run_qc_with_retries` result), also written to
        `out_path` via `concept_qc_pipeline.write_results_csv`.
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

    all_attempts: list[dict[str, Any]] = []
    for style_id in design_briefs:
        selected = selections[style_id]
        ground_truth = concept_qc_pipeline.parse_style_attributes(style_id)
        references = style_references[style_id]

        clip_refs = [clip_scoring.embed_image(p) for p in references]
        clip_control = [clip_scoring.embed_image(p) for p in control_images]
        dino_refs = [dino_scoring.embed_image(p) for p in references]
        dino_control = [dino_scoring.embed_image(p) for p in control_images]

        def judge_panel_fn(
            image_path: Path, ground_truth: dict[str, str] = ground_truth
        ) -> dict[str, Any]:
            return concept_qc_pipeline.run_judge_panel(
                image_path, ground_truth, backends.LOCAL_SDXL, groq_available, groq_detail
            )

        def margin_fn(
            image_path: Path,
            clip_refs: list[Any] = clip_refs,
            clip_control: list[Any] = clip_control,
            dino_refs: list[Any] = dino_refs,
            dino_control: list[Any] = dino_control,
        ) -> dict[str, Any]:
            return concept_qc_pipeline.score_margins(
                image_path, clip_refs, clip_control, dino_refs, dino_control, clip_band, dino_band
            )

        def generate_fn(scale: float, attempt_number: int) -> Path:
            raise RuntimeError(
                "run_pipeline's score stage runs with max_retries=0 -- a retry should never be "
                "triggered."
            )

        original_margin = {
            "clip_margin": float(selected["clip_margin"]),
            "clip_in_band": bool(selected["clip_in_band"]),
            "dino_margin": float(selected["dino_margin"]),
            "dino_in_band": bool(selected["dino_in_band"]),
        }
        attempts = concept_qc_pipeline.run_qc_with_retries(
            style_id=style_id,
            original_seed=int(selected["seed"]),
            original_scale=final_concepts.IP_ADAPTER_SCALE,
            original_image_path=Path(selected["image_path"]),
            original_margin=original_margin,
            ground_truth=ground_truth,
            generation_backend=backends.LOCAL_SDXL,
            clip_band=clip_band,
            dino_band=dino_band,
            judge_panel_fn=judge_panel_fn,
            margin_fn=margin_fn,
            generate_fn=generate_fn,
            max_retries=0,
        )
        final_attempt = attempts[-1]
        for attempt in attempts:
            attempt["style_final_pass"] = final_attempt["overall_pass"]
            attempt["n_attempts_for_style"] = len(attempts)
        all_attempts.extend(attempts)
        print(
            f"[score] {style_id}: overall_pass={final_attempt['overall_pass']} "
            f"(margin_band_pass={final_attempt['margin_band_pass']}, "
            f"mean_attribute_fidelity={final_attempt['mean_attribute_fidelity']:.3f}, "
            f"n_judges={final_attempt['n_contributing_judges']})"
        )

    concept_qc_pipeline.write_results_csv(all_attempts, out_path)
    print(f"[score] wrote {out_path}")
    return all_attempts


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
            forecast_result.final_three_path, args.tables_out_dir, args.images_dir
        ),
    )
    if _stage_index(args.stop_after) < _stage_index("generate"):
        _print_summary(timings, t_start)
        return

    design_briefs = final_concepts.load_design_briefs()
    style_references = final_concepts.load_final_three_references()
    control_images = load_control_pool(CONTROL_MANIFEST_PATH)
    clip_band = load_margin_band(CLIP_BAND_PATH)
    dino_band = load_margin_band(DINO_BAND_PATH)

    selections = _run_stage(
        "generate",
        lambda: run_generate_stage(
            design_briefs,
            style_references,
            control_images,
            clip_band,
            dino_band,
            args.n_seeds,
            args.generated_images_dir,
            args.pipeline_out_dir / "tables" / "final_concepts.csv",
        ),
    )
    if _stage_index(args.stop_after) < _stage_index("score"):
        _print_summary(timings, t_start)
        return

    qc_results_path = args.pipeline_out_dir / "tables" / "concept_qc_results.csv"
    _run_stage(
        "score",
        lambda: run_score_stage(
            design_briefs,
            style_references,
            control_images,
            selections,
            clip_band,
            dino_band,
            qc_results_path,
        ),
    )
    if _stage_index(args.stop_after) < _stage_index("hero"):
        _print_summary(timings, t_start)
        return

    hero_path, evidence_path = _run_stage(
        "hero",
        lambda: final_deliverables.main(
            qc_results_path=qc_results_path,
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
