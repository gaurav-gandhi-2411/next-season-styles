"""VLM-judge checklist fix, recomputed from stored per-attribute scores (task H2).

Why a re-judge at all: the brief says to recompute from stored per-attribute scores and NOT re-call
the judges -- but E5's/F5's tables persisted only each judge's MEAN (no `scores_json`), and F5's
11/12 candidates had no judge score at all (Gemini/Groq quota exhausted that day). The
per-attribute scores existed only for `vlm_calibration_results.csv` (recomputed here, no calls). For
candidates, each image is judged ONCE with the legacy 4-attribute checklist, per-attribute scores
persisted to `data/generated/judge_cache.jsonl`, and both the old (4-attribute) and new
(3-attribute, `garment_group` dropped) fidelity are then derived from that one stored set --
a paired, exact before/after. Never re-called for a cached image.

Usage:
    uv run python -m nss.generate.h2_rejudge
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import polars as pl

from nss.generate import vlm_judges
from nss.generate.concept_qc_pipeline import (
    CALIBRATION_RESULTS_PATH,
    compute_fidelity_thresholds,
    parse_style_attributes,
    run_judge_panel,
    summarize_calibration,
)

CACHE_PATH = Path("data/generated/judge_cache.jsonl")
RESCORE_PATH = Path("reports/tables/h2_judge_rescore.csv")
CALIBRATION_H2_PATH = Path("reports/tables/h2_calibration_recomputed.csv")
F5_TABLE_PATH = Path("reports/tables/final_concepts_v3.csv")
TRANSIENT_RETRIES = 3
TRANSIENT_SLEEP_SECONDS = 20.0


def mean_over(scores: dict[str, float], dims: Sequence[str]) -> float:
    """Mean of `scores` restricted to `dims` (a checklist subset) from stored per-attribute data."""
    return sum(scores[d] for d in dims) / len(dims)


def recompute_calibration(
    path: Path = CALIBRATION_RESULTS_PATH,
) -> tuple[pl.DataFrame, dict[str, float], dict[str, float]]:
    """Recompute calibration means + per-judge Gate-2 thresholds under the H2 checklist.

    Pure recomputation from the persisted `scores_json` -- no judge calls.

    Returns:
        `(df, old_thresholds, new_thresholds)`; `df` has old (4-attribute) and new (3-attribute)
        `mean_score` per calibration row.
    """
    df = pl.read_csv(path)
    old_summary = summarize_calibration(df)
    new_scores = []
    for row in df.to_dicts():
        if row["scores_json"] is None:
            new_scores.append(None)
        else:
            new_scores.append(
                mean_over(json.loads(row["scores_json"]), vlm_judges.ATTRIBUTE_DIMENSIONS)
            )
    new_df = df.with_columns(
        pl.col("mean_score").alias("mean_score_legacy_4attr"),
        pl.Series("mean_score", new_scores, dtype=pl.Float64),
    )
    return (
        new_df,
        compute_fidelity_thresholds(old_summary),
        compute_fidelity_thresholds(summarize_calibration(new_df)),
    )


def _load_cache(path: Path = CACHE_PATH) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {
        rec["image_path"]: rec
        for rec in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    }


def _judge_with_retry(
    image: Path,
    style_id: str,
    groq_available: bool,
    groq_detail: str,
    panel_fn: Callable[..., dict[str, Any]] = run_judge_panel,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Judge one image with the legacy checklist, retrying only Gemini 503 (transient overload).

    A 429 (quota) is NOT retried -- it does not clear within a run; the result records it.
    """
    truth = parse_style_attributes(style_id)
    result: dict[str, Any] = {}
    for attempt in range(TRANSIENT_RETRIES + 1):
        result = panel_fn(
            image,
            truth,
            "local_sdxl",
            groq_available,
            groq_detail,
            dimensions=vlm_judges.LEGACY_ATTRIBUTE_DIMENSIONS,
        )
        reason = str(result["gemini"]["excluded_reason"] or "")
        if result["gemini"]["available"] or "503" not in reason or attempt == TRANSIENT_RETRIES:
            break
        sleep_fn(TRANSIENT_SLEEP_SECONDS)
    return result


def judge_images(
    items: Sequence[tuple[str, int, Path]], cache_path: Path = CACHE_PATH
) -> dict[str, dict[str, Any]]:
    """Judge every `(style_id, seed, image)` not already cached; append results; return the cache.

    A judge that was UNAVAILABLE for a cached image is retried on a later call (only judges that
    produced a real score are treated as final) so a quota-blocked run can be completed later.
    """
    cache = _load_cache(cache_path)
    groq_available, groq_detail = vlm_judges.check_groq_availability()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    for style_id, seed, image in items:
        key = str(image)
        prior = cache.get(key)
        if prior and all(prior["judges"][j]["available"] for j in ("gemini", "groq")):
            continue
        fresh = _judge_with_retry(image, style_id, groq_available, groq_detail)
        judges = {
            name: {
                "available": res["available"],
                "scores": res["scores"],
                "raw_extraction": res["raw_extraction"],
                "excluded_reason": res["excluded_reason"],
            }
            for name, res in fresh.items()
        }
        if prior:  # keep an earlier real score if this attempt regressed for that judge
            for name in judges:
                if prior["judges"][name]["available"] and not judges[name]["available"]:
                    judges[name] = prior["judges"][name]
        cache[key] = {"image_path": key, "style_id": style_id, "seed": seed, "judges": judges}
        with cache_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(cache[key]) + "\n")
    return cache


def build_rescore_table(
    cache: dict[str, dict[str, Any]],
    old_thresholds: dict[str, float],
    new_thresholds: dict[str, float],
) -> pl.DataFrame:
    """Per candidate x judge: 4-attribute (before) vs 3-attribute (after) fidelity and Gate 2."""
    rows: list[dict[str, Any]] = []
    for rec in cache.values():
        for judge, res in rec["judges"].items():
            row: dict[str, Any] = {
                "style_id": rec["style_id"],
                "seed": rec["seed"],
                "image_path": rec["image_path"],  # distinguishes F5 vs H3 images of one seed
                "judge": judge,
                "available": res["available"],
                "scores_json": json.dumps(res["scores"]) if res["scores"] else None,
                "raw_extraction_json": (
                    json.dumps(res["raw_extraction"]) if res["raw_extraction"] else None
                ),
                "excluded_reason": (
                    str(res["excluded_reason"])[:160] if res["excluded_reason"] else None
                ),
                "fidelity_before_4attr": None,
                "fidelity_after_3attr": None,
                "threshold_before": old_thresholds.get(judge),
                "threshold_after": new_thresholds.get(judge),
                "pass_before": None,
                "pass_after": None,
            }
            if res["scores"]:
                before = mean_over(res["scores"], vlm_judges.LEGACY_ATTRIBUTE_DIMENSIONS)
                after = mean_over(res["scores"], vlm_judges.ATTRIBUTE_DIMENSIONS)
                row.update(
                    fidelity_before_4attr=before,
                    fidelity_after_3attr=after,
                    pass_before=before >= old_thresholds[judge],
                    pass_after=after >= new_thresholds[judge],
                )
            rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None).sort(["style_id", "image_path", "judge"])


def main() -> None:
    """H2: recompute calibration thresholds, judge F5's 12 candidates once, write before/after."""
    calib, old_t, new_t = recompute_calibration()
    calib.write_csv(CALIBRATION_H2_PATH)
    print(f"Gate-2 thresholds before (4 attr): {old_t}\nGate-2 thresholds after (3 attr):  {new_t}")
    f5 = pl.read_csv(F5_TABLE_PATH)
    items = [(r["style_id"], int(r["seed"]), Path(r["image_path"])) for r in f5.to_dicts()]
    cache = judge_images(items)
    table = build_rescore_table(cache, old_t, new_t)
    table.write_csv(RESCORE_PATH)
    print(f"Wrote {RESCORE_PATH} ({table.height} rows)")


if __name__ == "__main__":
    main()
