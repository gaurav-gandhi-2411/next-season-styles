"""Judge repeat-scoring of the FINAL concepts: fidelity median + observed spread.

Why: the same T-shirt image scored 0.64 then 0.43 across two judge calls (~0.2 noise) against Gate-2
thresholds of 0.513 (Groq) / 0.667 (Gemini). A single call is not a precise measurement. Each final
concept is scored `N_REPEATS` times per available judge; the per-judge MEDIAN is the fidelity number
and the observed range (max - min) is its noise bound. Gate 2 uses the pre-declared protocol: every
judge that produced all repeats must have its median at or above that judge's own calibrated
threshold (`SKILL.fidelity_pass_from_per_judge`, the same function the pipeline uses).

Every raw call is appended to `data/generated/judge_repeat_j4.jsonl` (never discarded), and an
unavailable judge/call is recorded with its reason -- an unmeasured score is never imputed.

Usage:
    uv run python -m nss.generate.judge_repeat
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

import polars as pl

from nss.generate import judge_rescore, vlm_judges
from nss.generate.concept_qc_pipeline import parse_style_attributes
from nss.generate.vlm_judges import SKILL

N_REPEATS = 3
RAW_PATH = Path("data/generated/judge_repeat_j4.jsonl")
OUT_PATH = Path("reports/tables/judge_repeats.csv")
# Seconds between calls: keeps bursts under per-minute provider limits.
CALL_GAP_SECONDS = 4.0
CALLERS = {
    "groq": vlm_judges.extract_attributes_groq,
    "gemini": vlm_judges.extract_attributes_gemini,
}


def summarise(scores: list[float]) -> dict[str, float | None]:
    """Median and observed range of one judge's repeat scores (`None`s if no repeat succeeded)."""
    if not scores:
        return {"median": None, "min": None, "max": None, "spread": None}
    return {
        "median": statistics.median(scores),
        "min": min(scores),
        "max": max(scores),
        "spread": max(scores) - min(scores),
    }


def _load_successes() -> dict[tuple[str, str], list[float]]:
    """Successful repeat scores already logged, keyed `(image_path, judge)`, so reruns resume."""
    done: dict[tuple[str, str], list[float]] = {}
    if RAW_PATH.exists():
        for line in RAW_PATH.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec["available"] and rec["mean_score"] is not None:
                done.setdefault((rec["image_path"], rec["judge"]), []).append(rec["mean_score"])
    return done


def repeat_records() -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Successful raw repeat records keyed `(image_path, judge)`, in call order."""
    out: dict[tuple[str, str], list[dict[str, Any]]] = {}
    if RAW_PATH.exists():
        for line in RAW_PATH.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec["available"] and rec["mean_score"] is not None:
                out.setdefault((rec["image_path"], rec["judge"]), []).append(rec)
    return out


def judge_thresholds() -> dict[str, float]:
    """Per-judge Gate-2 thresholds from the calibration (`0.75 x` each judge's positive mean)."""
    _, _, thresholds = judge_rescore.recompute_calibration()
    return thresholds


def main() -> None:
    """Top up each scored image to N_REPEATS successful scores per judge; write the summary.

    Images: the 3 FINAL selected concepts plus `EXTRA_ITEMS`. Idempotent: only missing repeats are
    requested. A judge's first failure ends ALL its attempts for the rest of the run (a quota 429
    does not clear within seconds; hammering it only burns the retry window) and is recorded as
    the reason on every row it blocks. A judge with no calibrated threshold is never called.
    """
    from nss.generate.final_selection_figures import SELECTED, selected_seed  # lazy import

    thresholds = judge_thresholds()
    done = _load_successes()
    blocked: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    items = [(sid, selected_seed(sid), path) for sid, path in SELECTED.items()]
    for style_id, seed, image in items:
        truth = {d: parse_style_attributes(style_id)[d] for d in vlm_judges.ATTRIBUTE_DIMENSIONS}
        for judge, caller in CALLERS.items():
            scores = done.get((str(image), judge), [])[:N_REPEATS]
            reason = blocked.get(judge)
            if judge not in thresholds and not scores:
                reason = reason or f"{judge} has no calibrated threshold"
            while len(scores) < N_REPEATS and reason is None:
                res = SKILL.run_judge(
                    judge, caller, image, vlm_judges.ATTRIBUTE_DIMENSIONS, truth, "local_sdxl"
                )
                keep = ("available", "raw_extraction", "scores", "mean_score", "excluded_reason")
                rec = {"image_path": str(image), "style_id": style_id, "judge": judge}
                with RAW_PATH.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({**rec, "rep": len(scores), **{k: res[k] for k in keep}}))
                    fh.write("\n")
                if not (res["available"] and res["mean_score"] is not None):
                    reason = blocked[judge] = str(res["excluded_reason"])[:160]
                    break
                scores.append(res["mean_score"])
                time.sleep(CALL_GAP_SECONDS)
            complete = len(scores) == N_REPEATS and judge in thresholds
            rows.append(
                {
                    "style_id": style_id,
                    "seed": seed,
                    "image_path": str(image),
                    "judge": judge,
                    "n_ok": len(scores),
                    "n_requested": N_REPEATS,
                    "scores": json.dumps([round(s, 4) for s in scores]),
                    **summarise(scores),
                    "threshold": thresholds.get(judge),
                    "median_pass": (statistics.median(scores) >= thresholds[judge])
                    if complete
                    else None,
                    "failure_reasons": None if len(scores) == N_REPEATS else reason,
                }
            )
    table = pl.DataFrame(rows, infer_schema_length=None)
    table.write_csv(OUT_PATH)
    with pl.Config(tbl_cols=-1, tbl_width_chars=250):
        print(
            table.drop("image_path", "style_id", "failure_reasons").with_columns(
                pl.col(pl.Float64).round(3)
            )
        )


if __name__ == "__main__":
    main()
