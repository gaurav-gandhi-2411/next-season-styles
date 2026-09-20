"""Gemini judge with the refreshed key: calibration controls and the four final concepts.

The old key (an `AQ.` OAuth artifact) returned 401 for every call; the replacement is a proper
`AIza` API key in `.env`. This module (1) re-runs the six calibration controls (3 real positives
scored against their own style, 3 mismatched negatives) to confirm the judge is still calibrated,
and (2) reads each of the four final concepts three times (median, as for the API judges before).
Transient 503s ("high demand") are retried with backoff; anything else stops the run and is
reported. Every raw reading is written, so a quota stop leaves an honest partial table.

Outputs: `reports/tables/gemini_calibration.csv`, `r1_gemini_readings.csv` (per reading and
dimension; also read by `judge_panel_kappa`), `r1_gemini_gate2.csv` (per concept).

Usage:
    uv run python -m nss.generate.gemini_panel
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path

import polars as pl

from nss.generate import judge_repeat, vlm_judges
from nss.generate.concept_qc_pipeline import (
    NEGATIVE_CONTROLS,
    POSITIVE_CONTROLS,
    parse_style_attributes,
)
from nss.generate.fidelity import applicable_dimensions
from nss.generate.final_selection_figures import ALL_SELECTED
from nss.generate.vlm_judges import SKILL

N_READINGS = 3
GAP_SECONDS = 6.0
RETRIES = 6
CAL_OUT = Path("reports/tables/gemini_calibration.csv")
READINGS_OUT = Path("reports/tables/r1_gemini_readings.csv")
GATE2_OUT = Path("reports/tables/r1_gemini_gate2.csv")


def read(image: Path, dims: tuple[str, ...]) -> dict[str, str]:
    """One Gemini extraction, retrying transient 503s with backoff."""
    for attempt in range(RETRIES):
        try:
            return vlm_judges.extract_attributes_gemini(image, dims)
        except Exception as exc:  # noqa: BLE001 - only the documented transient case is retried
            if "503" in str(exc) and attempt < RETRIES - 1:
                time.sleep(15 * (attempt + 1))
                continue
            raise
    raise RuntimeError("unreachable")


def calibrate() -> pl.DataFrame:
    """Score the six controls; returns per-control rows with the mean attribute score."""
    rows = []
    for kind, controls in (("positive", POSITIVE_CONTROLS), ("negative", NEGATIVE_CONTROLS)):
        for style_id, image in controls:
            truth = parse_style_attributes(style_id)
            dims = applicable_dimensions(truth["graphical_treatment"])
            scores = SKILL.score_attributes(read(image, dims), {d: truth[d] for d in dims})
            rows.append(
                {
                    "control": kind,
                    "style_id": style_id,
                    "image_path": str(image),
                    "mean_score": SKILL.mean_score(scores),
                    **{f"score_{d}": v for d, v in scores.items()},
                }
            )
            time.sleep(GAP_SECONDS)
    return pl.DataFrame(rows, infer_schema_length=None)


def main() -> None:
    """Calibrate, read the four concepts, write the tables and print the summary."""
    cal = calibrate()
    cal.write_csv(CAL_OUT)
    pos = cal.filter(pl.col("control") == "positive")["mean_score"].mean()
    neg = cal.filter(pl.col("control") == "negative")["mean_score"].mean()
    print(f"calibration: positive {pos:.3f} negative {neg:.3f} gap {pos - neg:.3f}")
    threshold = judge_repeat.judge_thresholds()["gemini"]
    readings, rows = [], []
    try:
        for style_id, image in ALL_SELECTED.items():
            truth = parse_style_attributes(style_id)
            dims = applicable_dimensions(truth["graphical_treatment"])
            fids, extractions = [], []
            for i in range(N_READINGS):
                extraction = read(image, dims)
                scores = SKILL.score_attributes(extraction, {d: truth[d] for d in dims})
                fids.append(SKILL.mean_score(scores))
                extractions.append(str(extraction))
                readings += [
                    {
                        "image": str(image),
                        "style_id": style_id,
                        "reading": i,
                        "dim": d,
                        "score": float(v),
                    }
                    for d, v in scores.items()
                ]
                time.sleep(GAP_SECONDS)
            rows.append(
                {
                    "style_id": style_id,
                    "gemini_median": statistics.median(fids),
                    "gemini_range": max(fids) - min(fids),
                    "threshold_h2": threshold,
                    "gemini_pass": statistics.median(fids) >= threshold,
                    "extractions": " | ".join(extractions),
                }
            )
    except Exception as exc:  # noqa: BLE001 - a quota stop (429) must leave a partial, honest table
        print(f"STOPPED after {len(rows)} of {len(ALL_SELECTED)} concepts: {type(exc).__name__}")
        print(str(exc)[:200])
    pl.DataFrame(readings).write_csv(READINGS_OUT)
    out = pl.DataFrame(rows) if rows else pl.DataFrame({"style_id": []})
    out.write_csv(GATE2_OUT)
    if not rows:
        return
    with pl.Config(tbl_rows=10, tbl_width_chars=200, fmt_str_lengths=120, float_precision=3):
        print(out.select("style_id", "gemini_median", "gemini_range", "gemini_pass", "extractions"))


if __name__ == "__main__":
    main()
