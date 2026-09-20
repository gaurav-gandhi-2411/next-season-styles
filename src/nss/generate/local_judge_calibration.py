"""Calibrate a local VLM judge exactly as Groq and Gemini were (task N4).

Same procedure as `concept_qc_pipeline.run_calibration` / `summarize_calibration`:

- POSITIVE controls: real catalogue images scored against their OWN style's true attributes
  (expected HIGH).
- NEGATIVE controls: the same real images scored against a DIFFERENT style's attributes, mismatched
  on purpose (expected LOW).
- PASS iff mean(positive) - mean(negative) >= `CALIBRATION_PASS_GAP` (0.3, fixed before any local
  number was seen); the fidelity threshold is then `DEFAULT_FIDELITY_THRESHOLD_FRACTION` (0.75) x
  the judge's own positive mean, as for the other judges.

Larger than the original 3+3 controls: every hand-screened reference image of every style in
`exemplar_images_screened.csv` (real photos, so no generation contamination is possible).

Usage:
    NSS_LOCAL_VLM=moondream2 uv run python -m nss.generate.local_judge_calibration
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from nss.generate import local_vlm, screen_references
from nss.generate.concept_qc_pipeline import CALIBRATION_PASS_GAP, parse_style_attributes
from nss.generate.fidelity import applicable_dimensions
from nss.generate.vlm_judges import SKILL

OUT = "reports/tables/local_judge_calibration_{backend}.csv"


def controls() -> list[tuple[str, Path]]:
    """`(style_id, real_image_path)` for every screened reference image (deterministic order)."""
    refs = screen_references.load_screened_references()
    return [(sid, p) for sid, paths in refs.items() for p in paths]


def run(backend: str) -> pl.DataFrame:
    """Score all positive and negative controls with `backend`; write and return the long table."""
    local_vlm.load(backend)
    items = controls()
    styles = sorted({s for s, _ in items})
    rows = []
    for style_id, path in items:
        extraction = local_vlm.extract_attributes_local(path)
        other = styles[(styles.index(style_id) + 1) % len(styles)]
        for control_type, target in (("positive", style_id), ("negative", other)):
            truth = parse_style_attributes(target)
            dims = applicable_dimensions(truth["graphical_treatment"])
            truth = {d: truth[d] for d in dims}
            scores = SKILL.score_attributes(extraction, truth)
            rows.append(
                {
                    "backend": backend,
                    "control_type": control_type,
                    "style_id_checked_against": target,
                    "image_path": str(path),
                    "raw_extraction_json": json.dumps(extraction),
                    "mean_score": SKILL.mean_score(scores),
                }
            )
    df = pl.DataFrame(rows)
    df.write_csv(OUT.format(backend=backend))
    return df


def summarise(df: pl.DataFrame) -> dict[str, float | bool]:
    """Positive/negative means, gap, pass flag and the resulting fidelity threshold."""
    pos = df.filter(pl.col("control_type") == "positive")["mean_score"].mean()
    neg = df.filter(pl.col("control_type") == "negative")["mean_score"].mean()
    gap = float(pos) - float(neg)
    return {
        "positive_mean": float(pos),
        "negative_mean": float(neg),
        "gap": gap,
        "passed": gap >= CALIBRATION_PASS_GAP,
        "threshold": SKILL.judge_fidelity_threshold(float(pos)),
        "n_controls": df.height // 2,
    }


def main() -> None:
    """Calibrate every backend named in `NSS_LOCAL_VLM` (comma-separated) and print the summary."""
    import os

    for backend in os.environ.get("NSS_LOCAL_VLM", "moondream2").split(","):
        summary = summarise(run(backend))
        print(backend, json.dumps(summary, indent=1))
        local_vlm.unload()


if __name__ == "__main__":
    main()
