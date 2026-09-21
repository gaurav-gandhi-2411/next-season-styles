"""Track 1b: re-read the concepts with Gemini and compare it with the local judges.

Rules are pre-registered in `reports/v3/PREREGISTRATION.md` (Track 1b). Images: the four submitted
concepts plus the white-top rule pick (seed 47) and runner-up (seed 48).

- Gemini: 3 live readings per image, median score per attribute dimension (as `gemini_panel`).
  The free tier allows 20 requests/day/project, so every reading is cached the moment it is made
  (`data/generated/v3_gemini_cache.jsonl`) and a 429 stops the run cleanly; re-run after the daily
  reset and it resumes. 6 images x 3 readings = 18 requests.
- SmolVLM (gating) and Florence-2 (advisory): NOT re-run. Both decode greedily, so their readings
  are deterministic; the stored extractions in `candidates_scored.csv` are re-scored against the
  style's true attributes, which keeps the GPU free for generation. Stated, not hidden.
- Per-judge score = mean attribute score; per-judge pass at that judge's calibrated Gate 2 threshold
  (unchanged). Pairwise Cohen's kappa on binarised (score >= 0.5) per-(image, attribute) items,
  computed with the same `SKILL.cohens_kappa` as `judge_panel_kappa`, reported with n.

No verdict is changed by this; it reports agreement.

    uv run python -m nss.generate.v3_judge_reread
"""

from __future__ import annotations

import itertools
import json
import statistics
import time
from pathlib import Path

import polars as pl

from nss.generate import concept_scoring, gemini_panel, judge_repeat
from nss.generate.concept_qc_pipeline import parse_style_attributes
from nss.generate.fidelity import applicable_dimensions
from nss.generate.final_selection_figures import ALL_SELECTED
from nss.generate.vlm_judges import SKILL

SCORED = Path("reports/tables/candidates_scored.csv")
WHITE_TOP_ALTERNATES = (47, 48)
N_READINGS = 3
GAP_SECONDS = 6.0
OUT_ITEMS = Path("reports/tables/v3_judge_items.csv")
OUT_SCORES = Path("reports/tables/v3_judge_scores.csv")
OUT_KAPPA = Path("reports/tables/v3_judge_kappa.csv")
OUT_RAW = Path("reports/tables/v3_gemini_readings_raw.csv")
CACHE = Path("data/generated/v3_gemini_cache.jsonl")  # resumable across the daily quota reset


class QuotaStop(RuntimeError):
    """Gemini's free-tier daily quota (20 requests/day/project) is exhausted."""


def _norm(path: str | Path) -> str:
    return str(path).replace("\\", "/")


def images() -> list[tuple[str, str, str]]:
    """(label, style_id, image_path) for every image to re-read."""
    out = [(f"submitted:{sid.split(' || ')[1]}", sid, _norm(p)) for sid, p in ALL_SELECTED.items()]
    white = next(s for s in ALL_SELECTED if "White" in s)
    base = Path(ALL_SELECTED[white]).parent
    out += [
        (f"white_top_seed{s}", white, _norm(base / f"s0.35_seed{s}.png"))
        for s in WHITE_TOP_ALTERNATES
    ]
    return out


def local_items(scored: pl.DataFrame, label: str, style_id: str, path: str) -> list[dict]:
    """Per-dimension scores of the stored SmolVLM / Florence-2 extractions for one image."""
    row = scored.filter(pl.col("image_norm") == path).to_dicts()[0]
    truth = parse_style_attributes(style_id)
    dims = applicable_dimensions(truth["graphical_treatment"])
    out = []
    for judge in ("smolvlm", "florence2"):
        extraction = json.loads(row[f"{judge}_extraction"])
        scores = SKILL.score_attributes(extraction, {d: truth[d] for d in dims})
        out += [
            {"label": label, "image": path, "judge": judge, "dim": d, "score": float(v)}
            for d, v in scores.items()
        ]
    return out


def _cached(path: str, reading: int) -> dict | None:
    if not CACHE.exists():
        return None
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec["image"] == path and rec["reading"] == reading:
            return rec
    return None


def _read_or_stop(path: str, reading: int, dims: tuple[str, ...]) -> dict[str, str]:
    """Cached extraction, or one live call; every live reading is written to the cache at once."""
    rec = _cached(path, reading)
    if rec is not None:
        return json.loads(rec["extraction"])
    try:
        extraction = gemini_panel.read(Path(path), dims)
    except Exception as exc:  # noqa: BLE001 -- only the quota case is translated; the rest re-raises
        if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
            raise QuotaStop(str(exc)[:160]) from exc
        raise
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    record = {"image": path, "reading": reading, "extraction": json.dumps(extraction)}
    with CACHE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    time.sleep(GAP_SECONDS)
    return extraction


def gemini_items(label: str, style_id: str, path: str) -> tuple[list[dict], list[dict]]:
    """Gemini readings (cached or live): (median-per-dimension items, raw per-reading rows)."""
    truth = parse_style_attributes(style_id)
    dims = applicable_dimensions(truth["graphical_treatment"])
    per_dim: dict[str, list[float]] = {d: [] for d in dims}
    raw = []
    for i in range(N_READINGS):
        extraction = _read_or_stop(path, i, dims)
        scores = SKILL.score_attributes(extraction, {d: truth[d] for d in dims})
        for d, v in scores.items():
            per_dim[d].append(float(v))
        raw.append({"label": label, "reading": i, "extraction": json.dumps(extraction)})
    items = [
        {"label": label, "image": path, "judge": "gemini", "dim": d, "score": statistics.median(v)}
        for d, v in per_dim.items()
    ]
    return items, raw


def kappa_table(items: pl.DataFrame) -> pl.DataFrame:
    """Pairwise Cohen's kappa on binarised per-(image, dim) items, with n."""
    wide = items.pivot(on="judge", index=["image", "dim"], values="score")
    rows = []
    for a, b in itertools.combinations(sorted(items["judge"].unique().to_list()), 2):
        both = wide.filter(pl.col(a).is_not_null() & pl.col(b).is_not_null())
        ka = [int(v >= 0.5) for v in both[a].to_list()]
        kb = [int(v >= 0.5) for v in both[b].to_list()]
        agree = sum(x == y for x, y in zip(ka, kb, strict=True))
        rows.append(
            {
                "judge_a": a,
                "judge_b": b,
                "n_items": both.height,
                "raw_agreement": agree / both.height if both.height else None,
                "kappa": SKILL.cohens_kappa(ka, kb) if both.height >= 5 else None,
            }
        )
    return pl.DataFrame(rows)


def main() -> None:
    """Run the re-read and write the item, score and kappa tables."""
    scored = pl.read_csv(SCORED).with_columns(
        pl.col("image_path").str.replace_all("\\\\", "/").alias("image_norm")
    )
    thresholds = {
        "gemini": judge_repeat.judge_thresholds()["gemini"],
        **concept_scoring.judge_thresholds(["smolvlm", "florence2"]),
    }
    items: list[dict] = []
    raw: list[dict] = []
    todo = images()
    for label, style_id, path in todo:
        items += local_items(scored, label, style_id, path)
        try:
            g_items, g_raw = gemini_items(label, style_id, path)
        except QuotaStop as stop:
            done = len(CACHE.read_text(encoding="utf-8").splitlines()) if CACHE.exists() else 0
            print(
                f"STOPPED at {label}: Gemini quota exhausted ({done} of "
                f"{len(todo) * N_READINGS} readings cached; re-run after the daily reset). {stop}"
            )
            raise SystemExit(3) from stop
        items += g_items
        raw += g_raw
        print(f"read {label}", flush=True)
    frame = pl.DataFrame(items)
    frame.write_csv(OUT_ITEMS)
    pl.DataFrame(raw).write_csv(OUT_RAW)
    scores = (
        frame.group_by("label", "image", "judge")
        .agg(pl.col("score").mean().alias("mean_score"), pl.len().alias("n_dims"))
        .with_columns(pl.col("judge").replace_strict(thresholds).alias("threshold"))
        .with_columns((pl.col("mean_score") >= pl.col("threshold")).alias("judge_gate2_pass"))
        .sort("label", "judge")
    )
    scores.write_csv(OUT_SCORES)
    kappa = kappa_table(frame)
    kappa.write_csv(OUT_KAPPA)
    with pl.Config(tbl_rows=40, tbl_width_chars=200, tbl_formatting="ASCII_FULL"):
        print(scores.drop("image"))
        print(kappa)
    print("thresholds:", thresholds)


if __name__ == "__main__":
    main()
