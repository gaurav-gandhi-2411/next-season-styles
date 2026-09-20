"""Per-judge scores, panel median and pairwise Cohen's kappa for the judge panel.

The API judges (Groq, Gemini) are quota-limited and, in this run, unavailable (Gemini returns
401, Groq's daily token budget is spent), so their readings are taken from the logs already on disk
(`vlm_calibration_results.csv`, `judge_cache.jsonl`, `judge_repeat_*.jsonl`); the local judges are
run afresh on the SAME images against the SAME ground-truth style, so every pair is compared on
identical (image, attribute) items. An item is one (image, attribute dimension); a judge's reading
is binary: its attribute score against the style's true value is >= 0.5 (`binarize_scores`).

Reported: each judge's mean per-attribute score and accuracy on the shared items, the panel median
score per item, and Cohen's kappa for every judge pair on the items both scored, with n. Kappa on
"correct / incorrect" agreement is a measure of whether two judges make the same errors, not of
whether either is right.

Usage:
    NSS_JUDGES=smolvlm,florence2 uv run python -m nss.generate.judge_panel_kappa
"""

from __future__ import annotations

import itertools
import json
import os
from pathlib import Path

import polars as pl

from nss.generate import local_vlm
from nss.generate.concept_qc_pipeline import parse_style_attributes
from nss.generate.vlm_judges import ATTRIBUTE_DIMENSIONS, SKILL

CALIBRATION = Path("reports/tables/vlm_calibration_results.csv")
CACHE = Path("data/generated/judge_cache.jsonl")
OUT = Path("reports/tables/judge_panel_readings.csv")
KAPPA_OUT = Path("reports/tables/judge_panel_kappa.csv")


def api_items() -> list[dict]:
    """Every (image, style, judge, dim, score) reading the API judges produced, from disk."""
    rows: list[dict] = []
    cal = pl.read_csv(CALIBRATION).filter(pl.col("available"))
    for r in cal.iter_rows(named=True):
        for dim, score in json.loads(r["scores_json"]).items():
            if dim in ATTRIBUTE_DIMENSIONS:
                rows.append(
                    {
                        "image": r["image_path"],
                        "style_id": r["style_id_checked_against"],
                        "judge": r["judge_name"],
                        "dim": dim,
                        "score": float(score),
                    }
                )
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        for judge, res in rec["judges"].items():
            if res.get("available") and res.get("scores"):
                for dim, score in res["scores"].items():
                    if dim in ATTRIBUTE_DIMENSIONS:
                        rows.append(
                            {
                                "image": rec["image_path"],
                                "style_id": rec["style_id"],
                                "judge": judge,
                                "dim": dim,
                                "score": float(score),
                            }
                        )
    df = (
        pl.DataFrame(rows).group_by("image", "style_id", "judge", "dim").agg(pl.col("score").mean())
    )
    return df.to_dicts()


def local_items(backends: list[str], keys: list[tuple[str, str]]) -> list[dict]:
    """Run each local judge on the same (image, style) pairs the API judges scored."""
    rows: list[dict] = []
    for backend in backends:
        local_vlm.load(backend)
        for image, style_id in keys:
            path = Path(image.replace("\\", "/"))
            if not path.exists():
                continue
            truth = parse_style_attributes(style_id)
            extraction = local_vlm.extract_attributes_local(path, ATTRIBUTE_DIMENSIONS)
            scores = SKILL.score_attributes(extraction, {d: truth[d] for d in ATTRIBUTE_DIMENSIONS})
            rows += [
                {"image": image, "style_id": style_id, "judge": backend, "dim": d, "score": v}
                for d, v in scores.items()
            ]
        local_vlm.unload()
    return rows


def main() -> None:
    """Assemble readings, compute per-judge means, panel median and pairwise kappa."""
    backends = os.environ.get("NSS_JUDGES", "smolvlm,florence2").split(",")
    api = api_items()
    keys = sorted({(r["image"], r["style_id"]) for r in api})
    df = pl.DataFrame(api + local_items(backends, keys))
    df.write_csv(OUT)
    wide = df.pivot(on="judge", index=["image", "style_id", "dim"], values="score")
    judges = [j for j in df["judge"].unique().sort().to_list()]
    kappas = []
    for a, b in itertools.combinations(judges, 2):
        both = wide.filter(pl.col(a).is_not_null() & pl.col(b).is_not_null())
        if both.height < 5:
            continue
        ka = [int(v >= 0.5) for v in both[a].to_list()]
        kb = [int(v >= 0.5) for v in both[b].to_list()]
        kappas.append(
            {
                "judge_a": a,
                "judge_b": b,
                "n_items": both.height,
                "kappa": SKILL.cohens_kappa(ka, kb),
            }
        )
    kdf = pl.DataFrame(kappas)
    kdf.write_csv(KAPPA_OUT)
    summary = (
        df.group_by("judge")
        .agg(
            pl.col("score").mean().alias("mean_score"),
            (pl.col("score") >= 0.5).mean().alias("accuracy"),
            pl.len().alias("n_items"),
        )
        .sort("judge")
    )
    complete = wide.filter(pl.all_horizontal([pl.col(j).is_not_null() for j in judges]))
    print("items scored by ALL judges:", complete.height)
    if complete.height:
        med = complete.select(pl.concat_list(judges).list.median().alias("m"))["m"]
        print("panel median score over those items:", round(float(med.mean()), 3))
    with pl.Config(tbl_rows=30, tbl_width_chars=160):
        print(summary)
        print(kdf)


if __name__ == "__main__":
    main()
