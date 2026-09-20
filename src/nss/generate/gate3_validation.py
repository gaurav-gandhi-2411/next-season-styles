"""Validate Gate 3 and the integrity check against human labels (tasks N2 and N3).

A gate is only trusted after it has been run on cases with known answers:

- GATE 3: the two briefed changes of the sweater, labelled by eye on 23 lever images
  (`evals/fixtures/gate3_sweater_labels.json`). Reported per judge: per-question accuracy,
  precision and recall of "yes", and agreement of the Gate 3 verdict (majority present).
- INTEGRITY: the three malformed underwear candidates (cut-out defect, sheer mesh, folded object)
  MUST be judged incoherent, and known-good images coherent (`integrity_labels.json`). If the check
  does not fail the three malformed images it does not work, and that is reported as such.

Usage:
    NSS_LOCAL_VLM=smolvlm uv run python -m nss.generate.gate3_validation local
    uv run python -m nss.generate.gate3_validation groq
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import polars as pl

from nss.generate import gate3, local_vlm

LABELS = Path("evals/fixtures/gate3_sweater_labels.json")
INTEGRITY = Path("evals/fixtures/integrity_labels.json")
IMG_DIR = Path("data/generated/n1_levers/ladieswear_sweater_knitwear_beige_melange")
OUT_G3 = "reports/tables/gate3_validation_{judge}.csv"
OUT_INT = "reports/tables/integrity_validation_{judge}.csv"
API_GAP_SECONDS = 4.0


def _answer(judge: str, path: Path, changes: list[str]) -> tuple[list[bool], bool]:
    """`(per-change yes/no, coherent)` for one image with `judge` ('local' or an API name)."""
    if judge == "local":
        g3 = gate3.gate3_local(path, changes)
        coherent, _raw = gate3.integrity_local(path)
        return list(g3["answers"]), coherent
    res = gate3.gate3_api(judge, path, changes)
    time.sleep(API_GAP_SECONDS)
    return list(res["answers"]), bool(res["coherent"])


def _tag(judge: str) -> str:
    return f"local-{local_vlm.BACKEND}" if judge == "local" else judge


def validate_gate3(judge: str) -> pl.DataFrame:
    """Per-image, per-change judge answers vs the human labels."""
    spec = json.loads(LABELS.read_text(encoding="utf-8"))
    rows = []
    for name, truth in spec["labels"].items():
        answers, coherent = _answer(judge, IMG_DIR / name, spec["changes"])
        for i, (t, a) in enumerate(zip(truth, answers, strict=True)):
            rows.append({"image": name, "change_idx": i, "truth": t, "answer": a})
        rows.append({"image": name, "change_idx": -1, "truth": True, "answer": coherent})
    df = pl.DataFrame(rows)
    df.write_csv(OUT_G3.format(judge=_tag(judge)))
    return df


def summarise_gate3(df: pl.DataFrame) -> dict[str, float]:
    """Accuracy/precision/recall over change questions, and Gate 3 verdict agreement."""
    q = df.filter(pl.col("change_idx") >= 0)
    tp = q.filter(pl.col("truth") & pl.col("answer")).height
    fp = q.filter(~pl.col("truth") & pl.col("answer")).height
    fn = q.filter(pl.col("truth") & ~pl.col("answer")).height
    tn = q.filter(~pl.col("truth") & ~pl.col("answer")).height
    agree = 0
    images = q["image"].unique().to_list()
    for img in images:
        sub = q.filter(pl.col("image") == img)
        agree += int(
            gate3.majority_present(sub["truth"].to_list())
            == gate3.majority_present(sub["answer"].to_list())
        )
    return {
        "n_questions": q.height,
        "accuracy": (tp + tn) / q.height,
        "precision_yes": tp / (tp + fp) if tp + fp else float("nan"),
        "recall_yes": tp / (tp + fn) if tp + fn else float("nan"),
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),
        "gate3_verdict_agreement": agree / len(images),
    }


def _floor_for(case: dict) -> dict[str, float | bool]:
    """The reference-based integrity floor for one labelled case (own image left out of refs)."""
    from nss.generate import dino_scoring, qc_gates

    refs = qc_gates.reference_paths_for_style(case["style"])
    target = Path(case["path"]).resolve()
    idx = next((i for i, r in enumerate(refs) if Path(r).resolve() == target), None)
    emb = [dino_scoring.embed_image(r) for r in refs]
    return gate3.integrity_floor(dino_scoring.embed_image(target), emb, exclude_index=idx)


def validate_integrity(judge: str) -> pl.DataFrame:
    """Judge answer per labelled image; the 3 known-malformed images must come out incoherent."""
    spec = json.loads(INTEGRITY.read_text(encoding="utf-8"))
    rows = []
    for case in spec["cases"]:
        path = Path(case["path"])
        if judge == "local":
            coherent, raw = gate3.integrity_local(path)
        else:
            coherent = bool(gate3.gate3_api(judge, path, ["a garment"])["coherent"])
            raw = ""
            time.sleep(API_GAP_SECONDS)
        rows.append(
            {
                "path": case["path"],
                "truth_coherent": case["coherent"],
                "judge_coherent": coherent,
                "raw": raw,
                **{f"floor_{k}": v for k, v in _floor_for(case).items()},
                "note": case["note"],
            }
        )
    df = pl.DataFrame(rows)
    df.write_csv(OUT_INT.format(judge=_tag(judge)))
    return df


def main(judge: str) -> None:
    """Run both validations for `judge` and print the summary."""
    df3 = validate_gate3(judge)
    print("GATE 3", _tag(judge), summarise_gate3(df3))
    dfi = validate_integrity(judge)
    bad = dfi.filter(pl.col("note").str.starts_with("H3 seed") & ~pl.col("truth_coherent"))
    caught = bad.filter(~pl.col("judge_coherent")).height
    good = dfi.filter(pl.col("truth_coherent"))
    passed_good = good.filter(pl.col("judge_coherent")).height
    print(
        f"INTEGRITY {_tag(judge)}: malformed underwear failed {caught}/{bad.height}; "
        f"known-good passed {passed_good}/{good.height}; all-bad caught "
        f"{dfi.filter(~pl.col('truth_coherent') & ~pl.col('judge_coherent')).height}/"
        f"{dfi.filter(~pl.col('truth_coherent')).height}"
    )
    flo = dfi.filter(pl.col("note").str.starts_with("H3 seed") & ~pl.col("truth_coherent"))
    print(
        "INTEGRITY FLOOR (DINOv2 nearest reference >= p10 of real nearest-sibling): malformed "
        f"underwear failed {flo.filter(~pl.col('floor_pass')).height}/{flo.height}; known-good "
        f"passed {good.filter(pl.col('floor_pass')).height}/{good.height}; all-bad caught "
        f"{dfi.filter(~pl.col('truth_coherent') & ~pl.col('floor_pass')).height}/"
        f"{dfi.filter(~pl.col('truth_coherent')).height}"
    )
    with pl.Config(tbl_rows=20, fmt_str_lengths=40, tbl_width_chars=160):
        print(
            dfi.select(
                "note",
                "truth_coherent",
                "judge_coherent",
                "floor_max_sim",
                "floor_floor",
                "floor_pass",
            )
        )


if __name__ == "__main__":
    main(sys.argv[1])
