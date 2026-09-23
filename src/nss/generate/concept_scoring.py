"""Score every candidate through Gates 1, 1b, 2, 3 and the integrity check.

Per image:
- GATE 1 / 1b: CLIP and DINOv2 similarity to the widened screened reference base, against that
  base's own p90 limits (`within_style_benchmark`, `gate1b_nearest_reference`); the exact-clone
  control must fail Gate 1b, per style.
- INTEGRITY: the reference-based floor (`gate3.integrity_floor`) AND the VLM coherence question of
  every local judge. The VLM question alone was shown not to work (`gate3_validation`), so the
  embedding floor is the operative automatic check; both are recorded.
- GATE 2: each local judge's blind attribute extraction scored against the style's visible
  attributes, per-judge threshold = 0.75 x that judge's calibrated positive mean; the panel passes
  iff every judge passes. Local decoding is greedy, so repeated readings are identical (spread 0):
  one reading per judge is the reading, stated rather than padded to three.
- GATE 3: each local judge answers, per briefed change, "does this garment have <change>?"; a judge
  passes iff a strict majority of the changes are present; the panel passes iff every judge does.

Judges: comma-separated local backends in `NSS_JUDGES` (default `smolvlm`). One judge is loaded at a
time; all image embeddings are computed first (the GPU is shared with SDXL only in generation).

Usage:
    NSS_JUDGES=smolvlm,moondream2 uv run python -m nss.generate.concept_scoring <style-keyword|all>
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import polars as pl

from nss.generate import (
    clip_scoring,
    colour_check,
    concept_generation,
    dino_scoring,
    final_concepts,
    final_registry,
    gate3,
    integrity_global,
    local_judge_calibration,
    local_vlm,
    product_retrieval,
)
from nss.generate.concept_qc_pipeline import parse_style_attributes
from nss.generate.fidelity import applicable_dimensions
from nss.generate.gate1b_nearest_reference import gate1b_pass, gate1b_threshold
from nss.generate.vlm_judges import SKILL
from nss.generate.within_style_benchmark import concept_similarity, style_benchmark

OUT = Path("reports/tables/candidates_scored.csv")
# Panel rule. Gemini's key is invalid (401) and Groq's daily token budget is spent, so the
# panel is the two local judges. Florence-2's agreement with the API judges on binarised attribute
# calls is too low to carry a verdict (kappa 0.36-0.48; `judge_panel_kappa.csv`) while SmolVLM's is
# 0.68 with Groq, so Florence-2 is ADVISORY (reported, never gating) and SmolVLM gates Gate 2. This
# was decided on measured agreement, not to pass any concept, and it changes one verdict (the white
# top's Gate 2: fail -> pass; its overall verdict stays FAIL on Gates 3 and integrity).
# Integrity: the GLOBAL floor gates (garment coherence is a global property: p10 of real
# nearest-sibling similarity pooled over every style, `integrity_global`); the per-style floor is
# reported as advisory (`integrity_style_pass`). Per-style is structurally a similarity gate in
# near-identical styles (the white top's 19 near-duplicate articles put its floor at 0.922), and it
# false-alarmed on 1 of 9 known-good images where the global floor passed 9 of 9. Global misses one
# known-malformed image (seed 44, sheer mesh, closest reference 0.811); per-style catches it. This
# is a correction of mechanism and framing: no concept's verdict changes (the white top fails both).
GATING_JUDGES = ("smolvlm",)
ADVISORY_JUDGES = ("florence2",)
SPACES = ("clip", "dinov2")


def judge_thresholds(backends: list[str]) -> dict[str, float]:
    """Per-judge Gate-2 threshold from that judge's calibration table."""
    out = {}
    for b in backends:
        df = pl.read_csv(local_judge_calibration.OUT.format(backend=b))
        out[b] = float(local_judge_calibration.summarise(df)["threshold"])
    return out


def similarity_rows(style_id: str, images: list[Path]) -> list[dict[str, Any]]:
    """Gate 1, Gate 1b, clone control and integrity floor for every image of one style."""
    refs = concept_generation.load_refs()[style_id]
    ref_embs = {
        "clip": [clip_scoring.embed_image(p) for p in refs],
        "dinov2": [dino_scoring.embed_image(p) for p in refs],
    }
    limits = {s: style_benchmark(ref_embs[s])["pair_p90"] for s in SPACES}
    clone = gate1b_pass({s: ref_embs[s][0] for s in SPACES}, ref_embs)
    rows = []
    for path in images:
        emb = {"clip": clip_scoring.embed_image(path), "dinov2": dino_scoring.embed_image(path)}
        sims = {s: concept_similarity(emb[s], ref_embs[s])["mean"] for s in SPACES}
        b = gate1b_pass(emb, ref_embs)
        floor = gate3.integrity_floor(emb["dinov2"], ref_embs["dinov2"])
        global_limit = integrity_global.global_floor()
        rows.append(
            {
                "style_id": style_id,
                "image_path": str(path),
                "scale": json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))["scale"],
                "seed": json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))["seed"],
                "n_refs": len(refs),
                **{f"{s}_mean_sim": sims[s] for s in SPACES},
                **{f"{s}_p90_limit": limits[s] for s in SPACES},
                "gate1_pass": SKILL.within_style_novelty_pass(sims, limits),
                **{f"{s}_max_sim": b[f"{s}_max_sim"] for s in SPACES},
                **{f"{s}_gate1b_limit": gate1b_threshold(ref_embs[s]) for s in SPACES},
                "gate1b_pass": bool(b["joint_pass"]),
                "clone_fails_gate1b": not clone["joint_pass"],
                "floor_max_sim": floor["max_sim"],
                "floor_limit": floor["floor"],
                "integrity_style_pass": floor["pass"],  # advisory
                "global_floor_limit": global_limit,
                "integrity_floor_pass": floor["max_sim"] >= global_limit,  # gating
            }
        )
    return rows


def judge_rows(backend: str, rows: list[dict[str, Any]], thresholds: dict[str, float]) -> None:
    """Add Gate 2 / Gate 3 / VLM-integrity columns for `backend` to every row (in place)."""
    local_vlm.load(backend)
    for row in rows:
        path = Path(row["image_path"])
        # A caller scoring an image outside the candidate tree (the MCP `score_concept` tool) has no
        # sidecar: it passes the briefed changes in the row instead.
        changes = row.get("changes")
        if changes is None:
            changes = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))["changes"]
        truth = parse_style_attributes(row["style_id"])
        dims = applicable_dimensions(truth["graphical_treatment"])
        extraction = local_vlm.extract_attributes_local(path, dims)
        scores = SKILL.score_attributes(extraction, {d: truth[d] for d in dims})
        # The "All over pattern" mapping (pattern_label.py) is NOT applied here: its supplementary
        # negative control false-passed a solid dress ("Melange."), and the mandatory solid-bikini
        # control could not be run (Kaggle 429). See reports/v3/TRACK1e_pattern.md.
        fid = SKILL.mean_score(scores)
        row[f"{backend}_fidelity"] = fid
        # L3: for the GATING judge, Gate 2 = the unchanged averaged fidelity AND a MEASURED colour
        # (`colour_check`) AND the retrieval product type (`product_retrieval`). The VLM's own
        # colour and product readings stay in the average but no longer decide anything alone
        # (K4's VLM-reading constraints misread red as orange). An AND: it can only tighten Gate 2.
        gating = backend in GATING_JUDGES
        if gating:
            colour = colour_check.check(path, row["style_id"])
            product = product_retrieval.check(path, row["style_id"])
        else:
            colour = {"verdict": colour_check.PASS, "nearest_delta_e": None, "threshold": None}
            product = {"pass": True, "retrieved": None}
        row[f"{backend}_colour_verdict"] = colour["verdict"]
        row[f"{backend}_colour_delta_e"] = colour["nearest_delta_e"]
        row[f"{backend}_colour_threshold"] = colour["threshold"]
        row[f"{backend}_product_ok"] = product["pass"]
        row[f"{backend}_product_retrieved"] = product["retrieved"]
        # M3: a colour distance inside the noise band is unmeasured, not pass or fail: Gate 2 is
        # None (INCONCLUSIVE in `critic_rule`) unless something else already failed it.
        if not (fid >= thresholds[backend] and product["pass"]) or colour["verdict"] == "fail":
            row[f"{backend}_gate2_pass"] = False
        elif colour["verdict"] == colour_check.ESCALATE:
            row[f"{backend}_gate2_pass"] = None
        else:
            row[f"{backend}_gate2_pass"] = True
        row[f"{backend}_extraction"] = json.dumps(extraction)
        if backend != "florence2" and changes:  # a captioner cannot answer yes/no questions
            g3 = gate3.gate3_local(path, changes)
            row[f"{backend}_gate3_answers"] = "".join("Y" if a else "N" for a in g3["answers"])
            row[f"{backend}_gate3_pass"] = g3["pass"]
            coherent, _raw = gate3.integrity_local(path)
            row[f"{backend}_vlm_coherent"] = coherent
    local_vlm.unload()


def apply_panel_rule(row: dict[str, Any]) -> None:
    """Derive the panel Gate 2 / Gate 3 columns from per-judge columns (in place).

    Gate 2 and Gate 3 need every GATING judge that has a reading to pass; advisory judges are
    reported in `gate2_advisory_pass` and never gate.
    """
    if "integrity_style_pass" not in row:  # legacy table: its floor column was the per-style one
        row["integrity_style_pass"] = row["integrity_floor_pass"]
    row["global_floor_limit"] = integrity_global.global_floor()
    row["integrity_floor_pass"] = bool(row["floor_max_sim"] >= row["global_floor_limit"])
    gating = [j for j in GATING_JUDGES if f"{j}_gate2_pass" in row]
    values = [row[f"{j}_gate2_pass"] for j in gating]
    row["gate2_pass"] = False if False in values else (None if None in values else all(values))
    advisory = [j for j in ADVISORY_JUDGES if f"{j}_gate2_pass" in row]
    row["gate2_advisory_pass"] = all(row[f"{j}_gate2_pass"] for j in advisory) if advisory else None
    g3 = [row[f"{j}_gate3_pass"] for j in GATING_JUDGES if f"{j}_gate3_pass" in row]
    row["gate3_pass"] = all(g3) if g3 else None


def reapply(path: Path = OUT) -> None:
    """Re-derive the panel columns of an existing scored table without re-running any judge."""
    rows = pl.read_csv(path).to_dicts()
    for row in rows:
        apply_panel_rule(row)
    pl.DataFrame(rows, infer_schema_length=None).write_csv(path)


def main(keyword: str) -> None:
    """Score the styles matching `keyword` (or all); append to/replace rows in `OUT`."""
    backends = os.environ.get("NSS_JUDGES", "smolvlm").split(",")
    thresholds = judge_thresholds(backends)
    styles = (
        list(final_registry.STYLE_ORDER)
        if keyword == "all"
        else [concept_generation.style_id_for(keyword)]
    )
    rows: list[dict[str, Any]] = []
    for style_id in styles:
        images = sorted(
            (concept_generation.OUT_ROOT / final_concepts._slugify(style_id)).glob("*.png")
        )
        rows.extend(similarity_rows(style_id, images))
    for backend in backends:
        judge_rows(backend, rows, thresholds)
    for row in rows:
        apply_panel_rule(row)
    new = pl.DataFrame(rows, infer_schema_length=None)
    if OUT.exists():
        old = pl.read_csv(OUT).filter(~pl.col("style_id").is_in(styles))
        new = pl.concat([old, new], how="diagonal_relaxed")
    new.write_csv(OUT)
    with pl.Config(tbl_rows=60, tbl_width_chars=220, fmt_str_lengths=20):
        print(
            new.filter(pl.col("style_id").is_in(styles)).select(
                pl.col("style_id").str.slice(12, 10).alias("s"),
                pl.col("image_path").str.slice(-22),
                "gate1_pass",
                "gate1b_pass",
                "integrity_floor_pass",
                "gate2_pass",
                "gate3_pass",
                *[f"{b}_gate3_answers" for b in backends if b != "florence2"],
            )
        )


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # polars prints Unicode; avoid cp1252 crash
    if sys.argv[1] == "--reapply":
        reapply()
    else:
        main(sys.argv[1])
