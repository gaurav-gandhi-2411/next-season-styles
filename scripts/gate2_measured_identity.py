"""L1/L2/L3: score the measured colour check and the retrieval product type on every stored case.

Rules pre-registered in `reports/v3/PREREGISTRATION.md` (section L, commit c5799c3). For each of the
129 cases (115 H3 + 14 J6) and the 54 recorded candidates: measured colour (`colour_check`),
retrieval product type (`product_retrieval`), and Gate 2 rebuilt as the UNCHANGED SmolVLM fidelity
pass AND colour AND product type. Verdicts are compared with the pre-K4 and the K4 ones.

    uv run --no-sync python scripts/gate2_measured_identity.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent_eval import live_cases, stored_wrong_style  # noqa: E402
from gate2_identity_rescore import _identity, _rows  # noqa: E402

from nss.generate import colour_check, critic_rule, product_retrieval  # noqa: E402
from nss.generate import concept_forecast_index as _cfi  # noqa: E402,F401

TABLES = Path("reports/tables")
REQUIRED_FAIL = ("H1_wrong_attribute_Dress", "m3_coral_dress_attempt1", "m3_coral_dress_attempt2")
N9 = "data/generated/n9/ladieswear_{}/s{}_seed{}.png"
DRESS = "dress_dresses-ladies_red_solid"
SWEATER = "sweater_knitwear_beige_melange"
EXTRA_REQUIRED = {  # the K4 false rejects, scored as extra named cases
    "red_dress_seed47_s0.35": (
        "Ladieswear || Dress || Dresses Ladies || Red || Solid",
        N9.format(DRESS, "0.35", 47),
    ),
    "red_dress_seed43_s0.45": (
        "Ladieswear || Dress || Dresses Ladies || Red || Solid",
        N9.format(DRESS, "0.45", 43),
    ),
    "submitted_dress_seed44": (
        "Ladieswear || Dress || Dresses Ladies || Red || Solid",
        N9.format(DRESS, "0.35", 44),
    ),
    "submitted_sweater_seed45": (
        "Ladieswear || Sweater || Knitwear || Beige || Melange",
        N9.format(SWEATER, "0.35", 45),
    ),
}


def _paths() -> dict[str, tuple[str, str]]:
    """Case name -> (style, image path) for the 129 cases."""
    out: dict[str, tuple[str, str]] = {}
    for c in live_cases():
        out[c["name"]] = (c["style"], c["path"])
    for c in stored_wrong_style():
        out[c["name"]] = (c["style"], c["path"])
    for line in (TABLES / "v3_hard_negatives_scores.jsonl").read_text("utf-8").splitlines():
        r = json.loads(line)
        out[r["name"]] = (r["style"], r["path"])
    return out


def _measure(style: str, path: str) -> dict[str, Any]:
    col = colour_check.check(path, style)
    prod = product_retrieval.check(path, style)
    return {
        "colour_pass": col["pass"],
        "colour_de": round(col["nearest_delta_e"], 2),
        "colour_threshold": round(col["threshold"], 2),
        "mask_fallback": col["used_fallback"],
        "product_pass": prod["pass"],
        "product_retrieved": prod["retrieved"],
        "product_top1_style": prod["top1_style"],
    }


def cases() -> pl.DataFrame:
    """The 129 cases with measured identity and the three verdicts."""
    paths = _paths()
    rows = []
    for r in _rows():
        style, path = paths[r["name"]]
        m = _measure(style, path)
        prod_ok, col_ok = _identity(r["style"], r["reading"])  # K4 (VLM readings)
        fid = bool(r["old_gate2"])
        k4 = fid and prod_ok and col_ok
        l3 = fid and m["colour_pass"] and m["product_pass"]
        v_pre = critic_rule.decide(r["passes"], r["clone_ok"])
        v_k4 = critic_rule.decide({**r["passes"], "gate2": k4}, r["clone_ok"])
        v_l3 = critic_rule.decide({**r["passes"], "gate2": l3}, r["clone_ok"])
        rows.append(
            {
                "name": r["name"],
                "group": r["group"],
                "style": style.split(" || ")[1],
                "fidelity_pass": fid,
                "k4_gate2": k4,
                "l3_gate2": l3,
                **m,
                "verdict_pre_k4": v_pre,
                "verdict_k4": v_k4,
                "verdict_l3": v_l3,
            }
        )
    return pl.DataFrame(rows)


def candidates() -> pl.DataFrame:
    """The 54 recorded candidates: all-gate passes before K4, after K4 and under L3."""
    cand = pl.read_csv(TABLES / "candidates_scored.csv").to_dicts()
    k4 = pl.read_csv(TABLES / "v3_gate2_identity_rescore_candidates.csv").to_dicts()
    rows = []
    for c, k in zip(cand, k4, strict=True):
        style, path = c["style_id"], Path(c["image_path"]).as_posix()
        m = _measure(style, path)
        others = bool(c["gate1b_pass"] and c["integrity_floor_pass"] and c["gate3_pass"])
        fid = bool(c["gate2_pass"])  # SmolVLM fidelity pass, before K4
        rows.append(
            {
                "style": style.split(" || ")[1],
                "image": Path(path).name,
                "scale": c["scale"],
                "seed": c["seed"],
                "others_pass": others,
                "fidelity_pass": fid,
                "all_pre_k4": others and fid,
                "all_k4": others and bool(k["new_gate2"]),
                "all_l3": others and fid and m["colour_pass"] and m["product_pass"],
                **m,
            }
        )
    return pl.DataFrame(rows)


def extras() -> pl.DataFrame:
    """The four extra named cases (the K4 false rejects and the two submitted concepts, n9 files)."""
    rows = []
    for name, (style, path) in EXTRA_REQUIRED.items():
        rows.append({"name": name, "style": style.split(" || ")[1], **_measure(style, path)})
    return pl.DataFrame(rows)


def main() -> None:
    """Write the tables and print the required outcomes and the headline counts."""
    c = cases()
    c.write_csv(TABLES / "v3_l_identity_cases.csv")
    cand = candidates()
    cand.write_csv(TABLES / "v3_l_identity_candidates.csv")
    ex = extras()
    ex.write_csv(TABLES / "v3_l_identity_required_extras.csv")
    print("--- required (L1): FAIL expected")
    for n in REQUIRED_FAIL:
        r = c.filter(pl.col("name") == n).to_dicts()[0]
        print(
            n,
            "colour_pass",
            r["colour_pass"],
            r["colour_de"],
            "thr",
            r["colour_threshold"],
            "| product",
            r["product_pass"],
            r["product_retrieved"],
        )  # noqa: E501
    print("--- required (L1): PASS expected")
    for r in ex.to_dicts():
        print(
            r["name"],
            "colour_pass",
            r["colour_pass"],
            r["colour_de"],
            "thr",
            r["colour_threshold"],
            "| product",
            r["product_pass"],
            r["product_retrieved"],
        )  # noqa: E501
    for n in ("submitted_Sweater", "submitted_Dress", "submitted_Top", "submitted_Bikini top"):
        r = c.filter(pl.col("name") == n).to_dicts()[0]
        print(
            n,
            "colour",
            r["colour_pass"],
            r["colour_de"],
            "thr",
            r["colour_threshold"],
            "fallback",
            r["mask_fallback"],
            "| product",
            r["product_pass"],
            r["product_retrieved"],
        )  # noqa: E501
    print(
        "all-gate passes on 54 candidates: pre-K4",
        int(cand["all_pre_k4"].sum()),
        "K4",
        int(cand["all_k4"].sum()),
        "L3",
        int(cand["all_l3"].sum()),
    )  # noqa: E501
    ch = c.filter(
        (pl.col("verdict_pre_k4") != pl.col("verdict_l3"))
        | (pl.col("verdict_k4") != pl.col("verdict_l3"))
    )
    print("verdict changes vs pre-K4 or K4:", ch.height)


if __name__ == "__main__":
    main()
