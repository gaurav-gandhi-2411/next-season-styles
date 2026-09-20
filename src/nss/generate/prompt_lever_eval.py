"""Score every N1 lever image with Gate 3 + integrity (API judge, one batched call per image).

The acceptance test for a lever is a VLM asked directly, per briefed change, "does this garment
have <change>?" and answering yes. Results go to `reports/tables/prompt_lever_gate3.csv`
(append-safe: images already scored by the same judge are skipped, so reruns resume after a quota
stop).

Usage:
    uv run python -m nss.generate.prompt_lever_eval <judge: groq|gemini> <style-keyword>
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import polars as pl

from nss.generate import final_concepts, gate3, levers

OUT = Path("reports/tables/prompt_lever_gate3.csv")
CALL_GAP_SECONDS = 4.0


def main(judge: str, keyword: str) -> None:
    """Score all lever images of the style matching `keyword` that `judge` has not scored yet."""
    briefs = final_concepts.load_design_briefs()
    style_id = next(s for s in briefs if keyword.lower() in s.lower())
    changes = briefs[style_id]["applied_changes"]
    img_dir = levers.OUT_ROOT / final_concepts._slugify(style_id)
    done = pl.read_csv(OUT) if OUT.exists() else pl.DataFrame()
    seen = (
        set(done.filter(pl.col("judge") == judge)["image"].to_list())
        if not done.is_empty()
        else set()
    )
    rows = done.to_dicts() if not done.is_empty() else []
    for path in sorted(img_dir.glob("*.png")):
        if str(path) in seen:
            continue
        try:
            res = gate3.gate3_api(judge, path, changes)
        except Exception as exc:  # noqa: BLE001 - quota/parse errors: stop, keep what we have
            print(f"stopped at {path.name}: {type(exc).__name__}: {str(exc)[:120]}")
            break
        rows.append(
            {
                "judge": judge,
                "style_id": style_id,
                "image": str(path),
                "config": path.stem,
                "n_present": res["n_present"],
                "n_changes": len(changes),
                "answers": ",".join("Y" if a else "N" for a in res["answers"]),
                "coherent": res["coherent"],
                "gate3_pass": res["pass"],
            }
        )
        print(f"{path.stem}: {rows[-1]['answers']} coherent={res['coherent']}")
        time.sleep(CALL_GAP_SECONDS)
    pl.DataFrame(rows).write_csv(OUT)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
