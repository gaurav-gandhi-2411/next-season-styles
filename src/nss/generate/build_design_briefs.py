"""Adapt the final three styles' persisted data onto the `style-brief` skill's generic
input schema, run the skill, and write `reports/tables/design_briefs.json`.

HARD CONSTRAINT: modelling is FROZEN. Every value this module consumes (attributes, `predicted_
intensity`, `growth_ratio`, SHAP drivers, `dominant_mechanism`, exemplar image paths) is already
persisted on disk by earlier modules (`nss.models.diversity_forecast`,
`nss.models.final_three_shap_verdict`, `nss.data.select_final_three_exemplars`) -- nothing here
retrains a model, recomputes SHAP, or re-runs inference. This module is pure read-schema-adapt-
write glue.

DATASET-SPECIFIC MAPPING LIVES HERE, NOT IN THE SKILL: `skills/style-brief/SKILL.md` and
`generate_brief.py` deliberately know nothing about H&M's column names (`index_group_name`,
`perceived_colour_master_name`, `graphical_appearance_name`, ...). This module is exactly the
"calling code" SKILL.md's "Adapting a new dataset" section describes -- it maps this project's
actual columns onto the skill's generic `StyleProfile` schema (`garment_category`,
`construction_group`, `colour_name`, `pattern_or_finish`, ...) before calling
`generate_design_brief`.

The skill module lives outside the installed `nss` package (`skills/style-brief/generate_brief.py`,
not `src/nss/...`) so it stays a genuinely standalone, portable artifact -- loaded here via an
explicit file-path import rather than a normal package import.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import polars as pl

TOP_STYLES_PATH = Path("reports/tables/top_styles_final_three.csv")
SHAP_VERDICT_PATH = Path("reports/tables/final_three_shap_verdict.csv")
EXEMPLAR_IMAGES_PATH = Path("reports/tables/exemplar_images_final_three.csv")
OUT_PATH = Path("reports/tables/design_briefs.json")

_REPO_ROOT = Path(__file__).resolve().parents[3]
SKILL_MODULE_PATH = _REPO_ROOT / "skills" / "style-brief" / "generate_brief.py"

N_SHAP_DRIVERS = 5

# Generic keyword check (not an H&M column name) for categories that warrant tasteful, non-model
# framing downstream -- see SKILL.md's `sensitivity_tag` field and worked example 2.
_SENSITIVE_KEYWORDS = ("underwear", "night", "lingerie", "intimate")


def _load_skill_module(module_path: Path = SKILL_MODULE_PATH) -> ModuleType:
    """Load `generate_brief.py` from `skills/style-brief/` via an explicit file-path import.

    Args:
        module_path: Path to `generate_brief.py`.

    Returns:
        The loaded module, exposing `generate_design_brief` and `validate_style_profile`.

    Raises:
        FileNotFoundError: if `module_path` does not exist.
        ImportError: if the module spec/loader cannot be constructed.
    """
    if not module_path.exists():
        raise FileNotFoundError(f"style-brief skill module not found at {module_path}")
    spec = importlib.util.spec_from_file_location("style_brief_generate_brief", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not build an import spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def infer_sensitivity_tag(garment_category: str, construction_group: str) -> str | None:
    """Generic keyword check for categories warranting tasteful, non-model prompt framing.

    Args:
        garment_category: e.g. `product_type_name`.
        construction_group: e.g. `garment_group_name`.

    Returns:
        `"intimate_apparel"` if either field contains a sensitive-category keyword
        (case-insensitive), else `None`.
    """
    haystack = f"{garment_category} {construction_group}".lower()
    if any(keyword in haystack for keyword in _SENSITIVE_KEYWORDS):
        return "intimate_apparel"
    return None


def load_reference_image_paths(
    exemplar_images_path: Path = EXEMPLAR_IMAGES_PATH,
) -> dict[str, list[str]]:
    """Successfully-fetched exemplar image paths for the three `final_rank_*` styles.

    Args:
        exemplar_images_path: Path to `exemplar_images_final_three.csv`.

    Returns:
        Mapping of `style_key -> sorted list of local_image_path strings`, restricted to
        `role` starting with `final_rank_` and `fetch_success == True`.
    """
    manifest = pl.read_csv(exemplar_images_path)
    filtered = manifest.filter(
        pl.col("role").str.starts_with("final_rank_") & pl.col("fetch_success")
    )
    paths: dict[str, list[str]] = {}
    for row in filtered.sort(["style_key", "local_image_path"]).iter_rows(named=True):
        paths.setdefault(row["style_key"], []).append(row["local_image_path"])
    return paths


def build_style_profile(
    top_styles_row: dict[str, object],
    shap_verdict_row: dict[str, object],
    reference_image_paths: list[str],
) -> dict[str, object]:
    """Map one style's H&M-schema data onto the `style-brief` skill's generic `StyleProfile`.

    Args:
        top_styles_row: One row of `top_styles_final_three.csv` (must have `style_key`,
            `product_type_name`, `garment_group_name`, `perceived_colour_master_name`,
            `graphical_appearance_name`, `predicted_intensity`, `growth_ratio`).
        shap_verdict_row: The matching row of `final_three_shap_verdict.csv` (must have
            `dominant_mechanism`, `shap_driver_{1..5}_feature`, `shap_driver_{1..5}_value`).
        reference_image_paths: Successfully-fetched exemplar image paths for this style.

    Returns:
        A `StyleProfile`-shaped dict (see `skills/style-brief/SKILL.md`).
    """
    garment_category = str(top_styles_row["product_type_name"])
    construction_group = str(top_styles_row["garment_group_name"])
    growth_ratio = top_styles_row["growth_ratio"]
    return {
        "style_id": str(top_styles_row["style_key"]),
        "attributes": {
            "garment_category": garment_category,
            "construction_group": construction_group,
            "colour_name": str(top_styles_row["perceived_colour_master_name"]),
            "pattern_or_finish": str(top_styles_row["graphical_appearance_name"]),
        },
        "performance_signal": {
            "predicted_intensity": float(top_styles_row["predicted_intensity"]),
            "growth_ratio": None if growth_ratio is None else float(growth_ratio),
        },
        "dominant_mechanism": str(shap_verdict_row["dominant_mechanism"]),
        "top_drivers": [
            {
                "feature": str(shap_verdict_row[f"shap_driver_{i}_feature"]),
                "importance": float(shap_verdict_row[f"shap_driver_{i}_value"]),
            }
            for i in range(1, N_SHAP_DRIVERS + 1)
        ],
        "reference_image_paths": reference_image_paths,
        "sensitivity_tag": infer_sensitivity_tag(garment_category, construction_group),
    }


def build_all_design_briefs(
    top_styles_path: Path = TOP_STYLES_PATH,
    shap_verdict_path: Path = SHAP_VERDICT_PATH,
    exemplar_images_path: Path = EXEMPLAR_IMAGES_PATH,
    skill_module_path: Path = SKILL_MODULE_PATH,
) -> list[dict[str, object]]:
    """Build one design brief per final-three style, in `top_styles_final_three.csv` row order.

    Args:
        top_styles_path: Path to `top_styles_final_three.csv`.
        shap_verdict_path: Path to `final_three_shap_verdict.csv`.
        exemplar_images_path: Path to `exemplar_images_final_three.csv`.
        skill_module_path: Path to the `style-brief` skill's `generate_brief.py`.

    Returns:
        List of `DesignBrief` dicts (see `skills/style-brief/SKILL.md`), one per style.

    Raises:
        ValueError: if a `top_styles_final_three.csv` style_key has no matching row in
            `final_three_shap_verdict.csv`.
    """
    skill = _load_skill_module(skill_module_path)
    top_styles = pl.read_csv(top_styles_path)
    shap_verdict = pl.read_csv(shap_verdict_path)
    shap_by_style = {row["style_key"]: row for row in shap_verdict.iter_rows(named=True)}
    image_paths_by_style = load_reference_image_paths(exemplar_images_path)

    briefs: list[dict[str, object]] = []
    for row in top_styles.iter_rows(named=True):
        style_key = row["style_key"]
        if style_key not in shap_by_style:
            raise ValueError(f"no SHAP verdict found for style_key={style_key!r}")
        profile = build_style_profile(
            row, shap_by_style[style_key], image_paths_by_style.get(style_key, [])
        )
        briefs.append(skill.generate_design_brief(profile))
    return briefs


def main() -> None:
    """CLI entry point: build all three design briefs and write `design_briefs.json`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-styles-path", type=Path, default=TOP_STYLES_PATH)
    parser.add_argument("--shap-verdict-path", type=Path, default=SHAP_VERDICT_PATH)
    parser.add_argument("--exemplar-images-path", type=Path, default=EXEMPLAR_IMAGES_PATH)
    parser.add_argument("--skill-module-path", type=Path, default=SKILL_MODULE_PATH)
    parser.add_argument("--out-path", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    briefs = build_all_design_briefs(
        args.top_styles_path,
        args.shap_verdict_path,
        args.exemplar_images_path,
        args.skill_module_path,
    )
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(briefs, indent=2), encoding="utf-8")
    print(f"Wrote {args.out_path} ({len(briefs)} design briefs)")
    for brief in briefs:
        print(f"  {brief['style_id']}")
        print(f"    silhouette: {brief['silhouette']}")
        print(f"    colour_direction: {brief['colour_direction']}")


if __name__ == "__main__":
    sys.exit(main() or 0)
