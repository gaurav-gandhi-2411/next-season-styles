"""Subprocess worker for the cross-process determinism regression test.

Not collected by pytest (filename does not match `test_*.py`). Invoked as a genuinely separate OS
process (via `subprocess.run([sys.executable, __file__])`) by
`tests/test_determinism_cross_process.py`, twice, so the two runs cannot share any in-process
state (Rust-internal hasher seeds, `PYTHONHASHSEED`, or anything else that is fixed once per
process but varies between processes) -- exactly the scenario the original bug report described
(see `nss.features.model_features` DETERMINISM (A5 FOLLOW-UP) docstring section).

Builds a small deterministic synthetic panel, runs the real `build_model_frame` -> `train_lightgbm`
-> `predict_lightgbm` pipeline (the same functions `nss.models.final_forecast` uses), and prints
the resulting predictions as JSON (`{style_key: prediction}`) to stdout -- nothing else on stdout,
so the caller can `json.loads(result.stdout)` directly.

Deliberately does NOT set `PYTHONHASHSEED` itself -- each `uv run`/`python` invocation gets
whatever the OS's default (randomized) hash seed is, matching the real-world "just rerun the
script" scenario the bug report described, and the strongest test of the `pl.Enum` fix (which does
not depend on `PYTHONHASHSEED` at all -- see module docstring cross-reference above).
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nss.models.backtest import generate_origin_schedule  # noqa: E402
from nss.models.lightgbm_model import (  # noqa: E402
    build_model_frame,
    feature_columns,
    predict_lightgbm,
    train_lightgbm,
)

_WEEK0 = date(2018, 1, 1)
_N_WEEKS = 90

# Same tiny config used by tests/test_lightgbm_model.py's _FAST_CONFIG -- fast to train, not the
# point of this test (structural/determinism correctness is).
_FAST_CONFIG: dict[str, int | float] = {
    "num_leaves": 7,
    "learning_rate": 0.2,
    "n_estimators": 5,
    "min_child_samples": 2,
}


def _weeks(n: int, start: date = _WEEK0) -> list[date]:
    return [start + timedelta(weeks=i) for i in range(n)]


def _synthetic_panel(n_weeks: int = _N_WEEKS) -> pl.DataFrame:
    """8 styles spanning several distinct category values per STYLE_KEY_COLS column, so a
    category-code collision across runs (the actual bug) is very likely to be caught if it still
    exists -- a 2-style fixture (as in other test files) has too few categories to reliably expose
    a shuffled-dictionary bug via a value-level prediction difference.
    """
    weeks = _weeks(n_weeks)
    first_seen = weeks[0]
    last_seen = weeks[-1]

    specs = [
        ("A", "Ladieswear", "Trousers", "Trousers Denim", "Black", "Solid", 10.0),
        ("B", "Menswear", "T-shirt", "Jersey Basic", "White", "Stripe", 100.0),
        ("C", "Divided", "Blouse", "Blouses", "Blue", "Check", 50.0),
        ("D", "Baby/Children", "Dress", "Dresses Ladies", "Pink", "Dot", 30.0),
        ("E", "Sport", "Shorts", "Jersey Fancy", "Grey", "Melange", 70.0),
        ("F", "Ladieswear", "Skirt", "Skirts", "Green", "Solid", 20.0),
        ("G", "Menswear", "Jacket", "Outdoor", "Brown", "Denim", 90.0),
        ("H", "Divided", "Top", "Blouses", "Red", "Lace", 40.0),
    ]

    frames = []
    for style_key, index_group, product_type, garment_group, colour, appearance, base in specs:
        frames.append(
            pl.DataFrame(
                {
                    "style_key": [style_key] * n_weeks,
                    "index_group_name": [index_group] * n_weeks,
                    "product_type_name": [product_type] * n_weeks,
                    "garment_group_name": [garment_group] * n_weeks,
                    "perceived_colour_master_name": [colour] * n_weeks,
                    "graphical_appearance_name": [appearance] * n_weeks,
                    "week_start": weeks,
                    "first_week_seen": [first_seen] * n_weeks,
                    "last_week_seen": [last_seen] * n_weeks,
                    "units": [float(base + i) for i in range(n_weeks)],
                    "n_active_articles": [float(1 + (i % 5)) for i in range(n_weeks)],
                    "price_index": [1.0 + 0.01 * i for i in range(n_weeks)],
                    "intensity_shrunk": [base + 0.5 * i for i in range(n_weeks)],
                    "units_per_active_article": [base + 0.3 * i for i in range(n_weeks)],
                }
            )
        )
    return pl.concat(frames)


def main() -> None:
    """Train + predict on the synthetic panel; print every (style_key, origin_week, prediction)
    row, in output order, as a JSON list -- deliberately NOT collapsed into a
    `{style_key: prediction}` dict (a given style_key appears once per origin_week, and a dict
    would silently keep only the last one), and deliberately order-sensitive (row order is itself
    part of what this test verifies is now deterministic -- see `nss.features.model_features`'s
    `build_features` sort, and `nss.features.style_panel`'s panel-build sorts)."""
    panel = _synthetic_panel()
    origins = generate_origin_schedule(panel)
    origin_weeks = [o.origin_week for o in origins]

    frame = build_model_frame(panel, origin_weeks)
    columns = feature_columns(frame)

    model = train_lightgbm(frame, _FAST_CONFIG, columns)
    preds = predict_lightgbm(model, frame, columns)

    records = [
        {
            "style_key": style_key,
            "origin_week": origin_week.isoformat(),
            "prediction": float(pred),
        }
        for style_key, origin_week, pred in zip(
            frame["style_key"].to_list(), frame["origin_week"].to_list(), preds, strict=True
        )
    ]
    print(json.dumps(records))


if __name__ == "__main__":
    main()
