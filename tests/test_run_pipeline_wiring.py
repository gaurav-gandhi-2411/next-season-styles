from __future__ import annotations

import ast
import inspect
from pathlib import Path

from nss.generate import final_deliverables


def test_run_pipeline_hero_stage_kwargs_match_final_deliverables_main() -> None:
    """Regression: `final_deliverables.main`'s first parameter was renamed and `run_pipeline.py`
    kept passing the old `final_concepts_v2_path=`, so the hero stage crashed with a TypeError
    only after the whole GPU pipeline had run. Every keyword passed must exist in the signature.
    """
    tree = ast.parse(Path("scripts/run_pipeline.py").read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "main"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "final_deliverables"
    ]
    assert calls, "run_pipeline.py no longer calls final_deliverables.main -- update this test"
    params = set(inspect.signature(final_deliverables.main).parameters)
    for call in calls:
        assert {kw.arg for kw in call.keywords} <= params
