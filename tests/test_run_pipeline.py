"""Light tests for scripts/run_pipeline.py's pure wiring/scoping logic.

Only the fast, deterministic, no-I/O functions are tested here (`seeds_for_style`, `_stage_index`,
CLI defaults) -- the real end-to-end run itself is "the test" for the stage functions,
matching `tests/test_agent_demo.py`'s identical rationale for not mocking out a script
whose entire point is demonstrating real, live behaviour.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import run_pipeline  # noqa: E402 -- see sys.path.insert above; module object needed for monkeypatching
from run_pipeline import (  # noqa: E402 -- see sys.path.insert above
    DEFAULT_TABLES_DIR,
    EXPECTED_N_DESIGN_BRIEFS,
    STAGE_ORDER,
    _stage_index,
    build_arg_parser,
    build_design_briefs_mod,
    is_curated_design_briefs,
    run_briefs_stage,
    seeds_for_style,
)

from nss.generate import (  # noqa: E402 -- see sys.path.insert above
    final_concepts,
    final_concepts_v2,
)
from nss.models import final_three_shap_verdict  # noqa: E402 -- see sys.path.insert above


def test_stage_order_is_the_documented_seven_stages() -> None:
    """`STAGE_ORDER` matches the documented chain: panel -> ... -> hero."""
    assert STAGE_ORDER == ("panel", "features", "forecast", "briefs", "generate", "score", "hero")


def test_stage_index_is_monotonic_with_stage_order() -> None:
    """`_stage_index` returns each stage's position, so `--stop-after` comparisons are correct."""
    indices = [_stage_index(stage) for stage in STAGE_ORDER]
    assert indices == list(range(len(STAGE_ORDER)))


def test_seeds_for_style_returns_first_n_seeds_for_a_normal_style() -> None:
    """A fully-qualified style gets the first `n_seeds` of `final_concepts_v2.INITIAL_SEEDS`,
    unmodified."""
    assert seeds_for_style("Ladieswear || T-shirt || Jersey Basic || Black || Solid", 1) == (42,)
    assert seeds_for_style("Ladieswear || T-shirt || Jersey Basic || Black || Solid", 2) == (42, 43)


def test_seeds_for_style_ignores_disqualifications_outside_initial_seeds() -> None:
    """The T-shirt style's `VISUAL_QC_DISQUALIFIED_SEEDS` entry (46/47/48/49, a retry-round
    finding) has no overlap with `INITIAL_SEEDS` (42-45) -- a round-0-only run like this script's
    is never affected by it, so the first `n_seeds` of `INITIAL_SEEDS` are returned unchanged."""
    t_shirt = "Ladieswear || T-shirt || Jersey Basic || Black || Solid"
    assert final_concepts_v2.VISUAL_QC_DISQUALIFIED_SEEDS[t_shirt] == frozenset({46, 47, 48, 49})
    assert seeds_for_style(t_shirt, 2) == (42, 43)


def test_seeds_for_style_underwear_is_fully_qualified_under_v2() -> None:
    """Unlike the original run (`final_concepts.UNDERWEAR_VISUAL_QC_DISQUALIFIED_SEEDS`), the v2 run
    confirmed every underwear candidate is free of a human model -- the style is absent from
    `final_concepts_v2.VISUAL_QC_DISQUALIFIED_SEEDS`, so seed selection is unaffected."""
    underwear = final_concepts.UNDERWEAR_STYLE_KEY
    assert underwear not in final_concepts_v2.VISUAL_QC_DISQUALIFIED_SEEDS
    assert seeds_for_style(underwear, 1) == (42,)


def test_seeds_for_style_raises_when_n_seeds_exceeds_available() -> None:
    """Only 4 seeds exist in `INITIAL_SEEDS` -- requesting 5 raises a clear, actionable error
    rather than silently returning a short tuple."""
    t_shirt = "Ladieswear || T-shirt || Jersey Basic || Black || Solid"
    with pytest.raises(ValueError, match="exceeds the 4 non-disqualified seeds available"):
        seeds_for_style(t_shirt, 5)


def test_arg_parser_defaults_write_deterministic_stages_to_the_real_tables_dir() -> None:
    """Default `--tables-out-dir` is the REAL `reports/tables/` location (safe -- see module
    docstring STAGE-OUTPUT ISOLATION: panel/features/forecast/briefs are deterministic)."""
    args = build_arg_parser().parse_args([])
    assert args.tables_out_dir == DEFAULT_TABLES_DIR == Path("reports/tables")
    assert args.stop_after == "hero"
    assert args.n_seeds == 1


def test_arg_parser_rejects_n_seeds_outside_one_or_two() -> None:
    """`--n-seeds` is restricted to {1, 2} -- the scoped-down run never re-derives all 4 of the
    original seeds via this script."""
    with pytest.raises(SystemExit):
        build_arg_parser().parse_args(["--n-seeds", "4"])


def test_arg_parser_force_briefs_defaults_to_false() -> None:
    """`--force-briefs` is opt-in -- the safe default is to reuse an existing curated
    `design_briefs.json` rather than silently overwriting it (see `is_curated_design_briefs`)."""
    args = build_arg_parser().parse_args([])
    assert args.force_briefs is False
    assert build_arg_parser().parse_args(["--force-briefs"]).force_briefs is True


# --- is_curated_design_briefs / run_briefs_stage guard (this task) -----------------------------


def _curated_briefs(n: int = EXPECTED_N_DESIGN_BRIEFS) -> list[dict[str, object]]:
    """A synthetic, minimal "real, refined" `design_briefs.json` payload for guard tests."""
    return [{"style_id": f"style-{i}", "applied_changes": [f"change-{i}"]} for i in range(n)]


def test_is_curated_design_briefs_true_for_the_real_committed_file() -> None:
    """The actual, hand-refined `reports/tables/design_briefs.json` this task exists to protect
    passes the guard -- if this regresses, the guard would incorrectly treat the real file as a
    stub and let the briefs stage silently overwrite it."""
    assert is_curated_design_briefs(build_design_briefs_mod.OUT_PATH) is True


def test_is_curated_design_briefs_false_when_file_missing(tmp_path: Path) -> None:
    assert is_curated_design_briefs(tmp_path / "does_not_exist.json") is False


def test_is_curated_design_briefs_false_for_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "design_briefs.json"
    path.write_text("{not valid json", encoding="utf-8")
    assert is_curated_design_briefs(path) is False


def test_is_curated_design_briefs_false_for_wrong_style_count(tmp_path: Path) -> None:
    """A naive rebuild that only produced 2 of the 3 expected styles is not curated."""
    path = tmp_path / "design_briefs.json"
    path.write_text(json.dumps(_curated_briefs(n=2)), encoding="utf-8")
    assert is_curated_design_briefs(path) is False


def test_is_curated_design_briefs_false_when_applied_changes_missing(tmp_path: Path) -> None:
    """The exact stub shape `build_all_design_briefs()` produces (no `applied_changes` field) --
    this is the case the guard must catch to avoid the `hero`-stage `KeyError`."""
    stub = [{"style_id": f"style-{i}"} for i in range(EXPECTED_N_DESIGN_BRIEFS)]
    path = tmp_path / "design_briefs.json"
    path.write_text(json.dumps(stub), encoding="utf-8")
    assert is_curated_design_briefs(path) is False


def test_is_curated_design_briefs_false_when_applied_changes_empty(tmp_path: Path) -> None:
    stub = [
        {"style_id": f"style-{i}", "applied_changes": []} for i in range(EXPECTED_N_DESIGN_BRIEFS)
    ]
    path = tmp_path / "design_briefs.json"
    path.write_text(json.dumps(stub), encoding="utf-8")
    assert is_curated_design_briefs(path) is False


def _setup_briefs_stage_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, dict[str, bool], Path]:
    """Common fixture for `run_briefs_stage` guard tests: a scratch `tables_out_dir`, a minimal
    `final_three_path` CSV, and every I/O-heavy collaborator monkeypatched to a cheap stub --
    isolates the guard branch itself from the (already separately tested/proven) real
    SHAP-verdict/exemplar/brief-generation logic.

    Returns:
        `(tables_out_dir, design_briefs_path, called, final_three_path)` where `called["build_all"]`
        records whether `build_design_briefs_mod.build_all_design_briefs` was invoked.
    """
    tables_out_dir = tmp_path / "tables"
    tables_out_dir.mkdir()
    final_three_path = tmp_path / "top_styles_final_three.csv"
    pl.DataFrame({"style_key": ["a"]}).write_csv(final_three_path)

    monkeypatch.setattr(
        final_three_shap_verdict,
        "build_verdict_table",
        lambda _final_three: pl.DataFrame({"style_key": ["a"]}),
    )
    monkeypatch.setattr(
        run_pipeline, "run_exemplars_stage", lambda *_args, **_kwargs: tmp_path / "manifest.csv"
    )
    called = {"build_all": False}

    def _fake_build_all(**_kwargs: object) -> list[dict[str, object]]:
        called["build_all"] = True
        return _curated_briefs()

    monkeypatch.setattr(build_design_briefs_mod, "build_all_design_briefs", _fake_build_all)

    design_briefs_path = tables_out_dir / build_design_briefs_mod.OUT_PATH.name
    return tables_out_dir, design_briefs_path, called, final_three_path


def test_run_briefs_stage_skips_regeneration_when_curated_file_already_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default (`force_briefs=False`) path: an existing curated `design_briefs.json` is left
    byte-identical and `build_all_design_briefs` is never called."""
    tables_out_dir, design_briefs_path, called, final_three_path = _setup_briefs_stage_fixture(
        tmp_path, monkeypatch
    )
    original_text = json.dumps(_curated_briefs(), indent=4)  # deliberately different formatting
    design_briefs_path.write_text(original_text, encoding="utf-8")

    result = run_briefs_stage(final_three_path, tables_out_dir, tmp_path / "images")

    assert result == design_briefs_path
    assert called["build_all"] is False
    assert design_briefs_path.read_text(encoding="utf-8") == original_text  # untouched, byte-exact


def test_run_briefs_stage_regenerates_when_no_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A genuinely clean state (no `design_briefs.json` at all) still falls back to the normal
    from-scratch build -- the guard must not block a truly clean checkout."""
    tables_out_dir, design_briefs_path, called, final_three_path = _setup_briefs_stage_fixture(
        tmp_path, monkeypatch
    )
    assert not design_briefs_path.exists()

    run_briefs_stage(final_three_path, tables_out_dir, tmp_path / "images")

    assert called["build_all"] is True
    assert design_briefs_path.exists()


def test_run_briefs_stage_regenerates_when_existing_file_is_a_stub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-curated stub (no `applied_changes`) is treated the same as "missing" -- regenerated,
    not preserved."""
    tables_out_dir, design_briefs_path, called, final_three_path = _setup_briefs_stage_fixture(
        tmp_path, monkeypatch
    )
    stub = [{"style_id": f"style-{i}"} for i in range(EXPECTED_N_DESIGN_BRIEFS)]
    design_briefs_path.write_text(json.dumps(stub), encoding="utf-8")

    run_briefs_stage(final_three_path, tables_out_dir, tmp_path / "images")

    assert called["build_all"] is True


def test_run_briefs_stage_force_briefs_overrides_existing_curated_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`force_briefs=True` regenerates even when a valid curated file is already present."""
    tables_out_dir, design_briefs_path, called, final_three_path = _setup_briefs_stage_fixture(
        tmp_path, monkeypatch
    )
    original_text = json.dumps(_curated_briefs(), indent=4)
    design_briefs_path.write_text(original_text, encoding="utf-8")

    run_briefs_stage(final_three_path, tables_out_dir, tmp_path / "images", force_briefs=True)

    assert called["build_all"] is True
