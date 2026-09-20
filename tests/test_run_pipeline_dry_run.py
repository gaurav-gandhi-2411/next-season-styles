"""Tests for `scripts/run_pipeline.py --dry-run`: pure wiring/safety logic only.

The real end-to-end dry run is exercised by running the script itself; these
tests pin the properties that keep it safe to hand to a reviewer -- it can only write to a scratch
root outside `data/` and `reports/`, never rebuilds the panel, and builds candidates from the
committed concept images instead of calling SDXL.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import run_pipeline  # noqa: E402 -- see sys.path.insert above; module object needed for patching
from run_pipeline import (  # noqa: E402 -- see sys.path.insert above
    DRY_RUN_CONCEPT_FILES,
    DRY_RUN_CONCEPTS_DIR,
    REPO_ROOT,
    build_arg_parser,
    build_dry_run_candidates,
    default_dry_run_scratch_dir,
    dry_run_arg_conflicts,
    judges_disabled,
    prepare_dry_run_scratch,
    resolve_dry_run_paths,
)

from nss.generate import vlm_judges  # noqa: E402 -- see sys.path.insert above
from nss.generate.final_deliverables import STYLE_ORDER  # noqa: E402


def test_dry_run_flags_default_off() -> None:
    args = build_arg_parser().parse_args([])
    assert args.dry_run is False
    assert args.scratch_dir is None
    assert dry_run_arg_conflicts(args) == []


def test_dry_run_flag_parses_with_scratch_dir(tmp_path: Path) -> None:
    args = build_arg_parser().parse_args(["--dry-run", "--scratch-dir", str(tmp_path)])
    assert args.dry_run is True
    assert args.scratch_dir == tmp_path


@pytest.mark.parametrize(
    "extra",
    [
        ["--rebuild-panel"],
        ["--force-briefs"],
        ["--tables-out-dir", "somewhere"],
        ["--pipeline-out-dir", "somewhere"],
        ["--generated-images-dir", "somewhere"],
        ["--images-dir", "somewhere"],
    ],
)
def test_flags_that_would_write_outside_scratch_conflict_with_dry_run(extra: list[str]) -> None:
    args = build_arg_parser().parse_args(["--dry-run", *extra])
    assert dry_run_arg_conflicts(args)


def test_default_scratch_is_outside_the_repo() -> None:
    root = default_dry_run_scratch_dir().resolve()
    assert not root.is_relative_to(REPO_ROOT)
    paths = resolve_dry_run_paths(root)
    for p in (
        paths.tables_dir,
        paths.pipeline_out_dir,
        paths.generated_images_dir,
        paths.images_dir,
    ):
        assert p.is_relative_to(root)


@pytest.mark.parametrize(
    "unsafe",
    [
        "data",
        "data/processed",
        "reports",
        "reports/tables",
        "reports/pipeline_run",
        ".",  # the repo root is an ancestor of both protected directories
    ],
)
def test_scratch_inside_or_above_protected_dirs_is_refused(unsafe: str) -> None:
    with pytest.raises(ValueError, match="protected"):
        resolve_dry_run_paths(REPO_ROOT / unsafe)


def test_every_resolved_output_path_avoids_data_and_reports(tmp_path: Path) -> None:
    paths = resolve_dry_run_paths(tmp_path / "scratch")
    protected = [(REPO_ROOT / "data").resolve(), (REPO_ROOT / "reports").resolve()]
    for p in (
        paths.tables_dir,
        paths.pipeline_out_dir,
        paths.generated_images_dir,
        paths.images_dir,
    ):
        assert not any(p.is_relative_to(q) for q in protected)


def test_prepare_scratch_copies_briefs_and_never_alters_the_source(tmp_path: Path) -> None:
    source = tmp_path / "committed" / "design_briefs.json"
    source.parent.mkdir()
    source.write_text('[{"style_id": "x"}]', encoding="utf-8")
    paths = resolve_dry_run_paths(tmp_path / "scratch")
    dest = prepare_dry_run_scratch(paths, source)
    assert dest == paths.tables_dir / "design_briefs.json"
    assert dest.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")
    assert paths.pipeline_out_dir.is_dir() and paths.generated_images_dir.is_dir()


def test_concept_files_cover_every_style_and_are_committed() -> None:
    assert set(DRY_RUN_CONCEPT_FILES) == set(STYLE_ORDER)
    for name in DRY_RUN_CONCEPT_FILES.values():
        assert (REPO_ROOT / DRY_RUN_CONCEPTS_DIR / name).exists(), name


def test_candidates_come_from_committed_images_with_recorded_seeds() -> None:
    concepts_dir = REPO_ROOT / DRY_RUN_CONCEPTS_DIR
    cands = build_dry_run_candidates(list(STYLE_ORDER), concepts_dir)
    assert set(cands) == set(STYLE_ORDER)
    for style_id, lst in cands.items():
        assert len(lst) == 1
        assert lst[0].style_id == style_id
        assert lst[0].image_path.parent == concepts_dir
        assert lst[0].image_path.exists()
        assert str(lst[0].seed) in lst[0].image_path.name


def test_missing_concept_image_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing"):
        build_dry_run_candidates([STYLE_ORDER[0]], tmp_path)


def test_judges_disabled_raises_unavailable_and_restores() -> None:
    original = vlm_judges.extract_attributes_gemini
    with (
        judges_disabled(),
        pytest.raises(vlm_judges.JudgeUnavailableError, match="dry-run"),
    ):
        vlm_judges.extract_attributes_gemini(Path("x.png"))
    assert vlm_judges.extract_attributes_gemini is original


def test_judges_restored_even_when_body_raises() -> None:
    original = vlm_judges.extract_attributes_gemini
    with pytest.raises(RuntimeError), judges_disabled():
        raise RuntimeError("boom")
    assert vlm_judges.extract_attributes_gemini is original


def test_dry_run_is_documented_in_module_docstring() -> None:
    assert "DRY-RUN MODE" in (run_pipeline.__doc__ or "")
