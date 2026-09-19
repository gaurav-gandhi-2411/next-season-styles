from __future__ import annotations

import importlib.util
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from nss.share_safety import (
    MANUAL_HANDLING_FILES,
    UnshareableFileError,
    assert_shareable,
    denied_reason,
    manual_handling_warning,
    secret_content_reason,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "name",
    [
        ".env",
        ".env.local",
        "prod.env",
        "vertex-sa.json",
        "my-project-sa.json",
        "gcp-service-account-prod.json",
        "kaggle.json",
        "credentials.json",
        "client_secret_123.json",
        "server.pem",
        "signing.key",
        "id_rsa",
        "id_ed25519.pub",
        "secrets.yaml",
        "model.safetensors",
    ],
)
def test_denied_file_names(name: str) -> None:
    assert denied_reason(Path("reports") / name) is not None


@pytest.mark.parametrize(
    "path",
    ["data/raw/articles.csv", ".venv/lib/x.py", "src/nss/__pycache__/a.pyc", "a/.cache/b.txt"],
)
def test_denied_directories(path: str) -> None:
    assert "denied directory" in (denied_reason(Path(path)) or "")


def test_the_two_manual_handling_files_are_denied() -> None:
    assert set(MANUAL_HANDLING_FILES) == {".env", "vertex-sa.json"}
    for f in MANUAL_HANDLING_FILES:
        assert denied_reason(Path(f)) is not None


@pytest.mark.parametrize(
    "path", ["reports/WRITEUP.md", "reports/DEMO.html", "reports/figures/FINAL_concepts.png"]
)
def test_ordinary_deliverables_are_allowed(path: str) -> None:
    assert denied_reason(Path(path)) is None


def test_key_shaped_content_is_refused_under_an_innocent_name(tmp_path: Path) -> None:
    f = tmp_path / "notes.txt"
    f.write_text('{"type": "service_account", "private_key": "x"}', encoding="utf-8")
    assert secret_content_reason(f) is not None
    with pytest.raises(UnshareableFileError):
        assert_shareable(f)
    g = tmp_path / "readme.md"
    g.write_text("-----BEGIN PRIVATE KEY-----\nabc\n", encoding="utf-8")
    with pytest.raises(UnshareableFileError):
        assert_shareable(g)


def test_inline_base64_does_not_cause_false_positives(tmp_path: Path) -> None:
    f = tmp_path / "page.html"
    f.write_text(
        '<img src="data:image/png;base64,AIzaAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA==">',
        encoding="utf-8",
    )
    assert secret_content_reason(f) is None


def test_unreadable_file_fails_closed(tmp_path: Path) -> None:
    assert "unreadable" in (secret_content_reason(tmp_path / "missing.txt") or "")


def test_warning_names_both_files_when_present(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("K=v", encoding="utf-8")
    (tmp_path / "vertex-sa.json").write_text("{}", encoding="utf-8")
    text = manual_handling_warning(tmp_path)
    assert ".env" in text and "vertex-sa.json" in text and "WARNING" in text
    assert manual_handling_warning(tmp_path / "empty") == ""


def test_build_submission_refuses_to_copy_a_denied_source(tmp_path: Path, monkeypatch) -> None:
    """The bundle builder cannot copy `.env` or a key file even if one is put in SOURCES."""
    module = _load_script("build_submission")
    monkeypatch.chdir(tmp_path)
    env = tmp_path / ".env"
    env.write_text("GROQ_API_KEY=abc", encoding="utf-8")
    monkeypatch.setattr(module, "BUNDLE", tmp_path / "bundle")
    monkeypatch.setattr(module, "SOURCES", {"innocent.md": env})
    with pytest.raises(UnshareableFileError):
        module.build()
    assert not (tmp_path / "bundle" / "innocent.md").exists()

    key = tmp_path / "vertex-sa.json"
    key.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(module, "SOURCES", {"notes.md": key})
    with pytest.raises(UnshareableFileError):
        module.build()


def test_real_bundle_sources_are_all_shareable() -> None:
    module = _load_script("build_submission")
    for src in module.SOURCES.values():
        assert_shareable(REPO_ROOT / src)


def test_share_zip_excludes_secrets_and_data_and_aborts_on_a_tracked_secret(tmp_path: Path) -> None:
    module = _load_script("make_share_zip")
    repo = tmp_path / "repo"
    (repo / "data" / "raw").mkdir(parents=True)
    (repo / "src").mkdir()
    (repo / "src" / "a.py").write_text("x = 1\n", encoding="utf-8")
    (repo / "data" / "raw" / ".gitkeep").write_text("", encoding="utf-8")
    (repo / ".env").write_text("K=v", encoding="utf-8")
    (repo / "vertex-sa.json").write_text("{}", encoding="utf-8")
    (repo / ".gitignore").write_text(".env\n*-sa.json\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)

    out = tmp_path / "share.zip"
    n, size = module.build_zip(repo, out)
    names = zipfile.ZipFile(out).namelist()
    assert size > 0 and n == len(names)
    joined = "\n".join(names)
    assert "src/a.py" in joined
    assert ".env" not in joined and "vertex-sa" not in joined and "data/" not in joined

    # Even if a secret is force-tracked, the build aborts and writes nothing.
    subprocess.run(["git", "add", "-f", ".env"], cwd=repo, check=True)
    out2 = tmp_path / "share2.zip"
    with pytest.raises(UnshareableFileError):
        module.build_zip(repo, out2)
    assert not out2.exists()
