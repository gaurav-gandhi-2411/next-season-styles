"""Denylist for anything that leaves this machine (task L4): the reviewer bundle and the share zip.

The working folder holds two files that must never be shared -- `.env` (API keys) and
`vertex-sa.json` (a GCP service-account key). Both are git-ignored and were never committed, but
they sit next to the code, so a hand-made zip of the folder would leak them. Everything that copies
files for the reviewer goes through `assert_shareable`, which refuses on any of:

* a denied FILE NAME or key-shaped extension (`.env*`, `*-sa.json`, `*.pem`, `id_rsa*`, ...);
* a denied DIRECTORY component (`data/`, `.venv`, `__pycache__`, model caches, ...);
* key-shaped CONTENT (private-key blocks, service-account JSON, common API-key prefixes) in a file
  small enough to be text, so a secret cannot ride along under an innocent name.

Fail closed: an unreadable file is refused, never assumed safe.
"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path, PurePosixPath

# The two files GG must handle manually (named in every build's warning).
MANUAL_HANDLING_FILES: tuple[str, ...] = (".env", "vertex-sa.json")

DENY_FILE_PATTERNS: tuple[str, ...] = (
    ".env",
    ".env.*",
    "*.env",
    "*-sa.json",
    "*service-account*.json",
    "*serviceaccount*.json",
    "kaggle.json",
    "credentials*.json",
    "client_secret*.json",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "id_rsa*",
    "id_ed25519*",
    "*.kdbx",
    ".netrc",
    "secrets.*",
)
DENY_DIR_PARTS: frozenset[str] = frozenset(
    {
        "data",
        ".venv",
        "venv",
        "__pycache__",
        ".git",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".cache",
        "huggingface",
        ".huggingface",
        "node_modules",
        "dist",
    }
)
DENY_SUFFIXES: frozenset[str] = frozenset({".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".onnx"})
SECRET_CONTENT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----"),
    re.compile(r'"type"\s*:\s*"service_account"'),
    re.compile(r'"private_key"\s*:'),
    re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),
    re.compile(r"\bgsk_[0-9A-Za-z]{20,}"),
    re.compile(r"\bsk-or-[0-9A-Za-z\-]{20,}"),
    re.compile(r"\bghp_[0-9A-Za-z]{30,}"),
    re.compile(r"\bhf_[0-9A-Za-z]{30,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
)
INLINE_B64 = re.compile(r"data:[A-Za-z0-9/+.\-]+;base64,[A-Za-z0-9+/=]+")
MAX_SCAN_BYTES = 8_000_000  # bigger than the largest text file we ship (a 2.5 MB HTML page)


class UnshareableFileError(RuntimeError):
    """A file matched the denylist or contains key-shaped content."""


def denied_reason(path: Path, root: Path | None = None) -> str | None:
    """Why `path` must not be shared, or `None` if the NAME/LOCATION is acceptable.

    `root` (the repo root) makes the directory check relative, so a checkout that itself lives
    under a directory called e.g. `data` is not misjudged.
    """
    rel = path.relative_to(root) if root is not None and path.is_absolute() else path
    posix = PurePosixPath(rel.as_posix())
    for part in posix.parts[:-1]:
        if part in DENY_DIR_PARTS:
            return f"inside a denied directory ({part}/)"
    name = posix.name
    for pattern in DENY_FILE_PATTERNS:
        if fnmatch.fnmatch(name.lower(), pattern.lower()):
            return f"key-shaped or secret file name (matches {pattern})"
    if posix.suffix.lower() in DENY_SUFFIXES:
        return f"model/weights file ({posix.suffix})"
    return None


def secret_content_reason(path: Path) -> str | None:
    """Why the CONTENT of `path` looks like a secret, or `None` (fail closed if unreadable)."""
    try:
        size = path.stat().st_size
        if size > MAX_SCAN_BYTES:
            return f"too large to scan ({size} bytes)"
        data = path.read_bytes()
    except OSError as exc:
        return f"unreadable, refused: {exc}"
    text = data.decode("utf-8", errors="ignore")
    # Inline image payloads (DEMO.html) are random base64: scanning them would risk false matches,
    # and an image is not a place a text secret can hide from the file-name checks.
    text = INLINE_B64.sub("data:<b64>", text)
    for pattern in SECRET_CONTENT_PATTERNS:
        if pattern.search(text):
            return f"contains key-shaped content ({pattern.pattern[:40]}...)"
    return None


def assert_shareable(path: Path, root: Path | None = None, scan_content: bool = True) -> None:
    """Raise `UnshareableFileError` unless `path` is safe to hand to a reviewer."""
    reason = denied_reason(path, root)
    if reason is None and scan_content and path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        reason = secret_content_reason(path)
    if reason is not None:
        raise UnshareableFileError(f"refusing to share {path}: {reason}")


def manual_handling_warning(root: Path) -> str:
    """Loud multi-line warning naming the secret files present in `root` (empty if none)."""
    present = [f for f in MANUAL_HANDLING_FILES if (root / f).exists()]
    if not present:
        return ""
    bar = "!" * 78
    return (
        f"{bar}\n"
        "WARNING: this folder contains secrets that are NOT in the bundle or the share zip:\n"
        + "".join(f"    {f}\n" for f in present)
        + "Do NOT zip the folder by hand. Delete or rotate these before sharing it in any form\n"
        "(`.env` holds API keys; `vertex-sa.json` is a GCP service-account key).\n"
        f"{bar}"
    )
