"""INGEST: clone a repo into a temporary directory, load what matters into memory, delete the clone.

Everything downstream reads from RepoBuffer.files; nothing of the analyzed repo stays on disk.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from ..analyzer.clone import SourceError, split_tree_url, validate_url
from ..settings import BACKEND_ROOT

CODE_EXT = {".py", ".ipynb", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx"}
TEXT_EXT = {".md", ".txt", ".yaml", ".yml", ".toml", ".json", ".cfg", ".ini", ".jinja", ".j2", ".prompt", ".tmpl"}
SPECIAL_NAMES = {".env.example", ".env.sample", "requirements.txt", "pyproject.toml", "package.json", "Dockerfile"}
IGNORE_DIRS = {".git", ".venv", "venv", "env", "node_modules", "__pycache__", "dist", "build", "site-packages", ".next",
               ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache", "coverage", ".idea", ".vscode", "out", "target"}
IGNORE_SUFFIXES = (".min.js", ".d.ts", ".map", "-lock.json", ".lock")
MAX_FILES = 2500
MAX_FILE_BYTES = 300_000
MAX_TOTAL_BYTES = 25_000_000
CLONE_TIMEOUT_S = 180


@dataclass
class RepoBuffer:
    name: str
    files: dict[str, str] = field(default_factory=dict)   # repo-relative path -> text (notebooks as code)
    commit: str | None = None
    repo_url: str | None = None
    local_path: str | None = None
    truncated: bool = False
    skipped: int = 0
    other_paths: list[str] = field(default_factory=list)  # files seen but not loaded (data, images, ...): names only

    @property
    def digest(self) -> str:
        h = hashlib.sha256()
        for path in sorted(self.files):
            h.update(path.encode())
            h.update(hashlib.sha256(self.files[path].encode("utf-8", "replace")).digest())
        return h.hexdigest()[:16]

    def code_files(self) -> dict[str, str]:
        return {p: t for p, t in self.files.items() if Path(p).suffix in CODE_EXT}

    def line(self, path: str, line: int | None) -> str:
        if line is None or path not in self.files:
            return ""
        lines = self.files[path].splitlines()
        return lines[line - 1].strip() if 0 < line <= len(lines) else ""

    @property
    def size_bytes(self) -> int:
        return sum(len(t) for t in self.files.values())


def notebook_to_code(raw: str) -> str:
    """Code cells joined with cell markers; markdown and outputs dropped. Line numbers refer to this text."""
    try:
        nb = json.loads(raw)
    except json.JSONDecodeError:
        return ""
    parts = []
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        src = "".join(src) if isinstance(src, list) else str(src)
        # IPython magics/shell lines are not Python
        src = "\n".join(("# " + ln) if ln.lstrip().startswith(("%", "!")) else ln for ln in src.splitlines())
        parts.append(f"# %% [cell {i}]\n{src}")
    return "\n\n".join(parts)


def _ignored_dir(rel: Path) -> bool:
    return any(part in IGNORE_DIRS or (part.startswith(".") and part not in {".env.example", ".env.sample"})
               for part in rel.parts[:-1])


def _wanted(rel: Path) -> bool:
    if _ignored_dir(rel):
        return False
    name = rel.name
    if name.endswith(IGNORE_SUFFIXES):
        return False
    return rel.suffix in CODE_EXT or rel.suffix in TEXT_EXT or name in SPECIAL_NAMES


def load_dir(root: Path, buffer: RepoBuffer) -> RepoBuffer:
    total = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if not _wanted(rel):
            if not _ignored_dir(rel) and len(buffer.other_paths) < MAX_FILES:
                buffer.other_paths.append(rel.as_posix())
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > MAX_FILE_BYTES or total + size > MAX_TOTAL_BYTES:
            buffer.skipped += 1
            continue
        if len(buffer.files) >= MAX_FILES:
            buffer.truncated = True
            break
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if rel.suffix == ".ipynb":
            text = notebook_to_code(text)
        buffer.files[rel.as_posix()] = text
        total += size
    return buffer


def _git(*args: str, cwd: Path | None = None, timeout: int = 30) -> subprocess.CompletedProcess:
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env)


def _commit(root: Path) -> str | None:
    try:
        out = _git("rev-parse", "HEAD", cwd=root)
        return (out.stdout.strip() or None) if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _force_remove(func, path, _exc):  # git pack files are read-only on Windows
    os.chmod(path, stat.S_IWRITE)
    func(path)


def load(repo_url: str | None = None, local_path: str | None = None) -> RepoBuffer:
    if bool(repo_url) == bool(local_path):
        raise SourceError("provide exactly one of repo_url or local_path")
    if local_path:
        root = Path(local_path)
        root = (root if root.is_absolute() else BACKEND_ROOT / root).resolve()
        if not root.is_dir():
            raise SourceError(f"local_path not found: {root}")
        return load_dir(root, RepoBuffer(name=root.name, commit=_commit(root), local_path=str(root)))

    url = validate_url(repo_url or "")
    base, ref, sub = split_tree_url(url)
    name = (sub.rstrip("/").rsplit("/", 1)[-1] if sub else base.rstrip("/").rsplit("/", 1)[-1]).removesuffix(".git")
    tmp = tempfile.mkdtemp(prefix="evalgraph-")
    try:
        target = Path(tmp) / "repo"
        branch = ["--branch", ref] if ref else []
        try:
            out = _git("clone", "--depth", "1", "--single-branch", *branch, "-c", "core.longpaths=true", base, str(target),
                       timeout=CLONE_TIMEOUT_S)
        except subprocess.TimeoutExpired as exc:
            raise SourceError(f"git clone timed out after {CLONE_TIMEOUT_S}s") from exc
        if out.returncode != 0:
            raise SourceError(f"git clone failed: {out.stderr.strip()[-300:]}")
        root = target / sub if sub else target
        if not root.is_dir():
            raise SourceError(f"folder '{sub}' not found on branch '{ref}'")
        return load_dir(root, RepoBuffer(name=name, commit=_commit(target), repo_url=url))
    finally:
        import shutil

        shutil.rmtree(tmp, onerror=_force_remove)
