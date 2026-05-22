# MultiMolecule
# Copyright (C) 2024-Present  MultiMolecule

# This file is part of MultiMolecule.

# MultiMolecule is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# MultiMolecule is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://multimolecule.danling.org/about/license-faq>.

"""Resolve sparse upstream source checkouts for fixture generators."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import uuid
from contextlib import contextmanager
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback.
    fcntl = None  # type: ignore[assignment]


def _context(repository: str, commit: str, cache_path: Path) -> str:
    return f"repo={repository} commit={commit} cache={cache_path}"


def _run_git(
    command: list[str],
    *,
    action: str,
    repository: str,
    commit: str,
    cache_path: Path,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode == 0:
        return result

    details = [
        f"git {action} failed ({_context(repository, commit, cache_path)})",
        f"command: {shlex.join(command)}",
    ]
    if result.stdout.strip():
        details.append(f"stdout:\n{result.stdout.rstrip()}")
    if result.stderr.strip():
        details.append(f"stderr:\n{result.stderr.rstrip()}")
    raise RuntimeError("\n".join(details))


def _git_in_tree(tree: Path, *args: str) -> list[str]:
    return ["git", "-C", str(tree), "-c", f"safe.directory={tree}", *args]


@contextmanager
def _cache_lock(source_root: Path):
    lock_path = source_root.parent / f".{source_root.name}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as handle:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def ensure_source_tree(
    repository: str,
    commit: str,
    sparse_paths: tuple[str, ...],
    *,
    env_var: str,
    cache_prefix: str,
) -> Path:
    if not sparse_paths or any(not str(path).strip() for path in sparse_paths):
        raise ValueError("sparse_paths must contain at least one non-empty path")

    override = os.environ.get(env_var)
    if override:
        return Path(override).expanduser().resolve()

    cache_root = Path(os.environ.get("MULTIMOLECULE_UPSTREAM_CACHE", "~/.cache/multimolecule-upstream"))
    source_root = (cache_root.expanduser() / f"{cache_prefix}-{commit[:12]}").resolve()
    with _cache_lock(source_root):
        if (source_root / ".git").exists():
            current = _run_git(
                _git_in_tree(source_root, "rev-parse", "HEAD"),
                action="rev-parse",
                repository=repository,
                commit=commit,
                cache_path=source_root,
            ).stdout.strip()
            if current == commit:
                return source_root
            _run_git(
                _git_in_tree(source_root, "fetch", "--depth", "1", "origin", commit),
                action="fetch",
                repository=repository,
                commit=commit,
                cache_path=source_root,
            )
            _run_git(
                _git_in_tree(source_root, "checkout", "--detach", commit),
                action="checkout",
                repository=repository,
                commit=commit,
                cache_path=source_root,
            )
            return source_root
        if source_root.exists():
            raise RuntimeError(
                f"source cache exists but is not a git checkout ({_context(repository, commit, source_root)})"
            )

        source_root.parent.mkdir(parents=True, exist_ok=True)
        temporary = source_root.parent / f".{source_root.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
        try:
            _run_git(
                ["git", "clone", "--filter=blob:none", "--no-checkout", repository, str(temporary)],
                action="clone",
                repository=repository,
                commit=commit,
                cache_path=source_root,
            )
            _run_git(
                _git_in_tree(temporary, "sparse-checkout", "set", *sparse_paths),
                action="sparse-checkout",
                repository=repository,
                commit=commit,
                cache_path=source_root,
            )
            _run_git(
                _git_in_tree(temporary, "checkout", "--detach", commit),
                action="checkout",
                repository=repository,
                commit=commit,
                cache_path=source_root,
            )
            temporary.rename(source_root)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    return source_root
