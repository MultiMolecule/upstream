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

"""Shared checksum helpers for upstream fixture tooling."""

from __future__ import annotations

import hashlib
from pathlib import Path

_SHA256_CACHE: dict[tuple[Path, int, int, int], str] = {}


class ChecksumError(RuntimeError):
    """Raised when a checksum cannot be computed."""


def clear_sha256_cache() -> None:
    """Clear the process-local sha256 cache."""
    _SHA256_CACHE.clear()


def sha256_of_file(path: Path) -> str:
    """Return the sha256 hex digest for `path`, cached by path and file stat."""
    resolved = path.expanduser().resolve()
    try:
        stat = resolved.stat()
    except OSError as error:
        raise ChecksumError(f"{resolved}: unable to stat file for sha256: {error}") from error
    if not resolved.is_file():
        raise ChecksumError(f"{resolved}: expected a file for sha256")

    key = (resolved, stat.st_size, stat.st_mtime_ns, stat.st_ino)
    cached = _SHA256_CACHE.get(key)
    if cached is not None:
        return cached

    digest = hashlib.sha256()
    try:
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except OSError as error:
        raise ChecksumError(f"{resolved}: unable to read file for sha256: {error}") from error

    value = digest.hexdigest()
    _SHA256_CACHE[key] = value
    return value


def remember_sha256(path: Path, value: str) -> None:
    """Remember a just-computed sha256 for `path` in the process-local cache."""
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    _SHA256_CACHE[(resolved, stat.st_size, stat.st_mtime_ns, stat.st_ino)] = value
