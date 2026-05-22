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

"""Safe archive extraction helpers for upstream fixture generators."""

from __future__ import annotations

import os
import tarfile
import zipfile
from pathlib import Path


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath((str(root), str(path))) == str(root)
    except ValueError:
        return False


def _checked_member_path(root: Path, name: str) -> Path:
    target = (root / name).resolve(strict=False)
    if not _is_relative_to(target, root):
        raise ValueError(f"Archive member would extract outside destination: {name!r}")
    return target


def _checked_link_path(root: Path, member_target: Path, linkname: str, *, hardlink: bool) -> None:
    link_path = Path(linkname)
    if link_path.is_absolute():
        target = link_path.resolve(strict=False)
    elif hardlink:
        target = (root / link_path).resolve(strict=False)
    else:
        target = (member_target.parent / link_path).resolve(strict=False)
    if not _is_relative_to(target, root):
        raise ValueError(f"Archive link target would escape destination: {linkname!r}")


def _validate_tar_member(root: Path, member: tarfile.TarInfo) -> None:
    target = _checked_member_path(root, member.name)
    if member.issym() or member.islnk():
        _checked_link_path(root, target, member.linkname, hardlink=member.islnk())
    elif not (member.isfile() or member.isdir()):
        raise ValueError(f"Unsupported archive member type: {member.name!r}")


def safe_extract_tar(archive_path: Path, destination: Path) -> None:
    root = destination.expanduser().resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path) as archive:
        members = archive.getmembers()
        for member in members:
            _validate_tar_member(root, member)
        archive.extractall(root, members=members)


def safe_extract_zip(archive_path: Path, destination: Path) -> None:
    root = destination.expanduser().resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        for member in members:
            _checked_member_path(root, member.filename)
        archive.extractall(root, members=members)
