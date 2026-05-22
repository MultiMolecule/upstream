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

"""Hugging Face Hub cache helpers for upstream fixture generators."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence


def hf_snapshot_dir(
    repo_id: str,
    *,
    revision: str | None = None,
    allow_patterns: str | Sequence[str] | None = None,
    env_var: str | None = None,
    local_files_only: bool = False,
) -> Path:
    """Return a local Hugging Face snapshot directory, honoring an explicit override."""
    override = os.environ.get(env_var) if env_var else None
    if override:
        path = Path(override).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Hugging Face snapshot override ${env_var} does not exist: {path}")
        if not path.is_dir():
            raise NotADirectoryError(f"Hugging Face snapshot override ${env_var} is not a directory: {path}")
        return path

    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        override_hint = f" or set ${env_var} to a local snapshot directory" if env_var else ""
        raise RuntimeError(
            "huggingface_hub is required to resolve Hugging Face snapshots; " f"install huggingface_hub{override_hint}."
        ) from error

    patterns = [allow_patterns] if isinstance(allow_patterns, str) else list(allow_patterns or ())
    return Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=patterns or None,
            local_files_only=local_files_only,
        )
    ).resolve()
