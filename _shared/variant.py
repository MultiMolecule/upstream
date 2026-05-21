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

"""Shared helpers for variant-pair upstream fixtures."""

from __future__ import annotations

import numpy as np
import torch

DNA_BASE_IDS = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3}


def encode_dna_ids(sequence: str) -> torch.Tensor:
    """Encode a DNA sequence as the fixture ``input_ids`` tensor."""
    return torch.tensor([[DNA_BASE_IDS[base] for base in sequence.upper()]], dtype=torch.long)


def dna_one_hot(sequence: str) -> np.ndarray:
    """Encode a DNA sequence as ``(batch, length, channels)`` one-hot features."""
    one_hot = np.zeros((1, len(sequence), 4), dtype=np.float32)
    for index, base in enumerate(sequence.upper()):
        channel = DNA_BASE_IDS.get(base)
        if channel is not None:
            one_hot[0, index, channel] = 1.0
    return one_hot


def dna_one_hot_with_context(
    sequence: str,
    *,
    left_context: int,
    right_context: int,
) -> torch.Tensor:
    """Encode a DNA sequence as channel-first one-hot features padded with context."""
    one_hot = np.zeros((1, 4, left_context + len(sequence) + right_context), dtype=np.float32)
    for index, base in enumerate(sequence.upper()):
        channel = DNA_BASE_IDS.get(base)
        if channel is not None:
            one_hot[0, channel, left_context + index] = 1.0
    return torch.from_numpy(one_hot)
