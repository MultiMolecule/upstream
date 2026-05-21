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

"""Fixture vocabularies used by upstream generators."""

from __future__ import annotations

from itertools import product

SPECIAL_TOKENS = ("<pad>", "<cls>", "<eos>", "<unk>", "<mask>", "<null>")

DNA_STANDARD_ALPHABET = tuple("ACGTNRYSWKMBDHVX|.*-?")
DNA_STREAMLINE_ALPHABET = tuple("ACGTN")

RNA_STANDARD_ALPHABET = tuple("ACGUNRYSWKMBDHVIX|.*-?")
RNA_STREAMLINE_ALPHABET = tuple("ACGUN")

PROTEIN_STANDARD_ALPHABET = tuple("ACDEFGHIKLMNPQRSTVWYXZBJUO|.*-?")


def _kmer_vocabulary(tokens: tuple[str, ...], nmers: int) -> tuple[str, ...]:
    if nmers <= 1:
        return tokens
    return tuple("".join(kmer) for kmer in product(tokens, repeat=nmers))


def multimolecule_dna_vocabulary(*, nmers: int = 1) -> tuple[str, ...]:
    tokens = DNA_STANDARD_ALPHABET if nmers <= 1 else DNA_STREAMLINE_ALPHABET
    return SPECIAL_TOKENS + _kmer_vocabulary(tokens, nmers)


def multimolecule_protein_vocabulary() -> tuple[str, ...]:
    return SPECIAL_TOKENS + PROTEIN_STANDARD_ALPHABET


def multimolecule_rna_vocabulary(*, nmers: int = 1) -> tuple[str, ...]:
    tokens = RNA_STANDARD_ALPHABET if nmers <= 1 else RNA_STREAMLINE_ALPHABET
    return SPECIAL_TOKENS + _kmer_vocabulary(tokens, nmers)
