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

"""Generate HAL golden fixtures from the published hexamer coefficients."""

from __future__ import annotations

import argparse
import sys
from itertools import product
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
MODEL_DIR = HERE.parent
REPO_ROOT = MODEL_DIR.parent.parent

sys.path.insert(0, str(REPO_ROOT))
from _corpus.load import crop_record, sequence_sha256  # noqa: E402
from _shared.download import fetch_http_file  # noqa: E402
from _shared.fixture import (  # noqa: E402
    fixture_out_dir,
    inputs_source_from_anchor_crop,
    sha256_of_file,
    write_fixture_artifacts,
)

MODEL = "hal"
CASE = "hal"
CORPUS_RECORD_ID = "dna/grch38_chr21"
CROP_NAME = "hal_160bp"
CROP_CENTER = "center"
CROP_LENGTH = 160
UPSTREAM_REPO_URL = "https://github.com/Alex-Rosenberg/cell-2015"
UPSTREAM_COMMIT = "ca54d1117fd28375260bfde3d1b46f3d6074f306"
CHECKPOINT_SOURCE = "https://zenodo.org/record/1466088/files/HAL_mer_scores.npz?download=1"
CHECKPOINT_FILENAME = "HAL_mer_scores.npz"
CHECKPOINT_SHA256 = "961acd13238e19168b691d714d2f70c385fc87443cb493a9a9f54927a0e7e6af"
CHECKPOINT_ENV_VAR = "MULTIMOLECULE_UPSTREAM_HAL_CHECKPOINT"
# HAL is a deterministic linear score, so use a tighter tolerance than neural fixtures.
ATOL = 1e-5
RTOL = 1e-5
BASE_IDS = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3}
HAL_BASES = ("A", "T", "C", "G")


def checkpoint_path() -> Path:
    return fetch_http_file(
        CHECKPOINT_SOURCE,
        CHECKPOINT_FILENAME,
        cache_prefix="hal",
        env_var=CHECKPOINT_ENV_VAR,
        sha256=CHECKPOINT_SHA256,
        description="HAL published hexamer coefficient npz",
    )


def make_hexamer_list(kmer_size: int) -> list[str]:
    return ["".join(kmer) for kmer in product(HAL_BASES, repeat=kmer_size)]


def upstream_score(sequence: str, checkpoint: Path) -> float:
    with np.load(checkpoint) as data:
        weights = np.asarray(data["weights"], dtype=np.float64).mean(axis=1)
    coefficients = dict(zip(make_hexamer_list(6), weights))
    counts = dict.fromkeys(coefficients, 0)
    for start in range(len(sequence) - 5):
        counts[sequence[start : start + 6]] += 1
    total = sum(counts.values())
    return float(sum(coefficients[hexamer] * count / total for hexamer, count in counts.items()))


def encode_sequence(sequence: str) -> torch.Tensor:
    return torch.tensor([[BASE_IDS[base] for base in sequence.upper()]], dtype=torch.long)


def main_for_case(case_name: str) -> None:
    if case_name != CASE:
        raise ValueError(f"Unsupported HAL fixture case: {case_name}")
    checkpoint = checkpoint_path()
    checkpoint_sha256 = sha256_of_file(checkpoint)
    crop = crop_record(CORPUS_RECORD_ID, CROP_LENGTH, center=CROP_CENTER)
    sequence = crop["sequence"].upper()
    if len(sequence) != CROP_LENGTH:
        raise AssertionError(f"crop length {len(sequence)} != {CROP_LENGTH}")
    if sequence_sha256(sequence) != crop["sha256"]:
        raise AssertionError("crop sha256 mismatch")

    input_ids = encode_sequence(sequence)
    expected = {"pooler_output": torch.tensor([[upstream_score(sequence, checkpoint)]], dtype=torch.float32)}

    out_dir = fixture_out_dir(REPO_ROOT, MODEL, CASE)

    meta = {
        "version": 1,
        "model": MODEL,
        "case": CASE,
        "auto_model": "AutoModel",
        "outputs": sorted(expected.keys()),
        "tolerance": {"atol": ATOL, "rtol": RTOL},
        "inputs_source": inputs_source_from_anchor_crop(
            crop,
            crop_name=CROP_NAME,
        ),
        "upstream": {
            "repository": UPSTREAM_REPO_URL,
            "commit": UPSTREAM_COMMIT,
            "checkpoint_source": CHECKPOINT_SOURCE,
            "checkpoint_sha256": checkpoint_sha256,
            "feature": "normalized_hexamer_frequency",
        },
    }
    write_fixture_artifacts(
        out_dir,
        inputs={"input_ids": input_ids},
        expected=expected,
        meta=meta,
    )
    print(f"Wrote fixture to {out_dir}")
    print(f"  input_ids: {tuple(input_ids.shape)}")
    print(f"  pooler_output: {expected['pooler_output'].item():.8f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=[CASE], help="HAL fixture case.")
    args = parser.parse_args()
    main_for_case(args.case)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
