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

"""Generate APARENT checkpoint-parity golden fixture from the upstream Keras model."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tensorflow.keras.models import load_model

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

MODEL = "aparent"
CASE = "aparent"
UPSTREAM_REPOSITORY = "https://github.com/johli/aparent"
UPSTREAM_COMMIT = "69ad29791709b48689ff5d9e3a3daefc568de9ce"
CHECKPOINT_SOURCE = (
    "https://raw.githubusercontent.com/johli/aparent/"
    "69ad29791709b48689ff5d9e3a3daefc568de9ce/"
    "saved_models/aparent_large_lessdropout_all_libs_no_sampleweights.h5"
)
CHECKPOINT_FILENAME = "aparent_large_lessdropout_all_libs_no_sampleweights.h5"
CHECKPOINT_SHA256 = "1bb687d390daf04b7edfe825baa4b9293fec4ed2d3fa0bd0603ba6f8df04af0e"
CHECKPOINT_ENV_VAR = "MULTIMOLECULE_UPSTREAM_APARENT_CHECKPOINT"
CORPUS_RECORD_ID = "dna/grch38_chr21"
CROP_NAME = "polyadenylation_205bp"
CROP_CENTER = "center"
CROP_LENGTH = 205
ATOL = 1e-4
RTOL = 1e-4
DNA_CHANNELS = ("A", "C", "G", "T")
MM_RNA_IDS = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3, "N": 4}


def encode_input_ids(sequence: str) -> torch.Tensor:
    return torch.tensor(
        [[MM_RNA_IDS.get(base, MM_RNA_IDS["N"]) for base in sequence.upper()]],
        dtype=torch.long,
    )


def one_hot_dna(sequence: str) -> np.ndarray:
    array = np.zeros((1, CROP_LENGTH, len(DNA_CHANNELS), 1), dtype=np.float32)
    for index, base in enumerate(sequence.upper()):
        if base in DNA_CHANNELS:
            array[0, index, DNA_CHANNELS.index(base), 0] = 1.0
    return array


def logit(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(probability, 1e-7, 1 - 1e-7)
    return np.log(probability) - np.log1p(-probability)


def checkpoint_path() -> Path:
    return fetch_http_file(
        CHECKPOINT_SOURCE,
        CHECKPOINT_FILENAME,
        cache_prefix=MODEL,
        env_var=CHECKPOINT_ENV_VAR,
        sha256=CHECKPOINT_SHA256,
        description="APARENT Keras checkpoint",
    )


def write_meta(out_dir: Path, crop: dict[str, Any], checkpoint: Path, checkpoint_sha256: str) -> dict[str, Any]:
    meta = {
        "version": 1,
        "model": MODEL,
        "case": CASE,
        "auto_model": "AutoModelForSequencePrediction",
        "outputs": ["logits"],
        "tolerance": {"atol": ATOL, "rtol": RTOL},
        "inputs_source": inputs_source_from_anchor_crop(
            crop,
            crop_name=CROP_NAME,
        ),
        "upstream": {
            "repository": UPSTREAM_REPOSITORY,
            "commit": UPSTREAM_COMMIT,
            "checkpoint_source": CHECKPOINT_SOURCE,
            "checkpoint_sha256": checkpoint_sha256,
        },
    }
    return meta


def main() -> None:
    case = sys.argv[1] if len(sys.argv) > 1 else CASE
    if case != CASE:
        raise SystemExit(f"Unknown APARENT case {case!r}; expected {CASE!r}")
    checkpoint = checkpoint_path()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"APARENT checkpoint not found: {checkpoint}")

    crop = crop_record(CORPUS_RECORD_ID, CROP_LENGTH, center=CROP_CENTER)
    sequence = crop["sequence"].upper()
    if sequence_sha256(sequence) != crop["sha256"]:
        raise AssertionError("crop sha256 mismatch")

    input_ids = encode_input_ids(sequence)
    upstream_inputs = [
        one_hot_dna(sequence),
        np.zeros((1, 13), dtype=np.float32),
        np.ones((1, 1), dtype=np.float32),
    ]

    model = load_model(str(checkpoint), compile=False)
    isoform_probability, _cleavage_probability = model.predict(upstream_inputs, verbose=0)
    expected = {"logits": torch.from_numpy(logit(isoform_probability).astype(np.float32))}
    inputs = {"input_ids": input_ids}

    out_dir = fixture_out_dir(REPO_ROOT, MODEL, CASE)
    meta = write_meta(out_dir, crop, checkpoint, sha256_of_file(checkpoint))
    write_fixture_artifacts(
        out_dir,
        inputs=inputs,
        expected=expected,
        meta=meta,
    )
    print(f"Wrote fixture to {out_dir}")
    print(f"  input_ids: {tuple(input_ids.shape)}")
    print(f"  logits: {tuple(expected['logits'].shape)}")


if __name__ == "__main__":
    main()
