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

"""Generate MPRA-DragoNN golden fixtures from the upstream Keras HDF5 checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
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

MODEL = "mpradragonn"
CASE = "mpradragonn"
CHECKPOINT_SOURCE = (
    "https://raw.githubusercontent.com/kundajelab/MPRA-DragoNN/"
    "9d977d7e67f1c7bcfab5a8ef777fbd720a6ec3ea/kipoi/ConvModel/pretrained.hdf5"
)
CHECKPOINT_FILENAME = "pretrained.hdf5"
CHECKPOINT_SHA256 = "27eb5aee40317b012a17a6a409f545825292eb1787c2ad506d6d37b97990f006"
CHECKPOINT_ENV_VAR = "MULTIMOLECULE_UPSTREAM_MPRADRAGONN_CHECKPOINT"
UPSTREAM_REPO_URL = "https://github.com/kundajelab/MPRA-DragoNN"
UPSTREAM_COMMIT = "9d977d7e67f1c7bcfab5a8ef777fbd720a6ec3ea"
CORPUS_RECORD_ID = "dna/grch38_chr21"
CROP_NAME = "regulatory_145bp"
CROP_CENTER = "center"
CROP_LENGTH = 145
ATOL = 1e-4
RTOL = 1e-4
DNA = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
DNA_CHANNELS = ("A", "C", "G", "T")


def one_hot(sequence: str) -> np.ndarray:
    encoded = np.zeros((1, len(sequence), len(DNA_CHANNELS)), dtype=np.float32)
    for index, base in enumerate(sequence.upper()):
        if base in DNA_CHANNELS:
            encoded[0, index, DNA_CHANNELS.index(base)] = 1.0
    return encoded


def checkpoint_path() -> Path:
    return fetch_http_file(
        CHECKPOINT_SOURCE,
        CHECKPOINT_FILENAME,
        cache_prefix=MODEL,
        env_var=CHECKPOINT_ENV_VAR,
        sha256=CHECKPOINT_SHA256,
        description="MPRA-DragoNN Keras checkpoint",
    )


def build_upstream_model(checkpoint: Path) -> tf.keras.Model:
    model = tf.keras.Sequential(name="sequential_1")
    model.add(tf.keras.Input(shape=(CROP_LENGTH, len(DNA_CHANNELS)), name="input_1"))
    for index in range(1, 4):
        model.add(tf.keras.layers.Conv1D(120, 5, activation="relu", padding="valid", name=f"conv1d_{index}"))
        model.add(
            tf.keras.layers.BatchNormalization(
                axis=-1,
                momentum=0.99,
                epsilon=0.001,
                name=f"batch_normalization_{index}",
            )
        )
        model.add(tf.keras.layers.Dropout(0.1, name=f"dropout_{index}"))
    model.add(tf.keras.layers.Flatten(name="flatten_1"))
    model.add(tf.keras.layers.Dense(12, activation="linear", name="dense_1"))
    model.load_weights(str(checkpoint))
    return model


def upstream_forward(sequence: str, checkpoint: Path) -> torch.Tensor:
    logits = build_upstream_model(checkpoint)(one_hot(sequence), training=False).numpy()
    return torch.from_numpy(logits.astype(np.float32)).contiguous()


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
            "repository": UPSTREAM_REPO_URL,
            "commit": UPSTREAM_COMMIT,
            "checkpoint_source": CHECKPOINT_SOURCE,
            "checkpoint_sha256": checkpoint_sha256,
        },
    }
    return meta


def main() -> None:
    case = sys.argv[1] if len(sys.argv) > 1 else CASE
    if case != CASE:
        raise SystemExit(f"Unknown MPRA-DragoNN case {case!r}; expected {CASE!r}")
    checkpoint = checkpoint_path()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"MPRA-DragoNN checkpoint not found: {checkpoint}")
    crop = crop_record(CORPUS_RECORD_ID, CROP_LENGTH, center=CROP_CENTER)
    sequence = crop["sequence"].upper()
    if sequence_sha256(sequence) != crop["sha256"]:
        raise AssertionError("crop sha256 mismatch")
    input_ids = torch.tensor([[DNA[base] for base in sequence]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    expected = {"logits": upstream_forward(sequence, checkpoint)}
    out_dir = fixture_out_dir(REPO_ROOT, MODEL, CASE)
    meta = write_meta(out_dir, crop, checkpoint, sha256_of_file(checkpoint))
    write_fixture_artifacts(
        out_dir,
        inputs={"input_ids": input_ids, "attention_mask": attention_mask},
        expected=expected,
        meta=meta,
    )
    print(f"Wrote fixture to {out_dir}")
    print(f"  input_ids: {tuple(input_ids.shape)}")
    print(f"  shapes: {{'logits': {tuple(expected['logits'].shape)}}}")


if __name__ == "__main__":
    main()
