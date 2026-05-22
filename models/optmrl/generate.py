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

"""Generate the OptMRL checkpoint-parity golden fixture."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
import torch

HERE = Path(__file__).resolve()
MODEL_DIR = HERE.parent
REPO_ROOT = MODEL_DIR.parent.parent  # upstream/

sys.path.insert(0, str(REPO_ROOT))
from _corpus.load import crop_record, sequence_sha256  # noqa: E402
from _shared.download import fetch_http_file  # noqa: E402
from _shared.fixture import (  # noqa: E402
    fixture_out_dir,
    inputs_source_from_anchor_crop,
    sha256_of_file,
    write_fixture_artifacts,
)

MODEL = "optmrl"
CASE = "optmrl"
ATOL = 1e-4
RTOL = 1e-4
CHECKPOINT_SOURCE = "https://zenodo.org/records/11258762/files/OptMRL-weights.h5"
CHECKPOINT_FILENAME = "OptMRL-weights.h5"
CHECKPOINT_SHA256 = "228ded24d38a2ccd55fbfbebfa2ab1095277bc774228a4a871b1556ba0b931a0"
CHECKPOINT_ENV_VAR = "MULTIMOLECULE_UPSTREAM_OPTMRL_CHECKPOINT"
CORPUS_RECORD_ID = "rna/grch38_chr21_transcribed"
CROP_NAME = "rna_50nt"
CROP_CENTER = "center"
CROP_LENGTH = 50
RNA = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3, "N": 4}
UPSTREAM_CHANNELS = ("A", "C", "G", "U")


def encode(sequence: str) -> list[int]:
    return [RNA.get(base, RNA["N"]) for base in sequence.upper()]


def one_hot(sequence: str) -> np.ndarray:
    encoded = np.zeros((1, len(sequence), len(UPSTREAM_CHANNELS)), dtype=np.float32)
    for index, base in enumerate(sequence.upper().replace("T", "U")):
        if base in UPSTREAM_CHANNELS:
            encoded[0, index, UPSTREAM_CHANNELS.index(base)] = 1.0
    return encoded


def checkpoint_path() -> Path:
    return fetch_http_file(
        CHECKPOINT_SOURCE,
        CHECKPOINT_FILENAME,
        cache_prefix="optmrl",
        env_var=CHECKPOINT_ENV_VAR,
        sha256=CHECKPOINT_SHA256,
        description="OptMRL Keras checkpoint",
    )


def build_upstream_model(checkpoint: Path) -> tf.keras.Model:
    model = tf.keras.Sequential(name="sequential_1")
    model.add(tf.keras.Input(shape=(CROP_LENGTH, len(UPSTREAM_CHANNELS)), name="conv1d_1_input"))
    model.add(tf.keras.layers.Conv1D(120, 8, padding="same", activation="relu", name="conv1d_1"))
    model.add(tf.keras.layers.Conv1D(120, 8, padding="same", activation="relu", name="conv1d_2"))
    model.add(tf.keras.layers.Dropout(0.0, name="dropout_1"))
    model.add(tf.keras.layers.Conv1D(120, 8, padding="same", activation="relu", name="conv1d_3"))
    model.add(tf.keras.layers.Dropout(0.0, name="dropout_2"))
    model.add(tf.keras.layers.Flatten(name="flatten_1"))
    model.add(tf.keras.layers.Dense(40, activation="linear", name="dense_1"))
    model.add(tf.keras.layers.Activation("relu", name="activation_1"))
    model.add(tf.keras.layers.Dropout(0.2, name="dropout_3"))
    model.add(tf.keras.layers.Dense(1, activation="linear", name="dense_2"))
    model.add(tf.keras.layers.Activation("linear", name="activation_2"))
    model.load_weights(str(checkpoint))
    return model


def upstream_forward(sequence: str, checkpoint: Path) -> torch.Tensor:
    logits = build_upstream_model(checkpoint)(one_hot(sequence), training=False).numpy()
    return torch.from_numpy(logits.astype(np.float32)).contiguous()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=[CASE], help="OptMRL fixture case.")
    return parser.parse_args()


def main() -> None:
    parse_args()
    checkpoint = checkpoint_path()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"OptMRL checkpoint not found: {checkpoint}")

    crop = crop_record(CORPUS_RECORD_ID, CROP_LENGTH, center=CROP_CENTER)
    sequence = crop["sequence"]
    if sequence_sha256(sequence) != crop["sha256"]:
        raise AssertionError("crop sha256 mismatch")

    inputs = {"input_ids": torch.tensor([encode(sequence)], dtype=torch.long)}
    expected = {"logits": upstream_forward(sequence, checkpoint)}

    out_dir = fixture_out_dir(REPO_ROOT, MODEL, CASE)

    meta: dict[str, Any] = {
        "version": 1,
        "model": MODEL,
        "case": CASE,
        "auto_model": "AutoModelForSequencePrediction",
        "outputs": sorted(expected.keys()),
        "tolerance": {"atol": ATOL, "rtol": RTOL},
        "inputs_source": inputs_source_from_anchor_crop(
            crop,
            crop_name=CROP_NAME,
        ),
        "upstream": {
            "repository": "https://github.com/ohlerlab/mlcis",
            "commit": "7bbd3772cbe78b2bdca389d4eeee098b90bbd238",
            "checkpoint_source": CHECKPOINT_SOURCE,
            "checkpoint_sha256": sha256_of_file(checkpoint),
        },
    }
    write_fixture_artifacts(
        out_dir,
        inputs=inputs,
        expected=expected,
        meta=meta,
    )


if __name__ == "__main__":
    main()
