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

"""Generate DeepSTARR golden fixtures from the upstream Keras HDF5 checkpoint."""

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

MODEL = "deepstarr"
CASE = "deepstarr"
CHECKPOINT_SOURCE = "https://zenodo.org/records/5502060/files/DeepSTARR.model.h5?download=1"
CHECKPOINT_FILENAME = "DeepSTARR.model.h5"
CHECKPOINT_SHA256 = "b1077afeb01570b4e5b79936d1f5ad6725a3499919957492d6ef0ecf8b188b46"
CHECKPOINT_ENV_VAR = "MULTIMOLECULE_UPSTREAM_DEEPSTARR_CHECKPOINT"
UPSTREAM_REPO_URL = "https://github.com/bernardo-de-almeida/DeepSTARR"
UPSTREAM_COMMIT = "b02e460c7581934bb6c8910e53be04da10688781"
CORPUS_RECORD_ID = "dna/grch38_chr21"
CROP_NAME = "regulatory_249bp"
CROP_CENTER = "center"
CROP_LENGTH = 249
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
        description="DeepSTARR Keras checkpoint",
    )


def build_upstream_model(checkpoint: Path) -> tf.keras.Model:
    sequence = tf.keras.Input(shape=(CROP_LENGTH, len(DNA_CHANNELS)), name="input_12")
    current = sequence
    for conv_name, filters, kernel_size, bn_name, act_name, pool_name in (
        (
            "Conv1D_1st",
            256,
            7,
            "batch_normalization_60",
            "activation_60",
            "max_pooling1d_44",
        ),
        (
            "Conv1D_2",
            60,
            3,
            "batch_normalization_61",
            "activation_61",
            "max_pooling1d_45",
        ),
        (
            "Conv1D_3",
            60,
            5,
            "batch_normalization_62",
            "activation_62",
            "max_pooling1d_46",
        ),
        (
            "Conv1D_4",
            120,
            3,
            "batch_normalization_63",
            "activation_63",
            "max_pooling1d_47",
        ),
    ):
        current = tf.keras.layers.Conv1D(filters, kernel_size, padding="same", activation="linear", name=conv_name)(
            current
        )
        current = tf.keras.layers.BatchNormalization(
            axis=-1,
            momentum=0.99,
            epsilon=0.001,
            name=bn_name,
        )(current)
        current = tf.keras.layers.Activation("relu", name=act_name)(current)
        current = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding="valid", name=pool_name)(current)

    current = tf.keras.layers.Flatten(name="flatten_11")(current)
    current = tf.keras.layers.Dense(256, activation="linear", name="Dense_1")(current)
    current = tf.keras.layers.BatchNormalization(
        axis=-1,
        momentum=0.99,
        epsilon=0.001,
        name="batch_normalization_64",
    )(current)
    current = tf.keras.layers.Activation("relu", name="activation_64")(current)
    current = tf.keras.layers.Dropout(0.4, name="dropout_17")(current)
    current = tf.keras.layers.Dense(256, activation="linear", name="Dense_2")(current)
    current = tf.keras.layers.BatchNormalization(
        axis=-1,
        momentum=0.99,
        epsilon=0.001,
        name="batch_normalization_65",
    )(current)
    current = tf.keras.layers.Activation("relu", name="activation_65")(current)
    current = tf.keras.layers.Dropout(0.4, name="dropout_18")(current)
    dev = tf.keras.layers.Dense(1, activation="linear", name="Dense_Dev")(current)
    housekeeping = tf.keras.layers.Dense(1, activation="linear", name="Dense_Hk")(current)
    model = tf.keras.Model(sequence, [dev, housekeeping], name="model_9")
    model.load_weights(str(checkpoint))
    return model


def upstream_forward(sequence: str, checkpoint: Path) -> torch.Tensor:
    model = build_upstream_model(checkpoint)
    dev, housekeeping = model(one_hot(sequence), training=False)
    logits = np.concatenate([dev.numpy(), housekeeping.numpy()], axis=1)
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
        raise SystemExit(f"Unknown DeepSTARR case {case!r}; expected {CASE!r}")
    checkpoint = checkpoint_path()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"DeepSTARR checkpoint not found: {checkpoint}")
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
