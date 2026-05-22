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

"""Generate the scBasset golden fixture from the upstream Keras checkpoint."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.dont_write_bytecode = True

import h5py
import numpy as np
import tensorflow as tf

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

MODEL = "scbasset"
CASE = "scbasset"
UPSTREAM_REPO_URL = "https://github.com/calico/scBasset"
UPSTREAM_COMMIT = "aed3a6f713091fd988196b297e9c06e092ff1d22"
CHECKPOINT_SOURCE = "https://storage.googleapis.com/scbasset_tutorial_data/buen_model_sc.h5"
CHECKPOINT_FILENAME = "buen_model_sc.h5"
CHECKPOINT_SHA256 = "1481133d217cea8b764464817775045f79bbb9d980c671465d51fbd3007d675b"
CHECKPOINT_ENV_VAR = "MULTIMOLECULE_UPSTREAM_SCBASSET_CHECKPOINT"
CORPUS_RECORD_ID = "dna/grch38_chr21"
CROP_NAME = "regulatory_1344bp"
CROP_CENTER = "center"
CROP_LENGTH = 1344
ATOL = 1e-4
RTOL = 1e-4
DNA = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}


class GELU(tf.keras.layers.Layer):
    def call(self, x):
        return tf.keras.activations.sigmoid(tf.constant(1.702) * x) * x


def conv_block(inputs, filters=None, kernel_size=1, pool_size=1, batch_norm=True, bn_momentum=0.90):
    current = GELU()(inputs)
    current = tf.keras.layers.Conv1D(
        filters=filters if filters is not None else inputs.shape[-1],
        kernel_size=kernel_size,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
    )(current)
    if batch_norm:
        current = tf.keras.layers.BatchNormalization(momentum=bn_momentum, gamma_initializer="ones")(current)
    if pool_size > 1:
        current = tf.keras.layers.MaxPool1D(pool_size=pool_size, padding="same")(current)
    return current


def conv_tower(inputs, filters_init, filters_mult, repeat, **kwargs):
    current = inputs
    filters = filters_init
    for _ in range(repeat):
        current = conv_block(current, filters=int(np.round(filters)), **kwargs)
        filters *= filters_mult
    return current


def dense_block(inputs, units, flatten=False, dropout=0, batch_norm=True, bn_momentum=0.90):
    current = GELU()(inputs)
    if flatten:
        _, seq_len, seq_depth = current.shape
        current = tf.keras.layers.Reshape((1, seq_len * seq_depth))(current)
    current = tf.keras.layers.Dense(units=units, use_bias=not batch_norm, kernel_initializer="he_normal")(current)
    if batch_norm:
        current = tf.keras.layers.BatchNormalization(momentum=bn_momentum, gamma_initializer="ones")(current)
    if dropout > 0:
        current = tf.keras.layers.Dropout(rate=dropout)(current)
    return current


def final(inputs, units, activation="linear"):
    return tf.keras.layers.Dense(
        units=units,
        use_bias=True,
        activation=activation,
        kernel_initializer="he_normal",
    )(inputs)


def one_hot(sequence: str) -> np.ndarray:
    array = np.zeros((1, len(sequence), 4), dtype=np.float32)
    for index, base in enumerate(sequence.upper()):
        channel = DNA.get(base)
        if channel is not None and channel < 4:
            array[0, index, channel] = 1.0
    return array


def make_logits_model(num_cells: int) -> tf.keras.Model:
    sequence = tf.keras.Input(shape=(CROP_LENGTH, 4), name="sequence")
    current = sequence
    current = conv_block(current, filters=288, kernel_size=17, pool_size=3)
    current = conv_tower(
        current,
        filters_init=288,
        filters_mult=1.122,
        repeat=6,
        kernel_size=5,
        pool_size=2,
    )
    current = conv_block(current, filters=256, kernel_size=1)
    current = dense_block(current, flatten=True, units=32, dropout=0.2)
    current = GELU()(current)
    current = final(current, units=num_cells, activation="linear")
    current = tf.keras.layers.Flatten()(current)
    return tf.keras.Model(inputs=sequence, outputs=current)


def checkpoint_path() -> Path:
    return fetch_http_file(
        CHECKPOINT_SOURCE,
        CHECKPOINT_FILENAME,
        cache_prefix="scbasset",
        env_var=CHECKPOINT_ENV_VAR,
        sha256=CHECKPOINT_SHA256,
        description="scBasset Keras checkpoint",
    )


def write_meta(out_dir: Path, crop: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
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
            "checkpoint_sha256": sha256_of_file(checkpoint),
        },
    }
    return meta


def main() -> None:
    case = sys.argv[1] if len(sys.argv) > 1 else CASE
    if case != CASE:
        raise SystemExit(f"Unknown scBasset case {case!r}; expected {CASE!r}")
    checkpoint = checkpoint_path()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"scBasset checkpoint not found: {checkpoint}")

    crop = crop_record(CORPUS_RECORD_ID, CROP_LENGTH, center=CROP_CENTER)
    sequence = crop["sequence"].upper()
    if sequence_sha256(sequence) != crop["sha256"]:
        raise AssertionError("crop sha256 mismatch")

    with h5py.File(checkpoint, "r") as handle:
        num_cells = int(handle["dense_1"]["dense_1"]["bias:0"].shape[0])
    model = make_logits_model(num_cells)
    model.load_weights(str(checkpoint))
    logits = model(one_hot(sequence), training=False).numpy()

    input_ids = np.asarray([[DNA.get(base, 4) for base in sequence]], dtype=np.int64)
    attention_mask = np.ones_like(input_ids, dtype=np.int64)
    expected = {"logits": np.ascontiguousarray(logits, dtype=np.float32)}

    out_dir = fixture_out_dir(REPO_ROOT, MODEL, CASE)
    meta = write_meta(out_dir, crop, checkpoint)
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
