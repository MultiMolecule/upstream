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

"""Generate the BPNet golden fixture from the upstream Keras checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

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

MODEL = "bpnet"
CASE = "bpnet"
UPSTREAM_REPO_URL = "https://github.com/kundajelab/bpnet"
UPSTREAM_COMMIT = "0cb7277b736260f8b4084c9b0c5bd62b9edb5266"
CHECKPOINT_SOURCE = "https://zenodo.org/records/4294904/files/bpnet.model.h5"
CHECKPOINT_FILENAME = "bpnet.model.h5"
CHECKPOINT_SHA256 = "e773929293fa90b6c54b5d738a3ff028382f045dd8e57c51762371f66b8b8b90"
CHECKPOINT_ENV_VAR = "MULTIMOLECULE_UPSTREAM_BPNET_CHECKPOINT"
CORPUS_RECORD_ID = "dna/grch38_chr21"
CROP_NAME = "regulatory_1000bp"
CROP_CENTER = "center"
CROP_LENGTH = 1000
ATOL = 1e-4
RTOL = 1e-4
DNA = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
UPSTREAM_CHANNELS = ("A", "C", "G", "T")
DILATIONS = (2, 4, 8, 16, 32, 64, 128, 256, 512)
TASKS = ("Oct4", "Sox2", "Nanog", "Klf4")


def encode(sequence: str) -> np.ndarray:
    return np.array([[DNA.get(base, DNA["N"]) for base in sequence.upper()]], dtype=np.int64)


def one_hot(sequence: str) -> np.ndarray:
    encoded = np.zeros((1, len(sequence), len(UPSTREAM_CHANNELS)), dtype=np.float32)
    for index, base in enumerate(sequence.upper()):
        if base in UPSTREAM_CHANNELS:
            encoded[0, index, UPSTREAM_CHANNELS.index(base)] = 1.0
    return encoded


def _layer_weights(h5file: h5py.File, layer_name: str, weight_name: str) -> np.ndarray:
    return h5file["model_weights"][layer_name][layer_name][f"{weight_name}:0"][()]


def checkpoint_path() -> Path:
    return fetch_http_file(
        CHECKPOINT_SOURCE,
        CHECKPOINT_FILENAME,
        cache_prefix="bpnet",
        env_var=CHECKPOINT_ENV_VAR,
        sha256=CHECKPOINT_SHA256,
        description="BPNet Keras checkpoint",
    )


def build_upstream_model() -> tuple[tf.keras.Model, dict[str, tf.keras.layers.Layer]]:
    sequence = tf.keras.Input(shape=(CROP_LENGTH, len(UPSTREAM_CHANNELS)), name="seq")
    current = sequence
    layers: dict[str, tf.keras.layers.Layer] = {}

    stem = tf.keras.layers.Conv1D(64, 25, padding="same", activation="relu", name="conv1d_1")
    current = stem(current)
    layers[stem.name] = stem
    for index, dilation in enumerate(DILATIONS, start=2):
        conv = tf.keras.layers.Conv1D(
            64,
            3,
            padding="same",
            dilation_rate=dilation,
            activation="relu",
            name=f"conv1d_{index}",
        )
        residual = conv(current)
        current = tf.keras.layers.Add(name=f"add_{index - 1}")([current, residual])
        layers[conv.name] = conv

    profiles = []
    counts = []
    for task_index, _task in enumerate(TASKS):
        profile = tf.keras.layers.Reshape((-1, 1, 64), name=f"reshape_{2 * task_index + 1}")(current)
        deconv = tf.keras.layers.Conv2DTranspose(
            2,
            (25, 1),
            padding="same",
            name=f"conv2d_transpose_{task_index + 1}",
        )
        profile = deconv(profile)
        profile = tf.keras.layers.Reshape((-1, 2), name=f"reshape_{2 * task_index + 2}")(profile)
        layers[deconv.name] = deconv
        profiles.append(profile)

        pooled = tf.keras.layers.GlobalAveragePooling1D(name=f"global_average_pooling1d_{task_index + 1}")(current)
        dense = tf.keras.layers.Dense(2, name=f"dense_{2 * task_index + 1}")
        counts.append(dense(pooled))
        layers[dense.name] = dense

    profile_logits = tf.keras.layers.Concatenate(axis=-1, name="profile_logits")(profiles)
    count_logits = tf.keras.layers.Concatenate(axis=-1, name="count_logits")(counts)
    return (
        tf.keras.Model(sequence, [profile_logits, count_logits], name="bpnet_oskn"),
        layers,
    )


def load_upstream_model(checkpoint: Path) -> tf.keras.Model:
    model, layers = build_upstream_model()
    with h5py.File(checkpoint, "r") as h5file:
        for index in range(1, len(DILATIONS) + 2):
            layer = layers[f"conv1d_{index}"]
            layer.set_weights(
                [
                    _layer_weights(h5file, layer.name, "kernel"),
                    _layer_weights(h5file, layer.name, "bias"),
                ]
            )
        for task_index in range(len(TASKS)):
            deconv = layers[f"conv2d_transpose_{task_index + 1}"]
            deconv.set_weights(
                [
                    _layer_weights(h5file, deconv.name, "kernel"),
                    _layer_weights(h5file, deconv.name, "bias"),
                ]
            )
            dense = layers[f"dense_{2 * task_index + 1}"]
            dense.set_weights(
                [
                    _layer_weights(h5file, dense.name, "kernel"),
                    _layer_weights(h5file, dense.name, "bias"),
                ]
            )
    return model


def upstream_forward(sequence: str, checkpoint: Path) -> dict[str, np.ndarray]:
    profile_logits, count_logits = load_upstream_model(checkpoint)(one_hot(sequence), training=False)
    return {
        "profile_logits": profile_logits.numpy().astype(np.float32),
        "count_logits": count_logits.numpy().astype(np.float32),
    }


def write_meta(out_dir: Path, crop: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
    meta = {
        "version": 1,
        "model": MODEL,
        "case": CASE,
        "auto_model": "AutoModelForProfilePrediction",
        "outputs": ["profile_logits", "count_logits"],
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
        raise SystemExit(f"Unknown BPNet case {case!r}; expected {CASE!r}")
    checkpoint = checkpoint_path()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"BPNet checkpoint not found: {checkpoint}")

    crop = crop_record(CORPUS_RECORD_ID, CROP_LENGTH, center=CROP_CENTER)
    sequence = crop["sequence"].upper()
    if sequence_sha256(sequence) != crop["sha256"]:
        raise AssertionError("crop sha256 mismatch")

    input_ids = encode(sequence)
    attention_mask = np.ones_like(input_ids, dtype=np.int64)
    expected = upstream_forward(sequence, checkpoint)

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
    print(
        "  shapes: "
        f"{{'profile_logits': {tuple(expected['profile_logits'].shape)}, "
        f"'count_logits': {tuple(expected['count_logits'].shape)}}}"
    )


if __name__ == "__main__":
    main()
