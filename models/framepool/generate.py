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

"""Generate the Framepool checkpoint-parity golden fixture."""

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

MODEL = "framepool"
CASE = "framepool"
ATOL = 1e-4
RTOL = 1e-4
UPSTREAM_REPOSITORY = "https://github.com/Karollus/5UTR"
UPSTREAM_COMMIT = "c575f9cdca0cac1ffa88eb18e4435fdfbc674b08"
CHECKPOINT_SOURCE = "https://zenodo.org/record/3584238/files/Framepool_combined_residual.h5"
CHECKPOINT_FILENAME = "Framepool_combined_residual.h5"
CHECKPOINT_SHA256 = "f009163ecb33e64f7a40db4f8da474317a1481003fdbacda95205736d956ad3b"
CHECKPOINT_ENV_VAR = "MULTIMOLECULE_UPSTREAM_FRAMEPOOL_CHECKPOINT"
CORPUS_RECORD_ID = "rna/grch38_chr21_transcribed"
CROP_NAME = "rna_50nt"
CROP_CENTER = "center"
CROP_LENGTH = 50
RNA = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3, "N": 4}
UPSTREAM_CHANNELS = ("A", "C", "G", "U")
LIBRARY_SIZE = 2
LIBRARY_INDEX = 1


def encode(sequence: str) -> list[int]:
    return [RNA.get(base, RNA["N"]) for base in sequence.upper()]


def one_hot(sequence: str) -> np.ndarray:
    encoded = np.zeros((1, len(sequence), len(UPSTREAM_CHANNELS)), dtype=np.float32)
    for index, base in enumerate(sequence.upper().replace("T", "U")):
        if base in UPSTREAM_CHANNELS:
            encoded[0, index, UPSTREAM_CHANNELS.index(base)] = 1.0
    return encoded


def library_indicator(batch_size: int) -> np.ndarray:
    indicator = np.zeros((batch_size, LIBRARY_SIZE), dtype=np.float32)
    indicator[:, LIBRARY_INDEX] = 1.0
    return indicator


def apply_pad_mask(tensors):
    tensor, mask = tensors
    return tensor * tf.expand_dims(mask, axis=2)


def global_avg_pool_masked(tensors):
    tensor, mask = tensors
    return tf.reduce_sum(tensor, axis=1) / tf.reduce_sum(mask, axis=1, keepdims=True)


def checkpoint_path() -> Path:
    return fetch_http_file(
        CHECKPOINT_SOURCE,
        CHECKPOINT_FILENAME,
        cache_prefix="framepool",
        env_var=CHECKPOINT_ENV_VAR,
        sha256=CHECKPOINT_SHA256,
        description="Framepool combined residual Keras checkpoint",
    )


def build_upstream_model(checkpoint: Path) -> tf.keras.Model:
    input_seq = tf.keras.Input(shape=(None, len(UPSTREAM_CHANNELS)), name="input_seq")
    input_experiment = tf.keras.Input(shape=(LIBRARY_SIZE,), name="input_experiment")

    pad_mask = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=2), name="compute_pad_mask")(input_seq)
    current = input_seq
    for index in range(3):
        shortcut = current
        current = tf.keras.layers.Conv1D(
            128,
            7,
            padding="same",
            activation="relu",
            name=f"convolution_{index}",
        )(current)
        current = tf.keras.layers.Lambda(apply_pad_mask, name=f"apply_pad_mask_{index}")([current, pad_mask])
        if index > 0:
            current = tf.keras.layers.Add(name=f"add_residual_{index}")([current, shortcut])

    reversed_features = tf.keras.layers.Lambda(lambda x: tf.reverse(x, axis=[1]), name="frame_masking")(current)
    reversed_mask = tf.keras.layers.Lambda(lambda x: tf.reverse(x, axis=[1]), name="frame_masking_padmask")(pad_mask)
    frame_features = [
        tf.keras.layers.Lambda(lambda x, offset=offset: x[:, offset::3, :], name=f"frame_features_{offset}")(
            reversed_features
        )
        for offset in range(3)
    ]
    frame_masks = [
        tf.keras.layers.Lambda(lambda x, offset=offset: x[:, offset::3], name=f"frame_masks_{offset}")(reversed_mask)
        for offset in range(3)
    ]
    pooled = [
        tf.keras.layers.GlobalMaxPooling1D(name=f"pool_max_frame_conv_{offset}")(features)
        for offset, features in enumerate(frame_features)
    ]
    pooled.extend(
        tf.keras.layers.Lambda(global_avg_pool_masked, name=f"pool_avg_frame_conv_{offset}")([features, mask])
        for offset, (features, mask) in enumerate(zip(frame_features, frame_masks))
    )
    current = tf.keras.layers.Concatenate(axis=-1, name="concatenate_pooled")(pooled)
    current = tf.keras.layers.Dense(64, activation="relu", name="fully_connected_0")(current)
    current = tf.keras.layers.Dropout(0.2, name="fc_dropout_0")(current)
    unscaled = tf.keras.layers.Dense(1, name="mrl_output_unscaled")(current)
    interaction = tf.keras.layers.Lambda(lambda tensors: tensors[0] * tensors[1], name="interaction_term")(
        [unscaled, input_experiment]
    )
    regression_features = tf.keras.layers.Concatenate(axis=1, name="prepare_regression")(
        [interaction, input_experiment]
    )
    output = tf.keras.layers.Dense(1, use_bias=False, name="scaling_regression")(regression_features)

    model = tf.keras.Model([input_seq, input_experiment], output, name="model_3")
    # Pooling and Lambda layer names differ from the legacy Keras graph, but all learned
    # layers keep their upstream names and are loaded by name.
    model.load_weights(str(checkpoint), by_name=True)
    return model


def upstream_forward(sequence: str, checkpoint: Path) -> torch.Tensor:
    encoded = one_hot(sequence)
    logits = build_upstream_model(checkpoint)([encoded, library_indicator(encoded.shape[0])], training=False).numpy()
    return torch.from_numpy(logits.astype(np.float32)).contiguous()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=[CASE], help="Framepool fixture case.")
    return parser.parse_args()


def main() -> None:
    parse_args()
    checkpoint = checkpoint_path()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Framepool checkpoint not found: {checkpoint}")

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
            "repository": UPSTREAM_REPOSITORY,
            "commit": UPSTREAM_COMMIT,
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
