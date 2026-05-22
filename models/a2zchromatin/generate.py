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

"""Generate a2z-chromatin golden fixtures from upstream Keras checkpoints."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402

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

MODEL = "a2zchromatin"
UPSTREAM_REPO_URL = "https://github.com/twrightsman/a2z-regulatory"
UPSTREAM_COMMIT = "4360ccbba700737dfe4123723d71fd21466a2c02"
CORPUS_RECORD_ID = "dna/grch38_chr21"
CROP_NAME = "regulatory_600bp"
CROP_CENTER = "center"
CROP_LENGTH = 600
ATOL = 1e-4
RTOL = 1e-4

VARIANTS = {
    "a2zchromatin-accessibility": {
        "source": "https://zenodo.org/records/5724562/files/model-accessibility-full.h5?download=1",
        "filename": "model-accessibility-full.h5",
        "sha256": "93f650a1d94fc1430d2133559fa11d0fd1cc5fdecbeecc99f8b19e2e4d577ba0",
        "env_var": "MULTIMOLECULE_UPSTREAM_A2ZCHROMATIN_ACCESSIBILITY_CHECKPOINT",
    },
    "a2zchromatin-methylation": {
        "source": "https://zenodo.org/records/5724562/files/model-methylation-full.h5?download=1",
        "filename": "model-methylation-full.h5",
        "sha256": "1f193c08a2ae7a67e026af4a81c388a6b8a7a1570cd4149be4282b3426c3a353",
        "env_var": "MULTIMOLECULE_UPSTREAM_A2ZCHROMATIN_METHYLATION_CHECKPOINT",
    },
}

MM_DNA = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3,
    "N": 4,
    "R": 5,
    "Y": 6,
    "S": 7,
    "W": 8,
    "K": 9,
    "M": 10,
    "B": 11,
    "D": 12,
    "H": 13,
    "V": 14,
    ".": 15,
}
UPSTREAM_ONE_HOT = {
    "A": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    "C": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
    "G": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
    "T": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    "W": np.array([0.5, 0.0, 0.0, 0.5], dtype=np.float32),
    "S": np.array([0.0, 0.5, 0.5, 0.0], dtype=np.float32),
    "M": np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32),
    "K": np.array([0.0, 0.0, 0.5, 0.5], dtype=np.float32),
    "R": np.array([0.5, 0.0, 0.5, 0.0], dtype=np.float32),
    "Y": np.array([0.0, 0.5, 0.0, 0.5], dtype=np.float32),
    "B": np.array([0.0, 1.0 / 3, 1.0 / 3, 1.0 / 3], dtype=np.float32),
    "D": np.array([1.0 / 3, 0.0, 1.0 / 3, 1.0 / 3], dtype=np.float32),
    "H": np.array([1.0 / 3, 1.0 / 3, 0.0, 1.0 / 3], dtype=np.float32),
    "V": np.array([1.0 / 3, 1.0 / 3, 1.0 / 3, 0.0], dtype=np.float32),
    "N": np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32),
}


@tf.keras.utils.register_keras_serializable(package="multimolecule_upstream", name="CompatLSTM")
class CompatLSTM(tf.keras.layers.LSTM):
    """Keras 3 compatibility shim for Keras 2.x LSTM configs carrying time_major."""

    def __init__(self, *args: Any, time_major: bool = False, **kwargs: Any):
        super().__init__(*args, **kwargs)


def encode_input_ids(sequence: str) -> torch.Tensor:
    return torch.tensor([[MM_DNA[base] for base in sequence.upper()]], dtype=torch.long)


def encode_upstream_one_hot(sequence: str) -> np.ndarray:
    encoded = np.zeros((len(sequence), 4), dtype=np.float32)
    for index, base in enumerate(sequence.upper()):
        try:
            encoded[index] = UPSTREAM_ONE_HOT[base]
        except KeyError as error:
            raise ValueError(f"Unsupported a2z-chromatin DNA base {base!r} at offset {index}") from error
    return encoded


def load_upstream_model(checkpoint: Path) -> tf.keras.Model:
    custom_objects = {"LSTM": CompatLSTM, "CompatLSTM": CompatLSTM}
    return tf.keras.models.load_model(checkpoint, compile=False, custom_objects=custom_objects)


def checkpoint_path(case: str) -> Path:
    variant = VARIANTS[case]
    return fetch_http_file(
        variant["source"],
        variant["filename"],
        cache_prefix=f"{MODEL}/{case}",
        env_var=variant["env_var"],
        sha256=variant["sha256"],
        description=f"a2z-chromatin {case} Keras checkpoint",
    )


def upstream_logits(model: tf.keras.Model, one_hot: np.ndarray) -> torch.Tensor:
    final_layer = model.layers[-1]
    hidden_model = tf.keras.Model(model.inputs, final_layer.input)
    hidden = hidden_model(tf.convert_to_tensor(one_hot[None, ...], dtype=tf.float32), training=False)
    kernel, bias = final_layer.get_weights()
    logits = tf.linalg.matmul(hidden, tf.convert_to_tensor(kernel)) + tf.convert_to_tensor(bias)
    return torch.from_numpy(logits.numpy()).to(torch.float32).contiguous()


def write_meta(
    out_dir: Path,
    case: str,
    crop: dict[str, Any],
    checkpoint: Path,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    variant = VARIANTS[case]
    meta = {
        "version": 1,
        "model": MODEL,
        "case": case,
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
            "checkpoint_source": variant["source"],
            "checkpoint_sha256": checkpoint_sha256,
        },
    }
    return meta


def main() -> None:
    case = sys.argv[1] if len(sys.argv) > 1 else "a2zchromatin-accessibility"
    if case not in VARIANTS:
        raise SystemExit(f"Unknown a2z-chromatin case {case!r}; expected one of {sorted(VARIANTS)}")
    checkpoint = checkpoint_path(case)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"a2z-chromatin checkpoint not found: {checkpoint}")

    torch.manual_seed(0)
    tf.random.set_seed(0)
    crop = crop_record(CORPUS_RECORD_ID, CROP_LENGTH, center=CROP_CENTER)
    sequence = crop["sequence"].upper()
    if sequence_sha256(sequence) != crop["sha256"]:
        raise AssertionError("crop sha256 mismatch")

    input_ids = encode_input_ids(sequence)
    attention_mask = torch.ones_like(input_ids)
    one_hot = encode_upstream_one_hot(sequence)
    model = load_upstream_model(checkpoint)
    expected = {"logits": upstream_logits(model, one_hot)}

    out_dir = fixture_out_dir(REPO_ROOT, MODEL, case)
    meta = write_meta(out_dir, case, crop, checkpoint, sha256_of_file(checkpoint))
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
