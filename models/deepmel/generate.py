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

"""Generate DeepMEL golden fixtures from the upstream Keras checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

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

MODEL = "deepmel"
CASE = "deepmel"
CHECKPOINT_SOURCE = "https://zenodo.org/records/3592129/files/DeepMEL.hdf5?download=1"
CHECKPOINT_FILENAME = "DeepMEL.hdf5"
CHECKPOINT_SHA256 = "06eaf3474f382bfd5d269b70e223eae1ab41ab4ff2bb0d3fe69f00df81add1b7"
CHECKPOINT_ENV_VAR = "MULTIMOLECULE_UPSTREAM_DEEPMEL_CHECKPOINT"
MODEL_JSON_SOURCE = "https://zenodo.org/records/3592129/files/DeepMEL.json.txt?download=1"
MODEL_JSON_FILENAME = "DeepMEL.json.txt"
MODEL_JSON_ENV_VAR = "MULTIMOLECULE_UPSTREAM_DEEPMEL_MODEL_JSON"
UPSTREAM_REPO_URL = "https://github.com/aertslab/DeepMEL"
UPSTREAM_COMMIT = "f329c3c90ea66185b86d0b5a274c5cfb91a8e363"
CORPUS_RECORD_ID = "dna/grch38_chr21"
CROP_NAME = "regulatory_500bp"
CROP_CENTER = "center"
CROP_LENGTH = 500
ATOL = 1e-4
RTOL = 1e-4
DNA = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
UPSTREAM_CHANNELS = ("A", "C", "G", "T")


def encode(sequence: str) -> np.ndarray:
    return np.array([[DNA.get(base, DNA["N"]) for base in sequence.upper()]], dtype=np.int64)


def one_hot(sequence: str) -> np.ndarray:
    encoded = np.zeros((1, len(sequence), len(UPSTREAM_CHANNELS)), dtype=np.float32)
    for index, base in enumerate(sequence.upper()):
        if base in UPSTREAM_CHANNELS:
            encoded[0, index, UPSTREAM_CHANNELS.index(base)] = 1.0
    return encoded


def checkpoint_path() -> Path:
    return fetch_http_file(
        CHECKPOINT_SOURCE,
        CHECKPOINT_FILENAME,
        cache_prefix=MODEL,
        env_var=CHECKPOINT_ENV_VAR,
        sha256=CHECKPOINT_SHA256,
        description="DeepMEL Keras checkpoint",
    )


def model_json_path() -> Path:
    return fetch_http_file(
        MODEL_JSON_SOURCE,
        MODEL_JSON_FILENAME,
        cache_prefix=MODEL,
        env_var=MODEL_JSON_ENV_VAR,
        description="DeepMEL model JSON",
    )


def load_upstream_model(model_json: Path, checkpoint: Path) -> tf.keras.Model:
    try:
        model = tf.keras.models.model_from_json(model_json.read_text())
    except TypeError as error:
        raise RuntimeError(
            "DeepMEL uses a legacy Keras 2 JSON graph. Run this generator through the shared "
            "tensorflow2 Docker image declared in source.yaml."
        ) from error
    model.load_weights(str(checkpoint))
    return model


def upstream_forward(sequence: str, model_json: Path, checkpoint: Path) -> np.ndarray:
    forward = one_hot(sequence)
    reverse_complement = forward[:, ::-1, ::-1]
    probabilities = (
        load_upstream_model(model_json, checkpoint).predict([forward, reverse_complement], verbose=0).astype(np.float32)
    )
    probabilities = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    return np.log(probabilities / (1.0 - probabilities)).astype(np.float32)


def write_meta(
    out_dir: Path,
    crop: dict[str, Any],
    model_json: Path,
    checkpoint: Path,
    checkpoint_sha256: str,
) -> dict[str, Any]:
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
        raise SystemExit(f"Unknown DeepMEL case {case!r}; expected {CASE!r}")
    checkpoint = checkpoint_path()
    model_json = model_json_path()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"DeepMEL checkpoint not found: {checkpoint}")
    if not model_json.is_file():
        raise FileNotFoundError(f"DeepMEL model JSON not found: {model_json}")

    crop = crop_record(CORPUS_RECORD_ID, CROP_LENGTH, center=CROP_CENTER)
    sequence = crop["sequence"].upper()
    if sequence_sha256(sequence) != crop["sha256"]:
        raise AssertionError("crop sha256 mismatch")

    input_ids = encode(sequence)
    attention_mask = np.ones_like(input_ids, dtype=np.int64)
    expected = {"logits": upstream_forward(sequence, model_json, checkpoint)}

    out_dir = fixture_out_dir(REPO_ROOT, MODEL, CASE)
    meta = write_meta(out_dir, crop, model_json, checkpoint, sha256_of_file(checkpoint))
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
