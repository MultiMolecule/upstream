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

"""Generate the Basenji2 golden fixture from the original upstream Keras checkpoint."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.dont_write_bytecode = True

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
from _shared.source_tree import ensure_source_tree  # noqa: E402

MODEL = "basenji"
CASE = "basenji"
UPSTREAM_REPO_URL = "https://github.com/calico/basenji"
UPSTREAM_COMMIT = "06ce5d387e20b47184d05433b3983163c5f923cd"
CHECKPOINT_SOURCE = "https://storage.googleapis.com/basenji_barnyard2/model_human.h5"
CHECKPOINT_SHA256 = "3f74da002918d3e695bbf9e6f8a685e860196679372ba0964bd1faa26a82eed1"
CORPUS_RECORD_ID = "dna/grch38_chr21"
CROP_NAME = "coverage_131072bp"
CROP_CENTER = "center"
CROP_LENGTH = 131072
TARGET_SLICE = [0]
# TF/Keras checkpoint replay differs slightly from the converted PyTorch head.
ATOL = 1e-3
RTOL = 1e-4
DNA = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}

BASENJI_SOURCE_ROOT = ensure_source_tree(
    UPSTREAM_REPO_URL,
    UPSTREAM_COMMIT,
    ("basenji", "manuscripts/cross2020/params_human.json"),
    env_var="MULTIMOLECULE_UPSTREAM_BASENJI_SOURCE",
    cache_prefix="basenji",
)
sys.path.insert(0, str(BASENJI_SOURCE_ROOT))
from basenji import seqnn  # noqa: E402

CHECKPOINT = fetch_http_file(
    CHECKPOINT_SOURCE,
    "model_human.h5",
    cache_prefix="basenji/barnyard2",
    env_var="MULTIMOLECULE_UPSTREAM_BASENJI_CHECKPOINT",
    sha256=CHECKPOINT_SHA256,
    description="Basenji Cross2020 human Keras checkpoint",
)
PARAMS = BASENJI_SOURCE_ROOT / "manuscripts/cross2020/params_human.json"


def one_hot(sequence: str) -> np.ndarray:
    array = np.zeros((1, len(sequence), 4), dtype=np.float32)
    for index, base in enumerate(sequence.upper()):
        channel = DNA.get(base)
        if channel is not None and channel < 4:
            array[0, index, channel] = 1.0
    return array


def enable_keras3_legacy_shape_properties() -> None:
    if not hasattr(tf.keras.layers.Layer, "input_shape"):
        tf.keras.layers.Layer.input_shape = property(lambda self: tuple(self.input.shape))  # type: ignore[attr-defined]
    if not hasattr(tf.keras.layers.Layer, "output_shape"):
        tf.keras.layers.Layer.output_shape = property(  # type: ignore[attr-defined]
            lambda self: tuple(self.output.shape)
        )


def write_meta(crop: dict[str, Any], checkpoint_sha256: str) -> dict[str, Any]:
    meta = {
        "version": 1,
        "model": MODEL,
        "case": CASE,
        "auto_model": "AutoModelForTokenPrediction",
        "outputs": ["coverage"],
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
            "target_slice": TARGET_SLICE,
        },
    }
    return meta


def main() -> None:
    case = sys.argv[1] if len(sys.argv) > 1 else CASE
    if case != CASE:
        raise SystemExit(f"Unknown Basenji case {case!r}; expected {CASE!r}")
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Basenji checkpoint not found: {CHECKPOINT}")

    crop = crop_record(CORPUS_RECORD_ID, CROP_LENGTH, center=CROP_CENTER)
    sequence = crop["sequence"].upper()
    if sequence_sha256(sequence) != crop["sha256"]:
        raise AssertionError("crop sha256 mismatch")

    enable_keras3_legacy_shape_properties()
    params = json.loads(PARAMS.read_text())
    params["model"]["verbose"] = False
    upstream_model = seqnn.SeqNN(params["model"])
    upstream_model.restore(str(CHECKPOINT), 0)
    # Basenji2's final track projection applies softplus, so this is post-activation coverage, not logits.
    coverage = upstream_model.model(one_hot(sequence), training=False).numpy()[..., TARGET_SLICE]

    input_ids = np.asarray([[DNA.get(base, 4) for base in sequence]], dtype=np.int64)
    attention_mask = np.ones_like(input_ids, dtype=np.int64)
    expected = {"coverage": np.ascontiguousarray(coverage, dtype=np.float32)}

    out_dir = fixture_out_dir(REPO_ROOT, MODEL, CASE)
    meta = write_meta(crop, sha256_of_file(CHECKPOINT))
    write_fixture_artifacts(
        out_dir,
        inputs={"input_ids": input_ids, "attention_mask": attention_mask},
        expected=expected,
        meta=meta,
    )
    print(f"Wrote fixture to {out_dir}")
    print(f"  input_ids: {tuple(input_ids.shape)}")
    print(f"  shapes: {{'coverage': {tuple(expected['coverage'].shape)}}}")


if __name__ == "__main__":
    main()
