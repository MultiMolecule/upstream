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

"""Generate the Enformer golden fixture from the official DeepMind TFHub SavedModel."""

from __future__ import annotations

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
from _shared.archive import safe_extract_tar  # noqa: E402
from _shared.download import fetch_http_file  # noqa: E402
from _shared.fixture import (  # noqa: E402
    fixture_out_dir,
    inputs_source_from_anchor_crop,
    sha256_of_file,
    write_fixture_artifacts,
)
from _shared.source_tree import ensure_source_tree  # noqa: E402

MODEL = "enformer"
CASE = "enformer"
UPSTREAM_REPO_URL = "https://github.com/google-deepmind/deepmind-research"
UPSTREAM_COMMIT = "f5de0ede8430809180254ee957abf36ed62579ef"
CHECKPOINT_SOURCE = "https://tfhub.dev/deepmind/enformer/1"
CHECKPOINT_ARCHIVE_SOURCE = "https://tfhub.dev/deepmind/enformer/1?tf-hub-format=compressed"
CHECKPOINT_ARCHIVE_FILENAME = "enformer-tfhub-1.tar.gz"
CHECKPOINT_ARCHIVE_ENV_VAR = "MULTIMOLECULE_UPSTREAM_ENFORMER_TFHUB_ARCHIVE"
CHECKPOINT_CACHE_NAME = "enformer-tfhub-1"
SAVED_MODEL_ENV_VAR = "MULTIMOLECULE_UPSTREAM_ENFORMER_SAVED_MODEL"
SAVED_MODEL_SHA256 = {
    "saved_model.pb": (
        "saved_model.pb",
        "c8f0af88f3626e7823f2b8fb2b8b7b07d6f337ae0aadc5f70c5671d263c695d6",
    ),
    "variables.index": (
        "variables/variables.index",
        "7a30693fac902bd23140e962368a80577ed4a05e62c7602b25a820be32d53f91",
    ),
    "variables.data-00000-of-00001": (
        "variables/variables.data-00000-of-00001",
        "6a64f5fdb08b79efc5f18a1d18a804486390d5649046c8b63ac7bb6794cfc461",
    ),
}
BASENJI_REPO_URL = "https://github.com/calico/basenji"
BASENJI_COMMIT = "06ce5d387e20b47184d05433b3983163c5f923cd"
BASENJI_SOURCE_ENV_VAR = "MULTIMOLECULE_UPSTREAM_ENFORMER_BASENJI_SOURCE"
BASENJI_PARAMS_REL = "manuscripts/enformer/params_enformer.json"
BASENJI_PARAMS_SHA256 = "4b311d68dbe3b36850c59c2e3709a59185f69aa7120e097e99d83dabc3b4aa43"
CORPUS_RECORD_ID = "dna/grch38_chr21"
CROP_NAME = "coverage_196608bp"
CROP_CENTER = "center"
CROP_LENGTH = 196_608
TFHUB_INPUT_LENGTH = 393_216
TARGET_SLICE = [0]
ATOL = 1e-4
RTOL = 1e-4
DNA = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
TFHUB_ONE_HOT_CHANNELS = 4


def tokenize_dna(sequence: str) -> np.ndarray:
    return np.asarray([[DNA.get(base, DNA["N"]) for base in sequence.upper()]], dtype=np.int64)


def one_hot_for_tfhub(sequence: str) -> np.ndarray:
    array = np.zeros((1, TFHUB_INPUT_LENGTH, TFHUB_ONE_HOT_CHANNELS), dtype=np.float32)
    flank = (TFHUB_INPUT_LENGTH - len(sequence)) // 2
    if flank < 0 or flank * 2 + len(sequence) != TFHUB_INPUT_LENGTH:
        raise ValueError(f"sequence length {len(sequence)} cannot be centered in {TFHUB_INPUT_LENGTH}")
    for offset, base in enumerate(sequence.upper(), start=flank):
        channel = DNA.get(base)
        if channel is not None and channel < TFHUB_ONE_HOT_CHANNELS:
            array[0, offset, channel] = 1.0
    return array


def require_saved_model(path: Path, label: str) -> Path:
    if not (path / "saved_model.pb").is_file():
        raise FileNotFoundError(f"DeepMind Enformer SavedModel not found at {label}: {path}")
    return path


def verify_file_sha256(path: Path, expected: str, description: str) -> str:
    actual = sha256_of_file(path)
    if actual != expected:
        raise RuntimeError(f"{description} sha256 mismatch: expected {expected}, got {actual}")
    return actual


def verify_saved_model_sha256(saved_model: Path) -> dict[str, str]:
    return {
        name: verify_file_sha256(saved_model / rel_path, expected, f"Enformer SavedModel {rel_path}")
        for name, (rel_path, expected) in SAVED_MODEL_SHA256.items()
    }


def saved_model_path() -> Path:
    override = os.environ.get(SAVED_MODEL_ENV_VAR)
    if override:
        return require_saved_model(Path(override).expanduser().resolve(), f"${SAVED_MODEL_ENV_VAR}")

    archive = fetch_http_file(
        CHECKPOINT_ARCHIVE_SOURCE,
        CHECKPOINT_ARCHIVE_FILENAME,
        cache_prefix=f"{MODEL}/tfhub",
        env_var=CHECKPOINT_ARCHIVE_ENV_VAR,
        description="DeepMind Enformer TFHub SavedModel archive",
        timeout=None,
    )
    saved_model = archive.parent / CHECKPOINT_CACHE_NAME
    if not (saved_model / "saved_model.pb").is_file():
        safe_extract_tar(archive, saved_model)
    return require_saved_model(saved_model, CHECKPOINT_SOURCE)


def basenji_source_root() -> Path:
    return ensure_source_tree(
        BASENJI_REPO_URL,
        BASENJI_COMMIT,
        (BASENJI_PARAMS_REL,),
        env_var=BASENJI_SOURCE_ENV_VAR,
        cache_prefix="enformer-basenji",
    )


def load_official_model(saved_model: Path) -> Any:
    loaded = tf.saved_model.load(str(saved_model))
    if not hasattr(loaded, "model") or not hasattr(loaded.model, "predict_on_batch"):
        raise TypeError(f"{saved_model} does not expose the expected TFHub .model.predict_on_batch API")
    return loaded.model


def upstream_forward(sequence: str, saved_model: Path) -> np.ndarray:
    model = load_official_model(saved_model)
    outputs = model.predict_on_batch(one_hot_for_tfhub(sequence))
    if "human" not in outputs:
        raise KeyError("Official Enformer SavedModel did not return a 'human' output head")
    logits = tf.gather(outputs["human"], TARGET_SLICE, axis=-1).numpy()
    return np.ascontiguousarray(logits, dtype=np.float32)


def write_meta(
    crop: dict[str, Any],
    saved_model_sha256: str,
    variables_index_sha256: str,
    variables_data_sha256: str,
) -> dict[str, Any]:
    meta = {
        "version": 1,
        "model": MODEL,
        "case": CASE,
        "auto_model": "AutoModelForTokenPrediction",
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
            "checkpoint_sha256": {
                "saved_model.pb": saved_model_sha256,
                "variables.index": variables_index_sha256,
                "variables.data-00000-of-00001": variables_data_sha256,
            },
            "target_slice": TARGET_SLICE,
        },
    }
    return meta


def main() -> None:
    case = sys.argv[1] if len(sys.argv) > 1 else CASE
    if case != CASE:
        raise SystemExit(f"Unknown Enformer case {case!r}; expected {CASE!r}")

    crop = crop_record(CORPUS_RECORD_ID, CROP_LENGTH, center=CROP_CENTER)
    sequence = crop["sequence"].upper()
    if len(sequence) != CROP_LENGTH:
        raise AssertionError(f"crop length {len(sequence)} != {CROP_LENGTH}")
    if sequence_sha256(sequence) != crop["sha256"]:
        raise AssertionError("crop sha256 mismatch")

    saved_model = saved_model_path()
    saved_model_sha256 = verify_saved_model_sha256(saved_model)
    params = basenji_source_root() / BASENJI_PARAMS_REL
    verify_file_sha256(params, BASENJI_PARAMS_SHA256, "Enformer Basenji params")
    input_ids = tokenize_dna(sequence)
    expected = {"logits": upstream_forward(sequence, saved_model)}

    out_dir = fixture_out_dir(REPO_ROOT, MODEL, CASE)
    meta = write_meta(
        crop,
        saved_model_sha256["saved_model.pb"],
        saved_model_sha256["variables.index"],
        saved_model_sha256["variables.data-00000-of-00001"],
    )
    write_fixture_artifacts(
        out_dir,
        inputs={"input_ids": input_ids},
        expected=expected,
        meta=meta,
    )

    summary = {key: tuple(value.shape) for key, value in expected.items()}
    print(f"Wrote fixture to {out_dir}")
    print(f"  input_ids: {tuple(input_ids.shape)}")
    print(f"  shapes: {summary}")


if __name__ == "__main__":
    main()
