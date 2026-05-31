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

"""Generate the Xpresso golden fixture from the upstream Keras model."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.dont_write_bytecode = True

import numpy as np
import torch
from keras.models import load_model

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

MODEL = "xpresso"
CASE = "xpresso"
UPSTREAM_REPO_URL = "https://github.com/vagarwal87/Xpresso"
UPSTREAM_COMMIT = "b5d1da2b7f7e9376e8b6eca2b6cc73cd361734a3"
CHECKPOINT_SOURCE = (
    "https://raw.githubusercontent.com/vagarwal87/Xpresso/"
    "b5d1da2b7f7e9376e8b6eca2b6cc73cd361734a3/Fig5_S5/human_trainepoch.11-0.426.h5"
)
CHECKPOINT_FILENAME = "human_trainepoch.11-0.426.h5"
CHECKPOINT_SHA256 = "15a383648008df5f843ec8651e78be5faf82609c88fb068da890aa7a89387bb0"
CHECKPOINT_ENV_VAR = "MULTIMOLECULE_UPSTREAM_XPRESSO_CHECKPOINT"
CORPUS_RECORD_ID = "dna/grch38_chr21"
CROP_NAME = "promoter_10500bp"
CROP_CENTER = "center"
CROP_LENGTH = 10500
ATOL = 1e-4
RTOL = 1e-4
DNA = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}


def one_hot(sequence: str) -> np.ndarray:
    array = np.zeros((1, len(sequence), 4), dtype=np.float32)
    for index, base in enumerate(sequence.upper()):
        channel = DNA.get(base)
        if channel is not None and channel < 4:
            array[0, index, channel] = 1.0
    return array


def checkpoint_path() -> Path:
    return fetch_http_file(
        CHECKPOINT_SOURCE,
        CHECKPOINT_FILENAME,
        cache_prefix=MODEL,
        env_var=CHECKPOINT_ENV_VAR,
        sha256=CHECKPOINT_SHA256,
        description="Xpresso Keras checkpoint",
    )


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
        raise SystemExit(f"Unknown Xpresso case {case!r}; expected {CASE!r}")
    checkpoint = checkpoint_path()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Xpresso checkpoint not found: {checkpoint}")

    crop = crop_record(CORPUS_RECORD_ID, CROP_LENGTH, center=CROP_CENTER)
    sequence = crop["sequence"].upper()
    if sequence_sha256(sequence) != crop["sha256"]:
        raise AssertionError("crop sha256 mismatch")

    features = np.zeros((1, 6), dtype=np.float32)
    model = load_model(str(checkpoint), compile=False)
    logits = model.predict([one_hot(sequence), features], batch_size=1, verbose=0)

    input_ids = torch.tensor([[DNA.get(base, 4) for base in sequence]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    features_tensor = torch.from_numpy(features).contiguous()
    expected = {"logits": torch.from_numpy(np.asarray(logits)).to(torch.float32).contiguous()}

    out_dir = fixture_out_dir(REPO_ROOT, MODEL, CASE)
    meta = write_meta(out_dir, crop, checkpoint, sha256_of_file(checkpoint))
    write_fixture_artifacts(
        out_dir,
        inputs={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "features": features_tensor,
        },
        expected=expected,
        meta=meta,
    )
    print(f"Wrote fixture to {out_dir}")
    print(f"  input_ids: {tuple(input_ids.shape)}")
    print(f"  features: {tuple(features_tensor.shape)}")
    print(f"  shapes: {{'logits': {tuple(expected['logits'].shape)}}}")


if __name__ == "__main__":
    main()
