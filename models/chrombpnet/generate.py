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

"""Generate the ChromBPNet golden fixture from the official ENCODE Keras sub-models."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.dont_write_bytecode = True

import numpy as np
from tensorflow.keras.models import load_model

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

MODEL = "chrombpnet"
CASE = "chrombpnet"
UPSTREAM_REPO_URL = "https://github.com/kundajelab/chrombpnet"
UPSTREAM_COMMIT = "1ea660fef2644b55f418e1a1f05823a0836922e5"
CHECKPOINT_SOURCE = "https://www.encodeproject.org/files/ENCFF984RAF/@@download/ENCFF984RAF.tar.gz"
CHECKPOINT_ACCESSION = "ENCFF984RAF"
CHECKPOINT_FILENAME = "ENCFF984RAF.tar.gz"
CHECKPOINT_ARCHIVE_MD5 = "0da121e548f1fc977e6ea206ef5b7a52"
CHECKPOINT_DIR_ENV_VAR = "MULTIMOLECULE_UPSTREAM_CHROMBPNET_CHECKPOINT_DIR"
CHECKPOINT_ARCHIVE_ENV_VAR = "MULTIMOLECULE_UPSTREAM_CHROMBPNET_ARCHIVE"
CHECKPOINT_ROOT_NAME = "ENCFF984RAF.models.ENCSR868FGK"
FOLD = 0
NOBIAS_FILENAME = "model.chrombpnet_nobias.fold_0.ENCSR868FGK.h5"
BIAS_FILENAME = "model.bias_scaled.fold_0.ENCSR868FGK.h5"
CORPUS_RECORD_ID = "dna/grch38_chr21"
CROP_NAME = "regulatory_2114bp"
CROP_CENTER = "center"
CROP_LENGTH = 2114
ATOL = 1e-4
RTOL = 1e-4
DNA = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}


def required_checkpoint_dir(path: Path) -> Path:
    if (path / f"fold_{FOLD}").is_dir():
        path = path / f"fold_{FOLD}"
    missing = [name for name in (NOBIAS_FILENAME, BIAS_FILENAME) if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"ChromBPNet checkpoint directory {path} is missing: {', '.join(missing)}")
    return path


def checkpoint_dir() -> Path:
    override = os.environ.get(CHECKPOINT_DIR_ENV_VAR)
    if override:
        return required_checkpoint_dir(Path(override).expanduser().resolve())

    archive = fetch_http_file(
        CHECKPOINT_SOURCE,
        CHECKPOINT_FILENAME,
        cache_prefix=f"{MODEL}/models",
        env_var=CHECKPOINT_ARCHIVE_ENV_VAR,
        description="ChromBPNet ENCODE K562 model archive",
    )
    root = archive.parent / CHECKPOINT_ROOT_NAME
    if not (root / f"fold_{FOLD}" / NOBIAS_FILENAME).is_file():
        safe_extract_tar(archive, root)
    return required_checkpoint_dir(root)


def checkpoint_sha256(root: Path) -> str:
    nobias = root / NOBIAS_FILENAME
    bias = root / BIAS_FILENAME
    return f"{nobias.name}={sha256_of_file(nobias)};" f"{bias.name}={sha256_of_file(bias)}"


def one_hot(sequence: str) -> np.ndarray:
    array = np.zeros((1, len(sequence), 4), dtype=np.float32)
    for index, base in enumerate(sequence.upper()):
        channel = DNA.get(base)
        if channel is not None and channel < 4:
            array[0, index, channel] = 1.0
    return array


def write_meta(out_dir: Path, crop: dict[str, Any], checkpoint: Path, ckpt_sha256: str) -> dict[str, Any]:
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
            "checkpoint_accession": CHECKPOINT_ACCESSION,
            "checkpoint_archive_md5": CHECKPOINT_ARCHIVE_MD5,
            "checkpoint_archive_env_var": CHECKPOINT_ARCHIVE_ENV_VAR,
            "checkpoint_env_var": CHECKPOINT_DIR_ENV_VAR,
            "checkpoint_sha256": ckpt_sha256,
            "fold": FOLD,
        },
    }
    return meta


def main() -> None:
    case = sys.argv[1] if len(sys.argv) > 1 else CASE
    if case != CASE:
        raise SystemExit(f"Unknown ChromBPNet case {case!r}; expected {CASE!r}")
    checkpoint = checkpoint_dir()
    nobias = checkpoint / NOBIAS_FILENAME
    bias = checkpoint / BIAS_FILENAME

    crop = crop_record(CORPUS_RECORD_ID, CROP_LENGTH, center=CROP_CENTER)
    sequence = crop["sequence"].upper()
    if sequence_sha256(sequence) != crop["sha256"]:
        raise AssertionError("crop sha256 mismatch")

    seq = one_hot(sequence)
    nobias_model = load_model(str(nobias), compile=False, safe_mode=False)
    bias_model = load_model(str(bias), compile=False, safe_mode=False)
    nobias_profile, nobias_count = (np.asarray(value) for value in nobias_model(seq, training=False))
    bias_profile, bias_count = (np.asarray(value) for value in bias_model(seq, training=False))
    profile = (nobias_profile + bias_profile)[..., None]
    count = np.logaddexp(nobias_count, bias_count)

    input_ids = np.asarray([[DNA.get(base, 4) for base in sequence]], dtype=np.int64)
    attention_mask = np.ones_like(input_ids, dtype=np.int64)
    profile_tensor = np.ascontiguousarray(profile, dtype=np.float32)
    count_tensor = np.ascontiguousarray(count, dtype=np.float32)

    out_dir = fixture_out_dir(REPO_ROOT, MODEL, CASE)
    meta = write_meta(out_dir, crop, checkpoint, checkpoint_sha256(checkpoint))
    write_fixture_artifacts(
        out_dir,
        inputs={"input_ids": input_ids, "attention_mask": attention_mask},
        expected={"profile_logits": profile_tensor, "count_logits": count_tensor},
        meta=meta,
    )
    print(f"Wrote fixture to {out_dir}")
    print(f"  input_ids: {tuple(input_ids.shape)}")
    print(f"  shapes: {{'profile_logits': {tuple(profile_tensor.shape)}, 'count_logits': {tuple(count_tensor.shape)}}}")


if __name__ == "__main__":
    main()
