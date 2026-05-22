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

"""Generate DeepSEA fixtures from the original Torch7 checkpoint."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve()
MODEL_DIR = HERE.parent
REPO_ROOT = MODEL_DIR.parent.parent

sys.path.insert(0, str(REPO_ROOT))
from _corpus.load import crop_record, sequence_sha256  # noqa: E402
from _shared.download import fetch_http_file, upstream_cache_root  # noqa: E402
from _shared.fixture import (  # noqa: E402
    fixture_out_dir,
    inputs_source_from_anchor_crop,
    sha256_of_file,
    write_fixture_artifacts,
)

MODEL = "deepsea"
CASE = "deepsea"
PREDICT_LUA = MODEL_DIR / "predict.lua"
UPSTREAM_REPO_URL = "https://deepsea.princeton.edu/"
UPSTREAM_COMMIT = "deepsea_train.v0.9"
PACKAGE_SOURCE = "https://deepsea.princeton.edu/media/code/deepsea.v0.94b.tar.gz"
PACKAGE_FILENAME = "deepsea.v0.94b.tar.gz"
PACKAGE_SHA256 = "449ef2b69b82ccc140c91fe7ae555513b48ef6f2d55ac03d4f56a9290732a627"
CHECKPOINT_SOURCE = f"{PACKAGE_SOURCE}::DeepSEA-v0.94b/deepsea.cpu"
CHECKPOINT_FILENAME = "deepsea.cpu"
CHECKPOINT_SHA256 = "2af6abe4f422839c56ddbb9f9b14771a62ec49378903035c4212acdd1a370129"
CHECKPOINT_ENV_VAR = "MULTIMOLECULE_UPSTREAM_DEEPSEA_CHECKPOINT"
PACKAGE_ENV_VAR = "MULTIMOLECULE_UPSTREAM_DEEPSEA_PACKAGE"
CORPUS_RECORD_ID = "dna/grch38_chr21"
CROP_NAME = "regulatory_1000bp"
CROP_CENTER = "center"
CROP_LENGTH = 1000
ATOL = 1e-4
RTOL = 1e-4
DNA = {"A": 0, "C": 1, "G": 2, "T": 3}
ORIGINAL_CHECKPOINT_SUFFIXES = {".cpu", ".net", ".t7", ".th"}


def encode(sequence: str) -> np.ndarray:
    return np.asarray([[DNA[base] for base in sequence.upper()]], dtype=np.int64)


def verify_checkpoint(path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix not in ORIGINAL_CHECKPOINT_SUFFIXES:
        raise RuntimeError(
            f"DeepSEA checkpoint must be an original Torch7 file with suffix "
            f"{sorted(ORIGINAL_CHECKPOINT_SUFFIXES)}; got {path}"
        )
    actual = sha256_of_file(path)
    if actual != CHECKPOINT_SHA256:
        raise AssertionError(f"{path}: sha256 {actual} != {CHECKPOINT_SHA256}")


def checkpoint_path() -> Path:
    override = os.environ.get(CHECKPOINT_ENV_VAR)
    if override:
        checkpoint = Path(override).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"DeepSEA checkpoint not found at ${CHECKPOINT_ENV_VAR}: {checkpoint}")
        verify_checkpoint(checkpoint)
        return checkpoint

    checkpoint = upstream_cache_root() / "deepsea" / CHECKPOINT_FILENAME
    if checkpoint.is_file():
        verify_checkpoint(checkpoint)
        return checkpoint

    archive = fetch_http_file(
        PACKAGE_SOURCE,
        PACKAGE_FILENAME,
        cache_prefix="deepsea",
        env_var=PACKAGE_ENV_VAR,
        sha256=PACKAGE_SHA256,
        description="DeepSEA v0.94b official Torch7 package",
    )
    with tarfile.open(archive) as tar:
        member = tar.getmember("DeepSEA-v0.94b/deepsea.cpu")
        extracted = tar.extractfile(member)
        if extracted is None:
            raise FileNotFoundError("DeepSEA-v0.94b/deepsea.cpu not found in official package")
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        with checkpoint.open("wb") as handle:
            handle.write(extracted.read())
    verify_checkpoint(checkpoint)
    return checkpoint


def torch7_command(checkpoint: Path, sequence: str, output_path: Path) -> list[str]:
    return [
        "luajit",
        str(PREDICT_LUA),
        str(checkpoint),
        sequence,
        str(output_path),
    ]


def run_torch7(checkpoint: Path, sequence: str) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix="deepsea-torch7-") as tmpdir:
        output_path = Path(tmpdir) / "logits.tsv"
        subprocess.run(torch7_command(checkpoint, sequence, output_path), cwd=REPO_ROOT, check=True)
        values = np.loadtxt(output_path, dtype=np.float32, ndmin=2)
    return np.ascontiguousarray(values)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", nargs="?", choices=[CASE], default=CASE)
    return parser.parse_args()


def main() -> None:
    parse_args()
    if not PREDICT_LUA.is_file():
        raise FileNotFoundError(f"DeepSEA Torch7 helper not found: {PREDICT_LUA}")
    checkpoint = checkpoint_path()

    crop = crop_record(CORPUS_RECORD_ID, CROP_LENGTH, center=CROP_CENTER)
    sequence = crop["sequence"].upper()
    if sequence_sha256(sequence) != crop["sha256"]:
        raise AssertionError("crop sha256 mismatch")

    input_ids = encode(sequence)
    attention_mask = np.ones_like(input_ids, dtype=np.int64)
    expected = {"logits": run_torch7(checkpoint, sequence)}

    out_dir = fixture_out_dir(REPO_ROOT, MODEL, CASE)
    meta = write_meta(out_dir, crop, checkpoint, sha256_of_file(checkpoint))
    write_fixture_artifacts(
        out_dir,
        inputs={"input_ids": input_ids, "attention_mask": attention_mask},
        expected=expected,
        meta=meta,
    )
    print(f"Wrote fixture to {out_dir}")
    print(f"  checkpoint: {checkpoint}")
    print(f"  input_ids: {tuple(input_ids.shape)}")
    print(f"  shapes: {{'logits': {tuple(expected['logits'].shape)}}}")


if __name__ == "__main__":
    main()
