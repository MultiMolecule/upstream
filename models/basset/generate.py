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

"""Generate Basset golden fixtures with the original Torch7 implementation."""

from __future__ import annotations

import argparse
import gzip
import os
import shutil
import subprocess
import sys
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
from _shared.source_tree import ensure_source_tree  # noqa: E402

MODEL = "basset"
CASE = "basset"
PREDICT_LUA = MODEL_DIR / "predict.lua"
UPSTREAM_REPO_URL = "https://github.com/davek44/Basset"
UPSTREAM_COMMIT = "71cd8016b28b33e40357cac59ba5fbade3692ac2"
CHECKPOINT_SOURCE = "https://www.dropbox.com/s/rguytuztemctkf8/pretrained_model.th.gz?dl=1"
CHECKPOINT_FILENAME = "pretrained_model.th"
CHECKPOINT_GZIP_FILENAME = "pretrained_model.th.gz"
CHECKPOINT_SHA256 = "d228f99c8ca286f7f7684534fc66f8ad8c8099afa7380d35c15e88da341f1b94"
CHECKPOINT_ENV_VAR = "MULTIMOLECULE_UPSTREAM_BASSET_CHECKPOINT"
SOURCE_ENV_VAR = "MULTIMOLECULE_UPSTREAM_BASSET_SOURCE"
CORPUS_RECORD_ID = "dna/grch38_chr21"
CROP_NAME = "regulatory_600bp"
CROP_CENTER = "center"
CROP_LENGTH = 600
ATOL = 1e-4
RTOL = 1e-4
DNA = {"A": 0, "C": 1, "G": 2, "T": 3}


def encode(sequence: str) -> np.ndarray:
    return np.asarray([[DNA[base] for base in sequence.upper()]], dtype=np.int64)


def basset_source_root() -> Path:
    return ensure_source_tree(
        UPSTREAM_REPO_URL,
        UPSTREAM_COMMIT,
        (
            "src",
            "data/models/pretrained_params.txt",
            "data/models/targets.txt",
            "install_data.py",
        ),
        env_var=SOURCE_ENV_VAR,
        cache_prefix="basset",
    )


def checkpoint_path() -> Path:
    override = os.environ.get(CHECKPOINT_ENV_VAR)
    if override:
        checkpoint = Path(override).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Basset checkpoint not found at ${CHECKPOINT_ENV_VAR}: {checkpoint}")
        actual = sha256_of_file(checkpoint)
        if actual != CHECKPOINT_SHA256:
            raise AssertionError(f"{checkpoint}: sha256 {actual} != {CHECKPOINT_SHA256}")
        return checkpoint

    checkpoint = upstream_cache_root() / "basset" / CHECKPOINT_FILENAME
    if checkpoint.is_file():
        actual = sha256_of_file(checkpoint)
        if actual != CHECKPOINT_SHA256:
            raise AssertionError(f"{checkpoint}: sha256 {actual} != {CHECKPOINT_SHA256}")
        return checkpoint

    compressed = fetch_http_file(
        CHECKPOINT_SOURCE,
        CHECKPOINT_GZIP_FILENAME,
        cache_prefix="basset",
        description="Basset pretrained Torch7 checkpoint archive",
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(compressed, "rb") as src, checkpoint.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    actual = sha256_of_file(checkpoint)
    if actual != CHECKPOINT_SHA256:
        raise AssertionError(f"{checkpoint}: sha256 {actual} != {CHECKPOINT_SHA256}")
    return checkpoint


def run_torch7(sequence: str, checkpoint: Path, basset_src: Path) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix="basset-torch7-") as tmpdir:
        output_path = Path(tmpdir) / "logits.tsv"
        subprocess.run(
            [
                "luajit",
                str(PREDICT_LUA),
                str(checkpoint),
                sequence,
                str(output_path),
                str(basset_src),
            ],
            cwd=REPO_ROOT,
            check=True,
        )
        values = np.loadtxt(output_path, dtype=np.float32, ndmin=2)
    return np.ascontiguousarray(values)


def write_meta(
    out_dir: Path,
    crop: dict[str, Any],
    checkpoint: Path,
    basset_src: Path,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", nargs="?", choices=[CASE], default=CASE)
    return parser.parse_args()


def main() -> None:
    parse_args()
    if not PREDICT_LUA.is_file():
        raise FileNotFoundError(f"Basset Torch7 helper not found: {PREDICT_LUA}")
    checkpoint = checkpoint_path()
    basset_src = basset_source_root() / "src"

    crop = crop_record(CORPUS_RECORD_ID, CROP_LENGTH, center=CROP_CENTER)
    sequence = crop["sequence"].upper()
    if sequence_sha256(sequence) != crop["sha256"]:
        raise AssertionError("crop sha256 mismatch")

    input_ids = encode(sequence)
    attention_mask = np.ones_like(input_ids, dtype=np.int64)
    expected = {"logits": run_torch7(sequence, checkpoint, basset_src)}

    out_dir = fixture_out_dir(REPO_ROOT, MODEL, CASE)
    meta = write_meta(out_dir, crop, checkpoint, basset_src, sha256_of_file(checkpoint))
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
