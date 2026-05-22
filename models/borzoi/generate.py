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

"""Generate the borzoi golden fixture from the official Borzoi model.

Run in an upstream TensorFlow environment compatible with Borzoi v1.0.0 and
Baskerville. The checked-in fixture targets TensorFlow 2.15 / Keras 2; Keras 3
changes the `Layer.add_weight` signature used by the upstream custom attention
layer.

The input is the canonical chr21-derived 524,288 bp DNA crop. To keep the
golden tractable, the official `SeqNN.build_slice` graph transform is applied
before prediction so the saved output is one evidence-preserving human track
instead of all 7,611 human tracks.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

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
from _shared.source_tree import ensure_source_tree  # noqa: E402

MODEL = "borzoi"
TARGET_SUM = False
# TensorFlow/Baskerville replay against converted PyTorch is stable at 1e-3.
ATOL = 1e-3
RTOL = 1e-4

UPSTREAM_REPO_URL = "https://github.com/calico/borzoi"
UPSTREAM_COMMIT = "9736924e6c861f0d71978cc3012ff8d76d6fa91f"
BASKERVILLE_REPO_URL = "https://github.com/calico/baskerville"
BASKERVILLE_COMMIT = "544073b87245d9f43ba63442c75bdbf32a9f8720"
CORPUS_RECORD_ID = "dna/grch38_chr21"
CROP_NAME = "coverage_524288bp"
CROP_CENTER = "center"
CROP_LENGTH = 524_288

MM_DNA_VOCAB = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
UPSTREAM_DNA_VOCAB = {"A": 0, "C": 1, "G": 2, "T": 3}


@dataclass
class BorzoiCase:
    case: str
    species: str
    head_index: int
    target_slice: tuple[int, ...]
    checkpoint_filename: str
    checkpoint_source: str
    checkpoint_sha256: str

    @property
    def checkpoint_env_var(self) -> str:
        case_key = self.case.upper().replace("-", "_")
        return f"MULTIMOLECULE_UPSTREAM_{case_key}_CHECKPOINT"

    @property
    def checkpoint_path(self) -> Path:
        return fetch_http_file(
            self.checkpoint_source,
            self.checkpoint_filename,
            cache_prefix=MODEL,
            env_var=self.checkpoint_env_var,
            sha256=self.checkpoint_sha256,
            description=f"Borzoi checkpoint {self.checkpoint_filename}",
        )


CASES = {
    "borzoi-human": BorzoiCase(
        case="borzoi-human",
        species="human",
        head_index=0,
        target_slice=(0,),
        checkpoint_filename="model0_best.h5",
        checkpoint_source="https://storage.googleapis.com/seqnn-share/borzoi/f0/model0_best.h5",
        checkpoint_sha256="7661c8d8541ae39293b1bf25392a4ec7b97b76114be5ddf82764644c3094eb9f",
    ),
    "borzoi-mouse": BorzoiCase(
        case="borzoi-mouse",
        species="mouse",
        head_index=1,
        target_slice=(0,),
        checkpoint_filename="model1_best.h5",
        checkpoint_source="https://storage.googleapis.com/seqnn-share/borzoi/f0/model1_best.h5",
        checkpoint_sha256="14b1d2e2f94898dd4d83b7e228f302707d524e3a1feb2b986341b196bd0fd127",
    ),
}


def tokenize_dna(sequence: str) -> np.ndarray:
    ids = [MM_DNA_VOCAB.get(base, MM_DNA_VOCAB["N"]) for base in sequence.upper()]
    return np.asarray([ids], dtype=np.int64)


def dna_1hot(sequence: str) -> np.ndarray:
    one_hot = np.zeros((len(sequence), 4), dtype=np.float32)
    for index, base in enumerate(sequence.upper()):
        base_index = UPSTREAM_DNA_VOCAB.get(base)
        if base_index is not None:
            one_hot[index, base_index] = 1.0
    return one_hot


def install_optional_gcs_stubs() -> None:
    """Let inference import Baskerville when optional GCS training deps are absent."""
    try:
        import google  # noqa: F401
        import google.auth.exceptions  # noqa: F401
        import google.cloud.storage  # noqa: F401

        return
    except ModuleNotFoundError:
        pass

    import types

    try:
        import google  # noqa: F401
    except ModuleNotFoundError:
        google = types.ModuleType("google")
        google.__path__ = []  # type: ignore[attr-defined]
        sys.modules.setdefault("google", google)

    auth = types.ModuleType("google.auth")
    auth_exceptions = types.ModuleType("google.auth.exceptions")

    class DefaultCredentialsError(Exception):
        pass

    auth_exceptions.DefaultCredentialsError = DefaultCredentialsError
    cloud = types.ModuleType("google.cloud")
    storage = types.ModuleType("google.cloud.storage")
    storage.Client = object
    sys.modules.setdefault("google.auth", auth)
    sys.modules.setdefault("google.auth.exceptions", auth_exceptions)
    sys.modules.setdefault("google.cloud", cloud)
    sys.modules.setdefault("google.cloud.storage", storage)


def configure_imports() -> Path:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    borzoi_source = ensure_source_tree(
        UPSTREAM_REPO_URL,
        UPSTREAM_COMMIT,
        ("examples/params.json", "src"),
        env_var="MULTIMOLECULE_UPSTREAM_BORZOI_SOURCE",
        cache_prefix=MODEL,
    )
    baskerville_source = ensure_source_tree(
        BASKERVILLE_REPO_URL,
        BASKERVILLE_COMMIT,
        ("src",),
        env_var="MULTIMOLECULE_UPSTREAM_BASKERVILLE_SOURCE",
        cache_prefix="baskerville",
    )
    sys.path.insert(0, str(baskerville_source / "src"))
    sys.path.insert(0, str(borzoi_source / "src"))
    install_optional_gcs_stubs()
    return borzoi_source


def upstream_forward(sequence: str, case: BorzoiCase) -> np.ndarray:
    borzoi_source = configure_imports()

    from baskerville import seqnn

    with (borzoi_source / "examples" / "params.json").open() as handle:
        params = json.load(handle)
    params["model"]["verbose"] = False

    model = seqnn.SeqNN(params["model"])
    if int(params["model"]["seq_length"]) != len(sequence):
        raise AssertionError(f"sequence length {len(sequence)} != upstream seq_length {params['model']['seq_length']}")
    model.restore(str(case.checkpoint_path), head_i=case.head_index)
    model.build_slice(list(case.target_slice), target_sum=TARGET_SUM)

    return np.asarray(model.model(dna_1hot(sequence)[None, ...], training=False), dtype=np.float32)


def write_case(case_name: str) -> None:
    case = CASES[case_name]
    checkpoint_path = case.checkpoint_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing Borzoi checkpoint: {checkpoint_path}")
    observed_sha256 = sha256_of_file(checkpoint_path)
    if observed_sha256 != case.checkpoint_sha256:
        raise ValueError(f"{checkpoint_path}: sha256 {observed_sha256} != {case.checkpoint_sha256}")

    crop = crop_record(CORPUS_RECORD_ID, CROP_LENGTH, center=CROP_CENTER)
    sequence = crop["sequence"]
    if len(sequence) != CROP_LENGTH:
        raise AssertionError(f"crop length {len(sequence)} != {CROP_LENGTH}")
    if sequence_sha256(sequence) != crop["sha256"]:
        raise AssertionError("crop sha256 mismatch")

    inputs = {"input_ids": tokenize_dna(sequence)}
    expected = {"logits": upstream_forward(sequence, case)}

    out_dir = fixture_out_dir(REPO_ROOT, MODEL, case.case)

    meta = {
        "version": 1,
        "model": MODEL,
        "case": case.case,
        "auto_model": "AutoModelForTokenPrediction",
        "outputs": sorted(expected.keys()),
        "tolerance": {"atol": ATOL, "rtol": RTOL},
        "inputs_source": inputs_source_from_anchor_crop(
            crop,
            crop_name=CROP_NAME,
        ),
        "upstream": {
            "repository": UPSTREAM_REPO_URL,
            "commit": UPSTREAM_COMMIT,
            "checkpoint_source": case.checkpoint_source,
            "checkpoint_sha256": observed_sha256,
            "target_slice": list(case.target_slice),
        },
    }
    write_fixture_artifacts(
        out_dir,
        inputs=inputs,
        expected=expected,
        meta=meta,
    )

    summary = {key: tuple(value.shape) for key, value in expected.items()}
    print(f"Wrote fixture to {out_dir}")
    print(f"  shapes: {summary}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", nargs="?", default="borzoi-human", choices=sorted(CASES))
    args = parser.parse_args()
    write_case(args.case)


if __name__ == "__main__":
    main()
