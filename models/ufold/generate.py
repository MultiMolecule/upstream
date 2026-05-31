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

"""Generate UFold golden fixtures from the official upstream PyTorch model."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
MODEL_DIR = HERE.parent
REPO_ROOT = MODEL_DIR.parent.parent

sys.path.insert(0, str(REPO_ROOT))
from _corpus.load import crop_record, sequence_sha256  # noqa: E402
from _shared.download import fetch_google_drive_file  # noqa: E402
from _shared.fixture import (  # noqa: E402
    fixture_out_dir,
    inputs_source_from_anchor_crop,
    sha256_of_file,
    write_fixture_artifacts,
)
from _shared.source_tree import ensure_source_tree  # noqa: E402

MODEL = "ufold"
CASE = "ufold"
UPSTREAM_REPO_URL = "https://github.com/uci-cbcl/UFold"
UPSTREAM_COMMIT = "75bd9acc83826059682dfca9d3659df66b132cd1"
CHECKPOINT_SOURCE = "google_drive://1Sq7MVgFOshGPlumRE_hpNXadvhJKaryi/ufold_train.pt"
CHECKPOINT_SHA256 = "532adba4a3d7ae95cc7550e812af7c7e0113b688455530b1d212c59ca579625e"
CHECKPOINT_ENV_VAR = "MULTIMOLECULE_UPSTREAM_UFOLD_CHECKPOINT"
SOURCE_ENV_VAR = "MULTIMOLECULE_UPSTREAM_UFOLD_SOURCE"
CORPUS_RECORD_ID = "rna/grch38_chr21_transcribed"
CROP_NAME = "rna_50nt"
CROP_CENTER = "center"
CROP_LENGTH = 50
PADDED_LENGTH = 80
ATOL = 1e-4
RTOL = 1e-4

UPSTREAM_BASE_INDEX = {"A": 0, "U": 1, "C": 2, "G": 3}
MM_BASE_INDEX = {"A": 0, "C": 1, "G": 2, "U": 3, "N": 4}
PAIR_SCORES = {
    ("A", "U"): 2.0,
    ("U", "A"): 2.0,
    ("G", "C"): 3.0,
    ("C", "G"): 3.0,
    ("G", "U"): 0.8,
    ("U", "G"): 0.8,
}


def ufold_source_root() -> Path:
    return ensure_source_tree(
        UPSTREAM_REPO_URL,
        UPSTREAM_COMMIT,
        ("Network.py", "ufold", "ufold_predict.py", "ufold_test.py"),
        env_var=SOURCE_ENV_VAR,
        cache_prefix="ufold",
    )


def checkpoint_path() -> Path:
    return fetch_google_drive_file(
        CHECKPOINT_SOURCE,
        "ufold_train.pt",
        cache_prefix=MODEL,
        env_var=CHECKPOINT_ENV_VAR,
        sha256=CHECKPOINT_SHA256,
        description="UFold official checkpoint",
    )


def load_upstream_class(source_root: Path) -> type[torch.nn.Module]:
    spec = importlib.util.spec_from_file_location("ufold_official_network", source_root / "Network.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import UFold Network.py from {source_root}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.U_Net


def encode_input_ids(sequence: str) -> torch.Tensor:
    ids = [MM_BASE_INDEX.get(base.upper().replace("T", "U"), MM_BASE_INDEX["N"]) for base in sequence]
    return torch.tensor([ids], dtype=torch.long)


def one_hot_aucg(sequence: str, padded_length: int) -> np.ndarray:
    one_hot = np.zeros((padded_length, 4), dtype=np.float32)
    for index, base in enumerate(sequence.upper().replace("T", "U")):
        base_index = UPSTREAM_BASE_INDEX.get(base)
        if base_index is not None:
            one_hot[index, base_index] = 1.0
    return one_hot


def pairing_prior(sequence: str, padded_length: int) -> np.ndarray:
    sequence = sequence.upper().replace("T", "U")
    prior = np.zeros((padded_length, padded_length), dtype=np.float32)
    length = len(sequence)
    for i in range(length):
        for j in range(length):
            coefficient = 0.0
            for offset in range(30):
                left = i - offset
                right = j + offset
                if left < 0 or right >= length:
                    break
                score = PAIR_SCORES.get((sequence[left], sequence[right]), 0.0)
                if score == 0.0:
                    break
                coefficient += score * np.exp(-0.5 * offset * offset)
            if coefficient > 0.0:
                for offset in range(1, 30):
                    left = i + offset
                    right = j - offset
                    if left >= length or right < 0:
                        break
                    score = PAIR_SCORES.get((sequence[left], sequence[right]), 0.0)
                    if score == 0.0:
                        break
                    coefficient += score * np.exp(-0.5 * offset * offset)
            prior[i, j] = coefficient
    return prior


def upstream_features(sequence: str) -> torch.Tensor:
    one_hot = one_hot_aucg(sequence, PADDED_LENGTH)
    features = np.zeros((17, PADDED_LENGTH, PADDED_LENGTH), dtype=np.float32)
    channel = 0
    for left_index in range(4):
        for right_index in range(4):
            features[channel] = np.matmul(
                one_hot[:, left_index].reshape(-1, 1),
                one_hot[:, right_index].reshape(1, -1),
            )
            channel += 1
    features[16] = pairing_prior(sequence, PADDED_LENGTH)
    return torch.from_numpy(features).unsqueeze(0)


def load_upstream_model(source_root: Path, checkpoint: Path) -> torch.nn.Module:
    model_cls = load_upstream_class(source_root)
    model = model_cls(img_ch=17, output_ch=1)
    state_dict = torch.load(checkpoint, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict, strict=True)
    return model.eval()


def main_for_case(case: str) -> None:
    if case != CASE:
        raise SystemExit(f"Unsupported UFold case {case!r}; expected {CASE!r}")
    source_root = ufold_source_root()
    checkpoint = checkpoint_path()
    checkpoint_sha256 = sha256_of_file(checkpoint)
    if checkpoint_sha256 != CHECKPOINT_SHA256:
        raise AssertionError(f"{checkpoint}: sha256 {checkpoint_sha256} != {CHECKPOINT_SHA256}")

    crop = crop_record(CORPUS_RECORD_ID, CROP_LENGTH, center=CROP_CENTER)
    sequence = crop["sequence"]
    if len(sequence) != CROP_LENGTH:
        raise AssertionError(f"crop length {len(sequence)} != {CROP_LENGTH}")
    if sequence_sha256(sequence) != crop["sha256"]:
        raise AssertionError("crop sha256 mismatch")

    input_ids = encode_input_ids(sequence)
    model = load_upstream_model(source_root, checkpoint)
    with torch.no_grad():
        logits = model(upstream_features(sequence))[:, :CROP_LENGTH, :CROP_LENGTH].detach().cpu().contiguous()

    out_dir = fixture_out_dir(REPO_ROOT, MODEL, CASE)
    meta = {
        "version": 1,
        "model": MODEL,
        "case": CASE,
        "auto_model": "AutoModelForRnaSecondaryStructurePrediction",
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
            "network_py": str(source_root / "Network.py"),
            "padded_length": PADDED_LENGTH,
        },
    }
    write_fixture_artifacts(
        out_dir,
        inputs={"input_ids": input_ids},
        expected={"logits": logits},
        meta=meta,
    )

    print(f"Wrote fixture to {out_dir}")
    print(f"  input_ids: {tuple(input_ids.shape)}")
    print(f"  logits: {tuple(logits.shape)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", nargs="?", default=CASE, choices=[CASE])
    args = parser.parse_args()
    main_for_case(args.case)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
