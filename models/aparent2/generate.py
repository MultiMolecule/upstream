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

"""Generate APARENT2 checkpoint-parity golden fixture by replaying the upstream HDF5 graph."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import h5py
import torch
import torch.nn.functional as F

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

MODEL = "aparent2"
CASE = "aparent2"
UPSTREAM_REPOSITORY = "https://github.com/johli/aparent-resnet"
UPSTREAM_COMMIT = "a745736a4d9fbe411d1869200f2596a02f875532"
CHECKPOINT_SOURCE = (
    "https://raw.githubusercontent.com/johli/aparent-resnet/"
    "a745736a4d9fbe411d1869200f2596a02f875532/saved_models/"
    "aparent_all_libs_resnet_no_clinvar_wt_ep_5_var_batch_size_inference_mode.h5"
)
CHECKPOINT_FILENAME = "aparent_all_libs_resnet_no_clinvar_wt_ep_5_var_batch_size_inference_mode.h5"
CHECKPOINT_SHA256 = "0435e4a3388941cf5e1e80ee99855f55d98ff0f8a7d071f289eb26750401ccf3"
CHECKPOINT_ENV_VAR = "MULTIMOLECULE_UPSTREAM_APARENT2_CHECKPOINT"
CORPUS_RECORD_ID = "dna/grch38_chr21"
CROP_NAME = "polyadenylation_205bp"
CROP_CENTER = "center"
CROP_LENGTH = 205
ATOL = 1e-4
RTOL = 1e-4
DNA_CHANNELS = ("A", "C", "G", "T")
MM_RNA_IDS = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3, "N": 4}
DILATIONS = (1, 2, 4, 8, 4, 2, 1)
NUM_BLOCKS = 4
BATCH_NORM_EPS = 1e-3
LIBRARY_INDEX = 11


def encode_input_ids(sequence: str) -> torch.Tensor:
    return torch.tensor(
        [[MM_RNA_IDS.get(base, MM_RNA_IDS["N"]) for base in sequence.upper()]],
        dtype=torch.long,
    )


def one_hot_dna(sequence: str) -> torch.Tensor:
    array = torch.zeros(1, len(DNA_CHANNELS), CROP_LENGTH, dtype=torch.float32)
    for index, base in enumerate(sequence.upper()):
        if base in DNA_CHANNELS:
            array[0, DNA_CHANNELS.index(base), index] = 1.0
        else:
            array[0, :, index] = 0.25
    return array


class Aparent2H5Graph:
    """Minimal upstream APARENT2 graph replay using raw Keras HDF5 tensors."""

    def __init__(self, path: Path):
        self.file = h5py.File(path, "r")
        self.weights = self.file["model_weights"]

    def close(self) -> None:
        self.file.close()

    def get(self, layer: str, name: str) -> torch.Tensor:
        return torch.from_numpy(self.weights[layer][f"{layer}/{name}:0"][()].copy()).float()

    def conv1d_weight(self, layer: str) -> torch.Tensor:
        kernel = self.get(layer, "kernel")
        kernel = kernel.squeeze(0)
        return kernel.permute(2, 1, 0).contiguous()

    def conv1d(self, hidden_state: torch.Tensor, layer: str, *, dilation: int = 1) -> torch.Tensor:
        return F.conv1d(
            hidden_state,
            self.conv1d_weight(layer),
            self.get(layer, "bias"),
            padding="same",
            dilation=dilation,
        )

    def batch_norm(self, hidden_state: torch.Tensor, layer: str) -> torch.Tensor:
        gamma = self.get(layer, "gamma").view(1, -1, 1)
        beta = self.get(layer, "beta").view(1, -1, 1)
        mean = self.get(layer, "moving_mean").view(1, -1, 1)
        variance = self.get(layer, "moving_variance").view(1, -1, 1)
        return (hidden_state - mean) * torch.rsqrt(variance + BATCH_NORM_EPS) * gamma + beta

    def residual_block(
        self,
        hidden_state: torch.Tensor,
        group_index: int,
        block_index: int,
        dilation: int,
    ) -> torch.Tensor:
        prefix = f"aparent_resblock_{group_index}_{block_index}"
        residual = hidden_state
        hidden_state = self.batch_norm(hidden_state, f"{prefix}_batch_norm_0")
        hidden_state = F.relu(hidden_state)
        hidden_state = self.conv1d(hidden_state, f"{prefix}_conv_0", dilation=dilation)
        hidden_state = self.batch_norm(hidden_state, f"{prefix}_batch_norm_1")
        hidden_state = F.relu(hidden_state)
        hidden_state = self.conv1d(hidden_state, f"{prefix}_conv_1", dilation=dilation)
        return hidden_state + residual

    def __call__(self, sequence_one_hot: torch.Tensor) -> torch.Tensor:
        hidden_state = self.conv1d(sequence_one_hot, "aparent_conv_0")
        group_skips = []
        for group_index, dilation in enumerate(DILATIONS):
            group_skips.append(self.conv1d(hidden_state, f"aparent_skip_conv_{group_index}"))
            for block_index in range(NUM_BLOCKS):
                hidden_state = self.residual_block(hidden_state, group_index, block_index, dilation)

        logits = self.conv1d(hidden_state, "aparent_last_block_conv")
        for group_skip in group_skips:
            logits = logits + group_skip
        logits = self.conv1d(logits, "aparent_final_conv").squeeze(1)
        logits = torch.cat([logits, torch.zeros(logits.size(0), 1, dtype=logits.dtype)], dim=1)

        lib_kernel = self.get("aparent_lib_conv", "kernel")[:, :, 0]
        lib_bias = self.get("aparent_lib_conv", "bias").reshape(-1)
        return logits + lib_kernel[:, LIBRARY_INDEX].unsqueeze(0) + lib_bias.unsqueeze(0)


def checkpoint_path() -> Path:
    return fetch_http_file(
        CHECKPOINT_SOURCE,
        CHECKPOINT_FILENAME,
        cache_prefix=MODEL,
        env_var=CHECKPOINT_ENV_VAR,
        sha256=CHECKPOINT_SHA256,
        description="APARENT2 Keras checkpoint",
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
            "repository": UPSTREAM_REPOSITORY,
            "commit": UPSTREAM_COMMIT,
            "checkpoint_source": CHECKPOINT_SOURCE,
            "checkpoint_sha256": checkpoint_sha256,
        },
    }
    return meta


def main() -> None:
    case = sys.argv[1] if len(sys.argv) > 1 else CASE
    if case != CASE:
        raise SystemExit(f"Unknown APARENT2 case {case!r}; expected {CASE!r}")
    checkpoint = checkpoint_path()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"APARENT2 checkpoint not found: {checkpoint}")

    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    crop = crop_record(CORPUS_RECORD_ID, CROP_LENGTH, center=CROP_CENTER)
    sequence = crop["sequence"].upper()
    if sequence_sha256(sequence) != crop["sha256"]:
        raise AssertionError("crop sha256 mismatch")

    input_ids = encode_input_ids(sequence)
    graph = Aparent2H5Graph(checkpoint)
    try:
        logits = graph(one_hot_dna(sequence)).detach().cpu().contiguous()
    finally:
        graph.close()

    out_dir = fixture_out_dir(REPO_ROOT, MODEL, CASE)
    meta = write_meta(out_dir, crop, checkpoint, sha256_of_file(checkpoint))
    write_fixture_artifacts(
        out_dir,
        inputs={"input_ids": input_ids},
        expected={"logits": logits},
        meta=meta,
    )
    print(f"Wrote fixture to {out_dir}")
    print(f"  input_ids: {tuple(input_ids.shape)}")
    print(f"  logits: {tuple(logits.shape)}")


if __name__ == "__main__":
    main()
