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

"""Generate the Malinois golden fixture from the local upstream PyTorch checkpoint."""

from __future__ import annotations

import io
import math
import os
import sys
import tarfile
from collections import OrderedDict
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

import torch
from torch import nn

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

MODEL = "malinois"
CASE = "malinois"
UPSTREAM_REPO_URL = "https://github.com/sjgosai/boda2"
UPSTREAM_COMMIT = "d74deca7a754f98f5dfefc1fd3f4506fed764947"
CHECKPOINT_SOURCE = (
    "https://storage.googleapis.com/tewhey-public-data/CODA_resources/"
    "malinois_artifacts__20211113_021200__287348.tar.gz"
)
CHECKPOINT_FILENAME = "malinois_artifacts__20211113_021200__287348.tar.gz"
CHECKPOINT_SHA256 = "06e926e42304b8207138f1fb871ec19e0654dcdb6b26a62ed23fe1e9ac8cc592"
CHECKPOINT_ENV_VAR = "MULTIMOLECULE_UPSTREAM_MALINOIS_CHECKPOINT"
SOURCE_ENV_VAR = "MULTIMOLECULE_UPSTREAM_MALINOIS_SOURCE"
CORPUS_RECORD_ID = "dna/grch38_chr21"
CROP_NAME = "regulatory_600bp"
CROP_CENTER = "center"
CROP_LENGTH = 600
ATOL = 1e-4
RTOL = 1e-4
DNA = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}


def malinois_source_root() -> Path:
    return ensure_source_tree(
        UPSTREAM_REPO_URL,
        UPSTREAM_COMMIT,
        ("boda/model", "boda/common", "boda/__init__.py"),
        env_var=SOURCE_ENV_VAR,
        cache_prefix="malinois",
    )


def checkpoint_path() -> Path:
    return fetch_http_file(
        CHECKPOINT_SOURCE,
        CHECKPOINT_FILENAME,
        cache_prefix="malinois",
        env_var=CHECKPOINT_ENV_VAR,
        sha256=CHECKPOINT_SHA256,
        description="Malinois CODA model artifact",
    )


def get_padding(kernel_size: int) -> tuple[int, int]:
    left = (kernel_size - 1) // 2
    return left, kernel_size - 1 - left


class Conv1dNorm(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, bias=True)
        self.bn_layer = nn.BatchNorm1d(out_channels, eps=1e-5, momentum=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn_layer(self.conv(x))


class LinearNorm(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.bn_layer = nn.BatchNorm1d(out_features, eps=1e-5, momentum=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn_layer(self.linear(x))


class GroupedLinear(nn.Module):
    def __init__(self, in_group_size: int, out_group_size: int, groups: int):
        super().__init__()
        self.in_group_size = in_group_size
        self.out_group_size = out_group_size
        self.groups = groups
        self.weight = nn.Parameter(torch.empty(groups, in_group_size, out_group_size))
        self.bias = nn.Parameter(torch.empty(groups, 1, out_group_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(3))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reorg = x.permute(1, 0).reshape(self.groups, self.in_group_size, -1).permute(0, 2, 1)
        hook = torch.bmm(reorg, self.weight) + self.bias
        return hook.permute(0, 2, 1).reshape(self.out_group_size * self.groups, -1).permute(1, 0)


class RepeatLayer(nn.Module):
    def __init__(self, *args: int):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.repeat(*self.args)


class BranchedLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_group_size: int,
        out_group_size: int,
        groups: int,
        layers: int,
    ):
        super().__init__()
        self.layers = layers
        self.nonlin = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5757068086404574)
        self.intake = RepeatLayer(1, groups)
        cur_size = in_features
        for index in range(layers):
            width = out_group_size if index + 1 == layers else hidden_group_size
            setattr(
                self,
                f"branched_layer_{index + 1}",
                GroupedLinear(cur_size, width, groups),
            )
            cur_size = hidden_group_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hook = self.intake(x)
        for index in range(self.layers - 1):
            hook = getattr(self, f"branched_layer_{index + 1}")(hook)
            hook = self.dropout(self.nonlin(hook))
        return getattr(self, f"branched_layer_{self.layers}")(hook)


class MalinoisUpstream(nn.Module):
    def __init__(self):
        super().__init__()
        self.pad1 = nn.ConstantPad1d(get_padding(19), 0.0)
        self.conv1 = Conv1dNorm(4, 300, 19)
        self.pad2 = nn.ConstantPad1d(get_padding(11), 0.0)
        self.conv2 = Conv1dNorm(300, 200, 11)
        self.pad3 = nn.ConstantPad1d(get_padding(7), 0.0)
        self.conv3 = Conv1dNorm(200, 200, 7)
        self.pad4 = nn.ConstantPad1d((1, 1), 0.0)
        self.maxpool_3 = nn.MaxPool1d(3, padding=0)
        self.maxpool_4 = nn.MaxPool1d(4, padding=0)
        self.linear1 = LinearNorm(200 * 13, 1000)
        self.branched = BranchedLinear(1000, 140, 140, 3, 3)
        self.output = GroupedLinear(140, 1, 3)
        self.nonlin = nn.ReLU()
        self.dropout = nn.Dropout(p=0.11625456877954289)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hook = self.nonlin(self.conv1(self.pad1(x)))
        hook = self.maxpool_3(hook)
        hook = self.nonlin(self.conv2(self.pad2(hook)))
        hook = self.maxpool_4(hook)
        hook = self.nonlin(self.conv3(self.pad3(hook)))
        hook = self.maxpool_4(self.pad4(hook))
        hook = torch.flatten(hook, start_dim=1)
        hook = self.dropout(self.nonlin(self.linear1(hook)))
        hook = self.branched(hook)
        return self.output(hook)


def load_original_state_dict(checkpoint: Path) -> OrderedDict[str, torch.Tensor]:
    with tarfile.open(checkpoint) as tar:
        member = next(
            item for item in tar.getmembers() if item.isfile() and os.path.basename(item.name) == "torch_checkpoint.pt"
        )
        file = tar.extractfile(member)
        if file is None:
            raise FileNotFoundError("Could not read torch_checkpoint.pt from Malinois tarball")
        checkpoint = torch.load(io.BytesIO(file.read()), map_location="cpu", weights_only=False)
    return OrderedDict(checkpoint["model_state_dict"])


def one_hot_tensor(sequence: str) -> torch.Tensor:
    tensor = torch.zeros(1, 4, len(sequence), dtype=torch.float32)
    for index, base in enumerate(sequence.upper()):
        channel = DNA.get(base)
        if channel is not None and channel < 4:
            tensor[0, channel, index] = 1.0
    return tensor


def write_meta(
    out_dir: Path,
    crop: dict[str, Any],
    checkpoint: Path,
    source_root: Path,
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
        raise SystemExit(f"Unknown Malinois case {case!r}; expected {CASE!r}")
    checkpoint = checkpoint_path()
    source_root = malinois_source_root()
    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    crop = crop_record(CORPUS_RECORD_ID, CROP_LENGTH, center=CROP_CENTER)
    sequence = crop["sequence"].upper()
    if sequence_sha256(sequence) != crop["sha256"]:
        raise AssertionError("crop sha256 mismatch")

    model = MalinoisUpstream().eval()
    model.load_state_dict(load_original_state_dict(checkpoint))
    logits = model(one_hot_tensor(sequence)).detach().cpu().contiguous()

    input_ids = torch.tensor([[DNA.get(base, 4) for base in sequence]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    out_dir = fixture_out_dir(REPO_ROOT, MODEL, CASE)
    meta = write_meta(out_dir, crop, checkpoint, source_root, sha256_of_file(checkpoint))
    write_fixture_artifacts(
        out_dir,
        inputs={"input_ids": input_ids, "attention_mask": attention_mask},
        expected={"logits": logits},
        meta=meta,
    )
    print(f"Wrote fixture to {out_dir}")
    print(f"  input_ids: {tuple(input_ids.shape)}")
    print(f"  shapes: {{'logits': {tuple(logits.shape)}}}")


if __name__ == "__main__":
    main()
