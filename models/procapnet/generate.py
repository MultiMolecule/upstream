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

"""Generate the ProCapNet golden fixture from the official ENCODE PyTorch checkpoint."""

from __future__ import annotations

import os
import sys
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
from _shared.archive import safe_extract_tar  # noqa: E402
from _shared.download import fetch_http_file  # noqa: E402
from _shared.fixture import (  # noqa: E402
    fixture_out_dir,
    inputs_source_from_anchor_crop,
    sha256_of_file,
    write_fixture_artifacts,
)

MODEL = "procapnet"
CASE = "procapnet"
UPSTREAM_REPO_URL = "https://github.com/kundajelab/ProCapNet"
UPSTREAM_COMMIT = "5664daa8c123d143275faae7c5385c99fa76bd5a"
CHECKPOINT_SOURCE = "https://www.encodeproject.org/files/ENCFF976FHE/@@download/ENCFF976FHE.tar.gz"
CHECKPOINT_ACCESSION = "ENCFF976FHE"
CHECKPOINT_FILENAME = "ENCFF976FHE.tar.gz"
CHECKPOINT_ARCHIVE_MD5 = "818c2ff02d75444e4e444fb7469e53b1"
CHECKPOINT_ENV_VAR = "MULTIMOLECULE_UPSTREAM_PROCAPNET_CHECKPOINT"
CHECKPOINT_DIR_ENV_VAR = "MULTIMOLECULE_UPSTREAM_PROCAPNET_CHECKPOINT_DIR"
CHECKPOINT_ARCHIVE_ENV_VAR = "MULTIMOLECULE_UPSTREAM_PROCAPNET_ARCHIVE"
CHECKPOINT_ROOT_NAME = "ENCFF976FHE.models.ENCSR261KBX"
FOLD = 0
CHECKPOINT_MEMBER = "ENCSR261KBX.procapnet_model.fold0.state_dict.torch"
CORPUS_RECORD_ID = "dna/grch38_chr21"
CROP_NAME = "regulatory_2114bp"
CROP_CENTER = "center"
CROP_LENGTH = 2114
ATOL = 1e-4
RTOL = 1e-4
DNA = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}


def required_checkpoint(path: Path) -> Path:
    if path.is_dir():
        if (path / f"fold_{FOLD}").is_dir():
            path = path / f"fold_{FOLD}"
        path = path / CHECKPOINT_MEMBER
    if not path.is_file():
        raise FileNotFoundError(f"ProCapNet checkpoint not found: {path}")
    return path


def checkpoint_path() -> Path:
    override = os.environ.get(CHECKPOINT_ENV_VAR)
    if override:
        return required_checkpoint(Path(override).expanduser().resolve())

    dir_override = os.environ.get(CHECKPOINT_DIR_ENV_VAR)
    if dir_override:
        return required_checkpoint(Path(dir_override).expanduser().resolve())

    archive = fetch_http_file(
        CHECKPOINT_SOURCE,
        CHECKPOINT_FILENAME,
        cache_prefix=f"{MODEL}/models",
        env_var=CHECKPOINT_ARCHIVE_ENV_VAR,
        description="ProCapNet ENCODE K562 model archive",
    )
    root = archive.parent / CHECKPOINT_ROOT_NAME
    if not (root / f"fold_{FOLD}" / CHECKPOINT_MEMBER).is_file():
        safe_extract_tar(archive, root)
    return required_checkpoint(root)


class ProCapNetUpstream(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_filters = 512
        self.n_layers = 8
        self.n_outputs = 2
        self.trimming = (2114 - 1000) // 2
        self.iconv = nn.Conv1d(4, self.n_filters, kernel_size=21, padding=10)
        self.rconvs = nn.ModuleList(
            [
                nn.Conv1d(
                    self.n_filters,
                    self.n_filters,
                    kernel_size=3,
                    padding=2**i,
                    dilation=2**i,
                )
                for i in range(1, self.n_layers + 1)
            ]
        )
        self.deconv_kernel_size = 75
        self.fconv = nn.Conv1d(self.n_filters, self.n_outputs, kernel_size=self.deconv_kernel_size)
        self.relus = nn.ModuleList([nn.ReLU() for _ in range(0, self.n_layers + 1)])
        self.linear = nn.Linear(self.n_filters, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        start, end = self.trimming, x.shape[2] - self.trimming
        x = self.relus[0](self.iconv(x))
        for i in range(self.n_layers):
            x_conv = self.relus[i + 1](self.rconvs[i](x))
            x = torch.add(x, x_conv)
        x = x[
            :,
            :,
            start - self.deconv_kernel_size // 2 : end + self.deconv_kernel_size // 2,
        ]
        profile = self.fconv(x)
        counts = self.linear(torch.mean(x, axis=2)).reshape(x.shape[0], 1)
        return profile, counts


def one_hot_tensor(sequence: str) -> torch.Tensor:
    tensor = torch.zeros(1, 4, len(sequence), dtype=torch.float32)
    for index, base in enumerate(sequence.upper()):
        channel = DNA.get(base)
        if channel is not None and channel < 4:
            tensor[0, channel, index] = 1.0
    return tensor


def write_meta(out_dir: Path, crop: dict[str, Any], checkpoint: Path, checkpoint_sha256: str) -> dict[str, Any]:
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
            "checkpoint_dir_env_var": CHECKPOINT_DIR_ENV_VAR,
            "checkpoint_env_var": CHECKPOINT_ENV_VAR,
            "checkpoint_sha256": checkpoint_sha256,
            "fold": FOLD,
        },
    }
    return meta


def main() -> None:
    case = sys.argv[1] if len(sys.argv) > 1 else CASE
    if case != CASE:
        raise SystemExit(f"Unknown ProCapNet case {case!r}; expected {CASE!r}")
    checkpoint = checkpoint_path()
    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    crop = crop_record(CORPUS_RECORD_ID, CROP_LENGTH, center=CROP_CENTER)
    sequence = crop["sequence"].upper()
    if sequence_sha256(sequence) != crop["sha256"]:
        raise AssertionError("crop sha256 mismatch")

    model = ProCapNetUpstream().eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    profile, count = model(one_hot_tensor(sequence))
    profile = profile.transpose(1, 2).detach().cpu().contiguous()
    count = count.detach().cpu().contiguous()

    input_ids = torch.tensor([[DNA.get(base, 4) for base in sequence]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    out_dir = fixture_out_dir(REPO_ROOT, MODEL, CASE)
    meta = write_meta(out_dir, crop, checkpoint, sha256_of_file(checkpoint))
    write_fixture_artifacts(
        out_dir,
        inputs={"input_ids": input_ids, "attention_mask": attention_mask},
        expected={"profile_logits": profile, "count_logits": count},
        meta=meta,
    )
    print(f"Wrote fixture to {out_dir}")
    print(f"  input_ids: {tuple(input_ids.shape)}")
    print(f"  shapes: {{'profile_logits': {tuple(profile.shape)}, 'count_logits': {tuple(count.shape)}}}")


if __name__ == "__main__":
    main()
