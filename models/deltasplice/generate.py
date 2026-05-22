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

"""Generate DeltaSplice golden fixtures from the upstream PyTorch ensemble."""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
MODEL_DIR = HERE.parent
REPO_ROOT = MODEL_DIR.parent.parent

sys.path.insert(0, str(REPO_ROOT))
from _corpus.load import crop_record, sequence_sha256  # noqa: E402
from _shared.fixture import (  # noqa: E402
    fixture_out_dir,
    inputs_source_from_anchor_crop,
    sha256_of_file,
    write_fixture_artifacts,
)
from _shared.source_tree import ensure_source_tree  # noqa: E402
from _shared.variant import dna_one_hot_with_context, encode_dna_ids  # noqa: E402

MODEL = "deltasplice"
CORPUS_RECORD_ID = "dna/grch38_chr21"
CROP_NAME = "splice_400bp"
CROP_CENTER = "center"
CROP_LENGTH = 400
UPSTREAM_REPO_URL = "https://github.com/chaolinzhanglab/DeltaSplice"
UPSTREAM_COMMIT = "71c2fd1629894956eaca08a8ead25aeb870ed2e2"
SOURCE_ENV_VAR = "MULTIMOLECULE_UPSTREAM_DELTASPLICE_SOURCE"
# DeltaSplice fixture generation runs the official 30kb context and five-member ensemble.
ATOL = 1e-5
RTOL = 1e-5
HIDDEN_SIZE = 64
CONTEXT = 30000
ENSEMBLE_MEMBERS = [f"model.ckpt-{index}" for index in range(5)]


@dataclass(frozen=True)
class DeltaSpliceCase:
    name: str
    checkpoint_dir: str
    checkpoint_source: str
    checkpoint_sha256: str


CASES = {
    "deltasplice": DeltaSpliceCase(
        name="deltasplice",
        checkpoint_dir="DeltaSplice_models",
        checkpoint_source="upstream-file://deltasplice/pretrained_models/DeltaSplice_models/model.ckpt-{0,1,2,3,4}",
        checkpoint_sha256="cd08712d3b9729edd3876ca28f1f192b6f5024f63fb06dc30e30d3e92bfc9fbc",
    ),
    "deltasplice-human": DeltaSpliceCase(
        name="deltasplice-human",
        checkpoint_dir="DeltaSplice_human",
        checkpoint_source="upstream-file://deltasplice/pretrained_models/DeltaSplice_human/model.ckpt-{0,1,2,3,4}",
        checkpoint_sha256="c4d5545c9b800b26f9c41be34a0ad8439a4638b6cee8b53610998d4baf2890bd",
    ),
}
WINDOWS = [
    11,
    11,
    11,
    11,
    19,
    19,
    19,
    19,
    25,
    25,
    25,
    25,
    33,
    33,
    33,
    33,
    43,
    43,
    85,
    85,
    85,
    85,
    85,
    85,
]
DILATIONS = [
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    2,
    2,
    2,
    2,
    8,
    8,
    8,
    8,
    16,
    16,
    16,
    16,
    16,
    16,
    32,
    32,
]


def deltasplice_root() -> Path:
    return ensure_source_tree(
        UPSTREAM_REPO_URL,
        UPSTREAM_COMMIT,
        ("deltasplice",),
        env_var=SOURCE_ENV_VAR,
        cache_prefix="deltasplice",
    )


def checkpoint_manifest(model_root: Path) -> tuple[str, dict[str, str]]:
    manifest = {}
    lines = []
    for filename in ENSEMBLE_MEMBERS:
        digest = sha256_of_file(model_root / filename)
        manifest[filename] = digest
        lines.append(f"{digest}  {filename}")
    text = "\n".join(lines) + "\n"
    return hashlib.sha256(text.encode()).hexdigest(), manifest


def load_upstream_encode(source_root: Path):
    sys.path.insert(0, str(source_root))
    from deltasplice.models.delta_pretrain import Encode  # noqa: WPS433

    return Encode, CONTEXT, WINDOWS, DILATIONS


def one_hot_with_context(sequence: str) -> torch.Tensor:
    one_hot = dna_one_hot_with_context(
        sequence,
        left_context=CONTEXT // 2,
        right_context=CONTEXT // 2,
    )
    return one_hot.transpose(1, 2).contiguous()


def load_member(
    filename: str,
    model_root: Path,
    model_cls: type[torch.nn.Module],
    context: int,
    windows: list[int],
    dilations: list[int],
) -> torch.nn.Module:
    model = model_cls(context, [HIDDEN_SIZE] * len(windows), windows, dilations, dropout=0.3)
    state = torch.load(model_root / filename, map_location="cpu", weights_only=True)
    state = {name.removeprefix("encode.module."): value for name, value in state.items()}
    model.load_state_dict(state)
    return model.eval()


def upstream_forward(sequence: str, source_root: Path, model_root: Path) -> torch.Tensor:
    model_cls, context, windows, dilations = load_upstream_encode(source_root)
    inputs = one_hot_with_context(sequence)
    member_outputs = []
    with torch.no_grad():
        for filename in ENSEMBLE_MEMBERS:
            member = load_member(filename, model_root, model_cls, context, windows, dilations)
            member_outputs.append(member(inputs))
    return torch.stack(member_outputs).mean(dim=0).contiguous()


def main_for_case(case_name: str) -> None:
    try:
        case = CASES[case_name]
    except KeyError as error:
        raise ValueError(f"Unsupported DeltaSplice fixture case: {case_name}") from error
    source_root = deltasplice_root()
    model_root = source_root / "deltasplice" / "pretrained_models" / case.checkpoint_dir
    manifest_sha256, manifest = checkpoint_manifest(model_root)
    if manifest_sha256 != case.checkpoint_sha256:
        raise AssertionError(f"DeltaSplice checkpoint manifest sha256 {manifest_sha256} != {case.checkpoint_sha256}")

    crop = crop_record(
        CORPUS_RECORD_ID,
        CROP_LENGTH,
        center=CROP_CENTER,
    )
    sequence = crop["sequence"].upper()
    if sequence_sha256(sequence) != crop["sha256"]:
        raise AssertionError("crop sha256 mismatch")
    inputs = {"input_ids": encode_dna_ids(sequence)}
    expected = {"logits": upstream_forward(sequence, source_root, model_root)}

    out_dir = fixture_out_dir(REPO_ROOT, MODEL, case.name)
    meta = {
        "version": 1,
        "model": MODEL,
        "case": case.name,
        "auto_model": "AutoModel",
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
            "checkpoint_sha256": manifest_sha256,
            "checkpoint_manifest": manifest,
            "ensemble_members": ENSEMBLE_MEMBERS,
        },
    }
    write_fixture_artifacts(out_dir, inputs=inputs, expected=expected, meta=meta)
    print(f"Wrote fixture to {out_dir}")
    print(f"  input_ids: {tuple(inputs['input_ids'].shape)}")
    print(f"  logits: {tuple(expected['logits'].shape)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=sorted(CASES), help="DeltaSplice fixture case.")
    args = parser.parse_args()
    main_for_case(args.case)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
