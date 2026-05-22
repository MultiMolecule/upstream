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

"""Generate DeepCpG-DNA golden fixtures from upstream Keras 1.x weights."""

from __future__ import annotations

import json
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

MODEL = "deepcpgdna"
UPSTREAM_REPO_URL = "https://github.com/cangermueller/deepcpg"
UPSTREAM_COMMIT = "7f58da5423121168edabb27202c234df0f0e460d"
ZENODO_RECORD = "https://zenodo.org/records/1466079/files"
CORPUS_RECORD_ID = "dna/grch38_chr21"
CROP_NAME = "deepcpgdna_1001bp"
CROP_CENTER = "center"
CROP_LENGTH = 1001
ATOL = 1e-4
RTOL = 1e-4

VARIANTS = {
    "deepcpgdna-smallwood2014-serum": {
        "stem": "Smallwood2014_serum_dna-model",
        "sha256": "bdaf0c259e4920ee62b54a384b5b60a6dae6cb50b3661039e2d3f68125db68c8",
        "model_json_sha256": "65b0c335578fbe1c9e75b2e5b142f408b7f7ac8a587700a5a9b4c923498f3b63",
    },
    "deepcpgdna-smallwood2014-2i": {
        "stem": "Smallwood2014_2i_dna-model",
        "sha256": "bf484b6bfe21aa6a975c559f2940561ffe4421adb6c87befb11e75452c7f1552",
        "model_json_sha256": "16a0a9705a63ec080b1d6b67ad35619d03c3d1160a13e5c90b40bb955079197f",
    },
    "deepcpgdna-hou2016-hcc": {
        "stem": "Hou2016_HCC_dna-model",
        "sha256": "f198292e17074cd057a674a4d77df024641ad917202cee0d82bda321d73e4ea7",
        "model_json_sha256": "07cea665e946a079a4834fbad0217a47458fe11d7cd4813938bd59c1f4407a74",
    },
    "deepcpgdna-hou2016-hepg2": {
        "stem": "Hou2016_HepG2_dna-model",
        "sha256": "10ccfef61d685ba8d4d540beb00c808c1bc4940855519ec71cd109d76b6071b2",
        "model_json_sha256": "4de115f2b5dbe551a85a9edb75a62c60318b314327c0dc88548455be50123890",
    },
    "deepcpgdna-hou2016-mesc": {
        "stem": "Hou2016_mESC_dna-model",
        "sha256": "6385fec3512dd200773a86d73138ef8366ed2a0a3c5444d2bca267ee93bb672d",
        "model_json_sha256": "d736f31c9075be8c228cf9f99bc58043c314c3d3bcdee20646df33d7c07a66ee",
    },
}

MM_DNA = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
UPSTREAM_DNA = {"A": 0, "T": 1, "G": 2, "C": 3, "N": 4}


def encode_input_ids(sequence: str) -> torch.Tensor:
    return torch.tensor([[MM_DNA[base] for base in sequence.upper()]], dtype=torch.long)


def encode_upstream_one_hot(sequence: str) -> torch.Tensor:
    ids = torch.tensor([[UPSTREAM_DNA[base] for base in sequence.upper()]], dtype=torch.long)
    one_hot = torch.zeros((ids.size(0), ids.size(1), 4), dtype=torch.float32)
    valid = ids < 4
    one_hot[valid] = F.one_hot(ids[valid], num_classes=4).to(torch.float32)
    return one_hot


def load_model_spec(model_json: Path) -> dict[str, Any]:
    if not model_json.is_file():
        raise FileNotFoundError(f"DeepCpG-DNA model JSON not found: {model_json}")
    return json.loads(model_json.read_text())


def layer_specs(model_spec: dict[str, Any], class_name: str) -> list[dict[str, Any]]:
    return [layer for layer in model_spec["config"]["layers"] if layer["class_name"] == class_name]


def cell_names(model_spec: dict[str, Any]) -> list[str]:
    names = []
    for name, _, _ in model_spec["config"]["output_layers"]:
        names.append(name.split("/", 1)[1] if "/" in name else name)
    return names


def torch_weight(dataset: h5py.Dataset) -> torch.Tensor:
    return torch.from_numpy(dataset[()]).to(torch.float32)


def upstream_logits(model_spec: dict[str, Any], weights_path: Path, one_hot: torch.Tensor) -> torch.Tensor:
    conv_specs = layer_specs(model_spec, "Convolution1D")
    pool_specs = layer_specs(model_spec, "MaxPooling1D")
    if len(conv_specs) != len(pool_specs):
        raise ValueError(f"Expected one MaxPooling1D per Convolution1D, got {len(conv_specs)} and {len(pool_specs)}")

    hidden = one_hot.transpose(1, 2).contiguous()
    with h5py.File(weights_path, "r") as handle:
        dna = handle["model_weights"]["dna"]
        for index, (_conv_spec, pool_spec) in enumerate(zip(conv_specs, pool_specs), start=1):
            conv_name = f"convolution1d_{index}"
            kernel = torch_weight(dna[conv_name][f"{conv_name}_W:0"]).squeeze(1).permute(2, 1, 0).contiguous()
            bias = torch_weight(dna[conv_name][f"{conv_name}_b:0"]).contiguous()
            hidden = F.conv1d(hidden, kernel, bias)
            hidden = F.relu(hidden)
            pool_size = int(pool_spec["config"]["pool_length"])
            hidden = F.max_pool1d(hidden, kernel_size=pool_size, stride=pool_size)

        dense_w = torch_weight(dna["dense_1"]["dense_1_W:0"])
        dense_b = torch_weight(dna["dense_1"]["dense_1_b:0"])
        hidden = hidden.transpose(1, 2).flatten(1)
        hidden = hidden @ dense_w + dense_b
        hidden = F.relu(hidden)

        rows = []
        cpg = handle["model_weights"]["cpg"]
        for name in cell_names(model_spec):
            group = cpg[name]["cpg"]
            weight = torch_weight(group[f"{name}_W:0"])
            bias = torch_weight(group[f"{name}_b:0"])
            rows.append(hidden @ weight + bias)
    return torch.cat(rows, dim=1).to(torch.float32).contiguous()


def checkpoint_source(stem: str) -> str:
    return f"{ZENODO_RECORD}/{stem}_weights.h5?download=1"


def model_json_source(stem: str) -> str:
    return f"{ZENODO_RECORD}/{stem}?download=1"


def env_suffix(case: str) -> str:
    return f"MULTIMOLECULE_UPSTREAM_{case.upper().replace('-', '_')}"


def checkpoint_path(case: str, stem: str) -> Path:
    variant = VARIANTS[case]
    return fetch_http_file(
        checkpoint_source(stem),
        f"{stem}_weights.h5",
        cache_prefix=f"{MODEL}/{case}",
        env_var=f"{env_suffix(case)}_CHECKPOINT",
        sha256=variant["sha256"],
        description=f"DeepCpG-DNA {case} weights",
    )


def model_json_path(case: str, stem: str) -> Path:
    variant = VARIANTS[case]
    return fetch_http_file(
        model_json_source(stem),
        stem,
        cache_prefix=f"{MODEL}/{case}",
        env_var=f"{env_suffix(case)}_MODEL_JSON",
        sha256=variant["model_json_sha256"],
        description=f"DeepCpG-DNA {case} model JSON",
    )


def write_meta(
    out_dir: Path,
    case: str,
    stem: str,
    crop: dict[str, Any],
    model_json: Path,
    weights_path: Path,
    model_json_sha256: str,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    meta = {
        "version": 1,
        "model": MODEL,
        "case": case,
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
            "checkpoint_source": checkpoint_source(stem),
            "checkpoint_sha256": checkpoint_sha256,
        },
    }
    return meta


def main() -> None:
    case = sys.argv[1] if len(sys.argv) > 1 else "deepcpgdna-smallwood2014-serum"
    if case not in VARIANTS:
        raise SystemExit(f"Unknown DeepCpG-DNA case {case!r}; expected one of {sorted(VARIANTS)}")
    stem = VARIANTS[case]["stem"]
    model_json = model_json_path(case, stem)
    weights_path = checkpoint_path(case, stem)
    if not weights_path.is_file():
        raise FileNotFoundError(f"DeepCpG-DNA checkpoint not found: {weights_path}")

    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    crop = crop_record(CORPUS_RECORD_ID, CROP_LENGTH, center=CROP_CENTER)
    sequence = crop["sequence"].upper()
    if sequence_sha256(sequence) != crop["sha256"]:
        raise AssertionError("crop sha256 mismatch")

    input_ids = encode_input_ids(sequence)
    attention_mask = torch.ones_like(input_ids)
    one_hot = encode_upstream_one_hot(sequence)
    model_spec = load_model_spec(model_json)
    expected = {"logits": upstream_logits(model_spec, weights_path, one_hot)}

    out_dir = fixture_out_dir(REPO_ROOT, MODEL, case)
    meta = write_meta(
        out_dir,
        case,
        stem,
        crop,
        model_json,
        weights_path,
        sha256_of_file(model_json),
        sha256_of_file(weights_path),
    )
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
