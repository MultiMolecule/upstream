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

"""Generate CARP golden fixtures from the upstream Microsoft checkpoints."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

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

MODEL = "carp"
UPSTREAM_REPO_URL = "https://github.com/microsoft/protein-sequence-models"
UPSTREAM_COMMIT = "af695772c4a1c056d930c95ec7e6428aa042f5cd"
SOURCE_ENV_VAR = "MULTIMOLECULE_UPSTREAM_CARP_SOURCE"
CORPUS_RECORD_ID = "protein/grch38_chr21_orf"
CROP_NAME = "protein_128aa"
CROP_CENTER = "center"
CROP_LENGTH = 128
ATOL = 1e-4
RTOL = 1e-5

MM_VOCAB = ["<pad>", "<cls>", "<eos>", "<unk>", "<mask>", "<null>"] + list("ACDEFGHIKLMNPQRSTVWYXZBJUO|.*-?")
TARGET_TOKENS = list("ACDEFGHIKLMNPQRSTVWYBZXJOU") + ["*", "-", "<mask>", "<cls>"]
TARGET_SLICE = [MM_VOCAB.index(token) for token in TARGET_TOKENS]


@dataclass(frozen=True)
class CarpCase:
    case: str
    filename: str
    source: str
    sha256: str

    @property
    def env_var(self) -> str:
        return f"MULTIMOLECULE_UPSTREAM_{self.case.upper().replace('-', '_')}_CHECKPOINT"


CASES = {
    "carp-600k": CarpCase(
        case="carp-600k",
        filename="carp_600k.pt",
        source="https://zenodo.org/record/6564798/files/carp_600k.pt?download=1",
        sha256="787f618e90cfdd6cbe2d98ab944e89dea00e972d81019181aba14bbf31e96e1e",
    ),
    "carp-38m": CarpCase(
        case="carp-38m",
        filename="carp_38M.pt",
        source="https://zenodo.org/record/6564798/files/carp_38M.pt?download=1",
        sha256="b66b68bce0fd690c780032140297de8fea1f94bfa5ecd67fb408ab3a3794cc3b",
    ),
    "carp-76m": CarpCase(
        case="carp-76m",
        filename="carp_76M.pt",
        source="https://zenodo.org/record/6564798/files/carp_76M.pt?download=1",
        sha256="7d224abac5eb2d5d649353352acb58efcae94277d26e25f1ac2cf90598205264",
    ),
    "carp-640m": CarpCase(
        case="carp-640m",
        filename="carp_640M.pt",
        source="https://zenodo.org/record/6564798/files/carp_640M.pt?download=1",
        sha256="ae1d7e0cbe713c731cfde7ace213f2a73372c31d8df543089e6b6d371228c4f7",
    ),
}


def parse_case(case: str) -> CarpCase:
    try:
        return CASES[case]
    except KeyError as error:
        raise ValueError(f"Unsupported CARP case: {case}") from error


def checkpoint_path(case: CarpCase) -> Path:
    return fetch_http_file(
        case.source,
        case.filename,
        cache_prefix=MODEL,
        env_var=case.env_var,
        sha256=case.sha256,
        description=f"{case.case} PyTorch checkpoint",
    )


def source_path() -> Path:
    source = ensure_source_tree(
        UPSTREAM_REPO_URL,
        UPSTREAM_COMMIT,
        ("sequence_models",),
        env_var=SOURCE_ENV_VAR,
        cache_prefix=MODEL,
    )
    if not (source / "sequence_models").is_dir():
        subprocess.run(
            ["git", "-C", str(source), "sparse-checkout", "set", "sequence_models"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(source), "checkout", "--detach", UPSTREAM_COMMIT],
            check=True,
        )
    return source


def encode_mm_input_ids(sequence: str) -> np.ndarray:
    unk = MM_VOCAB.index("<unk>")
    ids = [MM_VOCAB.index(residue) if residue in MM_VOCAB else unk for residue in sequence.upper()]
    return np.asarray([ids], dtype=np.int64)


def upstream_outputs(sequence: str, checkpoint: Path, source: Path) -> dict[str, np.ndarray]:
    sys.path.insert(0, str(source))
    from sequence_models.constants import PROTEIN_ALPHABET  # noqa: PLC0415
    from sequence_models.pretrained import load_carp  # noqa: PLC0415
    from sequence_models.utils import Tokenizer  # noqa: PLC0415

    model_data = torch.load(checkpoint, map_location="cpu")
    model = load_carp(model_data).eval()
    tokenized = np.asarray(Tokenizer(PROTEIN_ALPHABET).tokenize(sequence), dtype=np.int64)
    input_ids = torch.from_numpy(tokenized).unsqueeze(0)
    repr_layers = list(range(1, int(model_data["n_layers"]) + 1))
    with torch.no_grad():
        outputs = model(input_ids, repr_layers=repr_layers, logits=True)
    hidden_states = torch.stack([outputs["representations"][layer] for layer in repr_layers], dim=0)
    return {
        "hidden_states": hidden_states.detach().cpu().numpy().astype(np.float32),
        "logits": outputs["logits"].detach().cpu().numpy().astype(np.float32),
    }


def write_meta(case: CarpCase, crop: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
    return {
        "version": 1,
        "model": MODEL,
        "case": case.case,
        "auto_model": "AutoModelForPreTraining",
        "outputs": ["hidden_states", "logits"],
        "tolerance": {"atol": ATOL, "rtol": RTOL},
        "inputs_source": inputs_source_from_anchor_crop(
            crop,
            crop_name=CROP_NAME,
        ),
        "upstream": {
            "repository": UPSTREAM_REPO_URL,
            "commit": UPSTREAM_COMMIT,
            "checkpoint_source": case.source,
            "checkpoint_sha256": sha256_of_file(checkpoint),
            "target_slice": TARGET_SLICE,
        },
    }


def main() -> None:
    case = parse_case(sys.argv[1] if len(sys.argv) > 1 else "carp-600k")
    checkpoint = checkpoint_path(case)
    source = source_path()
    crop = crop_record(CORPUS_RECORD_ID, CROP_LENGTH, center=CROP_CENTER)
    sequence = crop["sequence"].upper()
    if sequence_sha256(sequence) != crop["sha256"]:
        raise AssertionError("crop sha256 mismatch")

    input_ids = encode_mm_input_ids(sequence)
    attention_mask = np.ones_like(input_ids, dtype=np.int64)
    expected = upstream_outputs(sequence, checkpoint, source)

    out_dir = fixture_out_dir(REPO_ROOT, MODEL, case.case)
    meta = write_meta(case, crop, checkpoint)
    write_fixture_artifacts(
        out_dir,
        inputs={"input_ids": input_ids, "attention_mask": attention_mask},
        expected=expected,
        meta=meta,
    )
    print(f"Wrote fixture to {out_dir}")
    print(f"  input_ids: {tuple(input_ids.shape)}")
    shapes = {key: tuple(value.shape) for key, value in expected.items()}
    print(f"  shapes: {shapes}")


if __name__ == "__main__":
    main()
