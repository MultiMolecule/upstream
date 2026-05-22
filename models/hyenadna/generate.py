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

"""Generate HyenaDNA golden fixtures from upstream HF checkpoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

HERE = Path(__file__).resolve()
MODEL_DIR = HERE.parent
REPO_ROOT = MODEL_DIR.parent.parent  # upstream/

sys.path.insert(0, str(REPO_ROOT))
from _corpus.load import crop_record, sequence_sha256  # noqa: E402
from _shared.bert_probe import (  # noqa: E402
    build_vocab_remap,
    remap_logits_to_vocab_subset,
    stack_hidden_states,
)
from _shared.fixture import (  # noqa: E402
    fixture_out_dir,
    inputs_source_from_anchor_crop,
    sha256_of_file,
    write_fixture_artifacts,
)
from _shared.huggingface import hf_snapshot_dir  # noqa: E402

MODEL = "hyenadna"
ATOL = 1e-4
RTOL = 1e-4
UPSTREAM_REPO_URL = "https://github.com/HazyResearch/hyena-dna"
UPSTREAM_COMMIT = "d553021b483b82980aa4b868b37ec2d4332e198a"

CORPUS_RECORD_ID = "dna/grch38_chr21"
CROP_NAME = "genome_1024bp"
CROP_CENTER = "center"
CROP_LENGTH = 1024
HF_ALLOW_PATTERNS = ("*.json", "*.py", "model.safetensors")

CASES = {
    "hyenadna-tiny": {
        "repo_id": "LongSafari/hyenadna-tiny-16k-seqlen-d128-hf",
        "huggingface_revision": "d79fa37e2cd62dd338103c630f95be8f90812d46",
        "checkpoint_sha256": "9b30dba3d2d801c305d96cc4ac6229f1d6ce61567871e0b58754633ad764bca8",
        "variant": "tiny-16k-seqlen-d128",
    },
    "hyenadna-small": {
        "repo_id": "LongSafari/hyenadna-tiny-1k-seqlen-d256-hf",
        "huggingface_revision": "6036d4e144922b44470294f510cbfd2539ba3b7e",
        "checkpoint_sha256": "81c15c4042d5fae646f00b429b76c65644cdbe88aa7e98b61aa781c84ff09031",
        "variant": "tiny-1k-seqlen-d256",
    },
    "hyenadna-medium": {
        "repo_id": "LongSafari/hyenadna-small-32k-seqlen-hf",
        "huggingface_revision": "8fe770c78eb13fe33bf81501612faeddf4d6f331",
        "checkpoint_sha256": "b7a1c9479248edf27691b189bb6657256f1705e560cadc64d6431633932cafa1",
        "variant": "small-32k-seqlen",
    },
    "hyenadna-large": {
        "repo_id": "LongSafari/hyenadna-large-1m-seqlen-hf",
        "huggingface_revision": "0a629abf9c7f85b4ec9aa6a1aefa3adcf1907446",
        "checkpoint_sha256": "deafb53209bfafb314d14bd81108546a34d473c07ed7ab7a355674376b56ad12",
        "variant": "large-1m-seqlen",
    },
}

UPSTREAM_DNA_VOCAB = {
    "[CLS]": 0,
    "[SEP]": 1,
    "[BOS]": 2,
    "[MASK]": 3,
    "[PAD]": 4,
    "[RESERVED]": 5,
    "[UNK]": 6,
    "A": 7,
    "C": 8,
    "G": 9,
    "T": 10,
    "N": 11,
}
MM_DNA_VOCAB = {
    "<pad>": 0,
    "<cls>": 1,
    "<eos>": 2,
    "<unk>": 3,
    "<mask>": 4,
    "<null>": 5,
}
MM_DNA_VOCAB.update({"A": 6, "C": 7, "G": 8, "T": 9, "N": 10})
UPSTREAM_LOGITS_VOCAB = ["<cls>", "<eos>", "<null>", "<mask>", "<pad>", "<null>", "<unk>", "A", "C", "G", "T", "N"]
MM_LOGITS_VOCAB = [token for token, _ in sorted(MM_DNA_VOCAB.items(), key=lambda item: item[1])]
LOGITS_REMAP = build_vocab_remap(
    UPSTREAM_LOGITS_VOCAB,
    MM_LOGITS_VOCAB,
    model_name=MODEL,
    duplicate_policy="last",
)


def checkpoint_source(case_info: dict[str, str]) -> str:
    return f"huggingface://{case_info['repo_id']}@{case_info['huggingface_revision']}/model.safetensors"


def tokenize_for_upstream(sequence: str) -> list[int]:
    ids = [UPSTREAM_DNA_VOCAB.get(base, UPSTREAM_DNA_VOCAB["[UNK]"]) for base in sequence.upper()]
    ids.append(UPSTREAM_DNA_VOCAB["[SEP]"])
    return ids


def tokenize_for_multimolecule(sequence: str) -> list[int]:
    ids = [MM_DNA_VOCAB.get(base, MM_DNA_VOCAB["<unk>"]) for base in sequence.upper()]
    ids.append(MM_DNA_VOCAB["<eos>"])
    return ids


def upstream_root_for_case(case: str) -> Path:
    case_info = CASES[case]
    return hf_snapshot_dir(
        str(case_info["repo_id"]),
        revision=str(case_info["huggingface_revision"]),
        allow_patterns=HF_ALLOW_PATTERNS,
        env_var="MULTIMOLECULE_UPSTREAM_HYENADNA_SOURCE",
    )


def load_upstream_model(upstream_root: Path) -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(str(upstream_root), trust_remote_code=True)
    return model.eval()


def upstream_forward(upstream_root: Path, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    model = load_upstream_model(upstream_root)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_hidden_states=True, return_dict=True)
    return {
        "hidden_states": stack_hidden_states(tuple(outputs.hidden_states)),
        "logits": remap_logits_to_vocab_subset(outputs.logits, LOGITS_REMAP),
    }


def main_for_case(case: str) -> None:
    if case not in CASES:
        known = ", ".join(sorted(CASES))
        raise SystemExit(f"Unknown HyenaDNA case {case!r}; known cases: {known}")

    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    case_info = CASES[case]
    upstream_root = upstream_root_for_case(case)
    checkpoint_path = upstream_root / "model.safetensors"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"HyenaDNA checkpoint not found: {checkpoint_path}")
    checkpoint_sha256 = sha256_of_file(checkpoint_path)
    if checkpoint_sha256 != case_info["checkpoint_sha256"]:
        raise RuntimeError(
            f"HyenaDNA {case} checkpoint digest mismatch for {checkpoint_path}: "
            f"expected {case_info['checkpoint_sha256']}, got {checkpoint_sha256}"
        )

    crop = crop_record(CORPUS_RECORD_ID, CROP_LENGTH, center=CROP_CENTER)
    sequence = crop["sequence"]
    if len(sequence) != CROP_LENGTH:
        raise AssertionError(f"crop length {len(sequence)} != {CROP_LENGTH}")
    if sequence_sha256(sequence) != crop["sha256"]:
        raise AssertionError("crop sha256 mismatch")

    input_ids = torch.tensor([tokenize_for_multimolecule(sequence)], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    upstream_input_ids = torch.tensor([tokenize_for_upstream(sequence)], dtype=torch.long)
    expected = upstream_forward(upstream_root, upstream_input_ids)

    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    out_dir = fixture_out_dir(REPO_ROOT, MODEL, case)

    meta = {
        "version": 1,
        "model": MODEL,
        "case": case,
        "auto_model": "AutoModelForCausalLM",
        "outputs": sorted(expected.keys()),
        "tolerance": {"atol": ATOL, "rtol": RTOL},
        "inputs_source": inputs_source_from_anchor_crop(
            crop,
            crop_name=CROP_NAME,
        ),
        "upstream": {
            "repository": UPSTREAM_REPO_URL,
            "commit": UPSTREAM_COMMIT,
            "checkpoint_source": checkpoint_source(case_info),
            "checkpoint_sha256": checkpoint_sha256,
            "target_slice": LOGITS_REMAP["target_slice"],
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
    print(f"  input_ids: {tuple(input_ids.shape)}")
    print(f"  shapes: {summary}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", nargs="?", default="hyenadna-tiny", choices=sorted(CASES))
    args = parser.parse_args()
    main_for_case(args.case)


if __name__ == "__main__":
    main()
