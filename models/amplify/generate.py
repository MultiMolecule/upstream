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

"""Generate AMPLIFY golden fixtures from upstream HF checkpoints.

Inputs are taken from ``_corpus/protein/grch38_chr21_orf`` with the
``protein_128aa`` center crop. The saved ``input_ids`` use MultiMolecule's
protein vocabulary so the main faithfulness test can run without importing the
upstream tokenizer.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel

HERE = Path(__file__).resolve()
MODEL_DIR = HERE.parent
REPO_ROOT = MODEL_DIR.parent.parent  # upstream/

sys.path.insert(0, str(REPO_ROOT))
from _corpus.load import crop_record  # noqa: E402
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

MODEL = "amplify"
# AMPLIFY CPU replay uses an xformers fallback shim for upstream attention.
ATOL = 5e-3
RTOL = 1e-4

UPSTREAM_REPO_URL = "https://github.com/chandar-lab/AMPLIFY"
UPSTREAM_COMMIT = "7fb657a6bd411e0d51517ccde7cc35391c072b65"
CORPUS_RECORD_ID = "protein/grch38_chr21_orf"
CORPUS_RECORD_NAME = "grch38_chr21_orf"
CROP_NAME = "protein_128aa"
CROP_CENTER = "center"
CROP_LENGTH = 128
HF_ALLOW_PATTERNS = ("*.json", "*.py", "model.safetensors")

# AMPLIFY's tokenizer maps ``<bos>`` to id 3. MultiMolecule's protein
# tokenizer uses ``<cls>`` for the same sequence-start role.
UPSTREAM_VOCAB = [
    "<pad>",
    "<unk>",
    "<mask>",
    "<cls>",
    "<eos>",
    "|",
    "L",
    "A",
    "G",
    "V",
    "S",
    "E",
    "R",
    "T",
    "I",
    "D",
    "P",
    "K",
    "Q",
    "N",
    "F",
    "Y",
    "M",
    "H",
    "W",
    "C",
    "B",
]
MM_VOCAB = ["<pad>", "<cls>", "<eos>", "<unk>", "<mask>", "<null>"] + list("ACDEFGHIKLMNPQRSTVWYXZBJUO|.*-?")
LOGITS_REMAP = build_vocab_remap(UPSTREAM_VOCAB, MM_VOCAB, model_name=MODEL)


@dataclass
class AmplifyCase:
    case: str
    variant: str
    huggingface_revision: str
    checkpoint_sha256: str

    @property
    def weights_dir(self) -> Path:
        case_token = self.case.upper().replace("-", "_")
        return hf_snapshot_dir(
            f"chandar-lab/{self.variant}",
            revision=self.huggingface_revision,
            allow_patterns=HF_ALLOW_PATTERNS,
            env_var=f"MULTIMOLECULE_UPSTREAM_{case_token}_CHECKPOINT_DIR",
        )

    @property
    def checkpoint_source(self) -> str:
        return f"huggingface://chandar-lab/{self.variant}@{self.huggingface_revision}/model.safetensors"


CASES = {
    "amplify-120m": AmplifyCase(
        case="amplify-120m",
        variant="AMPLIFY_120M",
        huggingface_revision="d918a9e8c64dc43f6ec4996022c3deba14d66470",
        checkpoint_sha256="a2375f1f54cbe00bdbe27eedcd039c92d12f165720c0349bc582a6eb42c099ce",
    ),
    "amplify-350m": AmplifyCase(
        case="amplify-350m",
        variant="AMPLIFY_350M",
        huggingface_revision="223e35e4e2074fadfdc1617f3451b14fa5ace1c8",
        checkpoint_sha256="9df2a8e1c6c220914b2e231008d5372235192c98e47f28abc4bf17e0ea97b5ed",
    ),
}


def parse_case(case: str) -> AmplifyCase:
    try:
        return CASES[case]
    except KeyError as error:
        raise ValueError(f"Unsupported AMPLIFY case: {case}") from error


def _install_xformers_stub() -> None:
    """Provide the xformers APIs used by upstream AMPLIFY on CPU runs."""

    class SwiGLU(nn.Module):
        def __init__(
            self,
            in_features: int,
            hidden_features: int,
            out_features: int,
            bias: bool = False,
        ):
            super().__init__()
            self.w12 = nn.Linear(in_features, hidden_features * 2, bias=bias)
            self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            gate, up = self.w12(x).chunk(2, dim=-1)
            return self.w3(F.silu(gate) * up)

    def memory_efficient_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
        p: float = 0.0,
    ) -> torch.Tensor:
        return F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=attn_bias,
            dropout_p=p,
        ).transpose(1, 2)

    ops = types.ModuleType("xformers.ops")
    ops.SwiGLU = SwiGLU
    ops.memory_efficient_attention = memory_efficient_attention
    xformers = types.ModuleType("xformers")
    xformers.ops = ops
    sys.modules.setdefault("xformers", xformers)
    sys.modules.setdefault("xformers.ops", ops)


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.upper().encode()).hexdigest()


def tokenize(sequence: str, vocab: list[str]) -> list[int]:
    cls_id = vocab.index("<cls>")
    eos_id = vocab.index("<eos>")
    unk_id = vocab.index("<unk>")
    ids = [cls_id]
    for ch in sequence.upper():
        ids.append(vocab.index(ch) if ch in vocab else unk_id)
    ids.append(eos_id)
    return ids


def additive_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    return torch.where(
        attention_mask.bool(),
        torch.zeros((), dtype=torch.float32),
        torch.full((), float("-inf"), dtype=torch.float32),
    )


def load_upstream_model(case: AmplifyCase) -> nn.Module:
    _install_xformers_stub()
    model = AutoModel.from_pretrained(str(case.weights_dir), trust_remote_code=True)
    rotary_module = importlib.import_module(model.__class__.__module__)
    if getattr(model.freqs_cis, "device", None) is not None and model.freqs_cis.device.type == "meta":
        model.freqs_cis = rotary_module.precompute_freqs_cis(
            model.config.hidden_size // model.config.num_attention_heads,
            model.config.max_length,
        )
    model.eval()
    return model


def upstream_forward(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    case: AmplifyCase,
) -> dict[str, torch.Tensor]:
    model = load_upstream_model(case)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=additive_attention_mask(attention_mask),
            output_hidden_states=True,
            return_dict=True,
        )
    return {
        "hidden_states": stack_hidden_states(outputs.hidden_states),
        "logits": remap_logits_to_vocab_subset(outputs.logits, LOGITS_REMAP),
    }


def main_for_case(case_name: str) -> None:
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    case = parse_case(case_name)
    if not case.weights_dir.is_dir():
        raise FileNotFoundError(f"{case.variant} weights directory not found: {case.weights_dir}")
    checkpoint_path = case.weights_dir / "model.safetensors"
    checkpoint_sha256 = sha256_of_file(checkpoint_path)
    if checkpoint_sha256 != case.checkpoint_sha256:
        raise RuntimeError(
            f"AMPLIFY {case.case} checkpoint digest mismatch for {checkpoint_path}: "
            f"expected {case.checkpoint_sha256}, got {checkpoint_sha256}"
        )

    crop = crop_record(CORPUS_RECORD_ID, CROP_LENGTH, center=CROP_CENTER)
    sequence = crop["sequence"]
    if len(sequence) != CROP_LENGTH:
        raise AssertionError(f"crop length {len(sequence)} != {CROP_LENGTH}")
    if sequence_sha256(sequence) != crop["sha256"]:
        raise AssertionError("crop sha256 mismatch")

    mm_ids = tokenize(sequence, MM_VOCAB)
    upstream_ids = tokenize(sequence, UPSTREAM_VOCAB)
    input_ids = torch.tensor([mm_ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    upstream_input_ids = torch.tensor([upstream_ids], dtype=torch.long)
    expected = upstream_forward(upstream_input_ids, attention_mask, case)

    inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
    out_dir = fixture_out_dir(REPO_ROOT, MODEL, case.case)

    checkpoint_path = case.weights_dir / "model.safetensors"
    meta = {
        "version": 1,
        "model": MODEL,
        "case": case.case,
        "auto_model": "AutoModelForMaskedLM",
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
    print(f"  sequence_sha256: {crop['sha256']}")
    print(f"  shapes: {summary}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", help="AMPLIFY fixture case, for example amplify-120m.")
    args = parser.parse_args()
    main_for_case(args.case)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
