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

"""Generate the dnabert2 golden fixture from upstream DNABERT-2.

Run with the local ``/opt/conda/bin/python`` environment. Inputs use a short
synthetic DNA probe passed through the upstream DNABERT-2 tokenizer.

The upstream checkpoint ships custom remote-code modules. The generator resolves
the pinned Hugging Face snapshot through the upstream cache helper, imports those
modules as a local package, and loads the state dict directly. It intentionally
avoids importing MultiMolecule's DNABERT2 implementation when producing expected
tensors.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

import torch
from transformers import AutoTokenizer

HERE = Path(__file__).resolve()
MODEL_DIR = HERE.parent
REPO_ROOT = MODEL_DIR.parent.parent  # upstream/

sys.path.insert(0, str(REPO_ROOT))
from _shared.bert_probe import (  # noqa: E402
    disable_mosaic_flash_attention,
    force_eager_attention,
    stack_mosaic_hidden_states,
)
from _shared.fixture import (  # noqa: E402
    fixture_out_dir,
    sha256_of_file,
    write_fixture_artifacts,
)
from _shared.huggingface import hf_snapshot_dir  # noqa: E402

MODEL = "dnabert2"
CASE = "dnabert2"
ATOL = 1e-4
RTOL = 1e-4

UPSTREAM_PACKAGE = "dnabert2_upstream"
UPSTREAM_REPO_URL = "https://huggingface.co/zhihan1996/DNABERT-2-117M"
UPSTREAM_COMMIT = "7bce263b15377fc15361f52cfab88f8b586abda0"
CHECKPOINT_SOURCE = f"huggingface://zhihan1996/DNABERT-2-117M@{UPSTREAM_COMMIT}/pytorch_model.bin"
CHECKPOINT_SHA256 = "7ff39ec77a484dd01070a41bfd6e95cdd7247bec80fe357ab43a4be33687aeba"
HF_ALLOW_PATTERNS = ("*.json", "*.py", "pytorch_model.bin")
PROBE_ID = f"{MODEL}/{CASE}/synthetic_bpe_probe"
PROBE_SEQUENCE = "ATCGATCGGCTAAGCTTAGC"
EXPECTED_PROBE_TOKENS = ("[CLS]", "A", "TCGA", "TCGG", "CTAA", "GCTTA", "GC", "[SEP]")


def upstream_root() -> Path:
    return hf_snapshot_dir(
        "zhihan1996/DNABERT-2-117M",
        revision=UPSTREAM_COMMIT,
        allow_patterns=HF_ALLOW_PATTERNS,
        env_var="MULTIMOLECULE_UPSTREAM_DNABERT2_SOURCE",
    )


def register_upstream_package(package_name: str, package_root: Path) -> None:
    """Register a directory with relative-import modules as an importable package."""
    package = ModuleType(package_name)
    package.__file__ = str(package_root / "__init__.py")
    package.__path__ = [str(package_root)]  # type: ignore[attr-defined]
    sys.modules[package_name] = package


def load_upstream_model(root: Path, checkpoint_path: Path) -> torch.nn.Module:
    register_upstream_package(UPSTREAM_PACKAGE, root)
    bert_layers = importlib.import_module(f"{UPSTREAM_PACKAGE}.bert_layers")
    configuration_bert = importlib.import_module(f"{UPSTREAM_PACKAGE}.configuration_bert")
    disable_mosaic_flash_attention(bert_layers)

    config = force_eager_attention(configuration_bert.BertConfig.from_pretrained(str(root)))
    # DNABERT-2's remote code was authored against transformers 4.28, where
    # these config attributes are materialized. Newer transformers keeps the
    # same forward behavior but no longer materializes absent default fields.
    if not hasattr(config, "is_decoder"):
        config.is_decoder = False
    if not hasattr(config, "pad_token_id"):
        config.pad_token_id = None
    model = bert_layers.BertForMaskedLM(config)
    state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    load_result = model.load_state_dict(state_dict, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            "DNABERT-2 state dict did not load exactly: "
            f"missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}"
        )
    return model.eval()


def upstream_forward(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    root: Path,
    checkpoint_path: Path,
) -> dict[str, torch.Tensor]:
    model = load_upstream_model(root, checkpoint_path)
    with torch.no_grad():
        hidden_states, _ = model.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_all_encoded_layers=True,
        )
        stacked_hidden_states = stack_mosaic_hidden_states(hidden_states, input_ids)
        logits = model.cls(stacked_hidden_states[-1])
    return {
        "hidden_states": stacked_hidden_states,
        "logits": logits.contiguous(),
    }


def main() -> None:
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    root = upstream_root()
    checkpoint_path = root / "pytorch_model.bin"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"DNABERT-2 checkpoint not found: {checkpoint_path}")
    checkpoint_sha256 = sha256_of_file(checkpoint_path)
    if checkpoint_sha256 != CHECKPOINT_SHA256:
        raise RuntimeError(
            f"DNABERT-2 checkpoint digest mismatch for {checkpoint_path}: "
            f"expected {CHECKPOINT_SHA256}, got {checkpoint_sha256}"
        )

    tokenizer = AutoTokenizer.from_pretrained(str(root), trust_remote_code=True, local_files_only=True)
    tokenized = tokenizer(PROBE_SEQUENCE, return_tensors="pt")
    input_ids = tokenized["input_ids"].to(dtype=torch.long)
    attention_mask = tokenized["attention_mask"].to(dtype=torch.long)
    probe_tokens = tuple(tokenizer.convert_ids_to_tokens(input_ids[0].tolist()))
    if probe_tokens != EXPECTED_PROBE_TOKENS:
        raise AssertionError(
            f"DNABERT-2 probe tokenization changed: expected {EXPECTED_PROBE_TOKENS}, got {probe_tokens}"
        )
    expected = upstream_forward(input_ids, attention_mask, root, checkpoint_path)

    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    out_dir = fixture_out_dir(REPO_ROOT, MODEL, CASE)

    meta = {
        "version": 1,
        "model": MODEL,
        "case": CASE,
        "auto_model": "AutoModelForMaskedLM",
        "outputs": sorted(expected.keys()),
        "tolerance": {"atol": ATOL, "rtol": RTOL},
        "inputs_source": {
            "type": "synthetic_token_probe",
            "id": PROBE_ID,
        },
        "upstream": {
            "repository": UPSTREAM_REPO_URL,
            "commit": UPSTREAM_COMMIT,
            "checkpoint_source": CHECKPOINT_SOURCE,
            "checkpoint_sha256": checkpoint_sha256,
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
    print(f"  tokens: {list(probe_tokens)}")
    print(f"  shapes: {summary}")


if __name__ == "__main__":
    main()
