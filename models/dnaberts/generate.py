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

"""Generate the dnaberts golden fixture from upstream DNABERT-S.

Run with the local ``/opt/conda/bin/python`` environment. The input is a short
synthetic DNA sequence passed through the pinned upstream tokenizer. The
upstream checkpoint ships custom remote-code modules, so the generator imports
those modules as a local package and loads the state dict directly. It avoids
Transformers-version-specific ``AutoModel.from_pretrained`` loading behavior and
does not import MultiMolecule when producing expected tensors.
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

MODEL = "dnaberts"
CASE = "dnaberts"
ATOL = 1e-4
RTOL = 1e-4

UPSTREAM_PACKAGE = "dnaberts_upstream"
UPSTREAM_REPO_URL = "https://huggingface.co/zhihan1996/DNABERT-S"
UPSTREAM_COMMIT = "00e47f96cdea35e4b6f5df89e5419cbe47d490c6"
CHECKPOINT_SOURCE = f"huggingface://zhihan1996/DNABERT-S@{UPSTREAM_COMMIT}/pytorch_model.bin"
CHECKPOINT_SHA256 = "f3cfc3d0541859df64759e758cc4bc40fe1d5760799a7d098a9f9f09501c1e5f"
HF_ALLOW_PATTERNS = ("*.json", "*.py", "pytorch_model.bin")

SYNTHETIC_SEQUENCE = "ACGTACGTACGTACGTACGTACGTACGTACGT"
PROBE_ID = f"{MODEL}/{CASE}/synthetic_sequence_probe"


def upstream_root() -> Path:
    return hf_snapshot_dir(
        "zhihan1996/DNABERT-S",
        revision=UPSTREAM_COMMIT,
        allow_patterns=HF_ALLOW_PATTERNS,
        env_var="MULTIMOLECULE_UPSTREAM_DNABERTS_SOURCE",
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
    model = bert_layers.BertModel(config)
    state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    load_result = model.load_state_dict(state_dict, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            "DNABERT-S state dict did not load exactly: "
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
        last_hidden_state, pooler_output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_all_encoded_layers=False,
        )
        pooler = model.pooler
        model.pooler = None
        try:
            hidden_states, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_all_encoded_layers=True,
            )
        finally:
            model.pooler = pooler
        stacked_hidden_states = stack_mosaic_hidden_states(hidden_states, input_ids)
    return {
        "hidden_states": stacked_hidden_states,
        "last_hidden_state": last_hidden_state.contiguous(),
        "pooler_output": pooler_output.contiguous(),
    }


def main() -> None:
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    root = upstream_root()
    checkpoint_path = root / "pytorch_model.bin"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"DNABERT-S checkpoint not found: {checkpoint_path}. "
            "Set $MULTIMOLECULE_UPSTREAM_DNABERTS_SOURCE to a pinned local snapshot or populate the Hugging Face cache."
        )
    checkpoint_sha256 = sha256_of_file(checkpoint_path)
    if checkpoint_sha256 != CHECKPOINT_SHA256:
        raise AssertionError(f"{checkpoint_path}: expected sha256 {CHECKPOINT_SHA256}, got {checkpoint_sha256}")

    sequence = SYNTHETIC_SEQUENCE
    tokenizer = AutoTokenizer.from_pretrained(str(root), trust_remote_code=True, local_files_only=True)
    tokenized = tokenizer(sequence, return_tensors="pt")
    input_ids = tokenized["input_ids"].to(dtype=torch.long)
    attention_mask = tokenized["attention_mask"].to(dtype=torch.long)
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
        "outputs": sorted(expected.keys()),
        "tolerance": {"atol": ATOL, "rtol": RTOL},
        "inputs_source": {
            "type": "synthetic_sequence_probe",
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
    print(f"  shapes: {summary}")


if __name__ == "__main__":
    main()
