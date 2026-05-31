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

"""Generate the UTR-LM TE/EL checkpoint-parity fixture from official code."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import types
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve()
CASE_DIR = HERE.parent
REPO_ROOT = CASE_DIR.parent.parent.parent  # upstream/

sys.path.insert(0, str(REPO_ROOT))
from _corpus.load import crop_record, sequence_sha256  # noqa: E402
from _shared.alphabet import multimolecule_rna_vocabulary  # noqa: E402
from _shared.bert_probe import (  # noqa: E402
    build_vocab_remap,
    remap_logits_to_vocab_subset,
)
from _shared.fixture import (  # noqa: E402
    fixture_out_dir,
    inputs_source_from_anchor_crop,
    sha256_of_file,
    write_fixture_artifacts,
)
from _shared.source_tree import ensure_source_tree  # noqa: E402


def resolve_upstream_source() -> Path:
    return ensure_source_tree(
        UPSTREAM_REPO_URL,
        UPSTREAM_COMMIT,
        (*REQUIRED_UPSTREAM_FILES, *CHECKPOINT_OFFICIAL_REL_PATHS),
        env_var=UPSTREAM_SRC_ENV,
        cache_prefix=MODEL,
    )


def resolve_checkpoint_path(env_var: str, official_rel_path: str) -> Path:
    if env_var in os.environ:
        return Path(os.environ[env_var])
    return resolve_upstream_source() / official_rel_path


MODEL = "utrlm"
CASE = "utrlm-te_el"
VARIANT = "TE_EL"
DESCRIPTION = __doc__
ATOL = 1e-4
RTOL = 1e-4
UPSTREAM_REPO_URL = "https://github.com/a96123155/UTR-LM"
UPSTREAM_COMMIT = "b77b589bf182eb9de6a1a5024fa09d44294d94fc"
UPSTREAM_SRC_ENV = "MULTIMOLECULE_UPSTREAM_UTRLM_SOURCE"
CHECKPOINT_ENV = "MULTIMOLECULE_UPSTREAM_UTRLM_TE_EL_CHECKPOINT"
CHECKPOINT_OFFICIAL_REL_PATH = (
    "Model/Pretrained/" "ESM2SI_3.1_fiveSpeciesCao_6layers_16heads_128embedsize_" "4096batchToks_MLMLossMin.pkl"
)
CHECKPOINT_SOURCE = f"upstream-file://{CHECKPOINT_OFFICIAL_REL_PATH}"
CHECKPOINT_SHA256 = "8b33611dfeee12c13d537f9b9aeab2fcf092b88093a82606499894b3ddf0808f"
CORPUS_RECORD_ID = "rna/grch38_chr21_transcribed"
CROP_NAME = "rna_50nt"
CROP_CENTER = "center"
CROP_LENGTH = 50
REQUIRED_UPSTREAM_FILES = (
    "Scripts/esm/axial_attention.py",
    "Scripts/esm/constants.py",
    "Scripts/esm/data.py",
    "Scripts/esm/model/esm2.py",
    "Scripts/esm/modules.py",
    "Scripts/esm/multihead_attention.py",
    "Scripts/esm/rotary_embedding.py",
)
CHECKPOINT_OFFICIAL_REL_PATHS = (
    CHECKPOINT_OFFICIAL_REL_PATH,
    "Model/Pretrained/"
    "ESM2SISS_FS4.1_fiveSpeciesCao_6layers_16heads_128embedsize_"
    "4096batchToks_lr1e-05_supervisedweight1.0_structureweight1.0_MLMLossMin_epoch93.pkl",
)
OFFICIAL_STANDARD_TOKS = "AGCT"
OFFICIAL_TOKEN_IDS = {
    "<pad>": 0,
    "<eos>": 1,
    "<unk>": 2,
    "A": 3,
    "G": 4,
    "C": 5,
    "T": 6,
    "<cls>": 7,
    "<mask>": 8,
    "<sep>": 9,
}
OFFICIAL_LOGITS_VOCAB = ("<pad>", "<eos>", "<unk>", "A", "G", "C", "U", "<cls>", "<mask>", "<sep>")
MM_VOCAB = multimolecule_rna_vocabulary()
LOGITS_REMAP = build_vocab_remap(OFFICIAL_LOGITS_VOCAB, MM_VOCAB, model_name=MODEL)
ALLOWED_UNEXPECTED_KEYS = {
    "supervised_linear.weight",
    "supervised_linear.bias",
    "structure_linear.weight",
    "structure_linear.bias",
}


def configure_case(
    *,
    case: str,
    variant: str,
    checkpoint_env: str,
    checkpoint_official_rel_path: str,
    checkpoint_sha256: str,
    description: str | None = None,
) -> None:
    global CASE, VARIANT, DESCRIPTION, CHECKPOINT_ENV, CHECKPOINT_OFFICIAL_REL_PATH
    global CHECKPOINT_SOURCE, CHECKPOINT_SHA256

    CASE = case
    VARIANT = variant
    if description is not None:
        DESCRIPTION = description
    CHECKPOINT_ENV = checkpoint_env
    CHECKPOINT_OFFICIAL_REL_PATH = checkpoint_official_rel_path
    CHECKPOINT_SHA256 = checkpoint_sha256
    CHECKPOINT_SOURCE = f"upstream-file://{CHECKPOINT_OFFICIAL_REL_PATH}"


def checkpoint_missing_message() -> str:
    return (
        f"UTR-LM {CASE} checkpoint not found: {resolve_checkpoint_path(CHECKPOINT_ENV, CHECKPOINT_OFFICIAL_REL_PATH)}. "
        f"Set {CHECKPOINT_ENV} or use {UPSTREAM_SRC_ENV} to point at an "
        f"official checkout containing {CHECKPOINT_OFFICIAL_REL_PATH}."
    )


def encode(sequence: str) -> list[int]:
    normalized = sequence.upper().replace("U", "T")
    unknown = sorted(set(normalized) - set(OFFICIAL_STANDARD_TOKS))
    if unknown:
        raise ValueError(f"UTR-LM upstream vocabulary cannot encode bases: {unknown}")
    return [
        OFFICIAL_TOKEN_IDS["<cls>"],
        *(OFFICIAL_TOKEN_IDS[base] for base in normalized),
        OFFICIAL_TOKEN_IDS["<eos>"],
    ]


def encode_mm(sequence: str) -> list[int]:
    token_to_id = {token: index for index, token in enumerate(MM_VOCAB)}
    normalized = sequence.upper().replace("T", "U")
    return [
        token_to_id["<cls>"],
        *(token_to_id.get(base, token_to_id["<unk>"]) for base in normalized),
        token_to_id["<eos>"],
    ]


def ensure_upstream_code() -> Path:
    root = resolve_upstream_source()
    for rel_path in REQUIRED_UPSTREAM_FILES:
        target = root / rel_path
        if not target.is_file():
            raise FileNotFoundError(
                f"missing UTR-LM upstream file: {target}. "
                f"Set {UPSTREAM_SRC_ENV} to a checkout of {UPSTREAM_REPO_URL} "
                f"at commit {UPSTREAM_COMMIT}."
            )
    return root


def load_module(module_name: str, module_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load UTR-LM upstream module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_upstream_classes() -> tuple[type, type[torch.nn.Module]]:
    root = ensure_upstream_code()
    sys.dont_write_bytecode = True
    esm_root = root / "Scripts" / "esm"
    for module_name in list(sys.modules):
        if module_name == "esm" or module_name.startswith("esm."):
            del sys.modules[module_name]

    esm_package = types.ModuleType("esm")
    esm_package.__path__ = [str(esm_root)]
    sys.modules["esm"] = esm_package
    model_package = types.ModuleType("esm.model")
    model_package.__path__ = [str(esm_root / "model")]
    sys.modules["esm.model"] = model_package

    constants = load_module("esm.constants", esm_root / "constants.py")
    esm_package.constants = constants
    data = load_module("esm.data", esm_root / "data.py")
    esm_package.data = data
    rotary_embedding = load_module("esm.rotary_embedding", esm_root / "rotary_embedding.py")
    esm_package.rotary_embedding = rotary_embedding
    multihead_attention = load_module("esm.multihead_attention", esm_root / "multihead_attention.py")
    esm_package.multihead_attention = multihead_attention
    axial_attention = load_module("esm.axial_attention", esm_root / "axial_attention.py")
    esm_package.axial_attention = axial_attention
    modules = load_module("esm.modules", esm_root / "modules.py")
    esm_package.modules = modules
    esm2 = load_module("esm.model.esm2", esm_root / "model" / "esm2.py")
    model_package.esm2 = esm2
    return data.Alphabet, esm2.ESM2


def load_upstream_weight_model() -> torch.nn.Module:
    checkpoint = resolve_checkpoint_path(CHECKPOINT_ENV, CHECKPOINT_OFFICIAL_REL_PATH)
    actual_sha256 = sha256_of_file(checkpoint)
    if actual_sha256 != CHECKPOINT_SHA256:
        raise RuntimeError(
            f"UTR-LM {CASE} checkpoint digest mismatch for {checkpoint}: "
            f"expected {CHECKPOINT_SHA256}, got {actual_sha256}"
        )
    Alphabet, ESM2 = load_upstream_classes()
    alphabet = Alphabet(mask_prob=0.15, standard_toks=OFFICIAL_STANDARD_TOKS)
    if alphabet.tok_to_idx != OFFICIAL_TOKEN_IDS:
        raise RuntimeError(f"unexpected UTR-LM upstream alphabet: {alphabet.tok_to_idx}")
    model = ESM2(num_layers=6, embed_dim=128, attention_heads=16, alphabet=alphabet)
    ckpt = torch.load(checkpoint, map_location=torch.device("cpu"))
    state_dict = {key.removeprefix("module."): value for key, value in ckpt.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        raise RuntimeError(f"missing UTR-LM upstream checkpoint keys: {sorted(missing)}")
    extra = set(unexpected) - ALLOWED_UNEXPECTED_KEYS
    if extra:
        raise RuntimeError(f"unexpected UTR-LM upstream checkpoint keys: {sorted(extra)}")
    return model.eval()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "case",
        nargs="?",
        default=CASE,
        choices=[CASE],
        help="UTR-LM fixture case.",
    )
    return parser.parse_args()


def main() -> None:
    parse_args()
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    if not resolve_checkpoint_path(CHECKPOINT_ENV, CHECKPOINT_OFFICIAL_REL_PATH).is_file():
        raise FileNotFoundError(checkpoint_missing_message())

    crop = crop_record(CORPUS_RECORD_ID, CROP_LENGTH, center=CROP_CENTER)
    sequence = crop["sequence"]
    if sequence_sha256(sequence) != crop["sha256"]:
        raise AssertionError("crop sha256 mismatch")

    inputs = {"input_ids": torch.tensor([encode_mm(sequence)], dtype=torch.long)}
    upstream_input_ids = torch.tensor([encode(sequence)], dtype=torch.long)
    model = load_upstream_weight_model()
    with torch.no_grad():
        outputs = model(
            upstream_input_ids,
            return_representation=False,
            repr_layers=[6],
            return_contacts=False,
        )
    expected = {"logits": remap_logits_to_vocab_subset(outputs["logits"], LOGITS_REMAP)}

    out_dir = fixture_out_dir(REPO_ROOT, MODEL, CASE)

    meta: dict[str, Any] = {
        "version": 1,
        "model": MODEL,
        "case": CASE,
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
            "checkpoint_source": CHECKPOINT_SOURCE,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "target_slice": LOGITS_REMAP["target_slice"],
        },
    }
    write_fixture_artifacts(
        out_dir,
        inputs=inputs,
        expected=expected,
        meta=meta,
    )


if __name__ == "__main__":
    main()
