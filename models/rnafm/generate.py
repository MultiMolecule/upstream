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

"""Generate RNA-FM golden fixtures from official RNA-FM implementations."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import types
from argparse import Namespace
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve()
MODEL_DIR = HERE.parent
REPO_ROOT = MODEL_DIR.parent.parent

sys.path.insert(0, str(REPO_ROOT))
from _corpus.load import crop_record  # noqa: E402
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
from _shared.huggingface import hf_snapshot_dir  # noqa: E402
from _shared.source_tree import ensure_source_tree  # noqa: E402

MODEL = "rnafm"
UPSTREAM_REPO_URL = "https://github.com/ml4bio/RNA-FM"
UPSTREAM_COMMIT = "348951516e0963d22bbb33b3c9fc18c89081d38e"
UPSTREAM_SOURCE_ENV_VAR = "MULTIMOLECULE_UPSTREAM_RNAFM_SOURCE"
UPSTREAM_CACHE_PREFIX = "rnafm"
HF_REPO_ID = "cuhkaih/rnafm"
HF_REVISION = "91d4a46d28d8054a7b429955e8fc0c253ba0afd6"
HF_SNAPSHOT_ENV_VAR = "MULTIMOLECULE_UPSTREAM_RNAFM_SNAPSHOT"
CORPUS_RECORD_ID = "rna/grch38_chr21_transcribed"
CROP_NAME = "rna_50nt"
CROP_CENTER = "center"
ATOL = 1e-4
RTOL = 1e-4

CASES = {
    "rnafm": {
        "source": f"huggingface://{HF_REPO_ID}@{HF_REVISION}/RNA-FM_pretrained.pth",
        "filename": "RNA-FM_pretrained.pth",
        "env_var": "MULTIMOLECULE_UPSTREAM_RNAFM_CHECKPOINT",
        "sha256": "5b5d7d87b37c291ef42c140ef9edf7aea29f255fa2a4fd435f776c52e93d5e99",
        "codon": False,
        "length": 50,
        "theme": "rna",
        "state": "fairseq",
    },
    "rnafm-mrna": {
        "source": f"huggingface://{HF_REPO_ID}@{HF_REVISION}/mRNA-FM_pretrained.pth",
        "filename": "mRNA-FM_pretrained.pth",
        "env_var": "MULTIMOLECULE_UPSTREAM_RNAFM_MRNA_CHECKPOINT",
        "sha256": "413324520d35f00bf5419a85033b18e11eb57bc1356a752660f8d75d3ef7624f",
        "codon": True,
        "length": 48,
        "theme": "rna-3mer",
        "state": "fairseq",
    },
    "rnafm-ss": {
        "source": f"huggingface://{HF_REPO_ID}@{HF_REVISION}/SS/RNA-FM-ResNet_bpRNA.pth",
        "filename": "SS/RNA-FM-ResNet_bpRNA.pth",
        "env_var": "MULTIMOLECULE_UPSTREAM_RNAFM_SS_CHECKPOINT",
        "sha256": "4ba2cde6b6ad75fb1c742cd5a7f40d4d0d4bcd63a329ff98347776e7f84630e6",
        "codon": False,
        "length": 50,
        "theme": "rna",
        "state": "downstream_backbone",
        "model_args": {
            "arch": "roberta_large",
            "layers": 12,
            "embed_dim": 640,
            "ffn_embed_dim": 5120,
            "attention_heads": 20,
            "max_positions": 1024,
            "dropout": 0.1,
        },
    },
}


def checkpoint_path(case: dict[str, Any]) -> Path:
    override = os.environ.get(str(case["env_var"]))
    if override:
        return Path(override).expanduser().resolve()
    snapshot = hf_snapshot_dir(
        HF_REPO_ID,
        revision=HF_REVISION,
        allow_patterns=str(case["filename"]),
        env_var=HF_SNAPSHOT_ENV_VAR,
    )
    return snapshot / str(case["filename"])


def install_fm_package(source_root: Path) -> None:
    """Expose the official fm package without executing fm/__init__.py."""

    fm_root = source_root / "fm"
    package = types.ModuleType("fm")
    package.__file__ = str(fm_root / "__init__.py")
    package.__path__ = [str(fm_root)]  # type: ignore[attr-defined]
    sys.modules["fm"] = package

    model_package = types.ModuleType("fm.model")
    model_package.__path__ = [str(fm_root / "model")]  # type: ignore[attr-defined]
    sys.modules["fm.model"] = model_package


def import_official_rnafm() -> tuple[Any, Any]:
    source_root = ensure_source_tree(
        UPSTREAM_REPO_URL,
        UPSTREAM_COMMIT,
        ("fm",),
        env_var=UPSTREAM_SOURCE_ENV_VAR,
        cache_prefix=UPSTREAM_CACHE_PREFIX,
    )
    install_fm_package(source_root)
    importlib.import_module("fm.constants")
    data = importlib.import_module("fm.data")
    importlib.import_module("fm.axial_attention")
    importlib.import_module("fm.multihead_attention")
    importlib.import_module("fm.modules")
    esm1 = importlib.import_module("fm.model.esm1")
    return data, esm1


def has_emb_layer_norm_before(model_state: dict[str, torch.Tensor]) -> bool:
    return any(key.startswith("emb_layer_norm_before") for key in model_state)


def fairseq_model_args(args: Namespace) -> dict[str, Any]:
    def strip_encoder_prefix(name: str) -> str:
        return "".join(name.split("encoder_")[1:] if "encoder" in name else name)

    return {strip_encoder_prefix(key): value for key, value in vars(args).items()}


def fairseq_model_state(
    model_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    def strip_encoder_dot(name: str) -> str:
        return "".join(name.split("encoder.")[1:] if "encoder" in name else name)

    def strip_sentence_encoder(name: str) -> str:
        return "".join(name.split("sentence_encoder.")[1:] if "sentence_encoder" in name else name)

    return {strip_encoder_dot(strip_sentence_encoder(key)): value for key, value in model_state.items()}


def load_model(case: dict[str, Any], checkpoint_path: Path) -> tuple[torch.nn.Module, Any]:
    data, esm1 = import_official_rnafm()
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=torch.device("cpu"))
    alphabet = data.Alphabet.from_architecture("roberta_large", str(case["theme"]))

    if case["state"] == "fairseq":
        model_args = fairseq_model_args(checkpoint["args"])
        model_state = fairseq_model_state(checkpoint["model"])
        model_state["embed_tokens.weight"][alphabet.mask_idx].zero_()
        model_args["emb_layer_norm_before"] = has_emb_layer_norm_before(model_state)
    elif case["state"] == "downstream_backbone":
        model_args = dict(case["model_args"])
        model_state = {
            key.removeprefix("backbone."): value for key, value in checkpoint.items() if key.startswith("backbone.")
        }
        model_args["emb_layer_norm_before"] = has_emb_layer_norm_before(model_state)
    else:
        raise ValueError(f"unknown RNA-FM state format: {case['state']}")

    model = esm1.BioBertModel(Namespace(**model_args), alphabet)
    expected_missing = {
        "contact_head.regression.weight",
        "contact_head.regression.bias",
    }
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    missing_set = set(missing)
    unexpected_set = set(unexpected)
    if missing_set - expected_missing or unexpected_set:
        raise RuntimeError(
            "official RNA-FM state mismatch: "
            f"missing={sorted(missing_set - expected_missing)} unexpected={sorted(unexpected_set)}"
        )
    return model.eval(), alphabet


def encode_mm(sequence: str, *, codon: bool) -> list[int]:
    sequence = sequence.upper().replace("T", "U")
    if codon and len(sequence) % 3:
        raise ValueError("mRNA-FM fixture sequence length must be divisible by 3")
    vocabulary = multimolecule_rna_vocabulary(nmers=3 if codon else 1)
    token_to_id = {token: index for index, token in enumerate(vocabulary)}
    tokens = [sequence[index : index + 3] for index in range(0, len(sequence), 3)] if codon else list(sequence)
    ids = [token_to_id["<cls>"]]
    ids.extend(token_to_id.get(token, token_to_id["<unk>"]) for token in tokens)
    ids.append(token_to_id["<eos>"])
    return ids


def encode_upstream(sequence: str, upstream_alphabet: Any) -> torch.Tensor:
    _, _, tokens = upstream_alphabet.get_batch_converter()([("", sequence.upper().replace("T", "U"))])
    return tokens


def normalized_upstream_vocab(upstream_alphabet: Any) -> list[str]:
    return [("<null>" if token.startswith("<null_") else token) for token in upstream_alphabet.all_toks]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=sorted(CASES))
    args = parser.parse_args()
    case = CASES[args.case]
    checkpoint = checkpoint_path(case)
    if sha256_of_file(checkpoint) != case["sha256"]:
        raise RuntimeError(f"checkpoint digest mismatch: {checkpoint}")

    crop = crop_record(CORPUS_RECORD_ID, int(case["length"]), center=CROP_CENTER)
    input_ids = torch.tensor([encode_mm(crop["sequence"], codon=bool(case["codon"]))], dtype=torch.long)
    model, upstream_alphabet = load_model(case, checkpoint)
    mm_vocab = multimolecule_rna_vocabulary(nmers=3 if case["codon"] else 1)
    upstream_vocab = normalized_upstream_vocab(upstream_alphabet)
    logits_remap = build_vocab_remap(upstream_vocab, mm_vocab, model_name=args.case, duplicate_policy="last")
    upstream_input_ids = encode_upstream(crop["sequence"], upstream_alphabet)
    with torch.no_grad():
        outputs = model(
            upstream_input_ids,
            repr_layers=list(range(model.num_layers + 1)),
        )
    expected = {
        "hidden_states": torch.stack(
            tuple(outputs["representations"][index] for index in range(model.num_layers + 1)),
            dim=0,
        )
        .detach()
        .cpu()
        .contiguous(),
        "logits": remap_logits_to_vocab_subset(outputs["logits"], logits_remap),
    }

    out_dir = fixture_out_dir(REPO_ROOT, MODEL, args.case)
    meta: dict[str, Any] = {
        "version": 1,
        "model": MODEL,
        "case": args.case,
        "auto_model": "AutoModelForMaskedLM",
        "outputs": sorted(expected),
        "tolerance": {"atol": ATOL, "rtol": RTOL},
        "inputs_source": inputs_source_from_anchor_crop(
            crop,
            crop_name=CROP_NAME,
        ),
        "upstream": {
            "repository": UPSTREAM_REPO_URL,
            "commit": UPSTREAM_COMMIT,
            "checkpoint_source": case["source"],
            "checkpoint_sha256": case["sha256"],
            "target_slice": logits_remap["target_slice"],
        },
    }
    write_fixture_artifacts(
        out_dir,
        inputs={"input_ids": input_ids},
        expected=expected,
        meta=meta,
    )
    print(f"Wrote fixture to {out_dir}")
    print(f"  input_ids: {tuple(input_ids.shape)}")
    print(f"  upstream_input_ids: {tuple(upstream_input_ids.shape)}")
    print(f"  expected: {{ {', '.join(f'{key}: {tuple(value.shape)}' for key, value in expected.items())} }}")


if __name__ == "__main__":
    main()
