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

"""Generate AbLang2 golden fixtures from the upstream implementation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
MODEL_DIR = HERE.parent
REPO_ROOT = MODEL_DIR.parent.parent  # upstream/

sys.path.insert(0, str(REPO_ROOT))
from _shared.archive import safe_extract_tar  # noqa: E402
from _shared.bert_probe import build_vocab_remap, remap_logits_to_vocab_subset  # noqa: E402
from _shared.download import fetch_http_file, upstream_cache_root  # noqa: E402
from _shared.fixture import fixture_out_dir, write_fixture_artifacts  # noqa: E402
from _shared.source_tree import ensure_source_tree  # noqa: E402

MODEL = "ablang2"
CASE = "ablang2"
UPSTREAM_REPO_URL = "https://github.com/oxpig/AbLang2"
UPSTREAM_COMMIT = "586af3083c32b5fb2f0a1c855f2ad4f1cad15ec3"
CHECKPOINT_SOURCE = "https://zenodo.org/records/10185169/files/ablang2-weights.tar.gz?download=1"
CHECKPOINT_FILENAME = "ablang2-weights.tar.gz"
CHECKPOINT_SHA256 = "7acb23b880e3b69b3fea8b7d6cacac0fda147daafd99edc411e0ff5c61252ac5"
CHECKPOINT_ENV_VAR = "MULTIMOLECULE_UPSTREAM_ABLANG2_CHECKPOINT_ARCHIVE"
CHECKPOINT_DIR_ENV_VAR = "MULTIMOLECULE_UPSTREAM_ABLANG2_CHECKPOINT_DIR"
SOURCE_ENV_VAR = "MULTIMOLECULE_UPSTREAM_ABLANG2_SOURCE"
PROBE_SEQUENCE = "<MRHKDESTNQC>|<GPAVIFYWL>"
ATOL = 2e-4
RTOL = 2e-4

UPSTREAM_VOCAB = [
    "<cls>",
    "M",
    "R",
    "H",
    "K",
    "D",
    "E",
    "S",
    "T",
    "N",
    "Q",
    "C",
    "G",
    "P",
    "A",
    "V",
    "I",
    "F",
    "Y",
    "W",
    "L",
    "<pad>",
    "<eos>",
    "<mask>",
    "X",
    "|",
]
MM_VOCAB = ["<pad>", "<cls>", "<eos>", "<unk>", "<mask>", "<null>"] + list("ACDEFGHIKLMNPQRSTVWYXZBJUO|.*-?")
LOGITS_REMAP = build_vocab_remap(UPSTREAM_VOCAB, MM_VOCAB, model_name=MODEL)


@dataclass(frozen=True)
class AbLang2Case:
    case: str = CASE
    checkpoint_source: str = CHECKPOINT_SOURCE
    checkpoint_sha256: str = CHECKPOINT_SHA256

    @property
    def checkpoint_dir(self) -> Path:
        override = os.environ.get(CHECKPOINT_DIR_ENV_VAR)
        if override:
            return Path(override).expanduser().resolve()
        archive = fetch_http_file(
            self.checkpoint_source,
            CHECKPOINT_FILENAME,
            cache_prefix=MODEL,
            env_var=CHECKPOINT_ENV_VAR,
            sha256=self.checkpoint_sha256,
            description="AbLang2 checkpoint archive",
        )
        destination = upstream_cache_root() / MODEL / "ablang2-weights"
        missing_checkpoint_files = (
            not (destination / "model.pt").is_file() or not (destination / "hparams.json").is_file()
        )
        if missing_checkpoint_files:
            safe_extract_tar(archive, destination)
        return destination


CASES = {CASE: AbLang2Case()}


def parse_case(case: str) -> AbLang2Case:
    try:
        return CASES[case]
    except KeyError as error:
        raise ValueError(f"Unsupported AbLang2 case: {case}") from error


def source_dir() -> Path:
    return ensure_source_tree(
        UPSTREAM_REPO_URL,
        UPSTREAM_COMMIT,
        ("ablang2",),
        env_var=SOURCE_ENV_VAR,
        cache_prefix=MODEL,
    )


def encode_probe(vocab: list[str]) -> torch.Tensor:
    char_map = {
        "<": "<cls>",
        ">": "<eos>",
        "-": "<pad>",
        "*": "<mask>",
    }
    ids = [vocab.index(char_map.get(token, token)) for token in PROBE_SEQUENCE]
    return torch.tensor([ids], dtype=torch.long)


def load_upstream_model(case: AbLang2Case):
    source = source_dir()
    checkpoint = case.checkpoint_dir
    sys.path.insert(0, str(source))
    from ablang2.models.ablang2 import ablang  # noqa: PLC0415

    with open(checkpoint / "hparams.json", encoding="utf-8") as handle:
        hparams = json.load(handle)
    model = ablang.AbLang(
        vocab_size=hparams["vocab_size"],
        hidden_embed_size=hparams["hidden_embed_size"],
        n_attn_heads=hparams["n_attn_heads"],
        n_encoder_blocks=hparams["n_encoder_blocks"],
        padding_tkn=hparams["pad_tkn"],
        mask_tkn=hparams["mask_tkn"],
        layer_norm_eps=hparams["layer_norm_eps"],
        a_fn=hparams["a_fn"],
    )
    model.load_state_dict(torch.load(checkpoint / "model.pt", map_location="cpu"))
    model.eval()
    return model, checkpoint


def upstream_outputs(case: AbLang2Case, input_ids: torch.Tensor) -> tuple[dict[str, torch.Tensor], Path]:
    model, checkpoint = load_upstream_model(case)
    with torch.no_grad():
        logits = model(input_ids)
        reps = model.AbRep(input_ids, return_rep_layers=range(13)).many_hidden_states
    hidden_states = torch.stack([reps[index] for index in range(13)], dim=0)
    return {
        "hidden_states": hidden_states,
        "logits": remap_logits_to_vocab_subset(logits, LOGITS_REMAP),
    }, checkpoint


def generate_case(case: AbLang2Case) -> None:
    upstream_input_ids = encode_probe(UPSTREAM_VOCAB)
    input_ids = encode_probe(MM_VOCAB)
    attention_mask = input_ids.ne(MM_VOCAB.index("<pad>")).long()
    expected, _checkpoint = upstream_outputs(case, upstream_input_ids)
    target_slice = LOGITS_REMAP["target_slice"]

    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    meta = {
        "version": 1,
        "model": MODEL,
        "case": case.case,
        "auto_model": "AutoModelForPreTraining",
        "outputs": sorted(expected.keys()),
        "tolerance": {"atol": ATOL, "rtol": RTOL},
        "inputs_source": {
            "type": "synthetic_token_probe",
            "id": "ablang2-paired-antibody-probe",
        },
        "upstream": {
            "repository": UPSTREAM_REPO_URL,
            "commit": UPSTREAM_COMMIT,
            "checkpoint_source": case.checkpoint_source,
            "checkpoint_sha256": case.checkpoint_sha256,
            "target_slice": target_slice,
        },
    }
    summary = write_fixture_artifacts(
        fixture_out_dir(REPO_ROOT, MODEL, case.case),
        inputs=inputs,
        expected=expected,
        meta=meta,
    )
    print(f"Wrote {summary['path']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", nargs="*", default=sorted(CASES))
    args = parser.parse_args()
    for case_name in args.case:
        generate_case(parse_case(case_name))


if __name__ == "__main__":
    main()
