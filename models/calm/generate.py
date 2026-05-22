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

"""Generate CaLM golden fixtures from the official CaLM implementation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve()
MODEL_DIR = HERE.parent
REPO_ROOT = MODEL_DIR.parent.parent

sys.path.insert(0, str(REPO_ROOT))
from _shared.alphabet import multimolecule_dna_vocabulary  # noqa: E402
from _shared.bert_probe import (  # noqa: E402
    build_vocab_remap,
    remap_logits_to_vocab_subset,
)
from _shared.download import fetch_http_file  # noqa: E402
from _shared.fixture import fixture_out_dir, sha256_of_file, write_fixture_artifacts  # noqa: E402
from _shared.source_tree import ensure_source_tree  # noqa: E402

MODEL = "calm"
CASE = "calm"
UPSTREAM_REPO_URL = "https://github.com/oxpig/CaLM"
UPSTREAM_COMMIT = "24187ea44744ca548b747ad262058f3463589f8e"
UPSTREAM_SOURCE_ENV_VAR = "MULTIMOLECULE_UPSTREAM_CALM_SOURCE"
UPSTREAM_CACHE_PREFIX = "calm"
CHECKPOINT_SOURCE = "http://opig.stats.ox.ac.uk/data/downloads/calm_weights.pkl"
CHECKPOINT_FILENAME = "calm_weights.pkl"
CHECKPOINT_ENV_VAR = "MULTIMOLECULE_UPSTREAM_CALM_CHECKPOINT"
CHECKPOINT_SHA256 = "bf536f8f19a5d3c6ebbdd53357a5e49bb0b8cc6ff7173f25ed178696e084ab11"
STANDARD_CODONS = tuple(first + second + third for first in "ACGT" for second in "ACGT" for third in "ACGT")
PROBE_SEQUENCE = "".join(STANDARD_CODONS)
PROBE_ID = f"{MODEL}/{CASE}/standard_codon_probe"
ATOL = 1e-4
RTOL = 1e-4


def encode_mm(sequence: str) -> list[int]:
    """Encode with the MultiMolecule DNA codon vocabulary for fixture inputs."""

    sequence = sequence.upper().replace("U", "T")
    if len(sequence) % 3:
        raise ValueError("CaLM fixture sequence length must be divisible by 3")
    vocabulary = multimolecule_dna_vocabulary(nmers=3)
    token_to_id = {token: index for index, token in enumerate(vocabulary)}
    tokens = ["<cls>", *(sequence[index : index + 3] for index in range(0, len(sequence), 3)), "<eos>"]
    try:
        return [token_to_id[token] for token in tokens]
    except KeyError as error:
        raise ValueError(f"unsupported MultiMolecule codon token {error.args[0]!r}") from error


def load_upstream_model_and_tokens(sequence: str) -> tuple[torch.nn.Module, torch.Tensor, list[str]]:
    source_root = ensure_source_tree(
        UPSTREAM_REPO_URL,
        UPSTREAM_COMMIT,
        ("calm",),
        env_var=UPSTREAM_SOURCE_ENV_VAR,
        cache_prefix=UPSTREAM_CACHE_PREFIX,
    )
    sys.path.insert(0, str(source_root))

    from calm.pretrained import CaLM  # noqa: WPS433
    from calm.sequence import CodonSequence  # noqa: WPS433

    wrapper = CaLM(weights_file=str(checkpoint_path()))
    tokens = wrapper.tokenize(CodonSequence(sequence))
    model = wrapper.model.eval()
    official_vocab = [wrapper.alphabet.get_tok(index) for index in range(model.alphabet_size)]
    return model, tokens, official_vocab


def checkpoint_path() -> Path:
    return fetch_http_file(
        CHECKPOINT_SOURCE,
        CHECKPOINT_FILENAME,
        cache_prefix=UPSTREAM_CACHE_PREFIX,
        env_var=CHECKPOINT_ENV_VAR,
        sha256=CHECKPOINT_SHA256,
        description="CaLM official checkpoint",
    )


def write_meta(official_vocab: list[str]) -> dict[str, Any]:
    mm_vocab = multimolecule_dna_vocabulary(nmers=3)
    logits_remap = build_vocab_remap(official_vocab, mm_vocab, model_name=MODEL)
    meta: dict[str, Any] = {
        "version": 1,
        "model": MODEL,
        "case": CASE,
        "auto_model": "AutoModelForPreTraining",
        "outputs": ["hidden_states", "logits"],
        "tolerance": {"atol": ATOL, "rtol": RTOL},
        "inputs_source": {
            "type": "synthetic_token_probe",
            "id": PROBE_ID,
        },
        "upstream": {
            "repository": UPSTREAM_REPO_URL,
            "commit": UPSTREAM_COMMIT,
            "checkpoint_source": CHECKPOINT_SOURCE,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "target_slice": logits_remap["target_slice"],
        },
    }
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=[CASE])
    args = parser.parse_args()
    if args.case != CASE:
        raise ValueError(f"Unsupported CaLM case: {args.case}")
    checkpoint = checkpoint_path()
    if sha256_of_file(checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError(f"checkpoint digest mismatch: {checkpoint}")

    torch.set_grad_enabled(False)
    sequence = PROBE_SEQUENCE
    input_ids = torch.tensor([encode_mm(sequence)], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    model, upstream_input_ids, official_vocab = load_upstream_model_and_tokens(sequence)
    logits_remap = build_vocab_remap(official_vocab, multimolecule_dna_vocabulary(nmers=3), model_name=MODEL)
    with torch.no_grad():
        outputs = model(upstream_input_ids, repr_layers=list(range(model.num_layers + 1)))
    expected = {
        "hidden_states": torch.stack(
            tuple(outputs["representations"][index] for index in range(model.num_layers + 1)), dim=0
        )
        .detach()
        .cpu()
        .contiguous(),
        "logits": remap_logits_to_vocab_subset(outputs["logits"], logits_remap),
    }

    out_dir = fixture_out_dir(REPO_ROOT, MODEL, CASE)
    meta = write_meta(official_vocab)
    write_fixture_artifacts(
        out_dir,
        inputs={"input_ids": input_ids, "attention_mask": attention_mask},
        expected=expected,
        meta=meta,
    )
    print(f"Wrote fixture to {out_dir}")
    print(f"  input_ids: {tuple(input_ids.shape)}")
    print(f"  upstream_input_ids: {tuple(upstream_input_ids.shape)}")
    print(f"  expected: {{ {', '.join(f'{key}: {tuple(value.shape)}' for key, value in expected.items())} }}")


if __name__ == "__main__":
    main()
