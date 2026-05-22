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

"""Generate old DNABERT k-mer golden fixtures from local HF checkpoints.

The original DNABERT tokenizer expects whitespace-separated overlapping k-mers,
for example ``ACGT`` with ``k=3`` becomes ``ACG CGT``. This generator performs
that k-merization explicitly and uses standard Transformers ``BertForMaskedLM``
to compute the upstream forward pass. No MultiMolecule model implementation is
imported for expected outputs.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertForMaskedLM

HERE = Path(__file__).resolve()
MODEL_DIR = HERE.parent
REPO_ROOT = MODEL_DIR.parent.parent  # upstream/

sys.path.insert(0, str(REPO_ROOT))
from _shared.bert_probe import (  # noqa: E402
    bert_mlm_probe_expected,
    build_vocab_remap,
    encode_tokens,
    force_eager_attention,
)
from _shared.fixture import (  # noqa: E402
    fixture_out_dir,
    sha256_of_file,
    write_fixture_artifacts,
)
from _shared.huggingface import hf_snapshot_dir  # noqa: E402

MODEL = "dnabert"
ATOL = 1e-4
RTOL = 1e-4

UPSTREAM_REPO_URL = "https://github.com/jerryji1993/DNABERT"
UPSTREAM_COMMIT = "b6da04ec9a7d4e53efe5b33a6ce1a21c0e7ac413"
FORWARD_PROBE_LENGTH = 64

SPECIAL_TOKEN_REMAP = {
    "[PAD]": "<pad>",
    "[CLS]": "<cls>",
    "[SEP]": "<eos>",
    "[UNK]": "<unk>",
    "[MASK]": "<mask>",
}
MM_SPECIAL_TOKENS = ("<pad>", "<cls>", "<eos>", "<unk>", "<mask>", "<null>")
NUCLEOBASES = ("A", "C", "G", "T")
HF_ALLOW_PATTERNS = ("config.json", "vocab.txt", "pytorch_model.bin")


@dataclass
class DnaBertCase:
    case: str
    variant: str
    kmer: int
    huggingface_revision: str
    checkpoint_sha256: str
    vocab_sha256: str

    @property
    def weights_dir(self) -> Path:
        case_token = self.case.upper().replace("-", "_")
        env_key = f"MULTIMOLECULE_UPSTREAM_{case_token}_SOURCE"
        return hf_snapshot_dir(
            f"zhihan1996/{self.variant}",
            revision=self.huggingface_revision,
            allow_patterns=HF_ALLOW_PATTERNS,
            env_var=env_key,
        )

    @property
    def checkpoint_path(self) -> Path:
        return self.weights_dir / "pytorch_model.bin"

    @property
    def checkpoint_source(self) -> str:
        return f"huggingface://zhihan1996/{self.variant}@{self.huggingface_revision}/pytorch_model.bin"


CASES = {
    "dnabert-6mer": DnaBertCase(
        case="dnabert-6mer",
        variant="DNA_bert_6",
        kmer=6,
        huggingface_revision="c56e67ea5827e0ddc67ef059addcf71569b1216e",
        checkpoint_sha256="e1688c3a3f881daff02a42b487e04612c4ec9ef33f1f02fa5f4e67d7b9865ca9",
        vocab_sha256="d8f61c247fc10f9b8c7f6baf4d31876f30076d4aaf9f31ff47b9e68badd62e61",
    ),
    "dnabert-3mer": DnaBertCase(
        case="dnabert-3mer",
        variant="DNA_bert_3",
        kmer=3,
        huggingface_revision="6531a531c2fba495bc9bf4ec6518f99ed0032fb1",
        checkpoint_sha256="8b20083ff1a993e022a89c09c2f7047102472fe5232c3815ee50b0b0359032ba",
        vocab_sha256="2b32fac2e2451a372dacc78c6953827052b8fdc70d9d7d3be7bc96016ae855f9",
    ),
    "dnabert-5mer": DnaBertCase(
        case="dnabert-5mer",
        variant="DNA_bert_5",
        kmer=5,
        huggingface_revision="4c35c45a815a80e02fd44b9dcd69d32bcef90d5e",
        checkpoint_sha256="232c27b78c04f9770973a018ad753d9ad182947b6b6194c4844da80e60f39de1",
        vocab_sha256="cb26cec621b5f135b11b043d52dff6d0366dc8ca290bafbab75fe38826055e9d",
    ),
    "dnabert-4mer": DnaBertCase(
        case="dnabert-4mer",
        variant="DNA_bert_4",
        kmer=4,
        huggingface_revision="1a4946417c1e08e588616bf9c8f999a543bb2c8b",
        checkpoint_sha256="c2914d5fe6ab35d7981d92f4eb9382e87d7caf20d0573651cdf82d5b0daf2d00",
        vocab_sha256="cbe4efeb5db2d70beca1ed612c2b9ef1ab9e762531c12bbed3f21c1dddea8f2e",
    ),
}


def parse_case(case: str) -> DnaBertCase:
    try:
        return CASES[case]
    except KeyError as error:
        raise ValueError(f"Unsupported DNABERT case: {case}") from error


def read_vocab(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def remap_vocab(vocab: list[str]) -> list[str]:
    return [SPECIAL_TOKEN_REMAP.get(token, token) for token in vocab]


def multimolecule_vocab(kmer: int) -> list[str]:
    kmers = ("".join(parts) for parts in itertools.product(NUCLEOBASES, repeat=kmer))
    return [*MM_SPECIAL_TOKENS, *kmers]


def forward_probe_tokens(kmer: int) -> list[str]:
    kmers = ["".join(parts) for parts in itertools.product(NUCLEOBASES, repeat=kmer)]
    middle_count = FORWARD_PROBE_LENGTH - 4
    return ["<cls>", "<mask>", "<unk>", *kmers[:middle_count], "<eos>"]


def load_upstream_model(case: DnaBertCase) -> BertForMaskedLM:
    if not case.checkpoint_path.is_file():
        raise FileNotFoundError(f"DNABERT checkpoint not found: {case.checkpoint_path}")
    config = force_eager_attention(BertConfig.from_pretrained(str(case.weights_dir)))
    model = BertForMaskedLM(config)
    state_dict = torch.load(case.checkpoint_path, map_location=torch.device("cpu"))
    load_result = model.load_state_dict(state_dict, strict=False)
    allowed_unexpected = {"bert.pooler.dense.weight", "bert.pooler.dense.bias"}
    unexpected = [key for key in load_result.unexpected_keys if key not in allowed_unexpected]
    if load_result.missing_keys or unexpected:
        raise RuntimeError(
            "DNABERT state dict did not load exactly: " f"missing={load_result.missing_keys}, unexpected={unexpected}"
        )
    return model.eval()


def main_for_case(case_name: str) -> None:
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    case = parse_case(case_name)

    vocab_path = case.weights_dir / "vocab.txt"
    if not vocab_path.is_file():
        raise FileNotFoundError(f"DNABERT vocab not found: {vocab_path}")
    checkpoint_sha256 = sha256_of_file(case.checkpoint_path)
    if checkpoint_sha256 != case.checkpoint_sha256:
        raise AssertionError(
            f"{case.checkpoint_path}: expected sha256 {case.checkpoint_sha256}, got {checkpoint_sha256}"
        )
    vocab_sha256 = sha256_of_file(vocab_path)
    if vocab_sha256 != case.vocab_sha256:
        raise AssertionError(f"{vocab_path}: expected sha256 {case.vocab_sha256}, got {vocab_sha256}")

    raw_old_vocab = read_vocab(vocab_path)
    old_vocab = remap_vocab(raw_old_vocab)
    new_vocab = multimolecule_vocab(case.kmer)
    vocab_remap = build_vocab_remap(old_vocab, new_vocab, model_name="DNABERT")

    probe_tokens = forward_probe_tokens(case.kmer)
    upstream_ids = encode_tokens(probe_tokens, old_vocab)
    mm_ids = encode_tokens(probe_tokens, new_vocab)
    input_ids = torch.tensor([mm_ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    upstream_input_ids = torch.tensor([upstream_ids], dtype=torch.long)

    model = load_upstream_model(case)
    expected = bert_mlm_probe_expected(model, upstream_input_ids, attention_mask, vocab_remap)

    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    out_dir = fixture_out_dir(REPO_ROOT, MODEL, case.case)

    meta = {
        "version": 1,
        "model": MODEL,
        "case": case.case,
        "auto_model": "AutoModelForMaskedLM",
        "outputs": sorted(expected.keys()),
        "tolerance": {"atol": ATOL, "rtol": RTOL},
        "inputs_source": {
            "type": "synthetic_token_probe",
            "id": f"{MODEL}/{case.case}/checkpoint_probe",
        },
        "upstream": {
            "repository": UPSTREAM_REPO_URL,
            "commit": UPSTREAM_COMMIT,
            "checkpoint_source": case.checkpoint_source,
            "checkpoint_sha256": checkpoint_sha256,
            "target_slice": vocab_remap["target_slice"],
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
    print(f"  kmer: {case.kmer}")
    print(f"  input_ids: {tuple(input_ids.shape)}")
    print(f"  shapes: {summary}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=sorted(CASES), help="DNABERT fixture case.")
    args = parser.parse_args()
    main_for_case(args.case)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
