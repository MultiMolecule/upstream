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

"""Generate UTRBERT k-mer golden fixtures from local upstream checkpoints.

The upstream 3UTRBERT checkpoints are BERT masked-LM checkpoints with
whitespace-separated overlapping RNA k-mers. The fixture intentionally does not
add BOS/EOS-equivalent tokens: the stored ``input_ids`` are exactly the
synthetic k-mer probe tokens.
"""

from __future__ import annotations

import argparse
import itertools
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertForMaskedLM

HERE = Path(__file__).resolve()
MODEL_DIR = HERE.parent
REPO_ROOT = MODEL_DIR.parent.parent  # upstream/

sys.path.insert(0, str(REPO_ROOT))
from _shared.archive import safe_extract_zip  # noqa: E402
from _shared.bert_probe import (  # noqa: E402
    bert_mlm_probe_expected,
    build_vocab_remap,
    encode_tokens,
    force_eager_attention,
)
from _shared.download import fetch_http_file, upstream_cache_root  # noqa: E402
from _shared.fixture import fixture_out_dir, sha256_of_file, write_fixture_artifacts  # noqa: E402

MODEL = "utrbert"
ATOL = 1e-4
RTOL = 1e-4

UPSTREAM_REPO_URL = "https://github.com/yangyn533/3UTRBERT"
UPSTREAM_COMMIT = "03a6fd5f7331141daf8cbc5f177491dbba90bc42"
FORWARD_PROBE_LENGTH = 64
REQUIRED_WEIGHT_FILES = ("config.json", "vocab.txt", "pytorch_model.bin")

SPECIAL_TOKEN_REMAP = {
    "[PAD]": "<pad>",
    "[CLS]": "<cls>",
    "[SEP]": "<eos>",
    "[UNK]": "<unk>",
    "[MASK]": "<mask>",
}
MM_SPECIAL_TOKENS = ("<pad>", "<cls>", "<eos>", "<unk>", "<mask>", "<null>")
NUCLEOBASES = ("A", "C", "G", "U", "N")


@dataclass
class UtrBertCase:
    case: str
    variant: str
    kmer: int
    checkpoint_sha256: str
    vocab_sha256: str
    size_bytes: int
    figshare_doi: str
    archive_url: str
    archive_filename: str

    @property
    def weights_dir(self) -> Path:
        case_token = self.case.upper().replace("-", "_")
        env_key = f"MULTIMOLECULE_UPSTREAM_{case_token}_SOURCE"
        override = os.environ.get(env_key)
        if override:
            return Path(override).expanduser().resolve()
        return ensure_utrbert_weights(self)

    @property
    def checkpoint_path(self) -> Path:
        return self.weights_dir / "pytorch_model.bin"

    @property
    def checkpoint_source(self) -> str:
        return f"figshare://{self.figshare_doi}/{self.archive_filename}#{self.kmer}-new-12w-0/pytorch_model.bin"


CASES = {
    "utrbert-6mer": UtrBertCase(
        case="utrbert-6mer",
        variant="3UTRBERT-6mer",
        kmer=6,
        checkpoint_sha256="e9f04fcdd2f7d6be8e64dc04c03c50bd278e3283ff49de0c95a663c9a8156faf",
        vocab_sha256="e7ee4a4483fd586df92054ea456c029fea7d250b4285414906b1d118b5ff2923",
        size_bytes=359229737,
        figshare_doi="10.6084/m9.figshare.22851272.v1",
        archive_url="https://ndownloader.figshare.com/files/40597961",
        archive_filename="6-new-12w-0.zip",
    ),
    "utrbert-5mer": UtrBertCase(
        case="utrbert-5mer",
        variant="3UTRBERT-5mer",
        kmer=5,
        checkpoint_sha256="ecda095cd77a892ab4f3599b8624d22a20b61c93663eaad0c81c5be3e2be2e0f",
        vocab_sha256="ef85059719284d8503c38e9c814b4e54d6d8474f479a6930905b0526adad5210",
        size_bytes=349780265,
        figshare_doi="10.6084/m9.figshare.22851191.v1",
        archive_url="https://ndownloader.figshare.com/files/40597919",
        archive_filename="5-new-12w-0.zip",
    ),
    "utrbert-4mer": UtrBertCase(
        case="utrbert-4mer",
        variant="3UTRBERT-4mer",
        kmer=4,
        checkpoint_sha256="8435c59bfc32a426da0da2e728de085d3eb74be5c49b4bd51848c9aee5d54d43",
        vocab_sha256="cd209e15bf117c1e1c6b8a90456007671997dd584567f361fd162079bd2bfc5b",
        size_bytes=347423195,
        figshare_doi="10.6084/m9.figshare.22851119.v1",
        archive_url="https://ndownloader.figshare.com/files/40597883",
        archive_filename="4-new-12w-0.zip",
    ),
    "utrbert-3mer": UtrBertCase(
        case="utrbert-3mer",
        variant="3UTRBERT-3mer",
        kmer=3,
        checkpoint_sha256="7aca71823ab74771006be1030d9e7239220bba40a16858575929a02e6d2a7471",
        vocab_sha256="ad559eeecdffde808141184b9ea77f9f71597e11224fc33355d0ca380dd90bb0",
        size_bytes=346827305,
        figshare_doi="10.6084/m9.figshare.22847354.v1",
        archive_url="https://ndownloader.figshare.com/files/40597877",
        archive_filename="3-new-12w-0.zip",
    ),
}


def parse_case(case: str) -> UtrBertCase:
    try:
        return CASES[case]
    except KeyError as error:
        raise ValueError(f"Unsupported UTRBERT case: {case}") from error


def has_required_files(path: Path) -> bool:
    return all((path / filename).is_file() for filename in REQUIRED_WEIGHT_FILES)


def ensure_utrbert_weights(case: UtrBertCase) -> Path:
    output_dir = upstream_cache_root() / "utrbert" / case.variant / f"{case.kmer}-new-12w-0"
    if has_required_files(output_dir):
        return output_dir

    archive_path = fetch_http_file(
        case.archive_url,
        case.archive_filename,
        cache_prefix=f"utrbert/{case.case}",
        env_var=f"MULTIMOLECULE_UPSTREAM_{case.case.upper().replace('-', '_')}_ARCHIVE",
        description=f"3UTRBERT {case.kmer}-mer Figshare archive",
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{case.case}-", dir=output_dir.parent) as tmp:
        tmp_dir = Path(tmp)
        safe_extract_zip(archive_path, tmp_dir)
        extracted = tmp_dir / f"{case.kmer}-new-12w-0"
        if not has_required_files(extracted):
            extracted = tmp_dir
        if not has_required_files(extracted):
            raise FileNotFoundError(f"3UTRBERT archive missing {case.kmer}-new-12w-0")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.copytree(extracted, output_dir)
    return output_dir


def read_vocab(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def remap_vocab(vocab: list[str]) -> list[str]:
    remapped = []
    for token in vocab:
        token = SPECIAL_TOKEN_REMAP.get(token, token)
        if token == "T":
            token = "U"
        remapped.append(token)
    return remapped


def multimolecule_vocab(kmer: int) -> list[str]:
    kmers = ("".join(parts) for parts in itertools.product(NUCLEOBASES, repeat=kmer))
    return [*MM_SPECIAL_TOKENS, *kmers]


def forward_probe_tokens(kmer: int) -> list[str]:
    kmers = ["".join(parts) for parts in itertools.product(("A", "C", "G", "U"), repeat=kmer)]
    middle_count = FORWARD_PROBE_LENGTH - 2
    return ["<mask>", "<unk>", *kmers[:middle_count]]


def load_upstream_model(case: UtrBertCase) -> BertForMaskedLM:
    if not case.checkpoint_path.is_file():
        raise FileNotFoundError(f"UTRBERT checkpoint not found: {case.checkpoint_path}")
    config = force_eager_attention(BertConfig.from_pretrained(str(case.weights_dir)))
    model = BertForMaskedLM(config)
    state_dict = torch.load(case.checkpoint_path, map_location=torch.device("cpu"))
    load_result = model.load_state_dict(state_dict, strict=False)
    allowed_unexpected = {"bert.pooler.dense.weight", "bert.pooler.dense.bias"}
    unexpected = [key for key in load_result.unexpected_keys if key not in allowed_unexpected]
    if load_result.missing_keys or unexpected:
        raise RuntimeError(
            "UTRBERT state dict did not load exactly: " f"missing={load_result.missing_keys}, unexpected={unexpected}"
        )
    return model.eval()


def main_for_case(case_name: str) -> None:
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    case = parse_case(case_name)

    vocab_path = case.weights_dir / "vocab.txt"
    if not vocab_path.is_file():
        raise FileNotFoundError(f"UTRBERT vocab not found: {vocab_path}")
    checkpoint_sha256 = sha256_of_file(case.checkpoint_path)
    if checkpoint_sha256 != case.checkpoint_sha256:
        raise AssertionError(
            f"{case.checkpoint_path}: expected sha256 {case.checkpoint_sha256}, got {checkpoint_sha256}"
        )
    vocab_sha256 = sha256_of_file(vocab_path)
    if vocab_sha256 != case.vocab_sha256:
        raise AssertionError(f"{vocab_path}: expected sha256 {case.vocab_sha256}, got {vocab_sha256}")
    if case.checkpoint_path.stat().st_size != case.size_bytes:
        raise AssertionError(f"{case.checkpoint_path}: expected {case.size_bytes} bytes")

    raw_old_vocab = read_vocab(vocab_path)
    old_vocab = remap_vocab(raw_old_vocab)
    new_vocab = multimolecule_vocab(case.kmer)
    vocab_remap = build_vocab_remap(old_vocab, new_vocab, model_name="UTRBERT")

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
    print("  add_special_tokens: False")
    print(f"  shapes: {summary}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=sorted(CASES), help="UTRBERT fixture case.")
    args = parser.parse_args()
    main_for_case(args.case)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
