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

"""Generate SpliceBERT golden fixtures from official Zenodo checkpoints.

The upstream artifacts are standard Transformers ``BertForMaskedLM`` checkpoints
inside the official Zenodo models archive. This generator keeps the upstream
forward pass in Transformers and only reindexes original-vocab logits onto the
shared MultiMolecule vocab columns.
"""

from __future__ import annotations

import argparse
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
from _shared.archive import safe_extract_tar  # noqa: E402
from _shared.bert_probe import (  # noqa: E402
    bert_mlm_probe_expected,
    build_vocab_remap,
    encode_tokens,
    force_eager_attention,
)
from _shared.download import fetch_http_file, upstream_cache_root  # noqa: E402
from _shared.fixture import fixture_out_dir, sha256_of_file, write_fixture_artifacts  # noqa: E402

MODEL = "splicebert"
ATOL = 1e-4
RTOL = 1e-4
UPSTREAM_REPO_URL = "https://github.com/chenkenbio/SpliceBERT"
UPSTREAM_COMMIT = "dc1d8781f6f167c70421c3f8b809772637031d98"
ZENODO_MODELS_URL = "https://zenodo.org/api/records/7995778/files/models.tar.gz/content"
ZENODO_DOI = "10.5281/zenodo.7995778"
ZENODO_MODELS_ARCHIVE = "models.tar.gz"
REQUIRED_WEIGHT_FILES = ("config.json", "vocab.txt", "pytorch_model.bin")

MM_VOCAB = (
    "<pad>",
    "<cls>",
    "<eos>",
    "<unk>",
    "<mask>",
    "<null>",
    "A",
    "C",
    "G",
    "U",
    "N",
    "R",
    "Y",
    "S",
    "W",
    "K",
    "M",
    "B",
    "D",
    "H",
    "V",
    "I",
    "X",
    "|",
    ".",
    "*",
    "-",
    "?",
)


@dataclass
class SpliceBertCase:
    case: str
    variant: str
    checkpoint_sha256: str
    config_sha256: str
    vocab_sha256: str
    context_nt: int

    @property
    def weights_dir(self) -> Path:
        case_token = self.case.upper().replace(".", "_").replace("-", "_")
        env_key = f"MULTIMOLECULE_UPSTREAM_{case_token}_SOURCE"
        override = os.environ.get(env_key)
        if override:
            return Path(override).expanduser().resolve()
        return ensure_splicebert_weights(self)

    @property
    def checkpoint_path(self) -> Path:
        return self.weights_dir / "pytorch_model.bin"

    @property
    def checkpoint_source(self) -> str:
        return f"zenodo://{ZENODO_DOI}/{ZENODO_MODELS_ARCHIVE}#models/{self.variant}/pytorch_model.bin"


CASES = {
    "splicebert": SpliceBertCase(
        case="splicebert",
        variant="SpliceBERT.1024nt",
        checkpoint_sha256="2ad91428c318e6c49233154073ca7a35f5f7899c9f4be3444775bae3dba0149d",
        config_sha256="51ea51e42951f18cc1d3455d7111fe0a41cebad476ba8314afc161a0e956303e",
        vocab_sha256="ba67bacb61d3de7d57e11e04ba59020a2cc67dbedbd9ae65715117b9fd4f30bc",
        context_nt=1024,
    ),
    "splicebert.510": SpliceBertCase(
        case="splicebert.510",
        variant="SpliceBERT.510nt",
        checkpoint_sha256="b3a48c768a029dc6291cde8085c0bc02d40a6d03007b0535e6788573e2ea7808",
        config_sha256="0e0daa1f987decd945cd76bc43a3ee4686e3715c470cb08fd2bf2f5c5296b4d8",
        vocab_sha256="ba67bacb61d3de7d57e11e04ba59020a2cc67dbedbd9ae65715117b9fd4f30bc",
        context_nt=510,
    ),
    "splicebert-human.510": SpliceBertCase(
        case="splicebert-human.510",
        variant="SpliceBERT-human.510nt",
        checkpoint_sha256="7d9e7b6d53f2e42efbd19de2ad01c195244c138bff1e4468334294e3a62cae90",
        config_sha256="9f3fb3b5f5a500b0aa600b063ecb78f1aefbe05a8bd9fab1843cb5894e70f048",
        vocab_sha256="ba67bacb61d3de7d57e11e04ba59020a2cc67dbedbd9ae65715117b9fd4f30bc",
        context_nt=510,
    ),
}


SPECIAL_TOKEN_REMAP = {
    "[PAD]": "<pad>",
    "[UNK]": "<unk>",
    "[CLS]": "<cls>",
    "[SEP]": "<eos>",
    "[MASK]": "<mask>",
}


def parse_case(case: str) -> SpliceBertCase:
    try:
        return CASES[case]
    except KeyError as error:
        raise ValueError(f"Unsupported SpliceBERT case: {case}") from error


def has_required_files(path: Path) -> bool:
    return all((path / filename).is_file() for filename in REQUIRED_WEIGHT_FILES)


def ensure_splicebert_weights(case: SpliceBertCase) -> Path:
    output_dir = upstream_cache_root() / "splicebert" / "models" / case.variant
    if has_required_files(output_dir):
        return output_dir

    archive_path = fetch_http_file(
        ZENODO_MODELS_URL,
        ZENODO_MODELS_ARCHIVE,
        cache_prefix="splicebert",
        env_var="MULTIMOLECULE_UPSTREAM_SPLICEBERT_ARCHIVE",
        description="SpliceBERT Zenodo models archive",
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{case.case}-", dir=output_dir.parent) as tmp:
        tmp_dir = Path(tmp)
        safe_extract_tar(archive_path, tmp_dir)
        extracted = tmp_dir / "models" / case.variant
        if not has_required_files(extracted):
            raise FileNotFoundError(f"SpliceBERT archive missing models/{case.variant}")
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


def forward_probe_tokens(shared_tokens: list[str]) -> list[str]:
    middle = [token for token in shared_tokens if token not in {"<pad>", "<cls>", "<eos>", "<mask>", "<unk>"}]
    return ["<cls>", "<mask>", "<unk>", *middle, "<eos>"]


def load_upstream_model(case: SpliceBertCase) -> BertForMaskedLM:
    if not case.checkpoint_path.is_file():
        raise FileNotFoundError(f"SpliceBERT checkpoint not found: {case.checkpoint_path}")
    config = force_eager_attention(BertConfig.from_pretrained(str(case.weights_dir)))
    model = BertForMaskedLM(config)
    state_dict = torch.load(case.checkpoint_path, map_location=torch.device("cpu"))
    load_result = model.load_state_dict(state_dict, strict=False)
    allowed_unexpected = {"bert.embeddings.position_ids"}
    unexpected = [key for key in load_result.unexpected_keys if key not in allowed_unexpected]
    if load_result.missing_keys or unexpected:
        raise RuntimeError(
            "SpliceBERT state dict did not load exactly: "
            f"missing={load_result.missing_keys}, unexpected={unexpected}"
        )
    return model.eval()


def main_for_case(case_name: str) -> None:
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    case = parse_case(case_name)

    config_path = case.weights_dir / "config.json"
    vocab_path = case.weights_dir / "vocab.txt"
    for path in (config_path, vocab_path, case.checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    checkpoint_sha256 = sha256_of_file(case.checkpoint_path)
    config_sha256 = sha256_of_file(config_path)
    vocab_sha256 = sha256_of_file(vocab_path)
    expected_hashes = {
        case.checkpoint_path: (checkpoint_sha256, case.checkpoint_sha256),
        config_path: (config_sha256, case.config_sha256),
        vocab_path: (vocab_sha256, case.vocab_sha256),
    }
    for path, (actual, expected) in expected_hashes.items():
        if actual != expected:
            raise AssertionError(f"{path}: expected sha256 {expected}, got {actual}")

    raw_old_vocab = read_vocab(vocab_path)
    old_vocab = remap_vocab(raw_old_vocab)
    vocab_remap = build_vocab_remap(old_vocab, MM_VOCAB, model_name="SpliceBERT")
    probe_tokens = forward_probe_tokens(vocab_remap["output_tokens"])
    upstream_ids = encode_tokens(probe_tokens, old_vocab)
    mm_ids = encode_tokens(probe_tokens, list(MM_VOCAB))
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
    print(f"  variant: {case.variant}")
    print(f"  input_ids: {tuple(input_ids.shape)}")
    print(f"  shapes: {summary}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", nargs="?", choices=sorted(CASES), help="SpliceBERT fixture case.")
    args = parser.parse_args()
    if args.case is not None:
        main_for_case(args.case)
    else:
        for case_name in sorted(CASES):
            main_for_case(case_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
