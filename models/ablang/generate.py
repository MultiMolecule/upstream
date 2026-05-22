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

"""Generate AbLang v1 golden fixtures from the upstream implementation."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import types
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
from _shared.fixture import fixture_out_dir, sha256_of_file, write_fixture_artifacts  # noqa: E402
from _shared.source_tree import ensure_source_tree  # noqa: E402

MODEL = "ablang"
UPSTREAM_REPO_URL = "https://github.com/oxpig/AbLang"
UPSTREAM_COMMIT = "8f901e99fd0ec6cedf12204f89b1de2cc76edbd4"
SOURCE_ENV_VAR = "MULTIMOLECULE_UPSTREAM_ABLANG_SOURCE"
PROBE_SEQUENCE = "QVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWV"

MM_VOCAB = ["<pad>", "<cls>", "<eos>", "<unk>", "<mask>", "<null>"] + list("ACDEFGHIKLMNPQRSTVWYXZBJUO|.*-?")


@dataclass(frozen=True)
class AbLangCase:
    case: str
    chain: str
    checkpoint_source: str
    checkpoint_sha256: str

    @property
    def env_prefix(self) -> str:
        return f"MULTIMOLECULE_UPSTREAM_{self.case.upper().replace('-', '_')}"

    @property
    def checkpoint_dir(self) -> Path:
        dir_env_var = f"{self.env_prefix}_CHECKPOINT_DIR"
        override = os.environ.get(dir_env_var)
        if override:
            return Path(override).expanduser().resolve()
        archive = fetch_http_file(
            self.checkpoint_source,
            f"{self.case}.tar.gz",
            cache_prefix=MODEL,
            env_var=f"{self.env_prefix}_CHECKPOINT_ARCHIVE",
            sha256=self.checkpoint_sha256,
            description=f"AbLang v1 {self.chain} checkpoint archive",
        )
        destination = upstream_cache_root() / MODEL / self.case
        required_files = ("amodel.pt", "hparams.json", "vocab.json")
        if not all((destination / filename).is_file() for filename in required_files):
            safe_extract_tar(archive, destination)
        return destination

    @property
    def checkpoint_archive(self) -> Path | None:
        dir_env_var = f"{self.env_prefix}_CHECKPOINT_DIR"
        if os.environ.get(dir_env_var):
            return None
        return fetch_http_file(
            self.checkpoint_source,
            f"{self.case}.tar.gz",
            cache_prefix=MODEL,
            env_var=f"{self.env_prefix}_CHECKPOINT_ARCHIVE",
            sha256=self.checkpoint_sha256,
            description=f"AbLang v1 {self.chain} checkpoint archive",
        )


CASES = {
    "ablang-heavy": AbLangCase(
        case="ablang-heavy",
        chain="heavy",
        checkpoint_source="https://opig.stats.ox.ac.uk/data/downloads/ablang-heavy.tar.gz",
        checkpoint_sha256="7c939d690cadbd2464901f80f58bf7018a6cd2b7edd73ff2ba6eb29f97ffa4f7",
    ),
    "ablang-light": AbLangCase(
        case="ablang-light",
        chain="light",
        checkpoint_source="https://opig.stats.ox.ac.uk/data/downloads/ablang-light.tar.gz",
        checkpoint_sha256="b5310bc75903feb781ef88adf3003f0f65595755755afa56b81c41a957c50cc9",
    ),
}


def parse_case(case: str) -> AbLangCase:
    try:
        return CASES[case]
    except KeyError as error:
        raise ValueError(f"Unsupported AbLang case: {case}") from error


def source_dir() -> Path:
    return ensure_source_tree(
        UPSTREAM_REPO_URL,
        UPSTREAM_COMMIT,
        ("ablang",),
        env_var=SOURCE_ENV_VAR,
        cache_prefix=MODEL,
    )


def upstream_vocab(weights_dir: Path) -> list[str]:
    with open(weights_dir / "vocab.json", encoding="utf-8") as handle:
        vocab = json.load(handle)
    by_index = [None] * len(vocab)
    for token, index in vocab.items():
        mapped = {"<": "<cls>", ">": "<eos>", "-": "<pad>", "*": "<mask>"}.get(token, token)
        by_index[index] = mapped
    if any(token is None for token in by_index):
        raise ValueError(f"{weights_dir / 'vocab.json'} has missing vocabulary ids")
    return by_index


def encode(sequence: str, vocab: list[str]) -> torch.Tensor:
    ids = [vocab.index("<cls>")]
    ids.extend(vocab.index(token) for token in sequence)
    ids.append(vocab.index("<eos>"))
    return torch.tensor([ids], dtype=torch.long)


def load_upstream_model(case: AbLangCase):
    source = source_dir()
    if not source.is_dir():
        raise FileNotFoundError(f"AbLang source checkout not found at {source}.")
    weights_dir = case.checkpoint_dir
    if not weights_dir.is_dir():
        raise FileNotFoundError(f"AbLang weights not found at {weights_dir}.")
    package = types.ModuleType("ablang")
    package.__path__ = [str(source / "ablang")]
    sys.modules["ablang"] = package
    ablang_model = importlib.import_module("ablang.model")

    with open(weights_dir / "hparams.json", encoding="utf-8") as handle:
        hparams = argparse.Namespace(**json.load(handle))
    model = ablang_model.AbLang(hparams)
    try:
        state_dict = torch.load(weights_dir / "amodel.pt", map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(weights_dir / "amodel.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def generate_case(case: AbLangCase) -> None:
    weights_dir = case.checkpoint_dir
    archive = case.checkpoint_archive
    old_vocab = upstream_vocab(weights_dir)
    remap = build_vocab_remap(old_vocab, MM_VOCAB, model_name=MODEL)
    upstream_input_ids = encode(PROBE_SEQUENCE, old_vocab)
    mm_input_ids = encode(PROBE_SEQUENCE, MM_VOCAB)
    attention_mask = torch.ones_like(mm_input_ids)

    model = load_upstream_model(case)
    with torch.no_grad():
        logits = model(upstream_input_ids)

    expected = {
        "logits": remap_logits_to_vocab_subset(logits, remap),
    }
    inputs = {
        "input_ids": mm_input_ids,
        "attention_mask": attention_mask,
    }
    meta = {
        "version": 1,
        "model": MODEL,
        "case": case.case,
        "auto_model": "AutoModelForPreTraining",
        "outputs": ["logits"],
        "tolerance": {"atol": 1e-4, "rtol": 1e-4},
        "inputs_source": {
            "type": "synthetic_token_probe",
            "id": f"ablang-v1-{case.chain}-antibody-probe",
        },
        "upstream": {
            "repository": UPSTREAM_REPO_URL,
            "commit": UPSTREAM_COMMIT,
            "checkpoint_source": case.checkpoint_source,
            "checkpoint_sha256": sha256_of_file(archive) if archive is not None else case.checkpoint_sha256,
            "target_slice": remap["target_slice"],
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
