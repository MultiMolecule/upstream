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

"""Shared OpenSpliceAI upstream fixture generation."""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import torch

HERE = Path(__file__).resolve()
MODEL_DIR = HERE.parent
REPO_ROOT = MODEL_DIR.parent.parent  # upstream/

sys.path.insert(0, str(REPO_ROOT))
from _corpus.load import crop_record, sequence_sha256  # noqa: E402
from _shared.fixture import (  # noqa: E402
    fixture_out_dir,
    inputs_source_from_anchor_crop,
    sha256_of_file,
    write_fixture_artifacts,
)
from _shared.source_tree import ensure_source_tree  # noqa: E402

MODEL = "openspliceai"
# SpliceAI-style fixtures compare local splice probabilities and should be tight.
ATOL = 1e-5
RTOL = 1e-5
SEED = 10
HIDDEN_SIZE = 32
UPSTREAM_REPO_URL = "https://github.com/Kuanhao-Chao/OpenSpliceAI"
UPSTREAM_COMMIT = "8f70d22bee10326630c397d2c7a9ff61f166f9cf"
SOURCE_ENV_VAR = "MULTIMOLECULE_UPSTREAM_OPENSPLICEAI_SOURCE"
SOURCE_YAML = MODEL_DIR / "source.yaml"
UPSTREAM_MODEL_SOURCE = "openspliceai/train_base"

CORPUS_RECORD_ID = "dna/grch38_chr21"
CROP_NAME = "splice_400bp"
CROP_CENTER = "center"
CROP_LENGTH = 400

CONTEXT_STAGE_SPECS = {
    80: ((4, 11, 1),),
    400: ((4, 11, 1), (4, 11, 4)),
    2000: ((4, 11, 1), (4, 11, 4), (4, 21, 10)),
    10000: ((4, 11, 1), (4, 11, 4), (4, 21, 10), (4, 41, 25)),
}
MM_RNA_VOCAB = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3, "N": 4}
UPSTREAM_DNA_VOCAB = {"A": 0, "C": 1, "G": 2, "T": 3}


@lru_cache(maxsize=1)
def source_yaml_metadata() -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - covered by generator environment checks.
        raise RuntimeError("PyYAML is required to read OpenSpliceAI source.yaml") from error

    metadata = yaml.safe_load(SOURCE_YAML.read_text())
    if not isinstance(metadata, dict):
        raise RuntimeError(f"{SOURCE_YAML}: expected a mapping")
    return metadata


def source_upstream_metadata() -> dict[str, Any]:
    upstream = source_yaml_metadata().get("upstream")
    if not isinstance(upstream, dict):
        raise RuntimeError(f"{SOURCE_YAML}: upstream must be a mapping")
    expected = {
        "repository": UPSTREAM_REPO_URL,
        "commit": UPSTREAM_COMMIT,
    }
    for key, value in expected.items():
        if upstream.get(key) != value:
            raise RuntimeError(f"{SOURCE_YAML}: upstream.{key}={upstream.get(key)!r} != {value!r}")
    return upstream


def source_checkpoint_metadata(case_name: str) -> dict[str, Any]:
    checkpoints = source_yaml_metadata().get("checkpoints")
    if not isinstance(checkpoints, dict):
        raise RuntimeError(f"{SOURCE_YAML}: checkpoints must be a mapping")
    checkpoint = checkpoints.get(case_name)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"{SOURCE_YAML}: missing checkpoint metadata for {case_name!r}")
    return checkpoint


def repository_slug(repository: str) -> str:
    parsed = urlparse(repository)
    slug = parsed.path.strip("/").removesuffix(".git")
    if not slug or "/" not in slug:
        raise ValueError(f"Unsupported GitHub repository URL: {repository}")
    return slug


def checkpoint_relative_path(source: str, repository: str, commit: str) -> Path:
    parsed = urlparse(source)
    source_path = parsed.path.lstrip("/")
    slug = repository_slug(repository)
    blob_prefix = f"{slug}/blob/{commit}/"
    raw_prefix = f"{slug}/{commit}/"
    if parsed.netloc == "github.com" and source_path.startswith(blob_prefix):
        relative = source_path[len(blob_prefix) :]
    elif parsed.netloc == "raw.githubusercontent.com" and source_path.startswith(raw_prefix):
        relative = source_path[len(raw_prefix) :]
    else:
        raise ValueError(f"Unsupported OpenSpliceAI checkpoint source URL: {source}")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe OpenSpliceAI checkpoint source path: {source}")
    return path


@lru_cache(maxsize=None)
def openspliceai_source_root(case_name: str, checkpoint_parent: str) -> Path:
    upstream = source_upstream_metadata()
    checkpoint_parents = {
        str(checkpoint_relative_path(str(checkpoint["source"]), UPSTREAM_REPO_URL, UPSTREAM_COMMIT).parent)
        for checkpoint in source_yaml_metadata()["checkpoints"].values()
    }
    return ensure_source_tree(
        str(upstream["repository"]),
        str(upstream["commit"]),
        tuple(sorted({UPSTREAM_MODEL_SOURCE, checkpoint_parent, *checkpoint_parents})),
        env_var=SOURCE_ENV_VAR,
        cache_prefix=MODEL,
    )


@dataclass
class OpenSpliceAiCase:
    case: str
    species: str
    context: int

    @property
    def checkpoint_metadata(self) -> dict[str, Any]:
        return source_checkpoint_metadata(self.case)

    @property
    def checkpoint_relative_path(self) -> Path:
        upstream = source_upstream_metadata()
        source = self.checkpoint_source
        return checkpoint_relative_path(source, str(upstream["repository"]), str(upstream["commit"]))

    @property
    def source_root(self) -> Path:
        return openspliceai_source_root(self.case, self.checkpoint_relative_path.parent.as_posix())

    @property
    def checkpoint_path(self) -> Path:
        case_token = self.case.upper().replace("-", "_")
        override = os.environ.get(f"MULTIMOLECULE_UPSTREAM_{case_token}_CHECKPOINT")
        if override:
            return Path(override).expanduser().resolve()
        return self.source_root / self.checkpoint_relative_path

    @property
    def checkpoint_source(self) -> str:
        source = self.checkpoint_metadata.get("source")
        if not isinstance(source, str):
            raise RuntimeError(f"{SOURCE_YAML}: checkpoint {self.case!r} requires a string source")
        return source

    @property
    def windows(self) -> np.ndarray:
        return np.asarray(
            [kernel for num_blocks, kernel, _ in CONTEXT_STAGE_SPECS[self.context] for _ in range(num_blocks)]
        )

    @property
    def atrous_rates(self) -> np.ndarray:
        return np.asarray(
            [rate for num_blocks, _, rate in CONTEXT_STAGE_SPECS[self.context] for _ in range(num_blocks)]
        )


def parse_case(case: str) -> OpenSpliceAiCase:
    match = re.fullmatch(r"openspliceai-(?P<species>[a-z]+)\.(?P<context>\d+)", case)
    if match is None:
        raise ValueError(f"Unsupported OpenSpliceAI case name: {case}")
    context = int(match.group("context"))
    if context not in CONTEXT_STAGE_SPECS:
        raise ValueError(f"Unsupported OpenSpliceAI context: {context}")
    return OpenSpliceAiCase(case=case, species=match.group("species"), context=context)


def tokenize_rna_streamline(sequence: str) -> torch.Tensor:
    ids = [MM_RNA_VOCAB.get(base, MM_RNA_VOCAB["N"]) for base in sequence.upper()]
    return torch.tensor([ids], dtype=torch.long)


def dna_1hot_with_context(sequence: str, context: int) -> torch.Tensor:
    padded = "N" * (context // 2) + sequence.upper() + "N" * (context // 2)
    one_hot = np.zeros((len(padded), 4), dtype=np.float32)
    for index, base in enumerate(padded):
        channel = UPSTREAM_DNA_VOCAB.get(base)
        if channel is not None:
            one_hot[index, channel] = 1.0
    return torch.from_numpy(one_hot.T[None, ...])


def load_upstream_model(case: OpenSpliceAiCase) -> torch.nn.Module:
    sys.path.insert(0, str(case.source_root))
    from openspliceai.train_base.openspliceai import SpliceAI

    model = SpliceAI(HIDDEN_SIZE, case.windows, case.atrous_rates, apply_softmax=False)
    state_dict = torch.load(case.checkpoint_path, map_location=torch.device("cpu"), weights_only=False)
    load_result = model.load_state_dict(state_dict, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            "OpenSpliceAI state dict did not load exactly: "
            f"missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}"
        )
    return model.eval()


def upstream_forward(sequence: str, case: OpenSpliceAiCase) -> dict[str, torch.Tensor]:
    model = load_upstream_model(case)
    inputs = dna_1hot_with_context(sequence, case.context)
    with torch.no_grad():
        logits = model(inputs).transpose(1, 2).contiguous()
    return {"logits": logits}


def main_for_case(case_name: str) -> None:
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    case = parse_case(case_name)
    if not case.checkpoint_path.is_file():
        raise FileNotFoundError(f"OpenSpliceAI checkpoint not found: {case.checkpoint_path}")

    crop = crop_record(CORPUS_RECORD_ID, CROP_LENGTH, center=CROP_CENTER)
    sequence = crop["sequence"]
    if len(sequence) != CROP_LENGTH:
        raise AssertionError(f"crop length {len(sequence)} != {CROP_LENGTH}")
    if sequence_sha256(sequence) != crop["sha256"]:
        raise AssertionError("crop sha256 mismatch")

    input_ids = tokenize_rna_streamline(sequence)
    attention_mask = torch.ones_like(input_ids)
    expected = upstream_forward(sequence, case)

    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    out_dir = fixture_out_dir(REPO_ROOT, MODEL, case.case)

    meta = {
        "version": 1,
        "model": MODEL,
        "case": case.case,
        "auto_model": "AutoModelForTokenPrediction",
        "outputs": sorted(expected.keys()),
        "tolerance": {"atol": ATOL, "rtol": RTOL},
        "inputs_source": inputs_source_from_anchor_crop(
            crop,
            crop_name=CROP_NAME,
        ),
        "upstream": {
            "repository": UPSTREAM_REPO_URL,
            "commit": UPSTREAM_COMMIT,
            "checkpoint_source": case.checkpoint_source,
            "checkpoint_sha256": sha256_of_file(case.checkpoint_path),
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", help="OpenSpliceAI fixture case, for example openspliceai-mane.80.")
    args = parser.parse_args()
    main_for_case(args.case)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
