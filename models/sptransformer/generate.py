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

"""Generate SpTransformer golden fixtures from the upstream PyTorch checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
MODEL_DIR = HERE.parent
REPO_ROOT = MODEL_DIR.parent.parent

sys.path.insert(0, str(REPO_ROOT))
from _corpus.load import crop_variant_reference  # noqa: E402
from _shared.download import fetch_google_drive_file  # noqa: E402
from _shared.fixture import (  # noqa: E402
    fixture_out_dir,
    inputs_source_from_variant_pair,
    sha256_of_file,
    write_fixture_artifacts,
)
from _shared.source_tree import ensure_source_tree  # noqa: E402
from _shared.variant import dna_one_hot_with_context, encode_dna_ids  # noqa: E402

MODEL = "sptransformer"
CASE = "sptransformer"
CORPUS_RECORD_ID = "dna/grch38_chr21_synthetic_variant"
CROP_NAME = "variant_400bp"
CROP_CENTER = "variant"
CROP_LENGTH = 400
UPSTREAM_REPO_URL = "https://github.com/ShenLab-Genomics/SpliceTransformer"
UPSTREAM_COMMIT = "b67a51dabf27e2980331cec197e4396513c0b34c"
CHECKPOINT_SOURCE = "google_drive://1d8n4vHDSbXqpPc_JFEswLomSUDBgHvno/SpTransformer_pytorch.ckpt"
CHECKPOINT_SHA256 = "bf9937c234139f850a770f84a26c65213a80d696d82426949a432ed1ccb9aa4e"
CHECKPOINT_ENV_VAR = "MULTIMOLECULE_UPSTREAM_SPTRANSFORMER_CHECKPOINT"
SOURCE_ENV_VAR = "MULTIMOLECULE_UPSTREAM_SPTRANSFORMER_SOURCE"
# SpliceTransformer output is a small local splice-effect delta.
ATOL = 1e-5
RTOL = 1e-5
CONTEXT = 4000


def sptransformer_source_root() -> Path:
    return ensure_source_tree(
        UPSTREAM_REPO_URL,
        UPSTREAM_COMMIT,
        ("model", "sptransformer.py", "custom_usage.py", "README.md"),
        env_var=SOURCE_ENV_VAR,
        cache_prefix="sptransformer",
    )


def checkpoint_path() -> Path:
    return fetch_google_drive_file(
        CHECKPOINT_SOURCE,
        "SpTransformer_pytorch.ckpt",
        cache_prefix=MODEL,
        env_var=CHECKPOINT_ENV_VAR,
        sha256=CHECKPOINT_SHA256,
        description="SpTransformer official checkpoint",
    )


def one_hot_with_context(sequence: str) -> torch.Tensor:
    return dna_one_hot_with_context(sequence, left_context=CONTEXT, right_context=CONTEXT)


def load_upstream_model(source_root: Path, checkpoint_path: Path) -> torch.nn.Module:
    sys.path.insert(0, str(source_root))
    from model.model import SpTransformer  # noqa: WPS433

    model = SpTransformer(
        128,
        context_len=CONTEXT,
        tissue_num=15,
        max_seq_len=8192,
        attn_depth=8,
        training=False,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = dict(checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint)
    # The local axial-positional-embedding package exposes weights as a ParameterList
    # (`weights.0`/`weights.1`), while the archived checkpoint uses older key names.
    for old, new in (
        ("attn.pos_emb.weights_0", "attn.pos_emb.weights.0"),
        ("attn.pos_emb.weights_1", "attn.pos_emb.weights.1"),
    ):
        if old in state_dict and new not in state_dict:
            state_dict[new] = state_dict.pop(old)
    model.load_state_dict(state_dict)
    return model.eval()


def upstream_forward(sequence: str, source_root: Path, checkpoint_path: Path) -> torch.Tensor:
    model = load_upstream_model(source_root, checkpoint_path)
    inputs = one_hot_with_context(sequence)
    with torch.no_grad():
        return model(inputs).transpose(1, 2).contiguous()


def main_for_case(case_name: str) -> None:
    if case_name != CASE:
        raise ValueError(f"Unsupported SpTransformer fixture case: {case_name}")
    source_root = sptransformer_source_root()
    checkpoint = checkpoint_path()
    checkpoint_sha256 = sha256_of_file(checkpoint)
    if checkpoint_sha256 != CHECKPOINT_SHA256:
        raise AssertionError(f"{checkpoint}: expected sha256 {CHECKPOINT_SHA256}, got {checkpoint_sha256}")

    crop, sequence, _ = crop_variant_reference(
        CORPUS_RECORD_ID,
        CROP_LENGTH,
        center=CROP_CENTER,
    )
    inputs = {"input_ids": encode_dna_ids(sequence)}
    expected = {"logits": upstream_forward(sequence, source_root, checkpoint)}

    out_dir = fixture_out_dir(REPO_ROOT, MODEL, CASE)

    meta = {
        "version": 1,
        "model": MODEL,
        "case": CASE,
        "auto_model": "AutoModel",
        "outputs": sorted(expected.keys()),
        "tolerance": {"atol": ATOL, "rtol": RTOL},
        "inputs_source": inputs_source_from_variant_pair(
            crop,
            crop_name=CROP_NAME,
        ),
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
    print(f"Wrote fixture to {out_dir}")
    print(f"  input_ids: {tuple(inputs['input_ids'].shape)}")
    print(f"  logits: {tuple(expected['logits'].shape)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=[CASE], help="SpTransformer fixture case.")
    args = parser.parse_args()
    main_for_case(args.case)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
