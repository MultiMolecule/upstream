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

"""Generate the ernierna-ss golden fixture from upstream ERNIE-RNA SS code.

Run this inside the existing ERNIE-RNA runtime image. The generator mounts the
workspace and reads the local fine-tuned secondary-structure checkpoint instead
of downloading another copy.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
CASE_DIR = HERE.parent
MODEL_DIR = CASE_DIR.parent
REPO_ROOT = MODEL_DIR.parent.parent  # upstream/

sys.path.insert(0, str(REPO_ROOT))
from _corpus.load import crop_record  # noqa: E402
from _shared.download import fetch_google_drive_file  # noqa: E402
from _shared.fixture import (  # noqa: E402
    fixture_out_dir,
    inputs_source_from_anchor_crop,
    sha256_of_file,
    write_fixture_artifacts,
)
from _shared.source_tree import ensure_source_tree  # noqa: E402

CASE = "ernierna-ss"
MODEL = "ernierna"
# Secondary-structure head replay spans Paddle and PyTorch numerics.
ATOL = 2e-3
RTOL = 1e-4

UPSTREAM_REPO_URL = "https://github.com/Bruce-ywj/ERNIE-RNA"
UPSTREAM_COMMIT = "43bc06de1088ed03ffd7de918ad4b2c2a3346a43"
UPSTREAM_SOURCE_ENV_VAR = "MULTIMOLECULE_UPSTREAM_ERNIERNA_SOURCE"
UPSTREAM_CACHE_PREFIX = "ernierna"
PRETRAIN_CHECKPOINT_ENV_VAR = "MULTIMOLECULE_UPSTREAM_ERNIERNA_PRETRAIN_CHECKPOINT"
SS_CHECKPOINT_ENV_VAR = "MULTIMOLECULE_UPSTREAM_ERNIERNA_SS_CHECKPOINT"
CHECKPOINT_SOURCE = (
    "google_drive://1tvoHc66uQ796mqlbKgJrl-MiRPQGylR5/" "ERNIE-RNA_attn-map_ss_prediction_bpRNA-1m_checkpoint.pt"
)
PRETRAIN_CHECKPOINT_SOURCE = "google_drive://1CmNxJxgjDhRoBdlDODFFjDNNzJMHVzmO"
PRETRAIN_CHECKPOINT_SHA256 = "50ac42d4a2af361b8a7d9fb62546e3f380c32893b179554609440339a6676851"
SS_CHECKPOINT_SHA256 = "c528cd8b4299fcb0a502093c617946596fb6e12b9db9090c5053e99c2e640653"
CORPUS_RECORD_ID = "rna/grch38_chr21_transcribed"
CROP_NAME = "rna_265nt"
CROP_CENTER = "center"
CROP_LENGTH = 265

UPSTREAM_VOCAB = [
    "<cls>",
    "<pad>",
    "<eos>",
    "<unk>",
    "G",
    "A",
    "U",
    "C",
    "N",
    "Y",
    "R",
    "S",
    "K",
    "W",
    "M",
    "D",
    "H",
    "V",
    "B",
    "X",
    "I",
    "<null>",
    "<null>",
    "<null>",
    "<mask>",
]
MM_VOCAB = ["<pad>", "<cls>", "<eos>", "<unk>", "<mask>", "<null>"] + list("ACGUNRYSWKMBDHVIX|.*-?")


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.upper().encode()).hexdigest()


def tokenize(sequence: str, vocab: list[str]) -> list[int]:
    cls_id = vocab.index("<cls>")
    eos_id = vocab.index("<eos>")
    unk_id = vocab.index("<unk>")
    ids = [cls_id]
    for ch in sequence.upper():
        ids.append(vocab.index(ch) if ch in vocab else unk_id)
    ids.append(eos_id)
    return ids


def upstream_root() -> Path:
    return ensure_source_tree(
        UPSTREAM_REPO_URL,
        UPSTREAM_COMMIT,
        ("src", "predict_ss_rna.py"),
        env_var=UPSTREAM_SOURCE_ENV_VAR,
        cache_prefix=UPSTREAM_CACHE_PREFIX,
    )


def required_checkpoint(env_var: str, source: str, filename: str) -> Path:
    expected_sha256 = PRETRAIN_CHECKPOINT_SHA256 if env_var == PRETRAIN_CHECKPOINT_ENV_VAR else SS_CHECKPOINT_SHA256
    return fetch_google_drive_file(
        source,
        filename,
        cache_prefix=MODEL,
        env_var=env_var,
        sha256=expected_sha256,
        description=f"ERNIE-RNA checkpoint {filename}",
    )


def load_upstream_ss_model() -> torch.nn.Module:
    source_root = upstream_root()
    dict_dir = source_root / "src" / "dict"
    pretrain_checkpoint = required_checkpoint(
        PRETRAIN_CHECKPOINT_ENV_VAR,
        PRETRAIN_CHECKPOINT_SOURCE,
        "ERNIE-RNA_pretrain.pt",
    )
    ss_checkpoint = required_checkpoint(
        SS_CHECKPOINT_ENV_VAR,
        CHECKPOINT_SOURCE,
        "ERNIE-RNA_attn-map_ss_prediction_bpRNA-1m_checkpoint.pt",
    )
    sys.path.insert(0, str(source_root))
    from src.utils import ChooseModel, load_pretrained_ernierna

    pretrained = load_pretrained_ernierna(str(pretrain_checkpoint), {"data": str(dict_dir)})
    model = ChooseModel(pretrained.encoder)
    state_dict = torch.load(ss_checkpoint, map_location=torch.device("cpu"))
    state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    load_result = model.load_state_dict(state_dict, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            "ERNIE-RNA SS state dict did not load exactly: "
            f"missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}"
        )
    return model.eval()


def upstream_forward(sequence: str) -> dict[str, torch.Tensor]:
    source_root = upstream_root()
    sys.path.insert(0, str(source_root))
    from predict_ss_rna import post_process, seq_to_rnaindex_and_onehot
    from src.utils import prepare_input_for_ernierna

    upstream_ids = tokenize(sequence, UPSTREAM_VOCAB)
    upstream_arr = np.asarray(upstream_ids, dtype=np.int64)
    one_d, two_d = prepare_input_for_ernierna(upstream_arr, len(sequence))
    _, data_seq = seq_to_rnaindex_and_onehot(sequence)

    model = load_upstream_ss_model()
    with torch.no_grad():
        raw_logits = model(one_d, two_d).squeeze(1)
        contact_map = post_process(raw_logits, data_seq, 0.01, 0.1, 100, 1.6, True, 1.5)
        contact_map = contact_map.clamp(0, 1)
        contact_map = contact_map.unsqueeze(-1).contiguous()
    return {"contact_map": contact_map}


def main() -> None:
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    pretrain_checkpoint = required_checkpoint(
        PRETRAIN_CHECKPOINT_ENV_VAR,
        PRETRAIN_CHECKPOINT_SOURCE,
        "ERNIE-RNA_pretrain.pt",
    )
    ss_checkpoint = required_checkpoint(
        SS_CHECKPOINT_ENV_VAR,
        CHECKPOINT_SOURCE,
        "ERNIE-RNA_attn-map_ss_prediction_bpRNA-1m_checkpoint.pt",
    )

    crop = crop_record(CORPUS_RECORD_ID, CROP_LENGTH, center=CROP_CENTER)
    cropped = crop["sequence"]
    if len(cropped) != CROP_LENGTH:
        raise AssertionError(f"crop length {len(cropped)} != {CROP_LENGTH}")
    if sequence_sha256(cropped) != crop["sha256"]:
        raise AssertionError("crop sha256 mismatch")

    mm_ids = tokenize(cropped, MM_VOCAB)
    if len(mm_ids) != CROP_LENGTH + 2:
        raise AssertionError(f"tokenized length {len(mm_ids)} != {CROP_LENGTH + 2}")

    expected = upstream_forward(cropped)

    inputs = {"input_ids": torch.tensor([mm_ids], dtype=torch.long)}
    out_dir = fixture_out_dir(REPO_ROOT, MODEL, CASE)
    meta = {
        "version": 1,
        "model": MODEL,
        "case": CASE,
        "auto_model": "AutoModelForRnaSecondaryStructurePrediction",
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
            "checkpoint_sha256": sha256_of_file(ss_checkpoint),
            "pretrain_checkpoint_source": PRETRAIN_CHECKPOINT_SOURCE,
            "pretrain_checkpoint_sha256": sha256_of_file(pretrain_checkpoint),
            "post_process": {
                "bounded_adjacency": True,
                "lr_min": 0.01,
                "lr_max": 0.1,
                "num_iters": 100,
                "sparsity": 1.6,
                "threshold": 1.5,
            },
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
    print(f"  shapes: {summary}")


if __name__ == "__main__":
    main()
