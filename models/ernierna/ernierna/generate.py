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

"""Generate the ernierna golden fixture from the upstream ERNIE-RNA.

Run in an upstream venv (py3.10 + fairseq v0.12.2 + torch). The MultiMolecule
package cannot be imported here because its package __init__ pulls danling and
torch._dynamo, which is incompatible with the older torch fairseq expects, so
this script keeps its sequence loading and RNA vocab mapping local.

Inputs are taken from ``_corpus/rna/grch38_chr21_transcribed`` with the
``rna_265nt`` center crop. Range fetching, caching, and the T-to-U
transcription transform are delegated to ``_corpus.load``.
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
from _shared.bert_probe import (  # noqa: E402
    build_vocab_remap,
    remap_logits_to_vocab_subset,
)
from _shared.download import fetch_google_drive_file  # noqa: E402
from _shared.fixture import (  # noqa: E402
    fixture_out_dir,
    inputs_source_from_anchor_crop,
    sha256_of_file,
    write_fixture_artifacts,
)
from _shared.source_tree import ensure_source_tree  # noqa: E402

CASE = "ernierna"
MODEL = "ernierna"
# Paddle-to-PyTorch replay is stable at 1e-3 for the saved hidden-state surface.
ATOL = 1e-3
RTOL = 1e-4

UPSTREAM_REPO_URL = "https://github.com/Bruce-ywj/ERNIE-RNA"
UPSTREAM_COMMIT = "43bc06de1088ed03ffd7de918ad4b2c2a3346a43"
UPSTREAM_SOURCE_ENV_VAR = "MULTIMOLECULE_UPSTREAM_ERNIERNA_SOURCE"
UPSTREAM_CACHE_PREFIX = "ernierna"
CHECKPOINT_ENV_VAR = "MULTIMOLECULE_UPSTREAM_ERNIERNA_CHECKPOINT"
CHECKPOINT_SOURCE = "google_drive://1CmNxJxgjDhRoBdlDODFFjDNNzJMHVzmO"
CHECKPOINT_SHA256 = "50ac42d4a2af361b8a7d9fb62546e3f380c32893b179554609440339a6676851"
CORPUS_RECORD_ID = "rna/grch38_chr21_transcribed"
CROP_NAME = "rna_265nt"
CROP_CENTER = "center"
CROP_LENGTH = 265

# Upstream ERNIE-RNA vocabulary from the fairseq dictionary order.
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

# MultiMolecule RNA vocabulary used by the converted checkpoint tokenizer.
MM_VOCAB = ["<pad>", "<cls>", "<eos>", "<unk>", "<mask>", "<null>"] + list("ACGUNRYSWKMBDHVIX|.*-?")
LOGITS_REMAP = build_vocab_remap(UPSTREAM_VOCAB, MM_VOCAB, model_name=MODEL, duplicate_policy="last")


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
        ("src",),
        env_var=UPSTREAM_SOURCE_ENV_VAR,
        cache_prefix=UPSTREAM_CACHE_PREFIX,
    )


def checkpoint_path() -> Path:
    return fetch_google_drive_file(
        CHECKPOINT_SOURCE,
        "ERNIE-RNA_pretrain.pt",
        cache_prefix=MODEL,
        env_var=CHECKPOINT_ENV_VAR,
        sha256=CHECKPOINT_SHA256,
        description="ERNIE-RNA pretraining checkpoint",
    )


def upstream_forward(sequence: str) -> dict[str, torch.Tensor]:
    """Load the upstream model and return the per-layer outputs for ``sequence``."""
    import importlib

    source_root = upstream_root()
    dict_dir = source_root / "src" / "dict"
    checkpoint = checkpoint_path()
    sys.path.insert(0, str(source_root))
    sys.path.insert(0, str(source_root / "src" / "ernie_rna" / "models"))

    # Side-effect imports register the fairseq model / task / criterion classes.
    importlib.import_module("src.ernie_rna.tasks.ernie_rna")
    importlib.import_module("src.ernie_rna.models.ernie_rna")
    importlib.import_module("src.ernie_rna.criterions.ernie_rna")

    from fairseq import checkpoint_utils
    from src.utils import prepare_input_for_ernierna

    upstream_ids = tokenize(sequence, UPSTREAM_VOCAB)
    upstream_arr = np.asarray(upstream_ids, dtype=np.int64)
    one_d, two_d = prepare_input_for_ernierna(upstream_arr, len(sequence))

    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        [str(checkpoint)], arg_overrides={"data": str(dict_dir)}
    )
    model = models[0].eval()
    encoder = model.encoder

    with torch.no_grad():
        logits, attn_bias_lst, out_dict = encoder(
            one_d,
            twod_tokens=two_d,
            is_twod=True,
            extra_only=False,
            masked_only=False,
        )

    inner_states = out_dict["inner_states"]
    if len(inner_states) != 13:
        raise AssertionError(f"expected 13 inner_states (embed + 12 layers), got {len(inner_states)}")
    if len(attn_bias_lst) != 13:
        raise AssertionError(f"expected 13 attn biases (initial + 12 layers), got {len(attn_bias_lst)}")

    hidden_states = torch.stack(tuple(state.transpose(0, 1).contiguous() for state in inner_states), dim=0)
    return {
        "hidden_states": hidden_states,
        "logits": remap_logits_to_vocab_subset(logits, LOGITS_REMAP),
    }


def main() -> None:
    torch.manual_seed(0)

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
        "auto_model": "AutoModelForMaskedLM",
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
            "checkpoint_sha256": sha256_of_file(checkpoint_path()),
            "target_slice": LOGITS_REMAP["target_slice"],
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
