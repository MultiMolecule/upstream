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

"""Generate ProteinBERT golden fixtures from the upstream Keras checkpoint."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

HERE = Path(__file__).resolve()
MODEL_DIR = HERE.parent
REPO_ROOT = MODEL_DIR.parent.parent

sys.path.insert(0, str(REPO_ROOT))
from _corpus.load import crop_record, sequence_sha256  # noqa: E402
from _shared.alphabet import multimolecule_protein_vocabulary  # noqa: E402
from _shared.download import fetch_http_file  # noqa: E402
from _shared.fixture import (  # noqa: E402
    fixture_out_dir,
    inputs_source_from_anchor_crop,
    sha256_of_file,
    write_fixture_artifacts,
)
from _shared.source_tree import ensure_source_tree  # noqa: E402

MODEL = "proteinbert"
CASE = "proteinbert"
UPSTREAM_REPO_URL = "https://github.com/nadavbra/protein_bert"
UPSTREAM_COMMIT = "69a1122bd7b590af7506981fb31db06e238a882c"
CHECKPOINT_SOURCE = "https://zenodo.org/records/10371965/files/full_go_epoch_92400_sample_23500000.pkl?download=1"
CHECKPOINT_FILENAME = "full_go_epoch_92400_sample_23500000.pkl"
CHECKPOINT_SHA256 = "cffb53e237e49fa91f19a4c8bc0898dcfd75fd12eab7833d6e95b36c02ebacbf"
CHECKPOINT_ENV_VAR = "MULTIMOLECULE_UPSTREAM_PROTEINBERT_CHECKPOINT"
SOURCE_ENV_VAR = "MULTIMOLECULE_UPSTREAM_PROTEINBERT_SOURCE"
CORPUS_RECORD_ID = "protein/grch38_chr21_orf"
CROP_NAME = "protein_128aa"
CROP_CENTER = "center"
CROP_LENGTH = 128
SEQ_LEN = CROP_LENGTH + 2
NUM_BLOCKS = 6
ATOL = 1e-4
RTOL = 1e-4

UPSTREAM_VOCAB = list("ACDEFGHIKLMNPQRSTUVWXY") + ["<unk>", "<cls>", "<eos>", "<pad>"]
MM_VOCAB = list(multimolecule_protein_vocabulary())
TARGET_SLICE = [MM_VOCAB.index(token) for token in UPSTREAM_VOCAB]


def checkpoint_path() -> Path:
    return fetch_http_file(
        CHECKPOINT_SOURCE,
        CHECKPOINT_FILENAME,
        cache_prefix=MODEL,
        env_var=CHECKPOINT_ENV_VAR,
        sha256=CHECKPOINT_SHA256,
        description="ProteinBERT Keras checkpoint dump",
    )


def source_path() -> Path:
    source = ensure_source_tree(
        UPSTREAM_REPO_URL,
        UPSTREAM_COMMIT,
        ("proteinbert",),
        env_var=SOURCE_ENV_VAR,
        cache_prefix=MODEL,
        submodules=("proteinbert/shared_utils",),
    )
    util_py = source / "proteinbert" / "shared_utils" / "util.py"
    if not util_py.is_file():
        raise FileNotFoundError(f"ProteinBERT upstream shared_utils submodule is missing: {util_py}")
    return source


def encode_mm_input_ids(sequence: str) -> np.ndarray:
    unk = MM_VOCAB.index("<unk>")
    ids = [MM_VOCAB.index("<cls>")]
    ids.extend(MM_VOCAB.index(residue) if residue in MM_VOCAB else unk for residue in sequence.upper())
    ids.append(MM_VOCAB.index("<eos>"))
    return np.asarray([ids], dtype=np.int64)


def load_upstream_checkpoint(checkpoint: Path, source: Path):
    sys.path.insert(0, str(source))
    from proteinbert.conv_and_global_attention_model import create_model  # noqa: PLC0415
    from proteinbert.model_generation import load_pretrained_model_from_dump  # noqa: PLC0415

    generator, input_encoder = load_pretrained_model_from_dump(
        str(checkpoint),
        create_model,
        load_optimizer_weights=False,
    )
    return generator.create_model(SEQ_LEN, compile=False), input_encoder


def hidden_model(model: tf.keras.Model) -> tf.keras.Model:
    sequence_layers = [model.get_layer(f"seq-merge2-norm-block{index}").output for index in range(1, NUM_BLOCKS + 1)]
    global_layer = model.get_layer(f"global-merge2-norm-block{NUM_BLOCKS}").output
    return tf.keras.Model(inputs=model.inputs, outputs=[*sequence_layers, global_layer])


def dense_logits(layer: tf.keras.layers.Dense, hidden: tf.Tensor) -> np.ndarray:
    kernel, bias = layer.get_weights()
    return (tf.linalg.matmul(hidden, tf.convert_to_tensor(kernel)) + tf.convert_to_tensor(bias)).numpy()


def upstream_outputs(sequence: str, checkpoint: Path, source: Path) -> dict[str, np.ndarray]:
    model, input_encoder = load_upstream_checkpoint(checkpoint, source)
    encoded = input_encoder.encode_X([sequence], SEQ_LEN)

    outputs = hidden_model(model)(encoded, training=False)
    sequence_states = [np.asarray(output.numpy(), dtype=np.float32) for output in outputs[:-1]]
    global_state = outputs[-1]

    logits = dense_logits(model.get_layer("output-seq"), outputs[-2])
    annotation_logits = dense_logits(model.get_layer("output-annotations"), global_state)
    return {
        "hidden_states": np.ascontiguousarray(np.stack(sequence_states, axis=0), dtype=np.float32),
        "logits": np.ascontiguousarray(logits, dtype=np.float32),
        "annotation_logits": np.ascontiguousarray(annotation_logits, dtype=np.float32),
    }


def write_meta(out_dir: Path, crop: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
    meta = {
        "version": 1,
        "model": MODEL,
        "case": CASE,
        "auto_model": "AutoModelForPreTraining",
        "outputs": ["hidden_states", "logits", "annotation_logits"],
        "tolerance": {"atol": ATOL, "rtol": RTOL},
        "inputs_source": inputs_source_from_anchor_crop(
            crop,
            crop_name=CROP_NAME,
        ),
        "upstream": {
            "repository": UPSTREAM_REPO_URL,
            "commit": UPSTREAM_COMMIT,
            "checkpoint_source": CHECKPOINT_SOURCE,
            "checkpoint_sha256": sha256_of_file(checkpoint),
            "target_slice": TARGET_SLICE,
        },
    }
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", nargs="?", default=CASE, choices=(CASE,))
    args = parser.parse_args()

    checkpoint = checkpoint_path()
    source = source_path()
    crop = crop_record(CORPUS_RECORD_ID, CROP_LENGTH, center=CROP_CENTER)
    sequence = crop["sequence"].upper()
    if sequence_sha256(sequence) != crop["sha256"]:
        raise AssertionError("crop sha256 mismatch")

    input_ids = encode_mm_input_ids(sequence)
    attention_mask = np.ones_like(input_ids, dtype=np.int64)
    expected = upstream_outputs(sequence, checkpoint, source)

    out_dir = fixture_out_dir(REPO_ROOT, MODEL, args.case)
    meta = write_meta(out_dir, crop, checkpoint)
    write_fixture_artifacts(
        out_dir,
        inputs={"input_ids": input_ids, "attention_mask": attention_mask},
        expected=expected,
        meta=meta,
    )
    print(f"Wrote fixture to {out_dir}")
    print(f"  input_ids: {tuple(input_ids.shape)}")
    shapes = {key: tuple(value.shape) for key, value in expected.items()}
    print(f"  shapes: {shapes}")


if __name__ == "__main__":
    main()
