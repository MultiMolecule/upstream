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

"""Generate SPOT-RNA checkpoint-parity fixtures from the upstream TF graphs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
tf.compat.v1.disable_eager_execution()

HERE = Path(__file__).resolve()
MODEL_DIR = HERE.parent
REPO_ROOT = MODEL_DIR.parent.parent

sys.path.insert(0, str(REPO_ROOT))
from _corpus.load import crop_record, sequence_sha256  # noqa: E402
from _shared.archive import safe_extract_tar  # noqa: E402
from _shared.download import fetch_http_file, upstream_cache_root  # noqa: E402
from _shared.fixture import (  # noqa: E402
    fixture_out_dir,
    inputs_source_from_anchor_crop,
    sha256_of_file,
    write_fixture_artifacts,
)
from _shared.source_tree import ensure_source_tree  # noqa: E402

MODEL = "spotrna"
CASE = "spotrna"
UPSTREAM_REPO_URL = "https://github.com/jaswindersingh2/SPOT-RNA"
UPSTREAM_COMMIT = "dfd027727b3890d2a80aee189b2a48ca27063163"
CHECKPOINT_SOURCE = "https://www.dropbox.com/s/dsrcf460nbjqpxa/SPOT-RNA-models.tar.gz?dl=1"
CHECKPOINT_ARCHIVE_FILENAME = "SPOT-RNA-models.tar.gz"
CHECKPOINT_SHA256 = "e9e897e869f045c6d7c03b9b1f70cc3fb10524d03dc8b04633c562121c8f07fb"
CHECKPOINT_ENV_VAR = "MULTIMOLECULE_UPSTREAM_SPOTRNA_CHECKPOINT_DIR"
SOURCE_ENV_VAR = "MULTIMOLECULE_UPSTREAM_SPOTRNA_SOURCE"
CORPUS_RECORD_ID = "rna/grch38_chr21_transcribed"
CROP_NAME = "rna_50nt"
CROP_CENTER = "center"
CROP_LENGTH = 50
ATOL = 1e-4
RTOL = 1e-4

MM_BASE_INDEX = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3, "N": 4}
CHECKPOINT_FILENAMES = tuple(
    f"model{index}.{suffix}" for index in range(5) for suffix in ("data-00000-of-00001", "index", "meta")
)


def spotrna_source_root() -> Path:
    return ensure_source_tree(
        UPSTREAM_REPO_URL,
        UPSTREAM_COMMIT,
        ("utils", "SPOT-RNA.py"),
        env_var=SOURCE_ENV_VAR,
        cache_prefix="spotrna",
    )


def checkpoint_dir() -> Path:
    override = os.environ.get(CHECKPOINT_ENV_VAR)
    if override:
        directory = Path(override).expanduser().resolve()
        if not directory.is_dir():
            raise FileNotFoundError(f"SPOT-RNA checkpoint directory not found at ${CHECKPOINT_ENV_VAR}: {directory}")
        return directory

    destination = upstream_cache_root() / "spotrna" / "SPOT-RNA-models"
    if all((destination / filename).is_file() for filename in CHECKPOINT_FILENAMES):
        return destination

    archive = fetch_http_file(
        CHECKPOINT_SOURCE,
        CHECKPOINT_ARCHIVE_FILENAME,
        cache_prefix="spotrna",
        description="SPOT-RNA model ensemble archive",
    )
    safe_extract_tar(archive, destination.parent)
    return destination


def checkpoint_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("model*.*"))


def checkpoint_manifest(paths: list[Path]) -> tuple[str, dict[str, str]]:
    manifest: dict[str, str] = {}
    lines = []
    for path in paths:
        digest = sha256_of_file(path)
        manifest[path.name] = digest
        lines.append(f"{digest}  {path.name}\n")
    return hashlib.sha256("".join(lines).encode()).hexdigest(), manifest


def load_spotrna_utils(source_root: Path) -> Any:
    spec = importlib.util.spec_from_file_location("spotrna_official_utils", source_root / "utils" / "utils.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import SPOT-RNA utils.py from {source_root}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def encode_input_ids(sequence: str) -> np.ndarray:
    ids = [MM_BASE_INDEX.get(base.upper(), MM_BASE_INDEX["N"]) for base in sequence]
    return np.asarray([ids], dtype=np.int64)


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values.astype(np.float64)))


def write_tfrecord(sequence: str, path: Path, utils: Any) -> np.ndarray:
    seq_len, feature, zero_mask, label_mask, true_label = utils.get_data(sequence)
    with tf.io.TFRecordWriter(str(path)) as writer:
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "rna_name": tf.train.Feature(bytes_list=tf.train.BytesList(value=[b"parity"])),
                    "seq_len": tf.train.Feature(int64_list=tf.train.Int64List(value=[seq_len])),
                    "feature": tf.train.Feature(float_list=tf.train.FloatList(value=feature)),
                    "zero_mask": tf.train.Feature(float_list=tf.train.FloatList(value=zero_mask)),
                    "label_mask": tf.train.Feature(float_list=tf.train.FloatList(value=label_mask)),
                    "true_label": tf.train.Feature(float_list=tf.train.FloatList(value=true_label)),
                }
            )
        )
        writer.write(example.SerializeToString())
    return np.asarray(label_mask, dtype=np.float32).reshape(seq_len, seq_len)


def run_member(checkpoint_path: Path, tfrecord_path: Path) -> np.ndarray:
    tf.compat.v1.reset_default_graph()
    config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    with tf.compat.v1.Session(config=config) as sess:
        saver = tf.compat.v1.train.import_meta_graph(str(checkpoint_path) + ".meta")
        saver.restore(sess, str(checkpoint_path))
        graph = tf.compat.v1.get_default_graph()
        sess.run(
            graph.get_operation_by_name("make_initializer_2"),
            feed_dict={graph.get_tensor_by_name("tensors_2/component_0:0"): [str(tfrecord_path)]},
        )
        logits = sess.run("output_FC/fully_connected/BiasAdd:0", feed_dict={"dropout:0": 1.0})
    return sigmoid(np.asarray(logits).reshape(-1))


def upstream_contact_map(sequence: str, source_root: Path, checkpoints: Path) -> np.ndarray:
    utils = load_spotrna_utils(source_root)
    with tempfile.TemporaryDirectory(prefix="spotrna-fixture-") as tmpdir:
        tfrecord_path = Path(tmpdir) / "parity.tfrecords"
        label_mask = write_tfrecord(sequence, tfrecord_path, utils)
        flat_probabilities = []
        for member_index in range(5):
            flat_probabilities.append(run_member(checkpoints / f"model{member_index}", tfrecord_path))

    probabilities = np.mean(np.stack(flat_probabilities, axis=0), axis=0).astype(np.float32)
    indices = np.where(label_mask == 1)
    if probabilities.shape[0] != indices[0].shape[0]:
        raise AssertionError(
            f"SPOT-RNA output length {probabilities.shape[0]} != label mask count {indices[0].shape[0]}"
        )
    contact_map = np.zeros(label_mask.shape, dtype=np.float32)
    contact_map[indices] = probabilities
    return np.ascontiguousarray(contact_map[None, ...], dtype=np.float32)


def main_for_case(case: str) -> None:
    if case != CASE:
        raise SystemExit(f"Unsupported SPOT-RNA case {case!r}; expected {CASE!r}")

    source_root = spotrna_source_root()
    checkpoints = checkpoint_dir()
    paths = checkpoint_files(checkpoints)
    if len(paths) != 15:
        raise FileNotFoundError(f"{checkpoints}: expected 15 SPOT-RNA checkpoint files, found {len(paths)}")
    checkpoint_sha256, manifest = checkpoint_manifest(paths)
    if checkpoint_sha256 != CHECKPOINT_SHA256:
        raise AssertionError(f"{checkpoints}: manifest sha256 {checkpoint_sha256} != {CHECKPOINT_SHA256}")

    crop = crop_record(CORPUS_RECORD_ID, CROP_LENGTH, center=CROP_CENTER)
    sequence = crop["sequence"].upper().replace("T", "U")
    if len(sequence) != CROP_LENGTH:
        raise AssertionError(f"crop length {len(sequence)} != {CROP_LENGTH}")
    if sequence_sha256(sequence) != crop["sha256"]:
        raise AssertionError("crop sha256 mismatch")

    input_ids = encode_input_ids(sequence)
    contact_map = upstream_contact_map(sequence, source_root, checkpoints)

    out_dir = fixture_out_dir(REPO_ROOT, MODEL, CASE)
    meta = {
        "version": 1,
        "model": MODEL,
        "case": CASE,
        "auto_model": "AutoModelForRnaSecondaryStructurePrediction",
        "outputs": ["contact_map"],
        "tolerance": {"atol": ATOL, "rtol": RTOL},
        "inputs_source": inputs_source_from_anchor_crop(
            crop,
            crop_name=CROP_NAME,
        ),
        "upstream": {
            "repository": UPSTREAM_REPO_URL,
            "commit": UPSTREAM_COMMIT,
            "checkpoint_source": CHECKPOINT_SOURCE,
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_manifest": manifest,
        },
    }
    write_fixture_artifacts(
        out_dir,
        inputs={"input_ids": input_ids},
        expected={"contact_map": contact_map},
        meta=meta,
    )

    print(f"Wrote fixture to {out_dir}")
    print(f"  input_ids: {tuple(input_ids.shape)}")
    print(f"  contact_map: {tuple(contact_map.shape)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", nargs="?", default=CASE, choices=[CASE])
    args = parser.parse_args()
    main_for_case(args.case)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
