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

"""Generate BPfold checkpoint-parity fixtures from the upstream PyTorch predictor."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import torch

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

MODEL = "bpfold"
CASE = "bpfold"
UPSTREAM_REPO_URL = "https://github.com/heqin-zhu/BPfold"
UPSTREAM_COMMIT = "545af8042835a0fa751d8b69cbb1b3d3d2d51579"
CHECKPOINT_SOURCE = "https://github.com/heqin-zhu/BPfold/releases/download/v0.2/model_predict.tar.gz"
CHECKPOINT_ARCHIVE_FILENAME = "model_predict.tar.gz"
CHECKPOINT_SHA256 = "51af350254b2bd3f4568b9fcaa9861b28f59568155088eae7a40902c6877d0bb"
CHECKPOINT_ENV_VAR = "MULTIMOLECULE_UPSTREAM_BPFOLD_CHECKPOINT_DIR"
SOURCE_ENV_VAR = "MULTIMOLECULE_UPSTREAM_BPFOLD_SOURCE"
CHECKPOINT_FILENAMES = tuple(f"BPfold_{index}-6.pth" for index in range(1, 7))
ENERGY_SOURCE = "upstream-file://src/BPfold/paras/key.energy"
ENERGY_SHA256 = "b657c5533bba393299c50e48e5310bd34e5f10b49fb36986a391944796bdce8b"
CORPUS_RECORD_ID = "rna/grch38_chr21_transcribed"
CROP_NAME = "rna_50nt"
CROP_CENTER = "center"
CROP_LENGTH = 50
ATOL = 1e-4
RTOL = 1e-4

MM_TOKEN_IDS = {
    "A": 6,
    "C": 7,
    "G": 8,
    "U": 9,
    "T": 9,
    "N": 10,
}


def bpfold_source_path() -> Path:
    return (
        ensure_source_tree(
            UPSTREAM_REPO_URL,
            UPSTREAM_COMMIT,
            ("src/BPfold",),
            env_var=SOURCE_ENV_VAR,
            cache_prefix="bpfold",
        )
        / "src"
    )


def checkpoint_dir() -> Path:
    override = os.environ.get(CHECKPOINT_ENV_VAR)
    if override:
        directory = Path(override).expanduser().resolve()
        if not directory.is_dir():
            raise FileNotFoundError(f"BPfold checkpoint directory not found at ${CHECKPOINT_ENV_VAR}: {directory}")
        return directory

    destination = upstream_cache_root() / "bpfold" / "model_predict"
    if all((destination / filename).is_file() for filename in CHECKPOINT_FILENAMES):
        return destination

    archive = fetch_http_file(
        CHECKPOINT_SOURCE,
        CHECKPOINT_ARCHIVE_FILENAME,
        cache_prefix="bpfold",
        description="BPfold v0.2 model_predict archive",
    )
    safe_extract_tar(archive, destination.parent)
    return destination


def checkpoint_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("BPfold_*-6.pth"))


def checkpoint_manifest(paths: list[Path]) -> tuple[str, dict[str, str]]:
    manifest: dict[str, str] = {}
    lines = []
    for path in paths:
        digest = sha256_of_file(path)
        manifest[path.name] = digest
        lines.append(f"{digest}  {path.name}\n")
    return hashlib.sha256("".join(lines).encode()).hexdigest(), manifest


def encode_input_ids(sequence: str) -> torch.Tensor:
    ids = [
        1,
        *[MM_TOKEN_IDS.get(base.upper(), MM_TOKEN_IDS["N"]) for base in sequence],
        2,
    ]
    return torch.tensor([ids], dtype=torch.long)


def upstream_logits(sequence: str, source_path: Path, checkpoints: Path) -> torch.Tensor:
    sys.path.insert(0, str(source_path))
    from BPfold.predict import BPfold_predict  # noqa: WPS433

    predictor = BPfold_predict(checkpoint_dir=str(checkpoints))
    loader = predictor.get_predict_loader(
        predictor.data_opts,
        predictor.device,
        input_seqs=[sequence],
        input_path=None,
        batch_size=1,
        num_workers=0,
        data_name="RNAseq",
    )
    for data, _ in loader:
        with torch.no_grad():
            pred_batch = torch.stack([model(data) for model in predictor.models], dim=0).mean(dim=0)
        network_length = int(data["forward_mask"].sum(dim=-1).max().item())
        base_length = network_length - 2
        logits = pred_batch[:, 1 : base_length + 1, 1 : base_length + 1].detach().cpu().contiguous()
        if logits.shape != (1, len(sequence), len(sequence)):
            raise AssertionError(f"BPfold logits shape {tuple(logits.shape)} does not match sequence length")
        return logits
    raise RuntimeError("BPfold predictor yielded no batches")


def main_for_case(case: str) -> None:
    if case != CASE:
        raise SystemExit(f"Unsupported BPfold case {case!r}; expected {CASE!r}")

    source_path = bpfold_source_path()
    checkpoints = checkpoint_dir()
    energy_path = source_path / "BPfold" / "paras" / "key.energy"
    paths = checkpoint_files(checkpoints)
    if len(paths) != 6:
        raise FileNotFoundError(f"{checkpoints}: expected 6 BPfold checkpoints, found {len(paths)}")
    checkpoint_sha256, manifest = checkpoint_manifest(paths)
    if checkpoint_sha256 != CHECKPOINT_SHA256:
        raise AssertionError(f"{checkpoints}: manifest sha256 {checkpoint_sha256} != {CHECKPOINT_SHA256}")
    energy_sha256 = sha256_of_file(energy_path)
    if energy_sha256 != ENERGY_SHA256:
        raise AssertionError(f"{energy_path}: sha256 {energy_sha256} != {ENERGY_SHA256}")

    crop = crop_record(CORPUS_RECORD_ID, CROP_LENGTH, center=CROP_CENTER)
    sequence = crop["sequence"].upper().replace("T", "U")
    if len(sequence) != CROP_LENGTH:
        raise AssertionError(f"crop length {len(sequence)} != {CROP_LENGTH}")
    if sequence_sha256(sequence) != crop["sha256"]:
        raise AssertionError("crop sha256 mismatch")

    input_ids = encode_input_ids(sequence)
    logits = upstream_logits(sequence, source_path, checkpoints)

    out_dir = fixture_out_dir(REPO_ROOT, MODEL, CASE)
    meta = {
        "version": 1,
        "model": MODEL,
        "case": CASE,
        "auto_model": "AutoModelForRnaSecondaryStructurePrediction",
        "outputs": ["logits"],
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
            "energy_source": ENERGY_SOURCE,
            "energy_sha256": energy_sha256,
        },
    }
    write_fixture_artifacts(
        out_dir,
        inputs={"input_ids": input_ids},
        expected={"logits": logits},
        meta=meta,
    )

    print(f"Wrote fixture to {out_dir}")
    print(f"  input_ids: {tuple(input_ids.shape)}")
    print(f"  logits: {tuple(logits.shape)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", nargs="?", default=CASE, choices=[CASE])
    args = parser.parse_args()
    main_for_case(args.case)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
