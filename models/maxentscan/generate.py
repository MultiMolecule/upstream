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

"""Generate MaxEntScan golden fixtures from the upstream maxentpy scorer."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve()
MODEL_DIR = HERE.parent
REPO_ROOT = MODEL_DIR.parent.parent

sys.path.insert(0, str(REPO_ROOT))

from _corpus.load import crop_record, sequence_sha256  # noqa: E402
from _shared.fixture import (  # noqa: E402
    fixture_out_dir,
    inputs_source_from_anchor_crop,
    sha256_of_file,
    write_fixture_artifacts,
)
from _shared.source_tree import ensure_source_tree  # noqa: E402

MODEL = "maxentscan"
UPSTREAM_REPO_URL = "https://github.com/kepbod/maxentpy"
UPSTREAM_COMMIT = "60b4fed4a169cc25264d83a7d50db6ba304c2104"
SOURCE_ENV_VAR = "MULTIMOLECULE_UPSTREAM_MAXENTPY_SOURCE"
# MaxEntScan emits scalar scores; strict tolerance catches table or window drift.
ATOL = 1e-5
RTOL = 1e-5

MM_BASE_IDS = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3}
CORPUS_RECORD_ID = "dna/grch38_chr21"
CROP_CENTER = "center"


@dataclass
class MaxEntScanCase:
    case: str
    mode: str
    window: int
    crop_name: str
    matrix_file: str
    matrix_sha256: str
    scorer_name: str
    matrix_loader_name: str

    def matrix_path(self, source_root: Path) -> Path:
        return source_root / "maxentpy" / "data" / self.matrix_file


CASES = {
    "maxentscan-score5": MaxEntScanCase(
        case="maxentscan-score5",
        mode="score5",
        window=9,
        crop_name="splice5_9bp",
        matrix_file="score5_matrix.txt",
        matrix_sha256="c64fa6aea3d8b71d6af69f5e3ece8f84de06f5450f6d4ba482fbbff6afcf8b5b",
        scorer_name="score5",
        matrix_loader_name="load_matrix5",
    ),
    "maxentscan-score3": MaxEntScanCase(
        case="maxentscan-score3",
        mode="score3",
        window=23,
        crop_name="splice3_23bp",
        matrix_file="score3_matrix.txt",
        matrix_sha256="9e8a74dc795ae5c9c5b611d4343a1a5a70fff863956f4e00ae23a85f21beb3ae",
        scorer_name="score3",
        matrix_loader_name="load_matrix3",
    ),
}


def encode_sequence(sequence: str) -> torch.Tensor:
    return torch.tensor([[MM_BASE_IDS[base] for base in sequence.upper()]], dtype=torch.long)


def maxentpy_root() -> Path:
    try:
        return ensure_source_tree(
            UPSTREAM_REPO_URL,
            UPSTREAM_COMMIT,
            ("maxentpy",),
            env_var=SOURCE_ENV_VAR,
            cache_prefix="maxentpy",
        )
    except Exception as exc:
        raise RuntimeError(
            "Unable to resolve the golden-pinned maxentpy upstream commit "
            f"{UPSTREAM_COMMIT} from {UPSTREAM_REPO_URL}. Keep this commit aligned "
            f"with golden metadata; set ${SOURCE_ENV_VAR} to a local checkout of that "
            "exact upstream tree if the public repository no longer serves it."
        ) from exc


def load_maxent_module() -> tuple[Path, Any]:
    source_root = maxentpy_root()
    sys.path.insert(0, str(source_root))
    from maxentpy import maxent  # noqa: WPS433

    return source_root, maxent


def main_for_case(case_name: str) -> None:
    case = CASES[case_name]
    source_root, maxent = load_maxent_module()
    matrix_path = case.matrix_path(source_root)
    matrix_sha256 = sha256_of_file(matrix_path)
    if matrix_sha256 != case.matrix_sha256:
        raise AssertionError(f"{matrix_path}: expected sha256 {case.matrix_sha256}, got {matrix_sha256}")

    crop = crop_record(CORPUS_RECORD_ID, case.window, center=CROP_CENTER)
    sequence = crop["sequence"].upper()
    if len(sequence) != case.window:
        raise AssertionError(f"crop length {len(sequence)} != {case.window}")
    if sequence_sha256(sequence) != crop["sha256"]:
        raise AssertionError("crop sha256 mismatch")

    input_ids = encode_sequence(sequence)
    matrix = getattr(maxent, case.matrix_loader_name)()
    score = getattr(maxent, case.scorer_name)(sequence, matrix=matrix)
    expected = {"logits": torch.tensor([[score]], dtype=torch.float32)}

    out_dir = fixture_out_dir(REPO_ROOT, MODEL, case.case)

    meta = {
        "version": 1,
        "model": MODEL,
        "case": case.case,
        "auto_model": "AutoModel",
        "outputs": sorted(expected.keys()),
        "tolerance": {"atol": ATOL, "rtol": RTOL},
        "inputs_source": inputs_source_from_anchor_crop(
            crop,
            crop_name=case.crop_name,
        ),
        "upstream": {
            "repository": UPSTREAM_REPO_URL,
            "commit": UPSTREAM_COMMIT,
            "checkpoint_source": f"upstream-file://maxentpy/data/{case.matrix_file}",
            "checkpoint_sha256": matrix_sha256,
        },
    }
    write_fixture_artifacts(
        out_dir,
        inputs={"input_ids": input_ids},
        expected=expected,
        meta=meta,
    )

    print(f"Wrote fixture to {out_dir}")
    print(f"  mode: {case.mode}")
    print(f"  input_ids: {tuple(input_ids.shape)}")
    print(f"  logits: {score:.8f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=sorted(CASES), help="MaxEntScan fixture case.")
    args = parser.parse_args()
    main_for_case(args.case)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
