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

"""Generate MTSplice golden fixtures from the upstream Keras checkpoint."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch
from tensorflow.keras.models import load_model

HERE = Path(__file__).resolve()
MODEL_DIR = HERE.parent
REPO_ROOT = MODEL_DIR.parent.parent

sys.path.insert(0, str(REPO_ROOT))
from _corpus.load import crop_variant_reference  # noqa: E402
from _shared.fixture import (  # noqa: E402
    fixture_out_dir,
    inputs_source_from_variant_pair,
    sha256_of_file,
    write_fixture_artifacts,
)
from _shared.source_tree import ensure_source_tree  # noqa: E402
from _shared.variant import (  # noqa: E402
    dna_one_hot,
    encode_dna_ids,
)

MODEL = "mtsplice"
CASE = "mtsplice"
CORPUS_RECORD_ID = "dna/grch38_chr21_synthetic_variant"
CROP_NAME = "variant_800bp"
CROP_CENTER = "variant"
CROP_LENGTH = 800
UPSTREAM_REPO_URL = "https://github.com/gagneurlab/MMSplice_MTSplice"
UPSTREAM_COMMIT = "31513da3846b187b3d7f96ad14f3c71e1177b0d3"
CHECKPOINT_SOURCE = "upstream-file://mmsplice/models/mtsplice_deep0.h5"
CHECKPOINT_SHA256 = "7f664348225cc2eaecd9b87bfc6b4903898c02a6a3a3f91cb1a4c7054db5bcdb"
SOURCE_ENV_VAR = "MULTIMOLECULE_UPSTREAM_MMSPLICE_SOURCE"
# Small splice delta outputs should stay tighter than the default neural tolerance.
ATOL = 1e-5
RTOL = 1e-5
OVERHANG = (300, 300)


def mmsplice_root() -> Path:
    return (
        ensure_source_tree(
            UPSTREAM_REPO_URL,
            UPSTREAM_COMMIT,
            ("mmsplice",),
            env_var=SOURCE_ENV_VAR,
            cache_prefix="mmsplice-mtsplice",
        )
        / "mmsplice"
    )


def load_mmsplice_layers(root: Path):
    package = types.ModuleType("mmsplice")
    package.__path__ = [str(root)]  # type: ignore[attr-defined]
    sys.modules["mmsplice"] = package
    spec = importlib.util.spec_from_file_location("mmsplice.layers", root / "layers.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import MMSplice layers from {root / 'layers.py'}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["mmsplice.layers"] = module
    spec.loader.exec_module(module)
    return module


def split_tissue_sequence(sequence: str, overhang: tuple[int, int] = OVERHANG) -> dict[str, str]:
    acceptor_intron, donor_intron = overhang
    tissue_acceptor_intron = 300
    tissue_acceptor_exon = 100
    tissue_donor_intron = 300
    tissue_donor_exon = 100

    diff_acceptor = acceptor_intron - tissue_acceptor_intron
    if diff_acceptor < 0:
        sequence = "N" * abs(diff_acceptor) + sequence
    elif diff_acceptor > 0:
        sequence = sequence[diff_acceptor:]

    diff_donor = donor_intron - tissue_donor_intron
    if diff_donor < 0:
        sequence = sequence + "N" * abs(diff_donor)
    elif diff_donor > 0:
        sequence = sequence[:-diff_donor]

    return {
        "acceptor": sequence[: tissue_acceptor_intron + tissue_acceptor_exon],
        "donor": sequence[-tissue_donor_exon - tissue_donor_intron :],
    }


def main_for_case(case_name: str) -> None:
    if case_name != CASE:
        raise ValueError(f"Unsupported MTSplice fixture case: {case_name}")
    root = mmsplice_root()
    checkpoint = root / "models" / "mtsplice_deep0.h5"
    checkpoint_sha256 = sha256_of_file(checkpoint)
    if checkpoint_sha256 != CHECKPOINT_SHA256:
        raise AssertionError(f"{checkpoint}: expected sha256 {CHECKPOINT_SHA256}, got {checkpoint_sha256}")

    crop, sequence, _ = crop_variant_reference(
        CORPUS_RECORD_ID,
        CROP_LENGTH,
        center=CROP_CENTER,
    )
    layers = load_mmsplice_layers(root)
    model = load_model(
        checkpoint,
        compile=False,
        custom_objects={"SplineWeight1D": layers.SplineWeight1D},
    )
    split = split_tissue_sequence(sequence)
    logits = model.predict(
        [dna_one_hot(split["acceptor"]), dna_one_hot(split["donor"])],
        verbose=0,
    ).astype(np.float32)
    inputs = {"input_ids": encode_dna_ids(sequence)}
    expected = {"logits": torch.from_numpy(logits)}

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
            "ensemble_member": "mtsplice_deep0.h5",
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
    parser.add_argument("case", choices=[CASE], help="MTSplice fixture case.")
    args = parser.parse_args()
    main_for_case(args.case)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
