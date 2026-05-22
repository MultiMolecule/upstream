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

"""Generate MMSplice golden fixtures from the upstream Keras module checkpoints."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tensorflow.keras.models import load_model

HERE = Path(__file__).resolve()
MODEL_DIR = HERE.parent
REPO_ROOT = MODEL_DIR.parent.parent

sys.path.insert(0, str(REPO_ROOT))
from _corpus.load import crop_variant_pair_sequences  # noqa: E402
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

MODEL = "mmsplice"
CASE = "mmsplice"
CORPUS_RECORD_ID = "dna/grch38_chr21_synthetic_variant"
CROP_NAME = "variant_400bp"
CROP_CENTER = "variant"
CROP_LENGTH = 400
UPSTREAM_REPO_URL = "https://github.com/gagneurlab/MMSplice_MTSplice"
UPSTREAM_COMMIT = "31513da3846b187b3d7f96ad14f3c71e1177b0d3"
CHECKPOINT_SOURCE = "upstream-file://mmsplice/models/{Intron3,Acceptor,Exon,Donor,Intron5}.h5"
CHECKPOINT_SHA256 = "0199327579b0104ed6b309cda07cfd20d243d870932db0fd4f54fef867e0d4b6"
SOURCE_ENV_VAR = "MULTIMOLECULE_UPSTREAM_MMSPLICE_SOURCE"
# Small splice delta outputs should stay tighter than the default neural tolerance.
ATOL = 1e-5
RTOL = 1e-5
OVERHANG = (100, 100)
MODULE_ORDER = ("acceptor_intron", "acceptor", "exon", "donor", "donor_intron")
MODULE_FILES = {
    "acceptor_intron": "Intron3.h5",
    "acceptor": "Acceptor.h5",
    "exon": "Exon.h5",
    "donor": "Donor.h5",
    "donor_intron": "Intron5.h5",
}
MODULE_SHA256 = {
    "Intron3.h5": "f52a1100429752c0992a555625a4cd19b7cfd6d85fccfc288b076f51d7299363",
    "Acceptor.h5": "e696c29ead85b167a9102dc24020d8c072a232236be6e75bbae4515efb46aa44",
    "Exon.h5": "aea7f0aff45e14e81c6c79bac0dad488721882c45fb39f6c66e4182a5d033c94",
    "Donor.h5": "d1e17b9fd7462e492d2c520af539d93e092527ba19d835cd8602f376912ff5c6",
    "Intron5.h5": "84738869bd2a1d07aa6a9a1c650d78ba893a973cb59d7909a3ca76efe24265f6",
}


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


def split_sequence(sequence: str, overhang: tuple[int, int] = OVERHANG) -> dict[str, str]:
    intron_left, intron_right = overhang
    acceptor_intron_length = 50
    acceptor_exon_length = 3
    donor_exon_length = 5
    donor_intron_length = 13
    acceptor_intron_cut = 6
    donor_intron_cut = 6

    lack_left = acceptor_intron_length - intron_left
    if lack_left >= 0:
        sequence = "N" * (lack_left + 1) + sequence
        intron_left += lack_left + 1
    lack_right = donor_intron_length - intron_right
    if lack_right >= 0:
        sequence = sequence + "N" * (lack_right + 1)
        intron_right += lack_right + 1

    exon = sequence[intron_left:-intron_right]
    return {
        "acceptor_intron": sequence[: intron_left - acceptor_intron_cut],
        "acceptor": sequence[intron_left - acceptor_intron_length : intron_left + acceptor_exon_length],
        "exon": exon or "N",
        "donor": sequence[-intron_right - donor_exon_length : -intron_right + donor_intron_length],
        "donor_intron": sequence[-intron_right + donor_intron_cut :],
    }


def logit(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 1e-5, 1 - 1e-5)
    return np.log(value) - np.log1p(-value)


def delta_logit_psi(delta: np.ndarray) -> np.ndarray:
    not_close = ~np.isclose(delta, 0)
    exon_overlap = (not_close[:, 1] & not_close[:, 2]) | (not_close[:, 2] & not_close[:, 3])
    acceptor_intron_overlap = not_close[:, 0] & not_close[:, 1]
    donor_intron_overlap = not_close[:, 3] & not_close[:, 4]
    features = np.concatenate(
        [
            delta,
            (delta[:, 2] * exon_overlap.astype(np.float32))[:, None],
            (delta[:, 4] * donor_intron_overlap.astype(np.float32))[:, None],
            (delta[:, 0] * acceptor_intron_overlap.astype(np.float32))[:, None],
        ],
        axis=1,
    )
    coefficients = np.asarray(
        [
            0.49685773,
            0.72322957,
            1.54760024,
            0.75011527,
            2.26187717,
            -0.69419094,
            2.40138709,
            0.88148553,
        ],
        dtype=np.float32,
    )
    return features @ coefficients[:, None] + np.float32(0.0006480262366686865)


def load_modules(root: Path, model_root: Path):
    layers = load_mmsplice_layers(root)
    return {
        "acceptor_intron": load_model(
            model_root / "Intron3.h5",
            compile=False,
            custom_objects={"ConvDNA": layers.ConvDNA},
        ),
        "acceptor": load_model(
            model_root / "Acceptor.h5",
            compile=False,
            custom_objects={"ConvDNA": layers.ConvDNA},
        ),
        "exon": load_model(
            model_root / "Exon.h5",
            compile=False,
            custom_objects={
                "ConvDNA": layers.ConvDNA,
                "GlobalAveragePooling1D_Mask0": layers.GlobalAveragePooling1D_Mask0,
            },
        ),
        "donor": load_model(
            model_root / "Donor.h5",
            compile=False,
            custom_objects={"ConvDNA": layers.ConvDNA},
        ),
        "donor_intron": load_model(
            model_root / "Intron5.h5",
            compile=False,
            custom_objects={"ConvDNA": layers.ConvDNA},
        ),
    }


def module_scores(sequence: str, modules: dict[str, Any]) -> np.ndarray:
    split = split_sequence(sequence)
    scores = []
    for name in MODULE_ORDER:
        score = modules[name].predict(dna_one_hot(split[name]), verbose=0)
        if name in {"acceptor", "donor"}:
            score = logit(score)
        scores.append(score)
    return np.concatenate(scores, axis=1).astype(np.float32)


def main_for_case(case_name: str) -> None:
    if case_name != CASE:
        raise ValueError(f"Unsupported MMSplice fixture case: {case_name}")
    root = mmsplice_root()
    model_root = root / "models"
    for filename, expected_sha256 in MODULE_SHA256.items():
        actual = sha256_of_file(model_root / filename)
        if actual != expected_sha256:
            raise AssertionError(f"{filename}: expected sha256 {expected_sha256}, got {actual}")

    crop, sequence, alternative_sequence, _ = crop_variant_pair_sequences(
        CORPUS_RECORD_ID,
        CROP_LENGTH,
        center=CROP_CENTER,
    )
    modules = load_modules(root, model_root)
    reference = module_scores(sequence, modules)
    alternative = module_scores(alternative_sequence, modules)
    expected = {"logits": torch.from_numpy(delta_logit_psi(alternative - reference).astype(np.float32))}
    inputs = {
        "input_ids": encode_dna_ids(sequence),
        "alternative_input_ids": encode_dna_ids(alternative_sequence),
    }

    out_dir = fixture_out_dir(REPO_ROOT, MODEL, CASE)

    meta = {
        "version": 1,
        "model": MODEL,
        "case": CASE,
        "auto_model": "AutoModelForSequencePrediction",
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
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "module_sha256": MODULE_SHA256,
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
    print(f"  logits: {expected['logits'].flatten().tolist()}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=[CASE], help="MMSplice fixture case.")
    args = parser.parse_args()
    main_for_case(args.case)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
