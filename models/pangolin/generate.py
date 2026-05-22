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

"""Generate Pangolin golden fixtures from the upstream PyTorch ensemble."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import torch

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
    dna_one_hot_with_context,
    encode_dna_ids,
)

MODEL = "pangolin"
CASE = "pangolin"
CORPUS_RECORD_ID = "dna/grch38_chr21_synthetic_variant"
CROP_NAME = "variant_400bp"
CROP_CENTER = "variant"
CROP_LENGTH = 400
UPSTREAM_REPO_URL = "https://github.com/tkzeng/Pangolin"
UPSTREAM_COMMIT = "5cf94b8db938c658391b4305cd7ce33297d44ff7"
SOURCE_ENV_VAR = "MULTIMOLECULE_UPSTREAM_PANGOLIN_SOURCE"
CHECKPOINT_SOURCE = "upstream-file://pangolin/models/final.{1,2,3}.{0,2,4,6}.3.v2"
CHECKPOINT_SHA256 = "288ccd001fa92a83e3a4ff43b482393b3794fe20cdcbf191889a18280ebd1ca3"
# Ensemble splice probabilities are small deltas, so keep tolerance tight.
ATOL = 1e-5
RTOL = 1e-5
CONTEXT = 10000
ENSEMBLE_MEMBERS = [
    ["final.1.0.3.v2", "final.2.0.3.v2", "final.3.0.3.v2"],
    ["final.1.2.3.v2", "final.2.2.3.v2", "final.3.2.3.v2"],
    ["final.1.4.3.v2", "final.2.4.3.v2", "final.3.4.3.v2"],
    ["final.1.6.3.v2", "final.2.6.3.v2", "final.3.6.3.v2"],
]


def pangolin_root() -> Path:
    return ensure_source_tree(
        UPSTREAM_REPO_URL,
        UPSTREAM_COMMIT,
        ("pangolin",),
        env_var=SOURCE_ENV_VAR,
        cache_prefix="pangolin",
    )


def checkpoint_manifest(model_root: Path) -> tuple[str, dict[str, str]]:
    manifest = {}
    lines = []
    for group in ENSEMBLE_MEMBERS:
        for filename in group:
            digest = sha256_of_file(model_root / filename)
            manifest[filename] = digest
            lines.append(f"{digest}  {filename}")
    text = "\n".join(lines) + "\n"
    return hashlib.sha256(text.encode()).hexdigest(), manifest


def load_pangolin_model_class(
    source_root: Path,
) -> tuple[type[torch.nn.Module], int, int, int]:
    sys.path.insert(0, str(source_root))
    from pangolin.model import AR, L, Pangolin, W  # noqa: WPS433

    return Pangolin, L, W, AR


def one_hot_with_context(sequence: str) -> torch.Tensor:
    left_context = CONTEXT // 2
    return dna_one_hot_with_context(
        sequence,
        left_context=left_context,
        right_context=CONTEXT - left_context,
    )


def load_member(
    filename: str,
    model_root: Path,
    model_cls: type[torch.nn.Module],
    length: int,
    width: int,
    ar: int,
) -> torch.nn.Module:
    model = model_cls(length, width, ar)
    state = torch.load(model_root / filename, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.eval()


def upstream_forward(sequence: str, source_root: Path, model_root: Path) -> torch.Tensor:
    model_cls, length, width, ar = load_pangolin_model_class(source_root)
    inputs = one_hot_with_context(sequence)
    tissue_logits = []
    with torch.no_grad():
        for tissue_index, group in enumerate(ENSEMBLE_MEMBERS):
            member_outputs = []
            for filename in group:
                member = load_member(filename, model_root, model_cls, length, width, ar)
                member_outputs.append(member(inputs).transpose(1, 2))
            logits = torch.stack(member_outputs).mean(dim=0)
            start = tissue_index * 3
            tissue_logits.append(logits[..., start : start + 3])
    return torch.cat(tissue_logits, dim=-1).contiguous()


def main_for_case(case_name: str) -> None:
    if case_name != CASE:
        raise ValueError(f"Unsupported Pangolin fixture case: {case_name}")
    source_root = pangolin_root()
    model_root = source_root / "pangolin" / "models"
    manifest_sha256, manifest = checkpoint_manifest(model_root)
    if manifest_sha256 != CHECKPOINT_SHA256:
        raise AssertionError(f"Pangolin checkpoint manifest sha256 {manifest_sha256} != {CHECKPOINT_SHA256}")

    crop, sequence, _ = crop_variant_reference(
        CORPUS_RECORD_ID,
        CROP_LENGTH,
        center=CROP_CENTER,
    )
    inputs = {"input_ids": encode_dna_ids(sequence)}
    expected = {"logits": upstream_forward(sequence, source_root, model_root)}

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
            "checkpoint_sha256": manifest_sha256,
            "checkpoint_manifest": manifest,
            "ensemble_members": ENSEMBLE_MEMBERS,
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
    parser.add_argument("case", choices=[CASE], help="Pangolin fixture case.")
    args = parser.parse_args()
    main_for_case(args.case)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
