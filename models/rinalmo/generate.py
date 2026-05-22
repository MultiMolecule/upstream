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

"""Generate RiNALMo golden fixtures from official RiNALMo implementations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve()
MODEL_DIR = HERE.parent
REPO_ROOT = MODEL_DIR.parent.parent

sys.path.insert(0, str(REPO_ROOT))
from _corpus.load import crop_record  # noqa: E402
from _shared.alphabet import multimolecule_rna_vocabulary  # noqa: E402
from _shared.bert_probe import (  # noqa: E402
    build_vocab_remap,
    remap_logits_to_vocab_subset,
)
from _shared.download import fetch_http_file  # noqa: E402
from _shared.fixture import (  # noqa: E402
    fixture_out_dir,
    inputs_source_from_anchor_crop,
    sha256_of_file,
    write_fixture_artifacts,
)
from _shared.source_tree import ensure_source_tree  # noqa: E402

MODEL = "rinalmo"
UPSTREAM_REPO_URL = "https://github.com/lbcb-sci/RiNALMo"
# Official CPU-compatible implementation branch; main imports flash_attn unconditionally.
UPSTREAM_COMMIT = "63c28a893e9c0347ec0539001ec6b1abe27d3446"
UPSTREAM_SOURCE_ENV_VAR = "MULTIMOLECULE_UPSTREAM_RINALMO_SOURCE"
UPSTREAM_CACHE_PREFIX = "rinalmo"
CHECKPOINT_ENV_VAR_PREFIX = "MULTIMOLECULE_UPSTREAM"
CORPUS_RECORD_ID = "rna/grch38_chr21_transcribed"
CROP_NAME = "rna_50nt"
CROP_CENTER = "center"
CROP_LENGTH = 50
ATOL = 1e-4
RTOL = 1e-4
OFFICIAL_VOCAB = (
    "<cls>",
    "<pad>",
    "<eos>",
    "<unk>",
    "<mask>",
    "A",
    "C",
    "G",
    "U",
    "I",
    "R",
    "Y",
    "K",
    "M",
    "S",
    "W",
    "B",
    "D",
    "H",
    "V",
    "N",
    "-",
)
MM_VOCAB = multimolecule_rna_vocabulary()
LOGITS_REMAP = build_vocab_remap(OFFICIAL_VOCAB, MM_VOCAB, model_name=MODEL)

CASES = {
    "rinalmo-micro": {
        "source": "https://zenodo.org/records/15043668/files/rinalmo_micro_pretrained.pt",
        "filename": "rinalmo_micro_pretrained.pt",
        "sha256": "910c73aa722061e8904a9b0d58e1e664c075fa78683fa364fcf099ae3f81ddf9",
        "size": "micro",
        "task": None,
    },
    "rinalmo-mega": {
        "source": "https://zenodo.org/records/15043668/files/rinalmo_mega_pretrained.pt",
        "filename": "rinalmo_mega_pretrained.pt",
        "sha256": "8e6d5799055dc96a556ac5dc4fbf882b677c86ccfeb072d42578d655890000b0",
        "size": "mega",
        "task": None,
    },
    "rinalmo-giga": {
        "source": "https://zenodo.org/records/15043668/files/rinalmo_giga_pretrained.pt",
        "filename": "rinalmo_giga_pretrained.pt",
        "sha256": "cd93c3f21eb3e767373c9491192686b5846247bd1110693e453c1dd0f321c0db",
        "size": "giga",
        "task": None,
    },
    "rinalmo-giga-ss": {
        "source": "https://zenodo.org/records/15043668/files/rinalmo_giga_ss_bprna_ft.pt",
        "filename": "rinalmo_giga_ss_bprna_ft.pt",
        "sha256": "44e377cb0c92f7b9cff8db8a38e33e5c0363d2f9a19831c6342346bb788a5e55",
        "size": "giga",
        "task": "ss",
    },
}


class SecondaryStructureModel(torch.nn.Module):
    def __init__(self, lm: torch.nn.Module, pred_head: torch.nn.Module):
        super().__init__()
        self.lm = lm
        self.pred_head = pred_head

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        representation = self.lm(tokens)["representation"]
        logits = self.pred_head(representation[..., 1:-1, :]).squeeze(-1)
        return {"logits": logits.unsqueeze(-1)}


def import_official_rinalmo() -> tuple[Any, Any, Any, Any]:
    source_root = ensure_source_tree(
        UPSTREAM_REPO_URL,
        UPSTREAM_COMMIT,
        ("rinalmo",),
        env_var=UPSTREAM_SOURCE_ENV_VAR,
        cache_prefix=UPSTREAM_CACHE_PREFIX,
    )
    sys.path.insert(0, str(source_root))
    from rinalmo.config import model_config  # noqa: WPS433
    from rinalmo.data.alphabet import Alphabet  # noqa: WPS433
    from rinalmo.model.downstream import SecStructPredictionHead  # noqa: WPS433
    from rinalmo.model.model import RiNALMo  # noqa: WPS433

    return model_config, Alphabet, RiNALMo, SecStructPredictionHead


def checkpoint_path(case_name: str, case: dict[str, Any]) -> Path:
    env_var = f"{CHECKPOINT_ENV_VAR_PREFIX}_{case_name.upper().replace('-', '_')}_CHECKPOINT"
    return fetch_http_file(
        str(case["source"]),
        str(case["filename"]),
        cache_prefix=f"{UPSTREAM_CACHE_PREFIX}/weights",
        env_var=env_var,
        sha256=str(case["sha256"]),
        description=f"RiNALMo {case_name} checkpoint",
    )


def encode_mm(sequence: str) -> list[int]:
    token_to_id = {token: index for index, token in enumerate(multimolecule_rna_vocabulary())}
    ids = [token_to_id["<cls>"]]
    ids.extend(token_to_id.get(base, token_to_id["<unk>"]) for base in sequence.upper().replace("T", "U"))
    ids.append(token_to_id["<eos>"])
    return ids


def validate_load_result(load_result: Any) -> None:
    unexpected = list(load_result.unexpected_keys)
    missing = list(load_result.missing_keys)
    allowed_missing = [key for key in missing if key.endswith(".rotary_emb.inv_freq")]
    if unexpected or len(allowed_missing) != len(missing):
        raise RuntimeError("Unexpected RiNALMo checkpoint mismatch: " f"missing={missing!r}, unexpected={unexpected!r}")


def load_model(case: dict[str, Any], checkpoint: Path) -> tuple[torch.nn.Module, Any]:
    model_config, Alphabet, RiNALMo, SecStructPredictionHead = import_official_rinalmo()
    config = model_config(str(case["size"]))
    state = torch.load(checkpoint, map_location=torch.device("cpu"))
    state = state.get("model", state)
    if case["task"] == "ss":
        lm = RiNALMo(config)
        lm_state = {key.removeprefix("lm."): value for key, value in state.items() if key.startswith("lm.")}
        validate_load_result(lm.load_state_dict(lm_state, strict=False))
        pred_head = SecStructPredictionHead(config["model"]["transformer"].embed_dim, num_blocks=2)
        head_state = {
            key.removeprefix("pred_head."): value for key, value in state.items() if key.startswith("pred_head.")
        }
        pred_head.load_state_dict(head_state, strict=True)
        return SecondaryStructureModel(lm, pred_head).eval(), Alphabet(**config["alphabet"])
    elif case["task"] is not None:
        raise ValueError(f"unknown RiNALMo task: {case['task']}")
    model = RiNALMo(config)
    validate_load_result(model.load_state_dict(state, strict=False))
    return model.eval(), Alphabet(**config["alphabet"])


def encode_upstream(sequence: str, alphabet: Any) -> torch.Tensor:
    return torch.tensor(alphabet.batch_tokenize([sequence.upper().replace("T", "U")]), dtype=torch.long)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=sorted(CASES))
    args = parser.parse_args()
    case = CASES[args.case]
    checkpoint = checkpoint_path(args.case, case)
    if sha256_of_file(checkpoint) != case["sha256"]:
        raise RuntimeError(f"checkpoint digest mismatch: {checkpoint}")

    crop = crop_record(CORPUS_RECORD_ID, CROP_LENGTH, center=CROP_CENTER)
    input_ids = torch.tensor([encode_mm(crop["sequence"])], dtype=torch.long)
    model, upstream_alphabet = load_model(case, checkpoint)
    upstream_input_ids = encode_upstream(crop["sequence"], upstream_alphabet)
    with torch.no_grad():
        outputs = model(upstream_input_ids)
    logits = outputs["logits"].detach().cpu().contiguous()
    if case["task"] is None:
        logits = remap_logits_to_vocab_subset(logits, LOGITS_REMAP)
    expected = {
        "logits": logits,
    }

    out_dir = fixture_out_dir(REPO_ROOT, MODEL, args.case)
    meta: dict[str, Any] = {
        "version": 1,
        "model": MODEL,
        "case": args.case,
        "auto_model": (
            "AutoModelForRnaSecondaryStructurePrediction" if case["task"] == "ss" else "AutoModelForPreTraining"
        ),
        "outputs": sorted(expected),
        "tolerance": {"atol": ATOL, "rtol": RTOL},
        "inputs_source": inputs_source_from_anchor_crop(
            crop,
            crop_name=CROP_NAME,
        ),
        "upstream": {
            "repository": UPSTREAM_REPO_URL,
            "commit": UPSTREAM_COMMIT,
            "checkpoint_source": case["source"],
            "checkpoint_sha256": case["sha256"],
            **(
                {
                    "target_slice": LOGITS_REMAP["target_slice"],
                }
                if case["task"] is None
                else {}
            ),
        },
    }
    write_fixture_artifacts(
        out_dir,
        inputs={"input_ids": input_ids},
        expected=expected,
        meta=meta,
    )
    print(f"Wrote fixture to {out_dir}")
    print(f"  input_ids: {tuple(input_ids.shape)}")
    print(f"  upstream_input_ids: {tuple(upstream_input_ids.shape)}")
    print(f"  expected: {{ {', '.join(f'{key}: {tuple(value.shape)}' for key, value in expected.items())} }}")


if __name__ == "__main__":
    main()
