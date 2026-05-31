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

"""Generate RNABERT golden fixtures from official upstream code/checkpoint."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import types
import zipfile
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve()
MODEL_DIR = HERE.parent
REPO_ROOT = MODEL_DIR.parent.parent

sys.path.insert(0, str(REPO_ROOT))
from _corpus.load import crop_record  # noqa: E402
from _shared.bert_probe import (  # noqa: E402
    build_vocab_remap,
    remap_logits_to_vocab_subset,
    stack_hidden_states,
)
from _shared.download import fetch_google_drive_file  # noqa: E402
from _shared.fixture import (  # noqa: E402
    fixture_out_dir,
    inputs_source_from_anchor_crop,
    sha256_of_file,
    write_fixture_artifacts,
)
from _shared.source_tree import ensure_source_tree  # noqa: E402

MODEL = "rnabert"
UPSTREAM_REPO_URL = "https://github.com/mana438/RNABERT"
UPSTREAM_COMMIT = "1aeebcb2823bc34fc37f6527d63fca06917e3919"
UPSTREAM_SOURCE_ENV_VAR = "MULTIMOLECULE_UPSTREAM_RNABERT_SOURCE"
UPSTREAM_CACHE_PREFIX = "rnabert"
CORPUS_RECORD_ID = "rna/grch38_chr21_transcribed"
CROP_NAME = "rna_50nt"
CROP_CENTER = "center"
CROP_LENGTH = 50
ATOL = 1e-4
RTOL = 1e-4
OFFICIAL_TOKEN_IDS = {"<pad>": 0, "<mask>": 1, "A": 2, "U": 3, "G": 4, "C": 5}
MM_VOCAB = ["<pad>", "<cls>", "<eos>", "<unk>", "<mask>", "<null>"] + list("ACGUNRYSWKMBDHVIX|.*-?")
LOGITS_REMAP = build_vocab_remap(tuple(OFFICIAL_TOKEN_IDS), MM_VOCAB, model_name=MODEL)
UPSTREAM_CODE_FILES = {
    "MLM_SFP.py": "0454e84c93de011207b01b56fa612207f25061e5988e6b149a408eeba30bdd50",
    "RNA_bert_config.json": "f97799857a2a035b1fcd07e4ac33522cb937050a0780c36fd728db68bef5e015",
    "utils/bert.py": "9f3a675a7a4090d4784c9a579b058b01b6aaaac78dad30452164118ccc3c7dfc",
}

CASES = {
    "rnabert": {
        "source": "google_drive://1sT6jlv9vrpX0npKmnbFeOqZ1JZDrZTQ2/RNABERT_pretrained.pth",
        "env_var": "MULTIMOLECULE_UPSTREAM_RNABERT_CHECKPOINT",
        "sha256": "c0038a6672191bcb5517f49be94db4692769e6bddcf1dc971a03688c64105d51",
    }
}


def extract_checkpoint(path: Path, output_dir: Path) -> Path:
    with zipfile.ZipFile(path) as archive:
        out = Path(archive.extract("bert_mul_2.pth", output_dir))
    return out


def encode(sequence: str) -> list[int]:
    normalized = sequence.upper().replace("T", "U")
    unknown = sorted(set(normalized) - {"A", "U", "G", "C"})
    if unknown:
        raise ValueError(f"RNABERT upstream vocabulary cannot encode bases: {unknown}")
    return [OFFICIAL_TOKEN_IDS[base] for base in normalized]


def encode_mm(sequence: str) -> list[int]:
    token_to_id = {token: index for index, token in enumerate(MM_VOCAB)}
    normalized = sequence.upper().replace("T", "U")
    return [token_to_id.get(base, token_to_id["<unk>"]) for base in normalized]


def checkpoint_path(case: dict[str, str]) -> Path:
    return fetch_google_drive_file(
        case["source"],
        "RNABERT_pretrained.pth",
        cache_prefix=MODEL,
        env_var=case["env_var"],
        sha256=case["sha256"],
        description="RNABERT official checkpoint",
    )


def ensure_upstream_code() -> Path:
    root = ensure_source_tree(
        UPSTREAM_REPO_URL,
        UPSTREAM_COMMIT,
        ("MLM_SFP.py", "RNA_bert_config.json", "utils/bert.py"),
        env_var=UPSTREAM_SOURCE_ENV_VAR,
        cache_prefix=UPSTREAM_CACHE_PREFIX,
    )
    for rel_path, expected_sha256 in UPSTREAM_CODE_FILES.items():
        target = root / rel_path
        if not target.is_file():
            raise FileNotFoundError(f"missing pinned RNABERT upstream file: {target}")
        digest = sha256_of_file(target)
        if digest != expected_sha256:
            raise RuntimeError(f"RNABERT upstream file digest mismatch for {target}: {digest}")
    return root


def upstream_code_manifest() -> list[dict[str, str]]:
    return [
        {
            "source": f"github://mana438/RNABERT@{UPSTREAM_COMMIT}/{rel_path}",
            "sha256": sha256,
        }
        for rel_path, sha256 in sorted(UPSTREAM_CODE_FILES.items())
    ]


def install_attrdict_shim() -> None:
    if "attrdict" in sys.modules:
        return

    class AttrDict(dict):
        def __getattr__(self, key: str) -> Any:
            try:
                return self[key]
            except KeyError as exc:
                raise AttributeError(key) from exc

    module = types.ModuleType("attrdict")
    module.AttrDict = AttrDict
    sys.modules["attrdict"] = module


def load_upstream_bert_module(root: Path) -> Any:
    install_attrdict_shim()
    sys.dont_write_bytecode = True
    module_path = root / "utils" / "bert.py"
    spec = importlib.util.spec_from_file_location("rnabert_upstream_bert", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load RNABERT upstream module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_model(checkpoint: Path) -> torch.nn.Module:
    torch.manual_seed(1016)
    upstream_root = ensure_upstream_code()
    bert = load_upstream_bert_module(upstream_root)
    config = types.SimpleNamespace(**json.loads((upstream_root / "RNA_bert_config.json").read_text()))
    config.hidden_size = config.num_attention_heads * config.multiple
    net_bert = bert.BertModel(config)
    model = bert.BertForMaskedLM(config, net_bert)
    with tempfile.TemporaryDirectory(prefix="rnabert_checkpoint_") as tmp_dir:
        state = torch.load(
            extract_checkpoint(checkpoint, Path(tmp_dir)),
            map_location=torch.device("cpu"),
        )
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=sorted(CASES))
    args = parser.parse_args()
    case = CASES[args.case]
    checkpoint = checkpoint_path(case)
    if sha256_of_file(checkpoint) != case["sha256"]:
        raise RuntimeError(f"checkpoint digest mismatch: {checkpoint}")

    crop = crop_record(CORPUS_RECORD_ID, CROP_LENGTH, center=CROP_CENTER)
    upstream_input_ids = torch.tensor([encode(crop["sequence"])], dtype=torch.long)
    input_ids = torch.tensor([encode_mm(crop["sequence"])], dtype=torch.long)
    model = load_model(checkpoint)
    with torch.no_grad():
        hidden_states, pooler_output = model.bert(
            upstream_input_ids,
            token_type_ids=None,
            attention_mask=None,
            output_all_encoded_layers=True,
            attention_show_flg=False,
        )
    stacked_hidden_states = stack_hidden_states(tuple(hidden_states))
    with torch.no_grad():
        logits_lm, logits_ss, logits_sa = model.cls(stacked_hidden_states[-1], pooler_output)
    expected = {
        "hidden_states": stacked_hidden_states,
        "logits_lm": remap_logits_to_vocab_subset(logits_lm, LOGITS_REMAP),
        "logits_sa": logits_sa.detach().cpu().contiguous(),
        "logits_ss": logits_ss.detach().cpu().contiguous(),
    }

    out_dir = fixture_out_dir(REPO_ROOT, MODEL, args.case)
    meta: dict[str, Any] = {
        "version": 1,
        "model": MODEL,
        "case": args.case,
        "auto_model": "AutoModelForPreTraining",
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
            "code_files": upstream_code_manifest(),
            "target_slice": LOGITS_REMAP["target_slice"],
        },
    }
    write_fixture_artifacts(
        out_dir,
        inputs={"input_ids": input_ids},
        expected=expected,
        meta=meta,
    )
    print(f"Wrote fixture to {out_dir}")


if __name__ == "__main__":
    main()
