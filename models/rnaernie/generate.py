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

"""Generate RNAErnie fixtures from the local official Paddle checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve()
MODEL_DIR = HERE.parent
REPO_ROOT = MODEL_DIR.parent.parent

sys.path.insert(0, str(REPO_ROOT))
from _corpus.load import crop_record  # noqa: E402
from _shared.bert_probe import build_vocab_remap  # noqa: E402
from _shared.download import fetch_google_drive_file  # noqa: E402
from _shared.fixture import (  # noqa: E402
    fixture_out_dir,
    inputs_source_from_anchor_crop,
    sha256_of_file,
    write_fixture_artifacts,
)

MODEL = "rnaernie"
UPSTREAM_REPO_URL = "https://github.com/CatIIIIIIII/RNAErnie"
UPSTREAM_COMMIT = "b5e4c1cfdc6f53101111823abaddcab00c43b0d3"
PADDLE_ROOT_ENV_VAR = "MULTIMOLECULE_UPSTREAM_RNAERNIE_CHECKPOINT_DIR"
CORPUS_RECORD_ID = "rna/grch38_chr21_transcribed"
CROP_NAME = "rna_50nt"
CROP_CENTER = "center"
CROP_LENGTH = 50
ATOL = 1e-4
RTOL = 1e-4

UPSTREAM_VOCAB = (
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[MASK]",
    "[DEL]",
    "[IND]",
    "RNaseMRPRNA",
    "RNasePRNA",
    "SRPRNA",
    "YRNA",
    "antisenseRNA",
    "autocatalyticallysplicedintron",
    "guideRNA",
    "hammerheadribozyme",
    "lncRNA",
    "miRNA",
    "miscRNA",
    "ncRNA",
    "other",
    "piRNA",
    "premiRNA",
    "precursorRNA",
    "rRNA",
    "ribozyme",
    "sRNA",
    "scRNA",
    "scaRNA",
    "siRNA",
    "snRNA",
    "snoRNA",
    "tRNA",
    "telomeraseRNA",
    "tmRNA",
    "vaultRNA",
    "A",
    "T",
    "C",
    "G",
)

MM_VOCAB = (
    "<pad>",
    "<cls>",
    "<eos>",
    "<unk>",
    "<mask>",
    "<null>",
    "A",
    "C",
    "G",
    "U",
    "N",
    "R",
    "Y",
    "S",
    "W",
    "K",
    "M",
    "B",
    "D",
    "H",
    "V",
    "I",
    "X",
    "|",
    ".",
    "*",
    "-",
    "?",
)
CONVERTED_UPSTREAM_VOCAB = (
    "<pad>",
    "<unk>",
    "<cls>",
    "<eos>",
    "<mask>",
    "<null>",
    "<null>",
    "<null>",
    "<null>",
    "<null>",
    "<null>",
    "<null>",
    "<null>",
    "<null>",
    "<null>",
    "<null>",
    "<null>",
    "<null>",
    "<null>",
    "<null>",
    "<null>",
    "<null>",
    "<null>",
    "<null>",
    "<null>",
    "<null>",
    "<null>",
    "<null>",
    "<null>",
    "<null>",
    "<null>",
    "<null>",
    "<null>",
    "<null>",
    "<null>",
    "A",
    "U",
    "C",
    "G",
)
LOGITS_REMAP = build_vocab_remap(
    CONVERTED_UPSTREAM_VOCAB,
    MM_VOCAB,
    model_name=MODEL,
    duplicate_policy="last",
)

CASES = {
    "rnaernie": {
        "source": ("google_drive://1Ls5k7hv83BLRTznB4XcegIa2yKkU40Ls/" "BERT,ERNIE,MOTIF,PROMPT/model_state.pdparams"),
        "config": ("google_drive://1Ls5k7hv83BLRTznB4XcegIa2yKkU40Ls/" "BERT,ERNIE,MOTIF,PROMPT/model_config.json"),
        "checkpoint_filename": "model_state.pdparams",
        "config_filename": "model_config.json",
        "sha256": "e5f17a98af4f1051cabc929b4b1e7d20a8bc8f989b296ef1b515e19b19a3867b",
        "config_sha256": "4fe00781c61d30151a797e902dbbc68d9c3937436b9f67fa478c563f4b6d8d2b",
    }
}


def case_paths(case: dict[str, Any]) -> tuple[Path, Path]:
    override = os.environ.get(PADDLE_ROOT_ENV_VAR)
    if override:
        root = Path(override).expanduser().resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"RNAErnie Paddle root not found at ${PADDLE_ROOT_ENV_VAR}: {root}")
        return root / str(case["checkpoint_filename"]), root / str(case["config_filename"])
    checkpoint = fetch_google_drive_file(
        case["source"],
        str(case["checkpoint_filename"]),
        cache_prefix=MODEL,
        sha256=str(case["sha256"]),
        description="RNAErnie Paddle checkpoint",
    )
    config = fetch_google_drive_file(
        case["config"],
        str(case["config_filename"]),
        cache_prefix=MODEL,
        sha256=str(case["config_sha256"]),
        description="RNAErnie Paddle config",
    )
    return checkpoint, config


def encode_upstream(sequence: str) -> list[int]:
    token_to_id = {token: index for index, token in enumerate(UPSTREAM_VOCAB)}
    ids = [token_to_id["[CLS]"]]
    unk = token_to_id["[UNK]"]
    for base in sequence.upper().replace("U", "T"):
        ids.append(token_to_id.get(base, unk))
    ids.append(token_to_id["[SEP]"])
    return ids


def encode_mm(sequence: str) -> list[int]:
    token_to_id = {token: index for index, token in enumerate(MM_VOCAB)}
    ids = [token_to_id["<cls>"]]
    unk = token_to_id["<unk>"]
    for base in sequence.upper().replace("T", "U"):
        ids.append(token_to_id.get(base, unk))
    ids.append(token_to_id["<eos>"])
    return ids


def load_config_kwargs(config_path: Path) -> dict[str, Any]:
    payload = json.loads(config_path.read_text())
    init_args = payload.get("init_args")
    if not isinstance(init_args, list) or not init_args or not isinstance(init_args[0], dict):
        raise RuntimeError(f"{config_path}: expected Paddle init_args[0] mapping")
    kwargs = dict(init_args[0])
    kwargs.pop("init_class", None)
    kwargs.setdefault("intermediate_size", int(kwargs["hidden_size"]) * 4)
    return kwargs


def import_runtime() -> tuple[Any, Any, Any]:
    try:
        import paddle
        from paddlenlp.transformers import ErnieForMaskedLM, ErnieModel
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "RNAErnie fixture generation requires PaddlePaddle and PaddleNLP. "
            "Run it in the Paddle runtime declared by source.yaml."
        ) from error
    return paddle, ErnieModel, ErnieForMaskedLM


def instantiate_model(config_path: Path) -> tuple[Any, Any]:
    paddle, ernie_model_cls, masked_lm_cls = import_runtime()
    paddle.set_device("cpu")
    paddle.seed(1016)
    kwargs = load_config_kwargs(config_path)
    try:
        ernie = ernie_model_cls(**kwargs)
        model = masked_lm_cls(ernie)
    except TypeError:
        from paddlenlp.transformers import ErnieConfig

        config = ErnieConfig(**kwargs)
        model = masked_lm_cls(config)
    return paddle, model


def load_pdparams(paddle: Any, checkpoint_path: Path) -> dict[str, Any]:
    try:
        state = paddle.load(str(checkpoint_path))
    except Exception:
        with checkpoint_path.open("rb") as handle:
            state = pickle.load(handle)
    state = dict(state)
    state.pop("StructuredToParameterName@@", None)
    return state


def state_value_shape(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        return ()
    return tuple(int(dim) for dim in shape)


def state_for_model(paddle: Any, model: Any, state: dict[str, Any]) -> dict[str, Any]:
    model_state = model.state_dict()
    prepared = {}
    consumed = set()
    remap = {
        "cls.predictions.decoder.weight": "cls.predictions.decoder_weight",
        "cls.predictions.decoder.bias": "cls.predictions.decoder_bias",
    }

    for key, target in model_state.items():
        source_key = key if key in state else remap.get(key)
        if source_key is None or source_key not in state:
            raise RuntimeError(f"Paddle checkpoint is missing key required by runtime model: {key}")
        value = state[source_key]
        if state_value_shape(value) != state_value_shape(target):
            raise RuntimeError(
                f"Paddle checkpoint shape mismatch for {key}: "
                f"expected {state_value_shape(target)}, got {state_value_shape(value)} from {source_key}"
            )
        prepared[key] = value if isinstance(value, paddle.Tensor) else paddle.to_tensor(value)
        consumed.add(source_key)

    unexpected = sorted(key for key in state if key not in consumed)
    if unexpected:
        raise RuntimeError(f"Paddle checkpoint has unexpected keys for runtime model: {unexpected[:8]}")
    return prepared


def load_model(checkpoint_path: Path, config_path: Path) -> tuple[Any, Any]:
    paddle, model = instantiate_model(config_path)
    state = load_pdparams(paddle, checkpoint_path)
    model.set_state_dict(state_for_model(paddle, model, state))
    model.eval()
    return paddle, model


def extract_logits(outputs: Any) -> Any:
    if isinstance(outputs, (tuple, list)):
        if not outputs:
            raise RuntimeError("Paddle ErnieForMaskedLM returned an empty output tuple")
        return outputs[0]
    if hasattr(outputs, "numpy") and callable(outputs.numpy):
        return outputs
    logits = getattr(outputs, "logits", None)
    if logits is None:
        raise RuntimeError("Paddle ErnieForMaskedLM output is missing logits")
    return logits


def to_numpy_array(value: Any) -> np.ndarray:
    return np.ascontiguousarray(value.numpy())


def remap_logits(value: Any) -> np.ndarray:
    logits = to_numpy_array(value)
    return np.ascontiguousarray(logits[..., LOGITS_REMAP["old_column_indices"]])


def upstream_forward(checkpoint_path: Path, config_path: Path, input_ids: list[int]) -> dict[str, np.ndarray]:
    paddle, model = load_model(checkpoint_path, config_path)
    paddle_ids = paddle.to_tensor([input_ids], dtype="int64")
    with paddle.no_grad():
        logits = extract_logits(model(paddle_ids))
    return {
        "logits": remap_logits(logits),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=sorted(CASES))
    args = parser.parse_args()
    case = CASES[args.case]
    checkpoint, config = case_paths(case)

    if sha256_of_file(checkpoint) != case["sha256"]:
        raise RuntimeError(f"checkpoint digest mismatch: {checkpoint}")
    if sha256_of_file(config) != case["config_sha256"]:
        raise RuntimeError(f"config digest mismatch: {config}")

    crop = crop_record(CORPUS_RECORD_ID, CROP_LENGTH, center=CROP_CENTER)
    upstream_ids = encode_upstream(crop["sequence"])
    mm_ids = encode_mm(crop["sequence"])
    if len(upstream_ids) != CROP_LENGTH + 2 or len(mm_ids) != CROP_LENGTH + 2:
        raise AssertionError("RNAErnie tokenized length must equal crop length plus [CLS]/[SEP]")

    expected = upstream_forward(checkpoint, config, upstream_ids)
    inputs = {"input_ids": np.asarray([mm_ids], dtype=np.int64)}

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
            "target_slice": LOGITS_REMAP["target_slice"],
        },
    }
    write_fixture_artifacts(
        out_dir,
        inputs=inputs,
        expected=expected,
        meta=meta,
    )
    print(f"Wrote fixture to {out_dir}")


if __name__ == "__main__":
    main()
