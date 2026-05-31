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

"""Generate ProGen2 upstream parity goldens from local official checkpoints."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tarfile
import types
from pathlib import Path
from typing import Any

import torch
import yaml
from tokenizers import Tokenizer

MODEL = "progen2"
DEFAULT_CASE = "progen2-small"
UPSTREAM_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = Path(__file__).with_name("source.yaml")
CODE_REPOSITORY = "https://github.com/salesforce/progen"
CODE_COMMIT = "485b2ea3db98f8d65d0cd86c2c85ae639b37a678"
CODE_SOURCE_ENV_VAR = "MULTIMOLECULE_UPSTREAM_PROGEN2_SOURCE"
CODE_CACHE_PREFIX = "progen2-code"
PROBE_ID = f"{MODEL}/standard_protein_probe"
PROBE_SEQUENCE = "ACDEFGHIKLMNPQRSTVWYBXZUO"
ORIGINAL_VOCAB = (
    "<pad>",
    "<cls>",
    "<eos>",
    "<null>",
    "<null>",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "<null>",
    "<null>",
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
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
    "X",
    "Z",
    "B",
    "J",
    "U",
    "O",
    "|",
    ".",
    "*",
    "-",
    "?",
)
MM_TOKEN_TO_ID = {token: index for index, token in enumerate(MM_VOCAB)}
DEFAULT_TOLERANCE = {"atol": 1e-4, "rtol": 1e-4}
TOLERANCE_BY_CASE = {
    "progen2-xlarge": {"atol": 6e-4, "rtol": 1e-4},
}

sys.path.insert(0, str(UPSTREAM_ROOT))
from _shared.archive import safe_extract_tar  # noqa: E402
from _shared.download import fetch_http_file  # noqa: E402
from _shared.fixture import fixture_out_dir, sha256_of_file, write_fixture_artifacts  # noqa: E402
from _shared.source_tree import ensure_source_tree  # noqa: E402


def load_source() -> dict[str, Any]:
    with SOURCE_PATH.open() as handle:
        return yaml.safe_load(handle)


def checkpoint_for(source: dict[str, Any], case: str) -> dict[str, Any]:
    checkpoints = source["checkpoints"]
    try:
        checkpoint = dict(checkpoints[case])
    except KeyError as error:
        choices = ", ".join(sorted(checkpoints))
        raise SystemExit(f"Unknown case {case!r}; choose one of: {choices}") from error
    checkpoint["case"] = case
    return checkpoint


def source_root() -> Path:
    return ensure_source_tree(
        CODE_REPOSITORY,
        CODE_COMMIT,
        ("progen2/models/progen", "progen2/tokenizer.json"),
        env_var=CODE_SOURCE_ENV_VAR,
        cache_prefix=CODE_CACHE_PREFIX,
    )


def code_root() -> Path:
    return source_root() / "progen2" / "models"


def tokenizer_path() -> Path:
    return source_root() / "progen2" / "tokenizer.json"


def checkpoint_dir_for(checkpoint: dict[str, Any]) -> Path:
    case = str(checkpoint["case"])
    case_token = case.upper().replace("-", "_")
    env_var = f"MULTIMOLECULE_UPSTREAM_{case_token}_CHECKPOINT_DIR"
    override = os.environ.get(env_var)
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_dir():
            raise NotADirectoryError(f"ProGen2 checkpoint root not found at ${env_var}: {path}")
        return path

    archive: Path | None = None
    destination: Path | None = None
    for attempt in range(2):
        archive = fetch_http_file(
            str(checkpoint["source"]),
            f"{case}.tar.gz",
            cache_prefix=f"{MODEL}/checkpoints",
            env_var=f"MULTIMOLECULE_UPSTREAM_{case_token}_ARCHIVE",
            description=f"ProGen2 {case} official checkpoint archive",
            timeout=300,
            retries=5,
        )
        destination = archive.parent / case
        direct_checkpoint = destination / "pytorch_model.bin"
        if not direct_checkpoint.is_file():
            try:
                safe_extract_tar(archive, destination)
            except (EOFError, OSError, tarfile.TarError):
                shutil.rmtree(destination, ignore_errors=True)
                archive.unlink(missing_ok=True)
                if attempt == 0:
                    continue
                raise
        if direct_checkpoint.is_file():
            return destination
        nested = destination / case
        if (nested / "pytorch_model.bin").is_file():
            return nested
    if archive is None:
        raise RuntimeError(f"{case}: checkpoint archive was not resolved")
    if destination is None:
        destination = archive.parent / case
    raise FileNotFoundError(f"{archive}: extracted ProGen2 checkpoint did not contain pytorch_model.bin")


def install_official_code_compat() -> None:
    """Patch removed Transformers v4 helpers needed by the local official code snapshot."""
    model_parallel = types.ModuleType("transformers.utils.model_parallel_utils")
    model_parallel.assert_device_map = lambda device_map, num_blocks: None
    model_parallel.get_device_map = lambda n_layers, devices: {device: [] for device in devices}
    sys.modules.setdefault("transformers.utils.model_parallel_utils", model_parallel)


def load_official_model(checkpoint_dir: Path):
    install_official_code_compat()
    sys.path.insert(0, str(code_root()))
    from progen.configuration_progen import ProGenConfig
    from progen.modeling_progen import ProGenAttention, ProGenForCausalLM, ProGenModel

    ProGenForCausalLM.all_tied_weights_keys = {}
    ProGenModel.all_tied_weights_keys = {}
    ProGenModel.get_head_mask = lambda self, head_mask, num_hidden_layers, is_attention_chunked=False: (
        [None] * num_hidden_layers if head_mask is None else head_mask
    )

    # The official ProGen2 classes predate current Transformers loading hooks.
    # `from_pretrained` can silently leave these models randomly initialized in
    # Transformers 5, so instantiate the config and assign the raw checkpoint
    # tensors explicitly.
    config = ProGenConfig.from_pretrained(str(checkpoint_dir))
    model = ProGenForCausalLM(config)
    state_dict = torch.load(checkpoint_dir / "pytorch_model.bin", map_location=torch.device("cpu"))
    model.load_state_dict(state_dict, strict=True, assign=True)
    model.eval()
    for module in model.modules():
        if isinstance(module, ProGenAttention):
            module.scale_attn = torch.sqrt(torch.tensor(module.head_dim, dtype=torch.float32))
            if getattr(module.bias, "is_meta", False):
                max_positions = module.bias.shape[-1]
                module.bias = torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                    1, 1, max_positions, max_positions
                )
            if getattr(module.masked_bias, "is_meta", False):
                module.masked_bias = torch.tensor(-1e9)
    return model


def original_vocab_for_size(vocab_size: int) -> list[str | None]:
    old_vocab: list[str | None] = list(ORIGINAL_VOCAB)
    if vocab_size > len(old_vocab):
        old_vocab.extend([None] * (vocab_size - len(old_vocab)))
    return old_vocab[:vocab_size]


def mm_id_for_upstream_token_id(token_id: int) -> int:
    try:
        token = ORIGINAL_VOCAB[token_id]
    except IndexError as error:
        raise ValueError(f"Upstream tokenizer produced token id {token_id}, outside ORIGINAL_VOCAB.") from error
    try:
        return MM_TOKEN_TO_ID[token]
    except KeyError as error:
        raise ValueError(f"Upstream token id {token_id} maps to {token!r}, which is not in the MM vocab.") from error


def build_vocab_remap(old_vocab: list[str | None], new_vocab: tuple[str, ...]) -> dict[str, Any]:
    """Map original upstream vocab columns onto the comparable MM vocab columns.

    The policy is intentionally column-only: run the original upstream LM head,
    then select columns for tokens that exist exactly once in the original vocab,
    ordered by the MultiMolecule vocab surface. Missing or ambiguous MM columns
    are not synthesized.
    """
    old_indices_by_token: dict[str, list[int]] = {}
    for old_index, old_token in enumerate(old_vocab):
        if old_token is None:
            continue
        old_indices_by_token.setdefault(old_token, []).append(old_index)

    duplicate_tokens = sorted(token for token, indices in old_indices_by_token.items() if len(indices) > 1)
    unique_old_index_by_token = {
        token: indices[0] for token, indices in old_indices_by_token.items() if len(indices) == 1
    }

    old_column_indices = []
    target_slice = []
    output_tokens = []
    omitted_missing_tokens = []
    omitted_duplicate_tokens = []
    for new_index, token in enumerate(new_vocab):
        if token in unique_old_index_by_token:
            old_column_indices.append(unique_old_index_by_token[token])
            target_slice.append(new_index)
            output_tokens.append(token)
        elif token in old_indices_by_token:
            omitted_duplicate_tokens.append(token)
        else:
            omitted_missing_tokens.append(token)

    if not output_tokens:
        raise ValueError("ProGen2 vocab remap produced no shared output columns.")
    return {
        "old_column_indices": old_column_indices,
        "target_slice": target_slice,
        "output_tokens": output_tokens,
        "omitted_missing_tokens": omitted_missing_tokens,
        "omitted_duplicate_tokens": omitted_duplicate_tokens,
        "duplicate_old_tokens": duplicate_tokens,
    }


def remap_logits_to_mm_vocab_subset(logits: torch.Tensor, remap: dict[str, Any]) -> torch.Tensor:
    column_index = torch.tensor(remap["old_column_indices"], dtype=torch.long, device=logits.device)
    return logits.index_select(-1, column_index).detach().cpu().contiguous()


def stack_outputs(values: tuple[torch.Tensor, ...] | None, name: str) -> torch.Tensor:
    if not values:
        raise RuntimeError(f"ProGen2 upstream did not return {name}")
    return torch.stack(tuple(value.detach().cpu() for value in values), dim=0).contiguous()


def encode_inputs(sequence: str) -> tuple[torch.Tensor, list[int], list[int], list[str]]:
    tokenizer = Tokenizer.from_file(str(tokenizer_path()))
    encoding = tokenizer.encode(sequence)
    old_ids = encoding.ids
    tokens = encoding.tokens

    mm_ids = [mm_id_for_upstream_token_id(token_id) for token_id in old_ids]
    return torch.tensor([mm_ids], dtype=torch.long), old_ids, mm_ids, tokens


def generate_case(case: str) -> Path:
    source = load_source()
    checkpoint = checkpoint_for(source, case)
    checkpoint_dir = checkpoint_dir_for(checkpoint)
    checkpoint_file = checkpoint_dir / "pytorch_model.bin"
    output_dir = fixture_out_dir(UPSTREAM_ROOT, MODEL, case)
    sequence = PROBE_SEQUENCE

    actual_sha = sha256_of_file(checkpoint_file)
    if checkpoint.get("sha256") != actual_sha:
        raise SystemExit(
            f"{checkpoint_file} sha256 is {actual_sha}, but source.yaml records {checkpoint.get('sha256')!r}."
        )

    input_ids, old_ids, mm_ids, tokens = encode_inputs(sequence)
    old_input_ids = torch.tensor([old_ids], dtype=torch.long)
    model = load_official_model(checkpoint_dir).float()
    old_vocab = original_vocab_for_size(model.config.vocab_size)
    vocab_remap = build_vocab_remap(old_vocab, MM_VOCAB)
    with torch.no_grad():
        upstream_outputs = model(
            old_input_ids,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        logits = remap_logits_to_mm_vocab_subset(upstream_outputs.logits, vocab_remap)
        hidden_states = stack_outputs(upstream_outputs.hidden_states, "hidden_states")

    upstream = source["upstream"]
    expected = {
        "hidden_states": hidden_states,
        "logits": logits.contiguous(),
    }
    meta = {
        "version": 1,
        "model": MODEL,
        "case": case,
        "auto_model": "AutoModelForCausalLM",
        "outputs": sorted(expected),
        "tolerance": TOLERANCE_BY_CASE.get(case, DEFAULT_TOLERANCE),
        "inputs_source": {
            "type": "synthetic_token_probe",
            "id": PROBE_ID,
        },
        "upstream": {
            "repository": upstream["repository"],
            "commit": upstream["commit"],
            "checkpoint_source": checkpoint["source"],
            "checkpoint_sha256": checkpoint["sha256"],
            "target_slice": vocab_remap["target_slice"],
        },
    }
    write_fixture_artifacts(
        output_dir,
        inputs={"input_ids": input_ids},
        expected=expected,
        meta=meta,
    )
    return output_dir


def main() -> None:
    source = load_source()
    choices = sorted(source["checkpoints"])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", nargs="?", choices=choices, default=DEFAULT_CASE)
    args = parser.parse_args()

    output_dir = generate_case(args.case)
    print(f"generated {output_dir.relative_to(UPSTREAM_ROOT)}")


if __name__ == "__main__":
    main()
