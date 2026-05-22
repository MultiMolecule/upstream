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

"""Shared helpers for BERT-style masked-LM checkpoint probes."""

from __future__ import annotations

from typing import Any, Sequence

import torch


def force_eager_attention(config: Any) -> Any:
    config._attn_implementation = "eager"
    config._attn_implementation_internal = "eager"
    return config


def disable_mosaic_flash_attention(bert_layers: Any) -> None:
    bert_layers.flash_attn_qkvpacked_func = None


def encode_tokens(tokens: Sequence[str], vocab: Sequence[str]) -> list[int]:
    ids_by_token = {token: index for index, token in enumerate(vocab)}
    missing = [token for token in tokens if token not in ids_by_token]
    if missing:
        raise ValueError(f"tokens missing from vocab: {missing}")
    return [ids_by_token[token] for token in tokens]


def build_vocab_remap(
    old_vocab: Sequence[str],
    new_vocab: Sequence[str],
    *,
    model_name: str,
    duplicate_policy: str = "error",
) -> dict[str, Any]:
    old_index_by_token: dict[str, int] = {}
    duplicate_old_tokens = []
    for old_index, old_token in enumerate(old_vocab):
        if old_token in old_index_by_token:
            duplicate_old_tokens.append(old_token)
            if duplicate_policy == "last":
                old_index_by_token[old_token] = old_index
            continue
        old_index_by_token[old_token] = old_index
    if duplicate_policy not in {"error", "first", "last"}:
        raise ValueError(f"{model_name} duplicate_policy must be 'error', 'first', or 'last'")
    if duplicate_old_tokens and duplicate_policy == "error":
        raise ValueError(f"{model_name} old vocab has duplicate tokens: {sorted(set(duplicate_old_tokens))}")

    old_column_indices = []
    target_slice = []
    output_tokens = []
    omitted_tokens = []
    for new_index, token in enumerate(new_vocab):
        old_index = old_index_by_token.get(token)
        if old_index is None:
            omitted_tokens.append(token)
            continue
        old_column_indices.append(old_index)
        target_slice.append(new_index)
        output_tokens.append(token)
    if not output_tokens:
        raise ValueError(f"{model_name} vocab remap produced no shared output columns.")
    return {
        "old_column_indices": old_column_indices,
        "target_slice": target_slice,
        "output_tokens": output_tokens,
        "omitted_tokens": omitted_tokens,
    }


def remap_logits_to_vocab_subset(logits: torch.Tensor, remap: dict[str, Any]) -> torch.Tensor:
    column_index = torch.tensor(remap["old_column_indices"], dtype=torch.long, device=logits.device)
    return logits.index_select(-1, column_index).detach().cpu().contiguous()


def shared_vocab_embeddings(model: torch.nn.Module, remap: dict[str, Any]) -> torch.Tensor:
    row_index = torch.tensor(remap["old_column_indices"], dtype=torch.long)
    with torch.no_grad():
        embeddings = model.get_input_embeddings()(row_index)
    return embeddings.detach().cpu().contiguous()


def stack_hidden_states(hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.stack(tuple(state.detach().cpu() for state in hidden_states), dim=0).contiguous()


def stack_mosaic_hidden_states(hidden_states: Sequence[torch.Tensor], input_ids: torch.Tensor) -> torch.Tensor:
    batch, seq_len = input_ids.shape[:2]
    layers = []
    for state in hidden_states:
        state = state.detach().cpu()
        if state.ndim == 2 and state.shape[0] == batch * seq_len:
            state = state.reshape(batch, seq_len, state.shape[-1])
        layers.append(state)
    return torch.stack(tuple(layers), dim=0).contiguous()


def bert_mlm_probe_expected(
    model: torch.nn.Module,
    upstream_input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    remap: dict[str, Any],
) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        outputs = model(
            input_ids=upstream_input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
    return {
        "hidden_states": stack_hidden_states(outputs.hidden_states),
        "logits": remap_logits_to_vocab_subset(outputs.logits, remap),
        "vocab_embeddings": shared_vocab_embeddings(model, remap),
    }
