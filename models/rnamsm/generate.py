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

"""Generate RNA-MSM golden fixtures from the official MSA Transformer checkpoint."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import __main__
import torch
from torch import nn
from torch.nn import functional as F

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
from _shared.download import fetch_google_drive_file  # noqa: E402
from _shared.fixture import (  # noqa: E402
    fixture_out_dir,
    inputs_source_from_anchor_crop,
    sha256_of_file,
    write_fixture_artifacts,
)

MODEL = "rnamsm"
UPSTREAM_REPO_URL = "https://github.com/yikunpku/RNA-MSM"
UPSTREAM_COMMIT = "faadd57559c1c146effbfd84a7628cd2042d5ede"
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
    "A",
    "G",
    "C",
    "U",
    "X",
    "N",
    "-",
    "<mask>",
)
MM_VOCAB = multimolecule_rna_vocabulary()
LOGITS_REMAP = build_vocab_remap(OFFICIAL_VOCAB, MM_VOCAB, model_name=MODEL)
TOKEN_TO_ID = {token: index for index, token in enumerate(OFFICIAL_VOCAB)}
PAD_TOKEN_ID = TOKEN_TO_ID["<pad>"]
UNK_TOKEN_ID = TOKEN_TO_ID["<unk>"]
BOS_TOKEN_ID = TOKEN_TO_ID["<cls>"]
NUM_LAYERS = 10
HIDDEN_SIZE = 768
NUM_ATTENTION_HEADS = 12
MAX_POSITION_EMBEDDINGS = 1024
MAX_TOKENS_PER_MSA = 2**14

CASES = {
    "rnamsm": {
        "source": "google_drive://11A-S13qAb5wiBi1YLs3EOrnixSDq7Q0q/RNA-MSM_pretrained.ckpt",
        "env_var": "MULTIMOLECULE_UPSTREAM_RNAMSM_CHECKPOINT",
        "sha256": "6aee49663a2959670c6080e7a3ab80e87a78dbf81e5e6b1ca7349a7849121f02",
    }
}


class OptimizerConfig:
    pass


class MSATransformerModelConfig:
    pass


class DataConfig:
    pass


class TrainConfig:
    pass


class LoggingConfig:
    pass


class Config:
    pass


for _name in (
    "OptimizerConfig",
    "MSATransformerModelConfig",
    "DataConfig",
    "TrainConfig",
    "LoggingConfig",
    "Config",
):
    setattr(__main__, _name, globals()[_name])


def checkpoint_path(case: dict[str, str]) -> Path:
    return fetch_google_drive_file(
        case["source"],
        "RNA-MSM_pretrained.ckpt",
        cache_prefix=MODEL,
        env_var=case["env_var"],
        sha256=case["sha256"],
        description="RNA-MSM official checkpoint",
    )


def encode_single_row_msa(sequence: str) -> list[list[int]]:
    ids = [BOS_TOKEN_ID]
    ids.extend(TOKEN_TO_ID.get(base, UNK_TOKEN_ID) for base in sequence.upper().replace("T", "U"))
    return [ids]


def encode_mm_single_row_msa(sequence: str) -> list[list[int]]:
    token_to_id = {token: index for index, token in enumerate(MM_VOCAB)}
    ids = [token_to_id["<cls>"]]
    ids.extend(token_to_id.get(base, token_to_id["<unk>"]) for base in sequence.upper().replace("T", "U"))
    return [ids]


class LearnedPositionalEmbedding(nn.Embedding):
    """Official RNA-MSM learned positional embedding."""

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int):
        super().__init__(num_embeddings + padding_idx + 1, embedding_dim, padding_idx)
        self.max_positions = num_embeddings

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        mask = input_ids.ne(self.padding_idx).int()
        positions = (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + self.padding_idx
        return F.embedding(
            positions,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


class RobertaLMHead(nn.Module):
    def __init__(self, embed_dim: int, output_dim: int, weight: nn.Parameter):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(features)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return F.linear(hidden_states, self.weight) + self.bias


class ContactPredictionHead(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.in_features = in_features
        self.prepend_bos = True
        self.append_eos = False
        self.eos_idx = TOKEN_TO_ID["<eos>"]
        self.regression = nn.Linear(in_features, 1)
        self.activation = nn.Sigmoid()


class NormalizedResidualBlock(nn.Module):
    def __init__(self, layer: nn.Module, embedding_dim: int, dropout: float):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.layer = layer
        self.dropout_module = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        outputs = self.layer(hidden_states, *args, **kwargs)
        if isinstance(outputs, tuple):
            hidden_states, *extra = outputs
        else:
            hidden_states = outputs
            extra = None
        hidden_states = self.dropout_module(hidden_states)
        hidden_states = residual + hidden_states
        if extra is not None:
            return (hidden_states,) + tuple(extra)
        return hidden_states


class FeedForwardNetwork(nn.Module):
    def __init__(self, embedding_dim: int, ffn_embedding_dim: int, activation_dropout: float):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.ffn_embedding_dim = ffn_embedding_dim
        self.max_tokens_per_msa = MAX_TOKENS_PER_MSA
        self.activation_fn = nn.GELU()
        self.activation_dropout_module = nn.Dropout(activation_dropout)
        self.fc1 = nn.Linear(embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, embedding_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.activation_dropout_module(hidden_states)
        return self.fc2(hidden_states)


class RowSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float, max_tokens_per_msa: int):
        super().__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.max_tokens_per_msa = max_tokens_per_msa
        self.attn_shape = "hnij"
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout_module = nn.Dropout(dropout)

    def align_scaling(self, query: torch.Tensor) -> float:
        return self.scaling / math.sqrt(query.size(0))

    def compute_attention_weights(
        self,
        hidden_states: torch.Tensor,
        scaling: float,
        self_attn_mask: torch.Tensor | None = None,
        self_attn_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_rows, num_cols, batch_size, _ = hidden_states.size()
        query = self.q_proj(hidden_states).view(num_rows, num_cols, batch_size, self.num_heads, self.head_dim)
        key = self.k_proj(hidden_states).view(num_rows, num_cols, batch_size, self.num_heads, self.head_dim)
        query *= scaling
        if self_attn_padding_mask is not None:
            query *= 1 - self_attn_padding_mask.permute(1, 2, 0).unsqueeze(3).unsqueeze(4).to(query)

        attention_weights = torch.einsum(f"rinhd,rjnhd->{self.attn_shape}", query, key)
        if self_attn_mask is not None:
            raise NotImplementedError("RNA-MSM row self-attention masks are not needed for fixtures.")
        if self_attn_padding_mask is not None:
            attention_weights = attention_weights.masked_fill(
                self_attn_padding_mask[:, 0].unsqueeze(0).unsqueeze(2),
                -10000,
            )
        return attention_weights

    def compute_attention_update(self, hidden_states: torch.Tensor, attention_probs: torch.Tensor) -> torch.Tensor:
        num_rows, num_cols, batch_size, embed_dim = hidden_states.size()
        value = self.v_proj(hidden_states).view(num_rows, num_cols, batch_size, self.num_heads, self.head_dim)
        context = torch.einsum(f"{self.attn_shape},rjnhd->rinhd", attention_probs, value)
        context = context.contiguous().view(num_rows, num_cols, batch_size, embed_dim)
        return self.out_proj(context)

    def _batched_forward(
        self,
        hidden_states: torch.Tensor,
        self_attn_mask: torch.Tensor | None = None,
        self_attn_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_rows, num_cols, _, _ = hidden_states.size()
        max_rows = max(1, self.max_tokens_per_msa // num_cols)
        attention_scores = 0
        scaling = self.align_scaling(hidden_states)
        for start in range(0, num_rows, max_rows):
            attention_scores += self.compute_attention_weights(
                hidden_states[start : start + max_rows],
                scaling,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=(
                    self_attn_padding_mask[:, start : start + max_rows] if self_attn_padding_mask is not None else None
                ),
            )
        attention_probs = attention_scores.softmax(-1)
        attention_probs = self.dropout_module(attention_probs)
        outputs = [
            self.compute_attention_update(hidden_states[start : start + max_rows], attention_probs)
            for start in range(0, num_rows, max_rows)
        ]
        return torch.cat(outputs, 0), attention_probs

    def forward(
        self,
        hidden_states: torch.Tensor,
        self_attn_mask: torch.Tensor | None = None,
        self_attn_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_rows, num_cols, _, _ = hidden_states.size()
        if (num_rows * num_cols > self.max_tokens_per_msa) and not torch.is_grad_enabled():
            return self._batched_forward(hidden_states, self_attn_mask, self_attn_padding_mask)
        scaling = self.align_scaling(hidden_states)
        attention_scores = self.compute_attention_weights(
            hidden_states,
            scaling,
            self_attn_mask,
            self_attn_padding_mask,
        )
        attention_probs = attention_scores.softmax(-1)
        attention_probs = self.dropout_module(attention_probs)
        output = self.compute_attention_update(hidden_states, attention_probs)
        return output, attention_probs


class ColumnSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float, max_tokens_per_msa: int):
        super().__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.max_tokens_per_msa = max_tokens_per_msa
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout_module = nn.Dropout(dropout)

    def compute_attention_update(
        self,
        hidden_states: torch.Tensor,
        self_attn_mask: torch.Tensor | None = None,
        self_attn_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_rows, num_cols, batch_size, embed_dim = hidden_states.size()
        if num_rows == 1:
            attention_probs = torch.ones(
                self.num_heads,
                num_cols,
                batch_size,
                num_rows,
                num_rows,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            output = self.out_proj(self.v_proj(hidden_states))
        else:
            query = self.q_proj(hidden_states).view(num_rows, num_cols, batch_size, self.num_heads, self.head_dim)
            key = self.k_proj(hidden_states).view(num_rows, num_cols, batch_size, self.num_heads, self.head_dim)
            value = self.v_proj(hidden_states).view(num_rows, num_cols, batch_size, self.num_heads, self.head_dim)
            query *= self.scaling
            attention_weights = torch.einsum("icnhd,jcnhd->hcnij", query, key)
            if self_attn_mask is not None:
                raise NotImplementedError("RNA-MSM column self-attention masks are not needed for fixtures.")
            if self_attn_padding_mask is not None:
                attention_weights = attention_weights.masked_fill(
                    self_attn_padding_mask.permute(2, 0, 1).unsqueeze(0).unsqueeze(3),
                    -10000,
                )
            attention_probs = attention_weights.softmax(-1)
            attention_probs = self.dropout_module(attention_probs)
            context = torch.einsum("hcnij,jcnhd->icnhd", attention_probs, value)
            context = context.contiguous().view(num_rows, num_cols, batch_size, embed_dim)
            output = self.out_proj(context)
        return output, attention_probs

    def _batched_forward(
        self,
        hidden_states: torch.Tensor,
        self_attn_mask: torch.Tensor | None = None,
        self_attn_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_rows, num_cols, _, _ = hidden_states.size()
        max_cols = max(1, self.max_tokens_per_msa // num_rows)
        outputs = []
        attention_chunks = []
        for start in range(0, num_cols, max_cols):
            output, attention = self(
                hidden_states[:, start : start + max_cols],
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=(
                    self_attn_padding_mask[:, :, start : start + max_cols]
                    if self_attn_padding_mask is not None
                    else None
                ),
            )
            outputs.append(output)
            attention_chunks.append(attention)
        return torch.cat(outputs, 1), torch.cat(attention_chunks, 1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        self_attn_mask: torch.Tensor | None = None,
        self_attn_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_rows, num_cols, _, _ = hidden_states.size()
        if (num_rows * num_cols > self.max_tokens_per_msa) and not torch.is_grad_enabled():
            return self._batched_forward(hidden_states, self_attn_mask, self_attn_padding_mask)
        return self.compute_attention_update(hidden_states, self_attn_mask, self_attn_padding_mask)


class AxialTransformerLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        ffn_embedding_dim: int,
        num_attention_heads: int,
        dropout: float,
        activation_dropout: float,
        max_tokens_per_msa: int,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.dropout_prob = dropout
        self.row_self_attention = self.build_residual(
            RowSelfAttention(embedding_dim, num_attention_heads, dropout, max_tokens_per_msa)
        )
        self.column_self_attention = self.build_residual(
            ColumnSelfAttention(embedding_dim, num_attention_heads, dropout, max_tokens_per_msa)
        )
        self.feed_forward_layer = self.build_residual(
            FeedForwardNetwork(embedding_dim, ffn_embedding_dim, activation_dropout)
        )

    def build_residual(self, layer: nn.Module) -> NormalizedResidualBlock:
        return NormalizedResidualBlock(layer, self.embedding_dim, self.dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        self_attn_mask: torch.Tensor | None = None,
        self_attn_padding_mask: torch.Tensor | None = None,
    ):
        hidden_states, _row_attention = self.row_self_attention(
            hidden_states,
            self_attn_mask=self_attn_mask,
            self_attn_padding_mask=self_attn_padding_mask,
        )
        hidden_states, _column_attention = self.column_self_attention(
            hidden_states,
            self_attn_mask=self_attn_mask,
            self_attn_padding_mask=self_attn_padding_mask,
        )
        hidden_states = self.feed_forward_layer(hidden_states)
        return hidden_states


class OfficialMSATransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_dim = HIDDEN_SIZE
        self.num_attention_heads = NUM_ATTENTION_HEADS
        self.num_layers = NUM_LAYERS
        self.embed_positions_msa = True
        self.dropout = 0.1
        self.attention_dropout = 0.1
        self.activation_dropout = 0.1
        self.max_tokens_per_msa = MAX_TOKENS_PER_MSA
        self.embed_tokens = nn.Embedding(len(OFFICIAL_VOCAB), HIDDEN_SIZE, padding_idx=PAD_TOKEN_ID)
        self.msa_position_embedding = nn.Parameter(0.01 * torch.randn(1, 1024, 1, 1), requires_grad=True)
        self.dropout_module = nn.Dropout(self.dropout)
        self.layers = nn.ModuleList(
            [
                AxialTransformerLayer(
                    embedding_dim=HIDDEN_SIZE,
                    ffn_embedding_dim=4 * HIDDEN_SIZE,
                    num_attention_heads=NUM_ATTENTION_HEADS,
                    dropout=self.dropout,
                    activation_dropout=self.activation_dropout,
                    max_tokens_per_msa=MAX_TOKENS_PER_MSA,
                )
                for _ in range(NUM_LAYERS)
            ]
        )
        self.contact_head = ContactPredictionHead(NUM_LAYERS * NUM_ATTENTION_HEADS)
        self.contact_head.requires_grad_(False)
        self.embed_positions = LearnedPositionalEmbedding(MAX_POSITION_EMBEDDINGS, HIDDEN_SIZE, PAD_TOKEN_ID)
        self.emb_layer_norm_before = nn.LayerNorm(HIDDEN_SIZE)
        self.emb_layer_norm_after = nn.LayerNorm(HIDDEN_SIZE)
        self.lm_head = RobertaLMHead(HIDDEN_SIZE, len(OFFICIAL_VOCAB), self.embed_tokens.weight)
        self.init_weights()

    def init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm) and module.elementwise_affine:
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        tokens: torch.Tensor,
        repr_layers: list[int] | None = None,
    ) -> dict[str, Any]:
        repr_layers = [] if repr_layers is None else repr_layers
        if tokens.ndim != 3:
            raise ValueError(f"Official RNA-MSM expects tokens with 3 dimensions, got {tuple(tokens.shape)}")
        batch_size, num_alignments, sequence_length = tokens.size()
        padding_mask = tokens.eq(PAD_TOKEN_ID)
        if not padding_mask.any():
            padding_mask = None

        hidden_states = self.embed_tokens(tokens.long())
        hidden_states += self.embed_positions(tokens.view(batch_size * num_alignments, sequence_length)).view(
            hidden_states.size()
        )
        if num_alignments > MAX_POSITION_EMBEDDINGS:
            raise RuntimeError(
                "Using model with MSA position embedding trained on maximum MSA "
                f"depth of {MAX_POSITION_EMBEDDINGS}, but received {num_alignments} alignments."
            )
        hidden_states += self.msa_position_embedding[:, :num_alignments]
        hidden_states = self.emb_layer_norm_before(hidden_states)
        hidden_states = self.dropout_module(hidden_states)
        if padding_mask is not None:
            hidden_states = hidden_states * (1 - padding_mask.unsqueeze(-1).type_as(hidden_states))

        repr_layer_set = set(repr_layers)
        hidden_representations = {}
        if 0 in repr_layer_set:
            hidden_representations[0] = hidden_states

        hidden_states = hidden_states.permute(1, 2, 0, 3)
        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                self_attn_padding_mask=padding_mask,
            )
            if (layer_idx + 1) in repr_layer_set:
                hidden_representations[layer_idx + 1] = hidden_states.permute(2, 0, 1, 3)

        hidden_states = self.emb_layer_norm_after(hidden_states)
        hidden_states = hidden_states.permute(2, 0, 1, 3)
        if NUM_LAYERS in repr_layer_set:
            hidden_representations[NUM_LAYERS] = hidden_states

        result = {
            "logits": self.lm_head(hidden_states),
            "representations": hidden_representations,
        }
        return result


def load_model(checkpoint: Path) -> OfficialMSATransformer:
    torch.manual_seed(1016)
    state = torch.load(checkpoint, weights_only=False, map_location=torch.device("cpu"))
    state_dict = state.get("state_dict", state)
    model = OfficialMSATransformer()
    model.load_state_dict(state_dict, strict=True)
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
    input_ids = torch.tensor([encode_mm_single_row_msa(crop["sequence"])], dtype=torch.long)
    upstream_input_ids = torch.tensor([encode_single_row_msa(crop["sequence"])], dtype=torch.long)
    model = load_model(checkpoint)
    with torch.no_grad():
        outputs = model(upstream_input_ids, repr_layers=list(range(NUM_LAYERS + 1)))
    expected = {
        "hidden_states": torch.stack(
            tuple(outputs["representations"][index] for index in range(NUM_LAYERS + 1)),
            dim=0,
        )
        .detach()
        .cpu()
        .contiguous(),
        "logits": remap_logits_to_vocab_subset(outputs["logits"], LOGITS_REMAP),
    }

    out_dir = fixture_out_dir(REPO_ROOT, MODEL, args.case)
    meta: dict[str, Any] = {
        "version": 1,
        "model": MODEL,
        "case": args.case,
        "auto_model": "AutoModelForMaskedLM",
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
        inputs={"input_ids": input_ids.contiguous()},
        expected=expected,
        meta=meta,
    )
    shapes = {key: tuple(value.shape) for key, value in expected.items()}
    print(f"Wrote fixture to {out_dir}")
    print(f"  inputs: {{'input_ids': {tuple(input_ids.shape)}}}")
    print(f"  expected: {shapes}")


if __name__ == "__main__":
    main()
