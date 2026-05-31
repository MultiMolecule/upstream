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

"""Generate the RibonanzaNet golden fixture from official upstream weights.

The official RibonanzaNet repository publishes the model as script-level
``Network.py`` and ``dropout.py`` modules rather than an importable package.
This generator keeps a compact, faithful reconstruction of the inference path
needed by ``RibonanzaNet.pt`` and loads the local checkpoint directly.

The fixture emits the two official chemical-mapping heads as
``logits_2a3`` and ``logits_dms`` so it validates the checkpoint head rather
than only the shared backbone.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from functools import partialmethod
from pathlib import Path

import torch
from torch import Tensor, einsum, nn
from torch.nn import functional as F

HERE = Path(__file__).resolve()
MODEL_DIR = HERE.parent
REPO_ROOT = MODEL_DIR.parent.parent  # upstream/

sys.path.insert(0, str(REPO_ROOT))
from _corpus.load import crop_record, sequence_sha256  # noqa: E402
from _shared.download import fetch_kaggle_file, verify_sha256  # noqa: E402
from _shared.fixture import (  # noqa: E402
    fixture_out_dir,
    inputs_source_from_anchor_crop,
    sha256_of_file,
    write_fixture_artifacts,
)

MODEL = "ribonanzanet"
DEFAULT_CASE = "ribonanzanet"
# RibonanzaNet fixtures compare multiple heads from an upstream PyTorch checkpoint.
ATOL = 1e-3
RTOL = 1e-4

UPSTREAM_REPO_URL = "https://github.com/Shujun-He/RibonanzaNet"
UPSTREAM_COMMIT = "dbf0fd5862e15e4d4f1bd72a082b2d94cfb2d706"
OFFICIAL_NETWORK_URL = "https://raw.githubusercontent.com/Shujun-He/RibonanzaNet/" f"{UPSTREAM_COMMIT}/Network.py"
OFFICIAL_DROPOUT_URL = "https://raw.githubusercontent.com/Shujun-He/RibonanzaNet/" f"{UPSTREAM_COMMIT}/dropout.py"
CHECKPOINT_ROOT_ENV_VAR = "MULTIMOLECULE_UPSTREAM_RIBONANZANET_CHECKPOINT_DIR"

CORPUS_RECORD_ID = "rna/grch38_chr21_transcribed"
CROP_NAME = "rna_50nt"
CROP_CENTER = "center"
CROP_LENGTH = 50

UPSTREAM_TOKEN_IDS = {"A": 0, "C": 1, "G": 2, "U": 3, "N": 4}
MM_VOCAB = ["<pad>", "<cls>", "<eos>", "<unk>", "<mask>", "<null>"] + list("ACGUNRYSWKMBDHVIX|.*-?")
MM_TOKEN_IDS = {token: index for index, token in enumerate(MM_VOCAB)}


@dataclass
class Variant:
    case: str
    checkpoint_filename: str
    checkpoint_source: str
    checkpoint_sha256: str
    auto_model: str
    variant: str
    outputs: tuple[str, ...]
    nclass: int = 2

    @property
    def checkpoint_env_var(self) -> str:
        return f"MULTIMOLECULE_UPSTREAM_{self.case.upper().replace('-', '_')}_CHECKPOINT"

    @property
    def checkpoint_path(self) -> Path:
        if self.checkpoint_env_var in os.environ:
            path = Path(os.environ[self.checkpoint_env_var]).expanduser().resolve()
            verify_sha256(
                path,
                self.checkpoint_sha256,
                description=f"RibonanzaNet {self.variant} checkpoint",
            )
            return path
        root = os.environ.get(CHECKPOINT_ROOT_ENV_VAR)
        if root:
            path = Path(root).expanduser().resolve() / self.checkpoint_filename
            verify_sha256(
                path,
                self.checkpoint_sha256,
                description=f"RibonanzaNet {self.variant} checkpoint",
            )
            return path
        return fetch_kaggle_file(
            self.checkpoint_source,
            self.checkpoint_filename,
            cache_prefix=MODEL,
            env_var=self.checkpoint_env_var,
            sha256=self.checkpoint_sha256,
            description=f"RibonanzaNet {self.variant} checkpoint",
        )


VARIANTS = {
    "ribonanzanet": Variant(
        case="ribonanzanet",
        checkpoint_filename="RibonanzaNet.pt",
        checkpoint_source="kaggle://datasets/shujun717/ribonanzanet-weights/RibonanzaNet.pt",
        checkpoint_sha256="c2aa45c14367863ece52d528d6c353ef40b66f7cb41539c19a042e87c7d3f215",
        auto_model="AutoModelForPreTraining",
        variant="chemical_mapping",
        outputs=("logits_2a3", "logits_dms"),
    ),
    "ribonanzanet-ss": Variant(
        case="ribonanzanet-ss",
        checkpoint_filename="RibonanzaNet-SS.pt",
        checkpoint_source="kaggle://datasets/shujun717/ribonanzanet-weights/RibonanzaNet-SS.pt",
        checkpoint_sha256="626060952368affbf61b78b532d6166387094754b68bc0553da376f2d2b00d56",
        auto_model="AutoModelForRnaSecondaryStructurePrediction",
        variant="secondary_structure",
        outputs=("logits_2a3", "logits_dms", "logits_ss"),
    ),
    "ribonanzanet-drop": Variant(
        case="ribonanzanet-drop",
        checkpoint_filename="RibonanzaNet-Drop.pt",
        checkpoint_source="kaggle://datasets/shujun717/ribonanzanet-weights/RibonanzaNet-Drop.pt",
        checkpoint_sha256="8e423126b5686a48499d0a22d8b234e01092b757fe773ed4e6519dd8c1cf0bdb",
        auto_model="RibonanzaNetForSequenceDropoutPrediction",
        variant="sequence_dropout",
        outputs=("logits_2a3", "logits_dms"),
    ),
    "ribonanzanet-deg": Variant(
        case="ribonanzanet-deg",
        checkpoint_filename="RibonanzaNet-Deg.pt",
        checkpoint_source="kaggle://datasets/shujun717/ribonanzanet-weights/RibonanzaNet-Deg.pt",
        checkpoint_sha256="8df072e9992cdff546f831095d8e518f7ba2421cbde371b0dad45a9f1476af37",
        auto_model="RibonanzaNetForDegradationPrediction",
        variant="degradation",
        outputs=(
            "logits_reactivity",
            "logits_deg_Mg_pH10",
            "logits_deg_pH10",
            "logits_deg_Mg_50C",
            "logits_deg_50C",
        ),
        nclass=5,
    ),
}


class OfficialConfig:
    """Hyperparameters from upstream ``configs/pairwise.yaml``."""

    def __init__(self, nclass: int = 2, use_triangular_attention: bool = False):
        self.dropout = 0.05
        self.k = 5
        self.ninp = 256
        self.nlayers = 9
        self.nclass = nclass
        self.ntoken = 5
        self.nhead = 8
        self.use_triangular_attention = use_triangular_attention
        self.pairwise_dimension = 64


class SharedDropout(nn.Module):
    """Dropout with a mask shared across selected dimensions."""

    def __init__(self, rate: float, batch_dim: int | list[int]):
        super().__init__()
        self.batch_dim = [batch_dim] if isinstance(batch_dim, int) else batch_dim
        self.dropout = nn.Dropout(rate)

    def forward(self, x: Tensor) -> Tensor:
        shape = list(x.shape)
        for batch_dim in self.batch_dim:
            shape[batch_dim] = 1
        mask = x.new_ones(shape)
        return x * self.dropout(mask)


class DropoutRowwise(SharedDropout):
    __init__ = partialmethod(SharedDropout.__init__, batch_dim=-3)


class DropoutColumnwise(SharedDropout):
    __init__ = partialmethod(SharedDropout.__init__, batch_dim=-2)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature: float, attn_dropout: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Tensor | None = None,
        attn_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        attn = torch.matmul(q, k.transpose(2, 3)) / self.temperature
        if mask is not None:
            attn = attn + mask
        if attn_mask is not None:
            attn = attn.float().masked_fill(attn_mask == -1, float("-1e-9"))
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_k: int, d_v: int, dropout: float = 0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        self.attention = ScaledDotProductAttention(temperature=d_k**0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Tensor | None = None,
        src_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        batch_size, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q

        q = self.w_qs(q).view(batch_size, len_q, n_head, d_k)
        k = self.w_ks(k).view(batch_size, len_k, n_head, d_k)
        v = self.w_vs(v).view(batch_size, len_v, n_head, d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if src_mask is not None:
            src_mask = src_mask.clone()
            src_mask[src_mask == 0] = -1
            src_mask = src_mask.unsqueeze(-1).float()
            attn_mask = torch.matmul(src_mask, src_mask.permute(0, 2, 1)).unsqueeze(1)
            q, attn = self.attention(q, k, v, mask=mask, attn_mask=attn_mask)
        else:
            q, attn = self.attention(q, k, v, mask=mask)

        q = q.transpose(1, 2).contiguous().view(batch_size, len_q, -1)
        q = self.dropout(self.fc(q))
        q = self.layer_norm(q + residual)
        return q, attn


class OuterProductMean(nn.Module):
    def __init__(self, in_dim: int = 256, dim_msa: int = 32, pairwise_dim: int = 64):
        super().__init__()
        self.proj_down1 = nn.Linear(in_dim, dim_msa)
        self.proj_down2 = nn.Linear(dim_msa**2, pairwise_dim)

    def forward(self, seq_rep: Tensor, pair_rep: Tensor | None = None) -> Tensor:
        seq_rep = self.proj_down1(seq_rep)
        outer_product = torch.einsum("bid,bjc->bijcd", seq_rep, seq_rep)
        outer_product = outer_product.reshape(*outer_product.shape[:-2], -1)
        outer_product = self.proj_down2(outer_product)
        if pair_rep is not None:
            outer_product = outer_product + pair_rep
        return outer_product


class RelativePosition(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.linear = nn.Linear(17, dim)

    def forward(self, src: Tensor) -> Tensor:
        length = src.shape[1]
        res_id = torch.arange(length, device=src.device).unsqueeze(0)
        bin_values = torch.arange(-8, 9, device=src.device)
        distance = res_id[:, :, None] - res_id[:, None, :]
        boundary = torch.tensor(8, device=src.device)
        distance = torch.minimum(torch.maximum(-boundary, distance), boundary)
        distance_onehot = (distance[..., None] == bin_values).float()
        if distance_onehot.sum(dim=-1).min() != 1:
            raise AssertionError("relative-position one-hot encoding is invalid")
        return self.linear(distance_onehot)


class TriangleMultiplicativeModule(nn.Module):
    def __init__(self, *, dim: int, hidden_dim: int | None = None, mix: str = "ingoing"):
        super().__init__()
        if mix not in {"ingoing", "outgoing"}:
            raise ValueError("mix must be either 'ingoing' or 'outgoing'")
        hidden_dim = dim if hidden_dim is None else hidden_dim
        self.norm = nn.LayerNorm(dim)
        self.left_proj = nn.Linear(dim, hidden_dim)
        self.right_proj = nn.Linear(dim, hidden_dim)
        self.left_gate = nn.Linear(dim, hidden_dim)
        self.right_gate = nn.Linear(dim, hidden_dim)
        self.out_gate = nn.Linear(dim, hidden_dim)
        for gate in (self.left_gate, self.right_gate, self.out_gate):
            nn.init.constant_(gate.weight, 0.0)
            nn.init.constant_(gate.bias, 1.0)
        self.mix_einsum_eq = (
            "... i k d, ... j k d -> ... i j d" if mix == "outgoing" else "... k j d, ... k i d -> ... i j d"
        )
        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim)

    def forward(self, x: Tensor, src_mask: Tensor) -> Tensor:
        src_mask = src_mask.unsqueeze(-1).float()
        mask = torch.matmul(src_mask, src_mask.permute(0, 2, 1)).unsqueeze(-1)
        if x.shape[1] != x.shape[2]:
            raise AssertionError("feature map must be symmetrical")

        x = self.norm(x)
        left = self.left_proj(x) * mask
        right = self.right_proj(x) * mask
        left = left * self.left_gate(x).sigmoid()
        right = right * self.right_gate(x).sigmoid()
        out_gate = self.out_gate(x).sigmoid()

        out = einsum(self.mix_einsum_eq, left, right)
        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)


class TriangleAttention(nn.Module):
    def __init__(self, in_dim: int = 128, dim: int = 32, n_heads: int = 4, wise: str = "row"):
        super().__init__()
        self.n_heads = n_heads
        self.wise = wise
        self.norm = nn.LayerNorm(in_dim)
        self.to_qkv = nn.Linear(in_dim, dim * 3 * n_heads, bias=False)
        self.linear_for_pair = nn.Linear(in_dim, n_heads, bias=False)
        self.to_gate = nn.Sequential(nn.Linear(in_dim, in_dim), nn.Sigmoid())
        self.to_out = nn.Linear(n_heads * dim, in_dim)

    def forward(self, z: Tensor, src_mask: Tensor) -> Tensor:
        src_mask = src_mask.clone()
        src_mask[src_mask == 0] = -1
        src_mask = src_mask.unsqueeze(-1).float()
        attn_mask = torch.matmul(src_mask, src_mask.permute(0, 2, 1))
        z = self.norm(z)
        q, k, v = torch.chunk(self.to_qkv(z), 3, -1)
        q, k, v = (item.reshape(*item.shape[:-1], self.n_heads, -1) for item in (q, k, v))
        bias = self.linear_for_pair(z)
        gate = self.to_gate(z)
        scale = q.size(-1) ** 0.5

        if self.wise == "row":
            logits = torch.einsum("brihd,brjhd->brijh", q, k) / scale
            bias = bias.unsqueeze(1)
            attn_mask = attn_mask[:, None, :, :, None]
            softmax_dim = 3
            multi_eq = "brijh,brjhd->brihd"
        elif self.wise == "col":
            logits = torch.einsum("bilhd,bjlhd->bijlh", q, k) / scale
            bias = bias.unsqueeze(3)
            attn_mask = attn_mask[:, :, :, None, None]
            softmax_dim = 2
            multi_eq = "bijlh,bjlhd->bilhd"
        else:
            raise ValueError("wise should be col or row")

        logits = (logits + bias).masked_fill(attn_mask == -1, float("-1e-9"))
        attn = logits.softmax(softmax_dim)
        out = torch.einsum(multi_eq, attn, v)
        out = gate * out.reshape(*out.shape[:-2], -1)
        return self.to_out(out)


class ConvTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        pairwise_dimension: int,
        use_triangular_attention: bool,
        dropout: float = 0.1,
        k: int = 3,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, d_model // nhead, d_model // nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.pairwise2heads = nn.Linear(pairwise_dimension, nhead, bias=False)
        self.pairwise_norm = nn.LayerNorm(pairwise_dimension)
        self.activation = nn.GELU()
        self.conv = nn.Conv1d(d_model, d_model, k, padding=k // 2)
        self.triangle_update_out = TriangleMultiplicativeModule(dim=pairwise_dimension, mix="outgoing")
        self.triangle_update_in = TriangleMultiplicativeModule(dim=pairwise_dimension, mix="ingoing")
        self.pair_dropout_out = DropoutRowwise(dropout)
        self.pair_dropout_in = DropoutRowwise(dropout)
        self.use_triangular_attention = use_triangular_attention
        if self.use_triangular_attention:
            self.triangle_attention_out = TriangleAttention(
                in_dim=pairwise_dimension,
                dim=pairwise_dimension // 4,
                wise="row",
            )
            self.triangle_attention_in = TriangleAttention(
                in_dim=pairwise_dimension,
                dim=pairwise_dimension // 4,
                wise="col",
            )
            self.pair_attention_dropout_out = DropoutRowwise(dropout)
            self.pair_attention_dropout_in = DropoutColumnwise(dropout)
        self.outer_product_mean = OuterProductMean(in_dim=d_model, pairwise_dim=pairwise_dimension)
        self.pair_transition = nn.Sequential(
            nn.LayerNorm(pairwise_dimension),
            nn.Linear(pairwise_dimension, pairwise_dimension * 4),
            nn.ReLU(inplace=True),
            nn.Linear(pairwise_dimension * 4, pairwise_dimension),
        )

    def forward(self, src: Tensor, pairwise_features: Tensor, src_mask: Tensor) -> tuple[Tensor, Tensor]:
        src = src * src_mask.float().unsqueeze(-1)
        src = src + self.conv(src.permute(0, 2, 1)).permute(0, 2, 1)
        src = self.norm3(src)

        pairwise_bias = self.pairwise2heads(self.pairwise_norm(pairwise_features)).permute(0, 3, 1, 2)
        src2, _ = self.self_attn(src, src, src, mask=pairwise_bias, src_mask=src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        pairwise_features = pairwise_features + self.outer_product_mean(src)
        pairwise_features = pairwise_features + self.pair_dropout_out(
            self.triangle_update_out(pairwise_features, src_mask)
        )
        pairwise_features = pairwise_features + self.pair_dropout_in(
            self.triangle_update_in(pairwise_features, src_mask)
        )
        if self.use_triangular_attention:
            pairwise_features = pairwise_features + self.pair_attention_dropout_out(
                self.triangle_attention_out(pairwise_features, src_mask)
            )
            pairwise_features = pairwise_features + self.pair_attention_dropout_in(
                self.triangle_attention_in(pairwise_features, src_mask)
            )
        pairwise_features = pairwise_features + self.pair_transition(pairwise_features)
        return src, pairwise_features


class OfficialRibonanzaNet(nn.Module):
    def __init__(self, config: OfficialConfig):
        super().__init__()
        self.config = config
        hidden_size = config.ninp * 4
        layers = []
        for index in range(config.nlayers):
            kernel = config.k if index != config.nlayers - 1 else 1
            layers.append(
                ConvTransformerEncoderLayer(
                    d_model=config.ninp,
                    nhead=config.nhead,
                    dim_feedforward=hidden_size,
                    pairwise_dimension=config.pairwise_dimension,
                    use_triangular_attention=config.use_triangular_attention,
                    dropout=config.dropout,
                    k=kernel,
                )
            )
        self.transformer_encoder = nn.ModuleList(layers)
        self.encoder = nn.Embedding(config.ntoken, config.ninp, padding_idx=4)
        self.decoder = nn.Linear(config.ninp, config.nclass)
        self.outer_product_mean = OuterProductMean(in_dim=config.ninp, pairwise_dim=config.pairwise_dimension)
        self.pos_encoder = RelativePosition(config.pairwise_dimension)

    def forward(
        self,
        src: Tensor,
        src_mask: Tensor,
        *,
        return_hidden: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        batch_size, length = src.shape
        src = self.encoder(src).reshape(batch_size, length, -1)
        pairwise_features = self.outer_product_mean(src)
        pairwise_features = pairwise_features + self.pos_encoder(src)
        for layer in self.transformer_encoder:
            src, pairwise_features = layer(src, pairwise_features, src_mask)
        logits = self.decoder(src).squeeze(-1) + pairwise_features.mean() * 0
        if return_hidden:
            return logits, src, pairwise_features
        return logits


class OfficialRibonanzaNetForSecondaryStructure(OfficialRibonanzaNet):
    def __init__(self, config: OfficialConfig):
        super().__init__(config)
        self.ct_predictor = nn.Linear(config.pairwise_dimension, 1)

    def forward(self, src: Tensor, src_mask: Tensor) -> dict[str, Tensor]:
        logits, _, pairwise_features = super().forward(src, src_mask, return_hidden=True)
        if logits.shape[-1] != 2:
            raise AssertionError(f"expected two chemical-mapping channels, got {tuple(logits.shape)}")
        pairwise_features = pairwise_features + pairwise_features.transpose(1, 2)
        return {
            "logits_2a3": logits[..., 0:1].contiguous(),
            "logits_dms": logits[..., 1:2].contiguous(),
            "logits_ss": self.ct_predictor(pairwise_features).contiguous(),
        }


class OfficialSequenceDropoutHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(hidden_size, num_labels)])

    def forward(self, hidden: Tensor) -> Tensor:
        return self.layers[0](hidden).mean(dim=1).exp()


class OfficialRibonanzaNetForSequenceDropout(nn.Module):
    def __init__(self, config: OfficialConfig):
        super().__init__()
        self.model = OfficialRibonanzaNet(config)
        self.head = OfficialSequenceDropoutHead(config.ninp, config.nclass)

    def forward(self, src: Tensor, src_mask: Tensor) -> dict[str, Tensor]:
        _, hidden, _ = self.model(src, src_mask, return_hidden=True)
        logits = self.head(hidden)
        if logits.shape[-1] != 2:
            raise AssertionError(f"expected two sequence-dropout channels, got {tuple(logits.shape)}")
        return {
            "logits_2a3": logits[..., 0:1].contiguous(),
            "logits_dms": logits[..., 1:2].contiguous(),
        }


def tokenize(sequence: str, vocab: dict[str, int]) -> list[int]:
    try:
        return [vocab[base] for base in sequence.upper()]
    except KeyError as error:
        raise ValueError(f"unsupported RibonanzaNet token {error.args[0]!r}") from error


def load_upstream_model(case: str) -> nn.Module:
    variant = VARIANTS[case]
    checkpoint_path = variant.checkpoint_path
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"RibonanzaNet checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    checkpoint = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    config = OfficialConfig(
        nclass=variant.nclass,
        use_triangular_attention=any("triangle_attention" in key for key in checkpoint),
    )
    if case == "ribonanzanet-ss":
        model = OfficialRibonanzaNetForSecondaryStructure(config)
    elif case == "ribonanzanet-drop":
        model = OfficialRibonanzaNetForSequenceDropout(config)
    else:
        model = OfficialRibonanzaNet(config)
    result = model.load_state_dict(checkpoint, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            "RibonanzaNet state dict did not load exactly: "
            f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )
    return model.eval()


def upstream_forward(case: str, upstream_ids: torch.Tensor, upstream_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    model = load_upstream_model(case)
    with torch.no_grad():
        output = model(upstream_ids, upstream_mask)
    if isinstance(output, dict):
        return output
    if case == "ribonanzanet-deg":
        if output.shape[-1] != 5:
            raise AssertionError(f"expected five degradation channels, got {tuple(output.shape)}")
        return {
            "logits_reactivity": output[..., 0:1].contiguous(),
            "logits_deg_Mg_pH10": output[..., 1:2].contiguous(),
            "logits_deg_pH10": output[..., 2:3].contiguous(),
            "logits_deg_Mg_50C": output[..., 3:4].contiguous(),
            "logits_deg_50C": output[..., 4:5].contiguous(),
        }
    if output.shape[-1] != 2:
        raise AssertionError(f"expected two chemical-mapping channels, got {tuple(output.shape)}")
    return {
        "logits_2a3": output[..., 0:1].contiguous(),
        "logits_dms": output[..., 1:2].contiguous(),
    }


def write_case(case: str) -> None:
    variant = VARIANTS[case]
    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    crop = crop_record(CORPUS_RECORD_ID, CROP_LENGTH, center=CROP_CENTER)
    sequence = crop["sequence"]
    if len(sequence) != CROP_LENGTH:
        raise AssertionError(f"crop length {len(sequence)} != {CROP_LENGTH}")
    if sequence_sha256(sequence) != crop["sha256"]:
        raise AssertionError("crop sha256 mismatch")

    upstream_ids = torch.tensor([tokenize(sequence, UPSTREAM_TOKEN_IDS)], dtype=torch.long)
    upstream_mask = torch.ones_like(upstream_ids)
    input_ids = torch.tensor([tokenize(sequence, MM_TOKEN_IDS)], dtype=torch.long)
    expected = upstream_forward(case, upstream_ids, upstream_mask)

    out_dir = fixture_out_dir(REPO_ROOT, MODEL, case)

    meta = {
        "version": 1,
        "model": MODEL,
        "case": case,
        "auto_model": variant.auto_model,
        "outputs": list(variant.outputs),
        "tolerance": {"atol": ATOL, "rtol": RTOL},
        "inputs_source": inputs_source_from_anchor_crop(
            crop,
            crop_name=CROP_NAME,
        ),
        "upstream": {
            "repository": UPSTREAM_REPO_URL,
            "commit": UPSTREAM_COMMIT,
            "checkpoint_source": variant.checkpoint_source,
            "checkpoint_sha256": sha256_of_file(variant.checkpoint_path),
        },
    }
    write_fixture_artifacts(
        out_dir,
        inputs={"input_ids": input_ids},
        expected=expected,
        meta=meta,
    )

    summary = {key: tuple(value.shape) for key, value in expected.items()}
    print(f"Wrote fixture to {out_dir}")
    print(f"  input_ids: {tuple(input_ids.shape)}")
    print(f"  sequence_sha256: {crop['sha256']}")
    print(f"  checkpoint_sha256: {meta['upstream']['checkpoint_sha256']}")
    print(f"  shapes: {summary}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", nargs="?", default=DEFAULT_CASE)
    args = parser.parse_args()
    if args.case not in VARIANTS:
        supported = ", ".join(sorted(VARIANTS))
        raise SystemExit(f"Unsupported RibonanzaNet case {args.case!r}; expected one of: {supported}")
    write_case(args.case)


if __name__ == "__main__":
    main()
