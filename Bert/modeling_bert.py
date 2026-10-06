"""BERT for sequence classification, written as plain PyTorch nn.Modules.

This is the architecture shared by the teacher (model_teacher.py) and the student
(model_student.py).

Module and parameter names deliberately mirror Hugging Face's
``BertForSequenceClassification`` (e.g. ``bert.encoder.layer.3.attention.self.query``).
This lets a pretrained HF checkpoint be loaded with a strict ``load_state_dict``.

Unlike the HF model, every encoder layer returns its projected Q, K, V tensors. The
MiniLM loss can then read them directly, without forward hooks.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

CONFIG_FILE = "config.json"
WEIGHTS_FILE = "model.safetensors"


@dataclass
class BertConfig:
    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-12
    initializer_range: float = 0.02
    pad_token_id: int = 0
    num_labels: int = 2

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size={self.hidden_size} is not divisible by num_attention_heads={self.num_attention_heads}"
            )

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


class QKV(NamedTuple):
    """Projected (pre-head-split) query/key/value of one layer, each [B, T, H]."""

    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor


class ClassifierOutput(NamedTuple):
    loss: torch.Tensor | None
    logits: torch.Tensor           # [B, num_labels]
    qkv: tuple[QKV, ...]           # one QKV per encoder layer, index -1 = last layer


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #


class BertEmbeddings(nn.Module):
    """word + position + token-type embeddings -> LayerNorm -> dropout."""

    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings)[None, :], persistent=False)

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = input_ids.size(1)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        embeddings = (
            self.word_embeddings(input_ids)
            + self.position_embeddings(self.position_ids[:, :seq_len])
            + self.token_type_embeddings(token_type_ids)
        )
        return self.dropout(self.LayerNorm(embeddings))


class BertSelfAttention(nn.Module):
    """Multi-head scaled dot-product self-attention that also returns Q, K, V."""

    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, H] -> [B, heads, T, head_dim]."""
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, QKV]:
        q, k, v = self.query(hidden_states), self.key(hidden_states), self.value(hidden_states)
        qh, kh, vh = self._split_heads(q), self._split_heads(k), self._split_heads(v)

        scores = qh @ kh.transpose(-1, -2) / math.sqrt(self.head_dim)            # [B, heads, T, T]
        key_is_padding = ~attention_mask[:, None, None, :].bool()                # [B, 1, 1, T]
        # masked_fill (not "+ mask") so fp16 scores can't overflow to -inf.
        scores = scores.masked_fill(key_is_padding, torch.finfo(scores.dtype).min)
        probs = self.dropout(F.softmax(scores, dim=-1))

        context = (probs @ vh).transpose(1, 2).reshape(hidden_states.shape)     # [B, T, H]
        return context, QKV(q, k, v)


class BertSelfOutput(nn.Module):
    """Attention output projection + residual + LayerNorm."""

    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, context: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return self.LayerNorm(self.dropout(self.dense(context)) + residual)


class BertAttention(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.self = BertSelfAttention(config)   # attribute name "self" matches HF's parameter names
        self.output = BertSelfOutput(config)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, QKV]:
        context, qkv = self.self(hidden_states, attention_mask)
        return self.output(context, hidden_states), qkv


class BertIntermediate(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.dense(hidden_states))  # exact (erf) GELU, as in BERT's "gelu"


class BertOutput(nn.Module):
    """Feed-forward output projection + residual + LayerNorm."""

    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return self.LayerNorm(self.dropout(self.dense(hidden_states)) + residual)


class BertLayer(nn.Module):
    """One Transformer encoder block: self-attention + feed-forward."""

    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, QKV]:
        attn_out, qkv = self.attention(hidden_states, attention_mask)
        return self.output(self.intermediate(attn_out), attn_out), qkv


class BertEncoder(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, tuple[QKV, ...]]:
        all_qkv = []
        for layer in self.layer:
            hidden_states, qkv = layer(hidden_states, attention_mask)
            all_qkv.append(qkv)
        return hidden_states, tuple(all_qkv)


class BertPooler(nn.Module):
    """tanh(W · h_[CLS]) — the sentence representation fed to the classifier."""

    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.dense(hidden_states[:, 0]))


class BertModel(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

    def forward(self, input_ids, attention_mask, token_type_ids=None) -> tuple[torch.Tensor, tuple[QKV, ...]]:
        hidden_states = self.embeddings(input_ids, token_type_ids)
        hidden_states, all_qkv = self.encoder(hidden_states, attention_mask)
        return self.pooler(hidden_states), all_qkv


# --------------------------------------------------------------------------- #
# Classifier (base class of TeacherBert and StudentBert)
# --------------------------------------------------------------------------- #


class BertClassifier(nn.Module):
    """BERT encoder + pooler + linear classification head."""

    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.config = config
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """BERT initialisation: N(0, 0.02) weights, zero biases, identity LayerNorm."""
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> ClassifierOutput:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        pooled, all_qkv = self.bert(input_ids, attention_mask, token_type_ids)
        logits = self.classifier(self.dropout(pooled))
        loss = F.cross_entropy(logits.float(), labels) if labels is not None else None
        return ClassifierOutput(loss=loss, logits=logits, qkv=all_qkv)

    # ---- persistence: config.json + model.safetensors ----

    def save_pretrained(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / CONFIG_FILE).write_text(json.dumps(asdict(self.config), indent=2))
        state = {k: v.detach().contiguous().cpu() for k, v in self.state_dict().items()}
        save_file(state, str(directory / WEIGHTS_FILE), metadata={"format": "pt"})

    @classmethod
    def from_pretrained(cls, directory: str | Path) -> "BertClassifier":
        directory = Path(directory)
        config = BertConfig(**json.loads((directory / CONFIG_FILE).read_text()))
        model = cls(config)
        model.load_state_dict(load_file(str(directory / WEIGHTS_FILE)), strict=True)
        return model
