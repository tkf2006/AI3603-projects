"""Decoder-only GPT model for the TinyStories assignment.

The evaluator imports ``GPT`` from this file and constructs it with
``GPT(GPTConfig(**config_json))``.  Training code belongs in a separate file so
importing this module never starts a training job.
"""

import math
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed():
    """Set the assignment's random seeds before constructing the model."""
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class GPTConfig:
    block_size: int = 512
    batch_size: int = 12  # Training choice; not part of the model weights.
    n_layer: int = 6
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    vocab_size: int = 8192


class SingleHeadAttention(nn.Module):
    """One head of causal self-attention: (B, T, C) -> (B, T, head_size)."""

    def __init__(self, config):
        super().__init__()
        self.head_size = config.n_embd // config.n_head
        self.query = nn.Linear(config.n_embd, self.head_size)
        self.key = nn.Linear(config.n_embd, self.head_size)
        self.value = nn.Linear(config.n_embd, self.head_size)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool)),
            persistent=False,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        _, seq_len, _ = x.shape
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        scores = query @ key.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_size)
        mask = self.causal_mask[:seq_len, :seq_len]
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        # Softmax in float32 is more stable when the V100 uses FP16 autocast.
        weights = F.softmax(scores.float(), dim=-1).to(value.dtype)
        weights = self.dropout(weights)
        return weights @ value


class CausalSelfAttention(SingleHeadAttention):
    """Named causal-attention component used by the multi-head layer."""


class MultiHeadAttention(nn.Module):
    """Concatenate independently masked heads and project back to n_embd."""

    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalSelfAttention(config) for _ in range(config.n_head)]
        )
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """Position-wise MLP with the GPT-style 4x hidden dimension."""

    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Pre-norm Transformer block with attention and MLP residuals."""

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.att = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffn = FeedForward(config)

    def forward(self, x):
        x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    """GPT-style decoder with token and learned position embeddings."""

    def __init__(self, config):
        super().__init__()
        if config.block_size < 1:
            raise ValueError("block_size must be positive")
        if config.n_layer < 1 or config.n_head < 1 or config.n_embd < 1:
            raise ValueError("n_layer, n_head and n_embd must be positive")
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if config.vocab_size < 1:
            raise ValueError("vocab_size must be positive")
        if not 0.0 <= config.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_final = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """Return ``(logits, loss)`` for token IDs with shape (B, T)."""
        if idx.ndim != 2:
            raise ValueError("idx must have shape (batch, sequence)")
        batch_size, seq_len = idx.shape
        if not 1 <= seq_len <= self.config.block_size:
            raise ValueError(
                f"sequence length must be between 1 and {self.config.block_size}"
            )

        positions = torch.arange(seq_len, device=idx.device)
        token_embedding = self.token_embedding_table(idx)
        position_embedding = self.position_embedding_table(positions)
        x = self.blocks(token_embedding + position_embedding)
        logits = self.lm_head(self.ln_final(x))

        loss = None
        if targets is not None:
            if targets.shape != idx.shape:
                raise ValueError("targets must have the same shape as idx")
            loss = F.cross_entropy(
                logits.reshape(batch_size * seq_len, self.config.vocab_size).float(),
                targets.reshape(batch_size * seq_len),
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, block_size=None):
        """Greedily append tokens, cropping context to the model's limit."""
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens cannot be negative")
        context_size = self.config.block_size if block_size is None else block_size
        if not 1 <= context_size <= self.config.block_size:
            raise ValueError("block_size must be within the model's context limit")

        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                logits, _ = self(idx[:, -context_size:])
                next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                idx = torch.cat((idx, next_id), dim=1)
        finally:
            self.train(was_training)
        return idx
