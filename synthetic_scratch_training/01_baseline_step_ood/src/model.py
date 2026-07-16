"""Tiny decoder-only causal transformer, learned absolute positional embeddings.

Custom (not HF `transformers`) so attention weights are easy to expose later for
mechanistic analysis, and so the model is exactly sized to the 51-token vocab.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import vocab

MAX_LEN = 320  # >= 2*12*12 + 6 = 294 (largest K=12,N=12 sequence)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x, return_attn=False):
        b, t, d = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (b, n_heads, t, head_dim)

        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
        attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=-1)

        out = attn_weights @ v  # (b, n_heads, t, head_dim)
        out = out.transpose(1, 2).reshape(b, t, d)
        out = self.proj(out)
        return (out, attn_weights) if return_attn else (out, None)


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, return_attn=False):
        attn_out, attn_weights = self.attn(self.ln1(x), return_attn=return_attn)
        x = x + self.drop(attn_out)
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x, attn_weights


class TinyTransformer(nn.Module):
    def __init__(self, n_layers=4, d_model=64, n_heads=4, d_ff=256, dropout=0.0,
                 vocab_size=vocab.VOCAB_SIZE, max_len=MAX_LEN):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids, return_attn=False):
        b, t = input_ids.shape
        assert t <= self.max_len, f"sequence length {t} exceeds max_len {self.max_len}"
        pos = torch.arange(t, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        x = self.drop(x)

        attn_maps = []
        for block in self.blocks:
            x, attn_weights = block(x, return_attn=return_attn)
            if return_attn:
                attn_maps.append(attn_weights)

        x = self.ln_f(x)
        logits = self.head(x)
        return (logits, attn_maps) if return_attn else (logits, None)
