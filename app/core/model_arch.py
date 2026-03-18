"""
model_arch.py
Copy chính xác toàn bộ kiến trúc model từ notebook training.
Không sửa gì để đảm bảo load checkpoint không bị lỗi.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# ── Constants ──────────────────────────────────────────────────────────────
FEATURE_COLS: list[str] = [f"feature_{i}" for i in range(1, 67)]


# ── Vocabulary ─────────────────────────────────────────────────────────────
class Vocabulary:
    PAD_IDX = 0
    UNK_IDX = 1

    def __init__(self):
        self.token2idx: dict = {}
        self._size = 2

    def build(self, X: pd.DataFrame, min_freq: int = 1) -> None:
        flat = X[FEATURE_COLS].values.ravel()
        freq = Counter(flat[~pd.isnull(flat)])
        for token, count in freq.items():
            if count >= min_freq:
                self.token2idx[token] = self._size
                self._size += 1

    def encode(self, val) -> int:
        return self.token2idx.get(val, self.UNK_IDX)

    @property
    def size(self):
        return self._size


# ── Dataset (inference-only, không cần labels) ────────────────────────────
class BehaviorDataset(Dataset):
    def __init__(self, X, vocab, max_seq_len, Y=None, class_maps=None):
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        raw = X[FEATURE_COLS].values
        N = len(raw)
        _enc = np.vectorize(
            lambda v: vocab.token2idx.get(v, vocab.UNK_IDX)
            if not (isinstance(v, float) and math.isnan(v))
            else vocab.PAD_IDX
        )
        encoded = _enc(raw).astype(np.int32)
        is_nan = pd.isnull(raw)
        lengths = (~is_nan).sum(axis=1)

        self.seqs  = np.zeros((N, max_seq_len), dtype=np.int32)
        self.masks = np.zeros((N, max_seq_len), dtype=bool)
        self.meta  = np.zeros((N, 3),           dtype=np.float32)

        for i in range(N):
            L   = int(lengths[i])
            enc = encoded[i, :L]
            if L > max_seq_len:
                enc = enc[-max_seq_len:]
                L   = max_seq_len
            self.seqs[i, :L]  = enc
            self.masks[i, :L] = True
            self.meta[i, 0]   = L / max_seq_len
            self.meta[i, 1]   = len(set(enc.tolist())) / max(L, 1)
            self.meta[i, 2]   = (enc == vocab.UNK_IDX).sum() / max(L, 1)

        if Y is not None and class_maps is not None:
            label_mat = np.zeros((N, len(class_maps)), dtype=np.int64)
            target_cols = list(class_maps.keys())
            for j, col in enumerate(target_cols):
                cmap = class_maps[col]
                label_mat[:, j] = [cmap.get(v, 0) for v in Y[col].values]
            self.labels = label_mat
        else:
            self.labels = None

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        seq  = torch.from_numpy(self.seqs[i]).long()
        mask = torch.from_numpy(self.masks[i])
        meta = torch.from_numpy(self.meta[i])
        if self.labels is not None:
            return seq, mask, meta, torch.from_numpy(self.labels[i])
        return seq, mask, meta


# ── Model Architecture ─────────────────────────────────────────────────────

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len + 1)

    def _build_cache(self, seq_len):
        t     = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb   = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cache", emb.cos())
        self.register_buffer("sin_cache", emb.sin())

    @staticmethod
    def _rotate_half(x):
        h = x.shape[-1] // 2
        return torch.cat([-x[..., h:], x[..., :h]], dim=-1)

    def apply_rope(self, x, seq_len):
        cos = self.cos_cache[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cache[:seq_len].unsqueeze(0).unsqueeze(0)
        return x * cos + self._rotate_half(x) * sin


class MultiHeadAttentionRoPE(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout, max_seq_len, rope_base=10000):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = embed_dim // n_heads
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj  = nn.Linear(embed_dim, embed_dim)
        self.dropout   = dropout
        self.rope      = RotaryEmbedding(self.head_dim, max_seq_len, rope_base)
        self.last_attn_weight = None  # lưu để phân tích heatmap

    def forward(self, x, key_padding_mask=None):
        B, T, D = x.shape
        H, Dh   = self.n_heads, self.head_dim
        qkv = self.qkv_proj(x).reshape(B, T, 3, H, Dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.rope.apply_rope(q, T)
        k = self.rope.apply_rope(k, T)
        scale      = self.head_dim ** -0.5
        attn_score = torch.matmul(q, k.transpose(-2, -1)) * scale
        if key_padding_mask is not None:
            bias = torch.zeros(B, 1, 1, T, device=x.device, dtype=x.dtype)
            bias = bias.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )
            attn_score = attn_score + bias
        attn_weight = torch.softmax(attn_score, dim=-1)
        self.last_attn_weight = attn_weight.detach()  # (B, H, T+1, T+1)
        if self.training and self.dropout > 0:
            attn_weight = F.dropout(attn_weight, p=self.dropout)
        out = torch.matmul(attn_weight, v).transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, embed_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout, max_seq_len, rope_base):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn  = MultiHeadAttentionRoPE(embed_dim, n_heads, dropout, max_seq_len, rope_base)
        self.ffn   = FeedForward(embed_dim, dropout)

    def forward(self, x, key_padding_mask=None):
        x = x + self.attn(self.norm1(x), key_padding_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerEncoderRoPE(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_layers, n_heads, max_seq_len, dropout, rope_base):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_emb   = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.seg_emb     = nn.Embedding(5, embed_dim)
        self.cls_token   = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.input_drop  = nn.Dropout(dropout)
        self.layers      = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, dropout, max_seq_len + 1, rope_base)
            for _ in range(n_layers)
        ])
        self.final_norm  = nn.LayerNorm(embed_dim)

    def _segment_ids(self, mask):
        ratio = mask.cumsum(dim=1).float() / (mask.sum(dim=1, keepdim=True).float() + 1e-9)
        seg   = torch.full(mask.shape, 4, dtype=torch.long, device=mask.device)
        seg[mask & (ratio <= 0.33)]                    = 1
        seg[mask & (ratio > 0.33) & (ratio <= 0.67)]  = 2
        seg[mask & (ratio > 0.67)]                     = 3
        return seg

    def forward(self, seq, mask):
        B, T    = seq.shape
        tok     = self.token_emb(seq)
        seg     = self.seg_emb(self._segment_ids(mask))
        cls     = self.cls_token.expand(B, -1, -1)
        cls_seg = self.seg_emb(torch.zeros(B, 1, dtype=torch.long, device=seq.device))
        x       = torch.cat([cls + cls_seg, tok + seg], dim=1)
        x       = self.input_drop(x)
        pad_mask = torch.cat(
            [torch.zeros(B, 1, dtype=torch.bool, device=mask.device), ~mask], dim=1
        )
        for layer in self.layers:
            x = layer(x, pad_mask)
        x        = self.final_norm(x)
        cls_out  = x[:, 0, :]
        weights  = mask.float().unsqueeze(-1)
        mean_out = (x[:, 1:, :] * weights).sum(1) / (weights.sum(1) + 1e-9)
        return cls_out, mean_out


class UserBehaviorModelCLF(nn.Module):
    def __init__(self, vocab_size, n_classes_list, cfg):
        super().__init__()
        D, H = cfg["embed_dim"], cfg["head_hidden"]
        self.encoder = TransformerEncoderRoPE(
            vocab_size, D, cfg["n_layers"], cfg["n_heads"],
            cfg["max_seq_len"], cfg["dropout"], cfg["rope_base"],
        )
        self.neck = nn.Sequential(
            nn.Linear(D * 2 + 3, D),
            nn.LayerNorm(D),
            nn.GELU(),
            nn.Dropout(cfg["dropout"]),
        )
        self.heads = nn.ModuleList([
            self._make_head(D, H, n_cls, deep=(j in {2, 5}), drop=cfg["dropout"])
            for j, n_cls in enumerate(n_classes_list)
        ])
        self._init_weights()

    @staticmethod
    def _make_head(D, H, n_cls, deep, drop):
        n_out = n_cls - 1
        if deep:
            return nn.Sequential(
                nn.Linear(D, H), nn.LayerNorm(H), nn.GELU(), nn.Dropout(drop),
                nn.Linear(H, H // 2), nn.GELU(), nn.Linear(H // 2, n_out),
            )
        return nn.Sequential(
            nn.Linear(D, H), nn.LayerNorm(H), nn.GELU(), nn.Dropout(drop),
            nn.Linear(H, n_out),
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()

    def forward(self, seq, mask, meta):
        cls_out, mean_out = self.encoder(seq, mask)
        fused = self.neck(torch.cat([cls_out, mean_out, meta], dim=-1))
        return [head(fused) for head in self.heads]


# ── Decode ordinal logits → class indices ─────────────────────────────────
def decode_predictions(logits_list: list[torch.Tensor]) -> torch.Tensor:
    """
    Giống OrdinalClassificationLoss.decode() trong training.
    logits_list: list[Tensor(B, n_cls-1)]
    Returns: Tensor(B, 6) class indices
    """
    return torch.stack(
        [(torch.sigmoid(l) > 0.5).sum(dim=-1) for l in logits_list],
        dim=1,
    )
