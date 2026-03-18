"""
model_service.py
Load checkpoint thật và chạy inference với Transformer model.

Pipeline:
  rows (feature_values) → BehaviorDataset → model.forward() →
  attention_weights (CLS→tokens) + predictions (6 target cols)
"""

from __future__ import annotations

import logging
import math
import os
import random
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from app.core.model_arch import (
    FEATURE_COLS,
    BehaviorDataset,
    UserBehaviorModelCLF,
    Vocabulary,
    decode_predictions,
)

logger = logging.getLogger(__name__)

MOCK_MODE = not os.path.exists(os.getenv("MODEL_PATH", "ml/model.pt"))
BATCH_SIZE = 64


class ModelService:
    """Singleton — load model 1 lần duy nhất khi app khởi động."""

    _instance: "ModelService | None" = None

    def __new__(cls) -> "ModelService":
        if cls._instance is None:
            inst = super().__new__(cls)
            inst._loaded       = False
            inst._model        = None
            inst._vocab        = None
            inst._cfg          = None
            inst._class_maps_c2v = None   # {col: {idx → value}}
            inst._target_cols  = None     # list[str] — thứ tự của 6 heads
            inst._device       = "cpu"
            cls._instance = inst
        return cls._instance

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def load(self, model_path: str) -> None:
        if self._loaded:
            return

        if not os.path.exists(model_path):
            logger.warning("MODEL_PATH '%s' không tìm thấy — MOCK MODE.", model_path)
            self._loaded = True
            return

        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._device = device
            logger.info("Loading checkpoint từ %s (device=%s)", model_path, device)

            ckpt = torch.load(model_path, map_location=device, weights_only=False)

            # ── 1. Vocabulary ──────────────────────────────────────────────
            vocab = Vocabulary()
            vocab.token2idx = ckpt["vocab_token2idx"]
            vocab._size     = ckpt.get("vocab_size", len(ckpt["vocab_token2idx"]) + 2)
            self._vocab = vocab

            # ── 2. Class maps ──────────────────────────────────────────────
            # c2v: {col: {class_idx → original_value}}
            self._class_maps_c2v = ckpt["class_maps_c2v"]
            # TARGET_COLS lấy từ keys của c2v (đúng thứ tự)
            self._target_cols = list(ckpt["class_maps_c2v"].keys())

            # ── 3. CFG — override device ───────────────────────────────────
            cfg = dict(ckpt["cfg"])
            cfg["device"] = device
            self._cfg = cfg

            # ── 4. Build & load model ──────────────────────────────────────
            n_classes_list = ckpt["n_classes_list"]
            model = UserBehaviorModelCLF(vocab.size, n_classes_list, cfg).to(device)

            state = ckpt["model_state"]
            # Xử lý nếu checkpoint được lưu từ DataParallel
            if any(k.startswith("module.") for k in state):
                state = {k[7:]: v for k, v in state.items()}
            model.load_state_dict(state)
            model.eval()
            self._model = model

            self._loaded = True
            logger.info(
                "✅ Model loaded | vocab=%d | heads=%d | cols=%s",
                vocab.size, len(n_classes_list), self._target_cols,
            )

        except Exception as exc:
            logger.error("Không load được model: %s", exc)
            raise

    # ── Public interface ───────────────────────────────────────────────────

    def predict(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Input:  list[{id, feature_values[66], sequence_length}]
        Output: list[{
            id,
            attention_weights: list[float],   # normalized, len=sequence_length
            predictions: dict[col → value],   # {attr_3: 42, attr_6: 31, ...}
        }]
        """
        if not os.path.exists(os.getenv("MODEL_PATH", "ml/model.pt")) or self._model is None:
            return [self._mock_predict_one(r) for r in rows]
        return self._real_predict(rows)

    @property
    def target_cols(self) -> list[str]:
        return self._target_cols or []

    # ── Real inference ─────────────────────────────────────────────────────

    def _real_predict(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        device = self._device
        cfg    = self._cfg
        vocab  = self._vocab

        # Tạo DataFrame giả từ rows để truyền vào BehaviorDataset
        records = []
        for r in rows:
            rec = {"id": r["id"]}
            for j, col in enumerate(FEATURE_COLS):
                rec[col] = r["feature_values"][j]  # int hoặc None
            records.append(rec)

        df = pd.DataFrame(records).set_index("id")

        # BehaviorDataset encode feature_values → token IDs
        ds = BehaviorDataset(df, vocab, cfg["max_seq_len"])
        dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        all_attn:  list[list[float]] = []
        all_preds: list[list[int]]   = []

        self._model.eval()
        with torch.no_grad():
            for seq, mask, meta in dl:
                seq  = seq.to(device)
                mask = mask.to(device)
                meta = meta.to(device)

                logits_list = self._model(seq, mask, meta)

                # ── Attention weights từ last transformer layer ────────────
                last_layer = self._model.encoder.layers[-1]
                attn = last_layer.attn.last_attn_weight  # (B, H, T+1, T+1)
                # CLS token (row 0) → input positions (cols 1..T)
                cls_attn = attn[:, :, 0, 1:]             # (B, H, T)
                avg_attn = cls_attn.mean(dim=1)          # (B, T) — avg over heads

                # Mask padding và normalize
                valid_mask = mask.float()                 # (B, T)
                avg_attn   = avg_attn * valid_mask
                sums = avg_attn.sum(dim=-1, keepdim=True).clamp(min=1e-9)
                norm_attn = (avg_attn / sums).cpu()       # (B, T)

                # ── Decode predictions → class indices → original values ───
                pred_idx = decode_predictions(logits_list)  # (B, 6)

                for i in range(pred_idx.shape[0]):
                    # Attention: cắt đến đúng sequence_length
                    seq_len = int(mask[i].sum().item())
                    attn_i  = norm_attn[i, :seq_len].tolist()
                    all_attn.append(attn_i)

                    # Predictions: map class_idx → original_value qua c2v
                    pred_vals: list[int] = []
                    for j, col in enumerate(self._target_cols):
                        class_idx = pred_idx[i, j].item()
                        value     = self._class_maps_c2v[col].get(class_idx, 0)
                        pred_vals.append(int(value))
                    all_preds.append(pred_vals)

        # Ghép kết quả với order IDs
        results = []
        for idx, row in enumerate(rows):
            results.append({
                "id":               row["id"],
                "attention_weights": all_attn[idx],
                "predictions": {
                    col: all_preds[idx][j]
                    for j, col in enumerate(self._target_cols)
                },
            })
        return results

    # ── Mock (khi chưa có model.pt) ───────────────────────────────────────

    def _mock_predict_one(self, row: dict[str, Any]) -> dict[str, Any]:
        seq_len = row["sequence_length"]
        rng     = random.Random(hash(row["id"]))

        is_dispersed = rng.random() < 0.35
        attn_weights = _mock_attention(seq_len, is_dispersed, rng)

        return {
            "id":               row["id"],
            "attention_weights": attn_weights,
            "predictions": {
                "attr_1": rng.randint(0, 99),
                "attr_2": rng.randint(0, 99),
                "attr_3": rng.randint(0, 99),
                "attr_4": rng.randint(0, 99),
                "attr_5": rng.randint(0, 99),
                "attr_6": rng.randint(0, 99),
            },
        }


# ── Helpers ────────────────────────────────────────────────────────────────

def _mock_attention(length: int, is_dispersed: bool, rng: random.Random) -> list[float]:
    if is_dispersed:
        weights = [0.3 + rng.random() * 0.7 for _ in range(length)]
    else:
        peak = length * 0.7
        weights = [
            math.exp(-abs(i - peak) / (length * 0.2)) + rng.random() * 0.2
            for i in range(length)
        ]
    total = sum(weights)
    return [w / total for w in weights]


# Singleton toàn cục
model_service = ModelService()
