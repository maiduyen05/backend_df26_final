"""
csv_parser.py
Parse và validate file CSV upload.

Yêu cầu CSV:
  - Cột bắt buộc: id, feature_1 ... feature_66
  - feature_i: integer 0–99, hoặc rỗng/NaN = padding
  - sequenceLength = số cột feature không null liên tiếp từ feature_1
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import pandas as pd

from app.core.exceptions import CSVValidationError

FEATURE_COLS = [f"feature_{i}" for i in range(1, 67)]   # 66 features
REQUIRED_COLS = {"id"} | set(FEATURE_COLS)


def parse_csv(content: bytes) -> list[dict[str, Any]]:
    """
    Đọc bytes CSV, trả về list dict mỗi phần tử là 1 đơn hàng:
    {
        "id": str,
        "feature_values": list[int | None],   # len=66, None=padding
        "sequence_length": int,
    }
    """
    try:
        df = pd.read_csv(io.BytesIO(content), dtype=str)
    except Exception as e:
        raise CSVValidationError(f"Không đọc được file: {e}") from e

    # ── Validate columns ────────────────────────────────────────────────────
    df.columns = df.columns.str.strip().str.lower()
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        sample = sorted(missing)[:5]
        raise CSVValidationError(f"Thiếu cột: {sample}")

    if df.empty:
        raise CSVValidationError("File không có dòng dữ liệu nào.")

    # ── Parse feature columns ───────────────────────────────────────────────
    rows: list[dict[str, Any]] = []

    for row_idx, row in df.iterrows():
        order_id = str(row["id"]).strip()
        if not order_id:
            raise CSVValidationError(f"Dòng {row_idx + 2}: cột 'id' rỗng.")

        feature_values: list[int | None] = []
        for col in FEATURE_COLS:
            raw = row.get(col, "")
            if pd.isna(raw) or str(raw).strip() in ("", "nan", "null", "none"):
                feature_values.append(None)
            else:
                try:
                    val = int(float(str(raw).strip()))
                except ValueError:
                    raise CSVValidationError(
                        f"Dòng {row_idx + 2}, '{col}': giá trị '{raw}' không phải số."
                    )
                if val < 0:
                    raise CSVValidationError(
                        f"Dòng {row_idx + 2}, '{col}': giá trị {val} ngoài khoảng 0–99."
                    )
                feature_values.append(val)

        # sequence_length = số cột đầu không null liên tiếp
        seq_len = _count_sequence_length(feature_values)
        if seq_len < 1:
            raise CSVValidationError(
                f"Dòng {row_idx + 2}: không có feature nào hợp lệ (toàn null)."
            )

        rows.append({
            "id": order_id,
            "feature_values": feature_values,
            "sequence_length": seq_len,
        })

    return rows


def _count_sequence_length(feature_values: list[int | None]) -> int:
    """Đếm số phần tử không-None liên tiếp từ đầu (trước padding)."""
    count = 0
    for v in feature_values:
        if v is None:
            break
        count += 1
    return count
