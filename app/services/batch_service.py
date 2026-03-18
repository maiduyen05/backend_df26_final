"""
batch_service.py
Orchestrate toàn bộ pipeline cho 1 batch:
  CSV → parse → model.predict → compute_metrics → aggregate → store

Mapping TARGET_COLS → fields:
  attr_3 → plant_signal_a (Chỉ số 1)
  attr_6 → plant_signal_b (Chỉ số 2)
"""

from __future__ import annotations

import random
import uuid
from collections import Counter
from datetime import datetime, timezone

from app.models.schemas import (
    BatchProfile,
    BatchRecord,
    BatchStats,
    BehavioralSegments,
    OrderMetrics,
    OverviewResponse,
    PlantSignalPoint,
    SequenceLengthBucket,
    UploadResponse,
)
from app.services.csv_parser import parse_csv
from app.services.metrics import compute_order_metrics
from app.services.model_service import model_service
from app.storage.in_memory import store


# ── Public API ─────────────────────────────────────────────────────────────

def process_upload(file_content: bytes) -> UploadResponse:
    """Full pipeline: CSV bytes → UploadResponse."""
    # 1. Parse CSV → list[{id, feature_values, sequence_length}]
    rows = parse_csv(file_content)

    # 2. Inference → list[{id, attention_weights, predictions}]
    predictions = model_service.predict(rows)
    pred_map    = {p["id"]: p for p in predictions}

    # 3. Tính metrics từng order
    orders: list[OrderMetrics] = []
    for row in rows:
        pred        = pred_map.get(row["id"], {})
        preds_dict  = pred.get("predictions", {})

        metrics = compute_order_metrics(
            order_id=row["id"],
            sequence_length=row["sequence_length"],
            feature_values=row["feature_values"],
            attention_weights=pred.get("attention_weights", []),
            predictions=preds_dict,
        )
        orders.append(metrics)

    # 4. Lưu vào store
    batch_id = str(uuid.uuid4())
    now      = datetime.now(tz=timezone.utc)
    record   = BatchRecord(
        batch_id=batch_id,
        created_at=now,
        status="completed",
        orders=orders,
    )
    store.save(batch_id, record)

    return UploadResponse(
        batch_id=batch_id,
        status="completed",
        total_rows=len(orders),
        created_at=now,
    )


def get_overview(batch_id: str) -> OverviewResponse:
    """Tổng hợp dữ liệu đã lưu → OverviewResponse."""
    record = store.get(batch_id)
    orders = record.orders

    return OverviewResponse(
        batch_id=batch_id,
        created_at=record.created_at,
        stats=_compute_stats(orders),
        sequence_length_distribution=_sequence_length_distribution(orders),
        plant_signal_sample=_plant_signal_sample(orders),
        segments=_compute_segments(orders),
        profile=_compute_profile(orders),
    )


# ── Aggregation ────────────────────────────────────────────────────────────

def _compute_stats(orders: list[OrderMetrics]) -> BatchStats:
    """
    Spec 2.1:
      - Phân loại theo risk score thresholds (không phải decision_state cũ)
      - Chỉ số 1 (ATTR_3): avg attr_3
      - Chỉ số 2 (ATTR_6): avg attr_6
      - Mức rủi ro hệ thống: avg overall_risk
      - Độ biến động: avg normalized_entropy
    """
    n = len(orders)

    release_count  = 0  # risk < 0.25
    throttle_count = 0  # 0.25 <= risk < 0.45
    buffer_count   = 0  # 0.45 <= risk < 0.65
    freeze_count   = 0  # risk >= 0.65
    recover_count  = 0  # decision_state == recover (legacy)

    sum_attr3 = sum_attr6 = sum_risk = sum_entropy = 0.0
    sum_conf  = sum_plant = 0

    for o in orders:
        r = o.overall_risk_score
        if r < 0.25:
            release_count  += 1
        elif r < 0.45:
            throttle_count += 1
        elif r < 0.65:
            buffer_count   += 1
        else:
            freeze_count   += 1

        if o.decision_state == "recover":
            recover_count += 1

        sum_attr3   += o.plant_signal_a
        sum_attr6   += o.plant_signal_b
        sum_risk    += o.overall_risk_score
        sum_entropy += o.normalized_entropy
        sum_conf    += o.confidence_score
        sum_plant   += o.plant_signal_a + o.plant_signal_b

    avg_attr3   = round(sum_attr3 / n, 2)
    avg_attr6   = round(sum_attr6 / n, 2)
    avg_risk    = round(sum_risk / n, 4)
    avg_entropy = round(sum_entropy / n, 4)

    return BatchStats(
        total_orders=n,
        release=release_count,
        throttle=throttle_count,
        buffer=buffer_count,
        freeze=freeze_count,
        recover=recover_count,
        avg_attr3=avg_attr3,
        avg_attr6=avg_attr6,
        avg_risk=avg_risk,
        avg_entropy=avg_entropy,
        # Legacy
        avg_confidence=round(sum_conf / n),
        avg_plant_load=round(sum_plant / n / 2),
        risky_exposure_pct=round((buffer_count + freeze_count) / n * 100),
    )


def _compute_segments(orders: list[OrderMetrics]) -> BehavioralSegments:
    """
    Spec 2.4:
      - Chu kỳ ngắn : seq_len < 5
      - Chu kỳ dài  : seq_len > 20
      - Luồng phổ biến: top 3 bộ (feature_1, feature_2, feature_3) đầu chuỗi
      - Cảnh báo rủi ro: số chuỗi có pattern ABAB >= 2 lần
    """
    short_cycle = sum(1 for o in orders if o.sequence_length < 5)
    long_cycle  = sum(1 for o in orders if o.sequence_length > 20)

    # Popular flows: top 3 bộ 3 hành vi đầu chuỗi
    popular_flows = _compute_popular_flows(orders)

    # ABAB risk count
    abab_risk_count = sum(1 for o in orders if _has_abab_pattern(o.feature_values))

    # Legacy
    short_decisive    = sum(1 for o in orders if o.sequence_length < 30 and o.confidence_score > 70)
    long_exploratory  = sum(1 for o in orders if o.sequence_length > 50 and o.confidence_score < 60)
    high_revisit      = sum(1 for o in orders if o.revisit_count > 4)
    unstable_rollback = sum(1 for o in orders if o.rollback_depth > 2)

    return BehavioralSegments(
        short_cycle=short_cycle,
        long_cycle=long_cycle,
        popular_flows=popular_flows,
        abab_risk_count=abab_risk_count,
        short_decisive=short_decisive,
        long_exploratory=long_exploratory,
        high_revisit=high_revisit,
        unstable_rollback=unstable_rollback,
    )


def _compute_profile(orders: list[OrderMetrics]) -> BatchProfile:
    """
    Spec 2.5:
      - max_sequence_length: chiều dài lớn nhất
      - avg_sequence_length: trung bình
      - data_quality_pct: 100% nếu CSV đúng định dạng
    """
    lengths = [o.sequence_length for o in orders]
    return BatchProfile(
        total_rows=len(orders),
        max_sequence_length=max(lengths),
        avg_sequence_length=round(sum(lengths) / len(lengths)),
        data_quality_pct=100,
        sequence_columns=66,
    )


def _sequence_length_distribution(orders: list[OrderMetrics]) -> list[SequenceLengthBucket]:
    """Phân phối theo bucket bội số 10."""
    bucket_map: dict[int, int] = {}
    for o in orders:
        bucket = (o.sequence_length // 10) * 10
        bucket_map[bucket] = bucket_map.get(bucket, 0) + 1
    return [
        SequenceLengthBucket(length=length, count=count)
        for length, count in sorted(bucket_map.items())
    ]


def _plant_signal_sample(orders: list[OrderMetrics]) -> list[PlantSignalPoint]:
    """Trả về toàn bộ orders cho chart Công suất hoạt động."""
    return [
        PlantSignalPoint(
            index=i,
            plant_signal_a=o.plant_signal_a,
            plant_signal_b=o.plant_signal_b,
        )
        for i, o in enumerate(orders)
    ]


# ── Spec 2.4 helpers ──────────────────────────────────────────────────────

def _compute_popular_flows(orders: list[OrderMetrics]) -> list:
    """
    Trả về top 3 bộ (f1, f2, f3) đầu chuỗi phổ biến nhất.
    Nếu nhiều bộ cùng tần suất, chọn ngẫu nhiên 3 trong số đó.
    """
    triplet_counter: Counter = Counter()
    for o in orders:
        seq = [v for v in o.feature_values if v is not None]
        if len(seq) >= 3:
            triplet = tuple(seq[:3])
            triplet_counter[triplet] += 1

    if not triplet_counter:
        return []

    # Lấy tần suất cao nhất và nhóm lại
    sorted_items = triplet_counter.most_common()

    # Lấy top 3 (nếu tie thì chọn ngẫu nhiên từ nhóm tie)
    result = []
    i = 0
    while len(result) < 3 and i < len(sorted_items):
        current_count = sorted_items[i][1]
        # Gom tất cả triplets cùng tần suất này
        same_freq = [item for item in sorted_items[i:] if item[1] == current_count]
        if len(result) + len(same_freq) <= 3:
            result.extend(same_freq)
            i += len(same_freq)
        else:
            # Chọn ngẫu nhiên
            needed = 3 - len(result)
            chosen = random.sample(same_freq, needed)
            result.extend(chosen)
            break

    return [{"actions": list(triplet), "count": count} for triplet, count in result]


def _has_abab_pattern(feature_values: list) -> bool:
    """
    Kiểm tra chuỗi có mẫu ABAB (A≠B) xuất hiện từ 2 lần trở lên.
    ABAB = 2 lần lặp, ABABAB = 3 lần lặp, v.v.
    Cần tìm ít nhất 1 cặp (A,B) với A≠B xuất hiện theo mẫu A,B,A,B trong chuỗi.
    """
    seq = [v for v in feature_values if v is not None]
    n   = len(seq)
    if n < 4:
        return False

    # Kiểm tra tất cả cặp (A, B) với A ≠ B
    # Tìm positions của từng token, rồi kiểm tra xen kẽ
    seen_pairs: set = set()
    for i in range(n - 3):
        a, b = seq[i], seq[i + 1]
        if a == b:
            continue
        if (a, b) in seen_pairs:
            continue
        # Tìm số lần lặp pattern AB trong chuỗi từ vị trí i
        count = 0
        j = i
        while j + 1 < n and seq[j] == a and seq[j + 1] == b:
            count += 1
            j += 2
        if count >= 2:
            return True
        seen_pairs.add((a, b))
    return False


# ── Workbench orders ───────────────────────────────────────────────────────

def get_orders(batch_id: str):
    """
    GET /batches/{id}/orders
    Trả về danh sách OrderSummary — mỗi phần tử là 1 row trong bảng Workbench
    kèm đủ dữ liệu cho panel chi tiết.
    """
    from app.models.schemas import OrderSummary, OrdersResponse

    record = store.get(batch_id)
    summaries = []
    for o in record.orders:
        sw = o.start_window       # {"month": m, "day": d}
        cw = o.completion_window
        summaries.append(OrderSummary(
            order_id    = o.order_id,
            start_month = sw.get("month", 1),
            start_day   = sw.get("day",   1),
            end_month   = cw.get("month", 1),
            end_day     = cw.get("day",   1),
            attr_3      = o.plant_signal_a,
            attr_6      = o.plant_signal_b,
            overall_risk_score = o.overall_risk_score,
            decision_state     = o.decision_state,
            confidence_score   = o.confidence_score,
            normalized_entropy = o.normalized_entropy,
            attention_weights  = o.attention_weights,
            feature_values     = o.feature_values,
            sequence_length    = o.sequence_length,
            repeat_ratio       = o.repeat_ratio,
            revisit_count      = o.revisit_count,
            rollback_depth     = o.rollback_depth,
            instability_score  = o.instability_score,
            delay_risk         = o.delay_risk,
        ))

    return OrdersResponse(
        batch_id = batch_id,
        total    = len(summaries),
        orders   = summaries,
    )
