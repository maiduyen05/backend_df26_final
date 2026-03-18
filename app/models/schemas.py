from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


# ── Enums / Literals ───────────────────────────────────────────────────────

DecisionState = Literal["release", "throttle", "buffer", "freeze", "recover"]


# ── Upload ─────────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    """Trả về ngay sau khi POST /upload hoàn tất xử lý."""
    batch_id: str = Field(..., description="UUID định danh batch")
    status: Literal["completed", "processing", "error"]
    total_rows: int = Field(..., ge=1)
    created_at: datetime


# ── Overview – sub-objects ─────────────────────────────────────────────────

class BatchStats(BaseModel):
    """
    Thống kê tổng hợp — dùng cho KPI cards, donut chart, recommendations.
    Decision states theo risk threshold spec 2.1:
      release  = Bình thường   (risk < 0.25)
      throttle = Cần theo dõi  (0.25 <= risk < 0.45)
      buffer   = Chờ xử lý    (0.45 <= risk < 0.65)
      freeze   = Cần tạm dừng (risk >= 0.65)
    """
    total_orders: int
    release: int          # Bình thường
    throttle: int         # Cần theo dõi
    buffer: int           # Chờ xử lý
    freeze: int           # Cần tạm dừng
    recover: int          # Đã xử lý (không bắt buộc trong spec nhưng giữ lại)

    avg_attr3: float      = Field(..., description="Trung bình attr_3 (Chỉ số 1)")
    avg_attr6: float      = Field(..., description="Trung bình attr_6 (Chỉ số 2)")
    avg_risk: float       = Field(..., ge=0, le=1, description="Mức rủi ro hệ thống trung bình")
    avg_entropy: float    = Field(..., ge=0, description="Độ biến động luồng hàng (entropy TB)")

    # Legacy fields — giữ để không break frontend cũ
    avg_confidence: int   = Field(0, ge=0, le=100)
    avg_plant_load: int   = Field(0, ge=0, le=99)
    risky_exposure_pct: int = Field(0, ge=0, le=100)


class SequenceLengthBucket(BaseModel):
    """1 cột trong bar chart 'Luồng hành động'."""
    length: int = Field(..., description="Giá trị bucket")
    count: int


class PlantSignalPoint(BaseModel):
    """1 điểm trong chart 'Công suất hoạt động'."""
    index: int
    plant_signal_a: int = Field(..., ge=0, le=99, description="attr_3 predicted")
    plant_signal_b: int = Field(..., ge=0, le=99, description="attr_6 predicted")


class BehavioralSegments(BaseModel):
    """4 segment cards trong 'Phân khúc giao dịch' (spec 2.4)."""
    short_cycle: int       = Field(..., description="Chuỗi có độ dài < 5")
    long_cycle: int        = Field(..., description="Chuỗi có độ dài > 20")
    popular_flows: list    = Field(..., description="Top 3 bộ 3 hành vi đầu chuỗi phổ biến nhất")
    abab_risk_count: int   = Field(..., description="Số chuỗi có mẫu ABAB từ 2 lần trở lên")

    # Legacy fields
    short_decisive: int    = Field(0)
    long_exploratory: int  = Field(0)
    high_revisit: int      = Field(0)
    unstable_rollback: int = Field(0)


class BatchProfile(BaseModel):
    """4 ô trong 'Thông tin dữ liệu' (spec 2.5)."""
    total_rows: int
    max_sequence_length: int  = Field(..., description="Hoạt động tối đa")
    avg_sequence_length: int  = Field(..., description="Hoạt động trung bình")
    data_quality_pct: int     = Field(100)

    # Legacy
    sequence_columns: int = Field(66)


# ── Overview – response ────────────────────────────────────────────────────

class OverviewResponse(BaseModel):
    """Response của GET /batches/{batch_id}/overview."""
    batch_id: str
    created_at: datetime
    stats: BatchStats
    sequence_length_distribution: list[SequenceLengthBucket]
    plant_signal_sample: list[PlantSignalPoint]
    segments: BehavioralSegments
    profile: BatchProfile


# ── Internal ──────────────────────────────────────────────────────────────

class OrderMetrics(BaseModel):
    """Kết quả tính toán cho 1 row sau khi chạy model."""
    order_id: str
    sequence_length: int
    feature_values: list           # 66 phần tử, None = padding
    attention_weights: list        # normalized, len = sequence_length
    attention_entropy: float
    normalized_entropy: float
    confidence_score: int          # 0–100
    repeat_ratio: float
    revisit_count: int
    rollback_depth: int
    sequence_entropy: float
    stagnation_score: float
    instability_score: float
    delay_risk: float
    plant_pressure: float
    window_uncertainty: float
    overall_risk_score: float
    decision_state: DecisionState
    plant_signal_a: int            # attr_3 (0–99)
    plant_signal_b: int            # attr_6 (0–99)
    start_window: dict
    completion_window: dict
    next_review_time: str


class BatchRecord(BaseModel):
    """Lưu trữ toàn bộ batch sau khi xử lý."""
    batch_id: str
    created_at: datetime
    status: Literal["completed", "processing", "error"]
    orders: list[OrderMetrics]


# ── Workbench — Order list ─────────────────────────────────────────────────

class OrderSummary(BaseModel):
    """1 row trong bảng Workbench + dữ liệu chi tiết cho panel."""
    order_id: str
    # Ngày bắt đầu: attr_1 = tháng, attr_2 = ngày
    start_month: int
    start_day:   int
    # Ngày kết thúc: attr_4 = tháng, attr_5 = ngày
    end_month: int
    end_day:   int
    # Chỉ số dự báo
    attr_3: int = Field(..., description="Chỉ số 1 (attr_3)")
    attr_6: int = Field(..., description="Chỉ số 2 (attr_6)")
    # Risk & state
    overall_risk_score: float
    decision_state: DecisionState
    # Chi tiết panel
    confidence_score:   int
    normalized_entropy: float
    attention_weights:  list[float]
    feature_values:     list
    sequence_length:    int
    repeat_ratio:       float
    revisit_count:      int
    rollback_depth:     int
    instability_score:  float
    delay_risk:         float


class OrdersResponse(BaseModel):
    """Response của GET /batches/{id}/orders."""
    batch_id: str
    total: int
    orders: list[OrderSummary]
