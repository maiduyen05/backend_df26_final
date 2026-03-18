"""
metrics.py
Tính toán tất cả metrics theo đúng công thức trong tài liệu đặc tả.

Công thức:
  3.2  repeat_ratio       = 1 - (unique / seq_len), clamp [0,1]
  3.3  revisit_count      = mọi lần gặp token đã thấy
  3.4  rollback_depth     = max(i - last_pos[token]), normalize / (L-1)
  3.5  stagnation         = revisit_count / seq_len
  3.6  seq_entropy        = H / log(U),  U>1; else 0
  3.7  instability        = 0.30*repeat + 0.25*revisit_norm + 0.25*rollback_norm + 0.20*stagnation
  3.8  attention_entropy  = -sum(p_i*ln(p_i)), normalize / ln(L)
  3.9  confidence         = 1 - NormalizedEntropy
  3.10 delay_risk & time_risk  (dùng attr_1,2,4,5)
  3.11 plant_pressure     = (attr_3 + attr_6) / 198
  3.12 window_uncertainty = 0.4*(1-Conf) + 0.3*DelayRisk + 0.3*TimeRisk
  3.13 overall_risk       = 0.30*AttnEntr + 0.20*Instab + 0.15*Delay
                           + 0.15*TimeRisk + 0.10*Plant + 0.10*WinUncert
"""

from __future__ import annotations

import math
from collections import Counter
from datetime import datetime, timezone

from app.models.schemas import DecisionState, OrderMetrics

# Risk thresholds (spec 2.1)
RELEASE_THRESH  = 0.25
THROTTLE_THRESH = 0.45
BUFFER_THRESH   = 0.65

# Config plant pressure direction
PLANT_SIGNAL_HIGH_MEANS_PRESSURE: bool = True


# ── Public entry point ─────────────────────────────────────────────────────

def compute_order_metrics(
    order_id: str,
    sequence_length: int,
    feature_values: list,
    attention_weights: list,
    predictions: dict,
) -> OrderMetrics:
    """
    predictions: dict {attr_1..attr_6: int}
    """
    seq = [v for v in feature_values if v is not None]
    L   = sequence_length

    # 3.2
    repeat_ratio = _repeat_ratio(seq, L)
    # 3.3
    revisit_count = _revisit_count(seq)
    # 3.4
    rollback_depth_raw, rollback_norm = _rollback_depth(seq, L)
    # 3.5
    stagnation = _stagnation(revisit_count, L)
    # 3.6
    seq_entropy = _sequence_entropy(seq)
    # 3.7
    revisit_norm = min(revisit_count / L, 1.0) if L > 0 else 0.0
    instability  = _instability(repeat_ratio, revisit_norm, rollback_norm, stagnation)
    # 3.8
    attn_entropy_raw, normalized_entropy = _attention_entropy(attention_weights, L)
    # 3.9
    confidence_score = max(0.0, min(1.0, 1.0 - normalized_entropy))
    confidence_int   = round(confidence_score * 100)

    # 3.10
    attr_1 = int(predictions.get("attr_1", 0))
    attr_2 = int(predictions.get("attr_2", 0))
    attr_4 = int(predictions.get("attr_4", 0))
    attr_5 = int(predictions.get("attr_5", 0))
    now = datetime.now(tz=timezone.utc)
    delay_risk, time_risk_score, start_window, completion_window = _delay_and_time_risk(
        attr_1, attr_2, attr_4, attr_5,
        confidence_score, now.month, now.day,
    )

    # 3.11
    attr_3 = int(predictions.get("attr_3", 0))
    attr_6 = int(predictions.get("attr_6", 0))
    plant_pressure = _plant_pressure(attr_3, attr_6)

    # 3.12
    window_uncertainty = _window_uncertainty(confidence_score, delay_risk, time_risk_score)

    # 3.13
    overall_risk = _overall_risk(
        normalized_entropy, instability,
        delay_risk, time_risk_score,
        plant_pressure, window_uncertainty,
    )

    decision_state = _get_decision_state(overall_risk)

    return OrderMetrics(
        order_id=order_id,
        sequence_length=L,
        feature_values=feature_values,
        attention_weights=attention_weights,
        attention_entropy=round(attn_entropy_raw, 6),
        normalized_entropy=round(normalized_entropy, 6),
        confidence_score=confidence_int,
        repeat_ratio=round(repeat_ratio, 4),
        revisit_count=revisit_count,
        rollback_depth=rollback_depth_raw,
        sequence_entropy=round(seq_entropy, 4),
        stagnation_score=round(stagnation, 4),
        instability_score=round(instability, 6),
        delay_risk=round(delay_risk, 4),
        plant_pressure=round(plant_pressure, 4),
        window_uncertainty=round(window_uncertainty, 4),
        overall_risk_score=round(overall_risk, 6),
        decision_state=decision_state,
        plant_signal_a=attr_3,
        plant_signal_b=attr_6,
        start_window=start_window,
        completion_window=completion_window,
        next_review_time="48h" if decision_state == "release"
                         else "12h" if decision_state in ("buffer", "freeze")
                         else "24h",
    )


# ── Helpers ────────────────────────────────────────────────────────────────

def _repeat_ratio(seq: list, L: int) -> float:
    if L == 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - len(set(seq)) / L))


def _revisit_count(seq: list) -> int:
    seen: set = set()
    count = 0
    for token in seq:
        if token in seen:
            count += 1
        else:
            seen.add(token)
    return count


def _rollback_depth(seq: list, L: int) -> tuple:
    if L < 2:
        return 0, 0.0
    last_pos: dict = {}
    max_depth = 0
    for i, token in enumerate(seq):
        if token in last_pos:
            depth = i - last_pos[token]
            if depth > max_depth:
                max_depth = depth
        last_pos[token] = i
    norm = max_depth / max(L - 1, 1)
    return max_depth, min(norm, 1.0)


def _stagnation(revisit_count: int, L: int) -> float:
    if L == 0:
        return 0.0
    return min(revisit_count / L, 1.0)


def _sequence_entropy(seq: list) -> float:
    if not seq:
        return 0.0
    counts = Counter(seq)
    U = len(counts)
    if U <= 1:
        return 0.0
    n = len(seq)
    h_raw = -sum((c / n) * math.log(c / n) for c in counts.values())
    return min(h_raw / math.log(U), 1.0)


def _instability(
    repeat_ratio: float,
    revisit_norm: float,
    rollback_norm: float,
    stagnation: float,
) -> float:
    score = (
        0.30 * repeat_ratio
        + 0.25 * revisit_norm
        + 0.25 * rollback_norm
        + 0.20 * stagnation
    )
    return min(score, 1.0)


def _attention_entropy(weights: list, L: int) -> tuple:
    if L <= 1 or not weights:
        return 0.0, 0.0
    valid = weights[:L]
    total = sum(valid)
    if total <= 0:
        return 0.0, 0.0
    h_raw = 0.0
    for w in valid:
        p = w / total
        if p > 0:
            h_raw -= p * math.log(p)
    normalized = min(h_raw / math.log(L), 1.0)
    return h_raw, normalized


def _delay_and_time_risk(
    attr_1: int, attr_2: int,
    attr_4: int, attr_5: int,
    confidence: float,
    current_month: int, current_day: int,
) -> tuple:
    start_index      = attr_1 * 31 + attr_2
    completion_index = attr_4 * 31 + attr_5
    raw_span         = max(completion_index - start_index, 0)
    delay_risk       = min(raw_span / 120, 1.0)

    current_index   = current_month * 31 + current_day
    days_to_start   = start_index - current_index
    start_proximity = 1.0 - min(max(days_to_start, 0) / 7, 1.0)
    overdue_penalty = 1.0 if days_to_start < 0 else 0.0

    time_risk_score = min(
        0.5 * delay_risk
        + 0.3 * (1.0 - start_proximity)
        + 0.2 * overdue_penalty,
        1.0,
    )

    start_window      = {"month": max(1, min(12, attr_1)), "day": max(1, min(31, attr_2))}
    completion_window = {"month": max(1, min(12, attr_4)), "day": max(1, min(31, attr_5))}
    return delay_risk, time_risk_score, start_window, completion_window


def _plant_pressure(attr_3: int, attr_6: int) -> float:
    raw = (attr_3 + attr_6) / 198.0
    if PLANT_SIGNAL_HIGH_MEANS_PRESSURE:
        return min(raw, 1.0)
    return max(0.0, 1.0 - raw)


def _window_uncertainty(confidence: float, delay_risk: float, time_risk: float) -> float:
    return min(
        0.4 * (1.0 - confidence)
        + 0.3 * delay_risk
        + 0.3 * time_risk,
        1.0,
    )


def _overall_risk(
    attn_entropy_norm: float,
    instability: float,
    delay_risk: float,
    time_risk_score: float,
    plant_pressure: float,
    window_uncertainty: float,
) -> float:
    R = (
        0.30 * attn_entropy_norm
        + 0.20 * instability
        + 0.15 * delay_risk
        + 0.15 * time_risk_score
        + 0.10 * plant_pressure
        + 0.10 * window_uncertainty
    )
    return max(0.0, min(R, 1.0))


def _get_decision_state(risk: float) -> DecisionState:
    """
    Bình thường  : risk < 0.25   → release
    Cần theo dõi : 0.25 <= risk < 0.45 → throttle
    Chờ xử lý   : 0.45 <= risk < 0.65 → buffer
    Cần tạm dừng : risk >= 0.65  → freeze
    """
    if risk < RELEASE_THRESH:
        return "release"
    if risk < THROTTLE_THRESH:
        return "throttle"
    if risk < BUFFER_THRESH:
        return "buffer"
    return "freeze"
