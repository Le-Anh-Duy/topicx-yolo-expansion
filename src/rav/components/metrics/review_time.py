"""`review_time`: thời gian người duyệt bỏ ra cho mỗi frame, từ log quyết định (#59) — nền cho kiểm chứng A của tài liệu focus."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from ...core.registry import register
from .base import Metric


@register("metric")
class ReviewTime(Metric):
    """Mỗi đợt (proposal) có quyết định keep / drop kèm `shown_at`: thời gian = (quyết định cuối − lúc lưới hiện ra) / số quyết định của đợt.
    `review_time` = trung vị giây / frame qua các đợt; `per_minute` = tổng số quyết định / tổng phút; `batches`, `decisions` = số đếm.
    Tính trên **mọi quyết định của phiên** (không phụ thuộc `unit_ids`); đợt dài hơn `max_batch_s` (bỏ dở rồi quay lại) không tính.
    Không đọc field → luôn độc lập. Chưa tách thời gian xem clip / chờ hệ thống."""

    name = "review_time"
    version = "1"

    class Params(Metric.Params):
        max_batch_s: float = 1800.0

    def evaluate(self, unit_ids, ctx) -> dict[str, float]:
        by_n = defaultdict(list)
        for d in ctx.state.history:
            if d.action in ("keep", "drop") and d.proposal_n is not None and d.shown_at is not None:
                by_n[d.proposal_n].append(d)
        per_frame, total_s, total_n = [], 0.0, 0
        for ds in by_n.values():
            span = (max(d.decided_at for d in ds) - min(d.shown_at for d in ds)).total_seconds()
            if 0 < span <= self.params.max_batch_s:
                per_frame.append(span / len(ds))
                total_s += span
                total_n += len(ds)
        return {self.name: float(np.median(per_frame)) if per_frame else 0.0,
                f"{self.name}/per_minute": 60.0 * total_n / total_s if total_s else 0.0,
                f"{self.name}/batches": float(len(per_frame)), f"{self.name}/decisions": float(total_n)}
