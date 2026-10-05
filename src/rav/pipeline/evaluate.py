"""P3 — chấm tập đã chọn bằng metric độc lập (ND-Rate@K pHash, CONTRACTS §9.10, §12.5). Lát: L4.

Metric lấy qua registry theo tên (`{"type", "params"}`), không import `components/`. Thuần: không đọc DB.
Kiểm độc lập ở đây, một lần cho mỗi metric: field metric khai báo (`fields`) có nằm trong `fields_used` của proposal không (#44).
`ctx` chỉ để đọc dữ liệu (pool + field) — không cần là ctx vừa propose; chấm lại proposal đã lưu được.
"""

from __future__ import annotations

from collections.abc import Iterable

from ..core.context import Context
from ..core.registry import REGISTRY, discover
from ..core.types import ComponentConfig, FieldKey


def evaluate(metrics: list[ComponentConfig | dict], unit_ids: list[str], fields_used: Iterable[FieldKey],
             ctx: Context) -> list[dict[str, float]]:
    """Một dict số đo cho mỗi metric (cùng thứ tự), kèm `independent` = 1.0 / 0.0."""
    discover()
    used = {k.name for k in fields_used}
    out = []
    for cfg in metrics:
        metric = REGISTRY.build("metric", cfg)
        out.append({**metric.evaluate(unit_ids, ctx), "independent": float(not used & set(metric.fields))})
    return out
