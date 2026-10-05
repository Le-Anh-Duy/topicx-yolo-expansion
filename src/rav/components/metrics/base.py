"""Base class của kind `metric` (chữ ký: CONTRACTS §5 dòng 11, §9.10, §12.5). Lát: L4 / L7."""

from __future__ import annotations

from typing import ClassVar

from ...core.component import Component


class Metric(Component):
    """Chấm một tập unit đã chọn. Trả dict phẳng: `{<name>: tổng, "by_video/<id>": …}`; P3 thêm `independent` (#42, #44)."""

    kind: ClassVar[str] = "metric"

    @property
    def fields(self) -> tuple[str, ...]:
        """Field metric đọc (khai báo, §12.5) — P3 so với `Proposal.fields_used` để kiểm độc lập."""
        return ()

    def evaluate(self, unit_ids: list[str], ctx) -> dict[str, float]:
        raise NotImplementedError
