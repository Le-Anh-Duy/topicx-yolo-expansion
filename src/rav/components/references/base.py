"""Base của kind `reference` (CONTRACTS §5 dòng 6): chọn tập unit để objective so sánh."""

from __future__ import annotations

from typing import ClassVar

from ...core.component import Component
from ...core.types import Unit


class Reference(Component):
    """Tập unit để so sánh (vd tập đã keep). Đọc `ctx.state`, không cache."""

    kind: ClassVar[str] = "reference"

    def select(self, ctx) -> list[Unit]:
        raise NotImplementedError
