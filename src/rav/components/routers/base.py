"""Base của kind `router` (CONTRACTS §5 dòng 16, §12). MVP chỉ có contract: người chọn recipe + bấm "Đề xuất" (quyết định #20)."""

from __future__ import annotations

from typing import ClassVar

from ...core.component import Component
from ...core.context import Context
from ...core.types import Action, History


class Router(Component):
    """Chọn việc tiếp theo: đề xuất với recipe nào, batch bao nhiêu, hay dừng."""

    kind: ClassVar[str] = "router"

    def next(self, history: History, ctx: Context) -> Action:
        raise NotImplementedError
