"""Base của kind `judge` (CONTRACTS §5 dòng 15, §12). MVP chỉ có contract: người làm judge qua UI (quyết định #20)."""

from __future__ import annotations

from typing import ClassVar

from ...core.component import Component
from ...core.context import Context
from ...core.types import Decision, JudgeRequest


class Judge(Component):
    """Quyết định keep/drop cho các frame của một proposal theo `goal` của phiên."""

    kind: ClassVar[str] = "judge"

    def judge(self, req: JudgeRequest, ctx: Context) -> list[Decision]:
        # `reason` bắt buộc với actor algo:/ai:; được bỏ qua frame (không trả = để người xem).
        raise NotImplementedError
