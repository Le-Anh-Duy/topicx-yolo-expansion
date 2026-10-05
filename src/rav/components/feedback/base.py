"""Base class của kind `feedback` (chữ ký: CONTRACTS §5 dòng 10, §12.3)."""

from __future__ import annotations

from typing import ClassVar

from ...core.component import Component
from ...core.types import Decision, ReviewState


class Feedback(Component):
    """Áp một lô `Decision` lên state, trả state **mới** (không sửa `state` truyền vào). `ReviewState = apply(log, ReviewState())`."""

    kind: ClassVar[str] = "feedback"

    def apply(self, decisions: list[Decision], state: ReviewState) -> ReviewState:
        raise NotImplementedError
