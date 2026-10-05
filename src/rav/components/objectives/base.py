"""Base của kind `objective` (CONTRACTS §2 Scores, §5 dòng 7, §9.1, §9.5)."""

from __future__ import annotations

from typing import ClassVar

from ...core.component import Component
from ...core.types import Scores, Unit


class Objective(Component):
    """Chấm từng unit độc lập, điểm trong [0,1], cao = nên chọn; `Scores.name` = `name` (= `type` trong recipe)."""

    kind: ClassVar[str] = "objective"
    uses_goal: ClassVar[bool] = False   # có đọc mục tiêu phiên không — UI nói thẳng khi recipe chưa dùng mục tiêu (#61)

    def score(self, units: list[Unit], ctx) -> Scores:
        raise NotImplementedError
