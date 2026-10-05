"""Base của kind `selector` (CONTRACTS §2 SelectResult, §5 dòng 9, §9.2, §9.6, §9.8)."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from ...core.component import Component
from ...core.types import SelectResult, Unit


class Selector(Component):
    """Chọn tối đa n unit từ `units` (candidates). Bất biến: đúng min(n, |mask True|) index, duy nhất, không vi phạm mask."""

    kind: ClassVar[str] = "selector"

    def pick(self, units: list[Unit], combined: np.ndarray, mask: np.ndarray, n: int, ctx) -> SelectResult:
        raise NotImplementedError
