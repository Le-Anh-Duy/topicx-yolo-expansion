"""Base của kind `combiner` (CONTRACTS §5 dòng 8, quyết định #40)."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from ...core.component import Component
from ...core.types import Scores


class Combiner(Component):
    """Gộp điểm các objective (đã chuẩn hoá [0,1], không chuẩn hoá lại) thành `combined` + `mask`, căn theo `unit_ids`."""

    kind: ClassVar[str] = "combiner"

    def combine(self, scores: list[tuple[Scores, float]], unit_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError
