"""`product_strength`: tích score ** strength, mask = AND; objectives rỗng → combined = 1."""

from __future__ import annotations

import numpy as np

from ...core.registry import register
from ...core.types import Scores
from .base import Combiner


@register("combiner")
class ProductStrength(Combiner):
    """combined = ∏ score_i ** strength_i; mask = AND các mask (None = toàn True).
    Không có objective → combined = 1, mask toàn True."""

    name = "product_strength"
    version = "1"

    def combine(self, scores: list[tuple[Scores, float]], unit_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
        combined, mask = np.ones(len(unit_ids)), np.ones(len(unit_ids), bool)
        for s, strength in scores:
            if s.unit_ids != unit_ids:
                raise ValueError(f"Scores {s.name!r} không căn theo unit_ids")
            combined *= np.asarray(s.values, np.float64) ** strength
            if s.mask is not None:
                mask &= np.asarray(s.mask, bool)
        return combined, mask
