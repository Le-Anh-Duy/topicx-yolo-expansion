"""`random`: chọn ngẫu nhiên theo ctx.rng. Baseline. (Tên file tránh trùng module chuẩn `random`.)"""

from __future__ import annotations

import numpy as np

from ...core.registry import register
from ...core.types import SelectResult
from .base import Selector


@register("selector")
class RandomSelect(Selector):
    """Hoán vị ngẫu nhiên (ctx.rng, propose đã seed lại) các unit có mask True, lấy n đầu. Baseline, bỏ qua combined."""

    name = "random"
    version = "1"

    def pick(self, units, combined, mask, n, ctx) -> SelectResult:
        return SelectResult([int(i) for i in ctx.rng.permutation(np.flatnonzero(mask))[:n]])
