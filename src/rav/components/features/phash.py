"""`hash.phash`: pHash 64-bit, dành cho metric ND-Rate (độc lập với embedding). per_sample=True."""

from __future__ import annotations

import imagehash
import numpy as np
from PIL import Image

from ...core.component import Feature
from ...core.registry import register
from ...core.types import FieldTable


@register("feature")
class PHash(Feature):
    """pHash của frame anchor (imagehash.phash): ảnh xám 32×32 → DCT → 8×8 hệ số tần số thấp, bit = hệ số > trung vị;
    lưu thành uint64. Khoảng cách Hamming nhỏ = hai ảnh gần trùng."""

    name = "phash"
    version = "1"
    provides = "hash.phash"
    per_sample = True

    def compute(self, units, ctx) -> FieldTable:
        values = np.array([int(str(imagehash.phash(Image.fromarray(ctx.pixels(u.anchor)))), 16) for u in units],
                          dtype=np.uint64)
        return FieldTable(self.key, [u.id for u in units], values, "scalar")
