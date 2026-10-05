"""`nd_rate_at_k`: ND-Rate@K bằng pHash (CONTRACTS §9.10). Lát: L4."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from pydantic import Field

from ...core.registry import register
from .base import Metric


@register("metric")
class NDRateAtK(Metric):
    """Tỉ lệ unit trong tập có ít nhất một unit khác trong chính tập đó cách Hamming pHash ≤ `threshold`. Thấp = tốt; tập rỗng → 0.
    `by_video/<id>` = tỉ lệ đó trên các unit của từng video (vẫn so với cả tập)."""

    name = "nd_rate_at_k"
    version = "1"

    class Params(Metric.Params):
        field: str = "hash.phash"
        threshold: int = Field(8, ge=0, le=64)   # chốt sau spot-check L7b (#51); pHash luôn chẵn

    @property
    def fields(self) -> tuple[str, ...]:
        return (self.params.field,)

    def evaluate(self, unit_ids, ctx) -> dict[str, float]:
        if not unit_ids:
            return {self.name: 0.0}
        by_id = {u.id: u for u in ctx.units}
        units = [by_id[i] for i in unit_ids]
        values = np.asarray(ctx.field(self.params.field, units).values)
        if values.dtype != np.uint64:
            raise ValueError(f"nd_rate_at_k cần field hash 64-bit (vd hash.phash); {self.params.field!r} có dtype {values.dtype}")
        h = values
        bits = np.unpackbits((h[:, None] ^ h[None, :]).view(np.uint8).reshape(len(h), len(h), 8), axis=-1)
        dist = bits.sum(-1)
        np.fill_diagonal(dist, 65)
        dup = (dist <= self.params.threshold).any(1)
        groups = defaultdict(list)
        for u, d in zip(units, dup):
            groups[u.video_id].append(d)
        return {self.name: float(dup.mean()),
                **{f"by_video/{v}": float(np.mean(ds)) for v, ds in groups.items()}}
