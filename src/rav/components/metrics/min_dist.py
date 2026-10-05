"""`min_dist`: độ phân tán của tập theo embedding — khoảng cách tới frame gần nhất trong chính tập (#59)."""

from __future__ import annotations

import numpy as np

from ...core.registry import register
from .base import Metric


@register("metric")
class MinDist(Metric):
    """Với mỗi unit trong tập: d = 1 − max cos(emb[u], emb[v]) trên các v khác trong tập. `min_dist` = trung bình d (cao = tập phân tán),
    `min_dist/min` = d nhỏ nhất (cặp gần nhau nhất). Tập < 2 unit → 0. Đọc cùng field embedding với recipe diversity → thường không độc lập."""

    name = "min_dist"
    version = "1"

    class Params(Metric.Params):
        field: str = "emb.dinov2_s"

    @property
    def fields(self) -> tuple[str, ...]:
        return (self.params.field,)

    def evaluate(self, unit_ids, ctx) -> dict[str, float]:
        if len(unit_ids) < 2:
            return {self.name: 0.0, f"{self.name}/min": 0.0}
        by_id = {u.id: u for u in ctx.units}
        x = np.asarray(ctx.field(self.params.field, [by_id[i] for i in unit_ids]).values, np.float64)
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        sim = x @ x.T
        np.fill_diagonal(sim, -np.inf)
        d = 1.0 - sim.max(axis=1)
        return {self.name: float(d.mean()), f"{self.name}/min": float(d.min())}
