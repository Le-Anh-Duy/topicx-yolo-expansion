"""`cluster_coverage`: tập phủ bao nhiêu cụm k-means của pool (#59) — số mô tả, cụm không mang nghĩa nghiệp vụ."""

from __future__ import annotations

import numpy as np

from ...core.registry import register
from .base import Metric


@register("metric")
class ClusterCoverage(Metric):
    """Đọc field cụm của cả pool (mặc định `cluster.kmeans`). `cluster_coverage` = số cụm có ít nhất một unit của tập / số cụm của pool;
    `cluster_coverage/entropy` = entropy phân bố tập theo cụm / log(số cụm) (1 = rải đều). Tập rỗng → 0."""

    name = "cluster_coverage"
    version = "1"

    class Params(Metric.Params):
        field: str = "cluster.kmeans"

    @property
    def fields(self) -> tuple[str, ...]:
        return (self.params.field,)

    def evaluate(self, unit_ids, ctx) -> dict[str, float]:
        pool = np.asarray(ctx.field(self.params.field).values)
        k = len(np.unique(pool))
        if not unit_ids or k == 0:
            return {self.name: 0.0, f"{self.name}/entropy": 0.0}
        by_id = {u.id: u for u in ctx.units}
        labels = np.asarray(ctx.field(self.params.field, [by_id[i] for i in unit_ids]).values)
        _, counts = np.unique(labels, return_counts=True)
        p = counts / counts.sum()
        entropy = float(-(p * np.log(p)).sum() / np.log(k)) if k > 1 else 1.0
        return {self.name: len(counts) / k, f"{self.name}/entropy": entropy}
