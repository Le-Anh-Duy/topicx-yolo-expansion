"""`cluster.kmeans`: cụm k-means trên emb.dinov2_s của cả pool, cho metric `cluster_coverage` (#59). per_sample=False."""

from __future__ import annotations

import numpy as np

from ...core.component import Feature
from ...core.registry import register
from ...core.types import FieldTable


@register("feature")
class KMeansClusters(Feature):
    """Gom emb.dinov2_s của cả pool thành `k` cụm bằng k-means (scikit-learn, `seed` cố định, n_init=4); giá trị = chỉ số cụm 0..k−1.
    Chỉ là số mô tả độ phủ: cụm không mang nghĩa nghiệp vụ (tài liệu focus §6). Pool ít hơn k unit → mỗi unit một cụm."""

    name = "kmeans"
    version = "1"
    provides = "cluster.kmeans"
    requires = ("emb.dinov2_s",)

    class Params(Feature.Params):
        k: int = 10
        seed: int = 0

    def compute(self, units, ctx) -> FieldTable:
        from sklearn.cluster import KMeans  # nạp lazy

        x = np.asarray(ctx.field("emb.dinov2_s", units).values, np.float32)
        k = min(self.params.k, len(units))
        labels = (KMeans(n_clusters=k, n_init=4, random_state=self.params.seed).fit_predict(x) if k > 1
                  else np.zeros(len(units), np.int64))
        return FieldTable(self.key, [u.id for u in units], labels.astype(np.int64), "scalar")
