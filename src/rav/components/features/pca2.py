"""`proj.pca2`: PCA 2 chiều của emb.dinov2_s, fit trên cả pool, cho view embedding_map. per_sample=False."""

from __future__ import annotations

import numpy as np

from ...core.component import Feature
from ...core.registry import register
from ...core.types import FieldTable


@register("feature")
class PCA2(Feature):
    """Chiếu emb.dinov2_s xuống 2 chiều bằng PCA fit trên cả pool: trừ trung bình, SVD, lấy 2 thành phần chính đầu;
    dấu mỗi trục cố định (hệ số có trị tuyệt đối lớn nhất là dương) để tái lập. Chỉ để khám phá, không dùng để chọn."""

    name = "pca2"
    version = "1"
    provides = "proj.pca2"
    requires = ("emb.dinov2_s",)

    def compute(self, units, ctx) -> FieldTable:
        x = np.asarray(ctx.field("emb.dinov2_s", units).values, np.float64)
        x = x - x.mean(axis=0) if len(x) else x
        xy = np.zeros((len(units), 2), np.float32)
        if len(units) >= 2:
            _, _, vt = np.linalg.svd(x, full_matrices=False)
            comps = vt[:2]
            comps *= np.sign(comps[np.arange(len(comps)), np.abs(comps).argmax(axis=1)])[:, None]
            xy[:, :len(comps)] = x @ comps.T
        return FieldTable(self.key, [u.id for u in units], xy, "vector")
