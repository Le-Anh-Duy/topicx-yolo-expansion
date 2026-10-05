"""`temporal_coverage`: tập có rải khắp thời lượng video không (#59)."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from ...core.registry import register
from .base import Metric


@register("metric")
class TemporalCoverage(Metric):
    """Chia mỗi video của pool thành các khoảng `bin_s` giây (theo t0 của unit trong pool); tỉ lệ khoảng có ít nhất một unit của tập.
    `temporal_coverage` = trên mọi video gộp lại; `by_video/<id>` = từng video. Không đọc field → luôn độc lập."""

    name = "temporal_coverage"
    version = "1"

    class Params(Metric.Params):
        bin_s: float = 10.0

    def evaluate(self, unit_ids, ctx) -> dict[str, float]:
        chosen = set(unit_ids)
        bins_all: dict[str, set] = defaultdict(set)
        bins_hit: dict[str, set] = defaultdict(set)
        for u in ctx.units:
            b = int(np.floor(u.t0 / self.params.bin_s))
            bins_all[u.video_id].add(b)
            if u.id in chosen:
                bins_hit[u.video_id].add(b)
        total = sum(len(v) for v in bins_all.values())
        out = {self.name: sum(len(v) for v in bins_hit.values()) / total if total else 0.0}
        out.update({f"by_video/{v}": len(bins_hit[v]) / len(b) for v, b in bins_all.items()})
        return out
