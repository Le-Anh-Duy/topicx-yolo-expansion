"""`greedy_nms`: tham lam + NMS thời gian (cùng video) + min_dist embedding; thiếu thì nới, relaxed=True."""

from __future__ import annotations

import numpy as np
from pydantic import Field

from ...core.registry import register
from ...core.types import SelectResult
from .base import Selector

RELAX = (1.0, 0.5, 0.25)   # hệ số nhân cả hai ngưỡng; hết mà vẫn thiếu thì ×0 = lấy top theo combined


@register("selector")
class GreedyNMS(Selector):
    """Duyệt candidates (mask True) theo combined giảm dần (hoà → (video_id, t0)); nhận unit nếu với mọi pick đã nhận
    trong batch: không (cùng video và |Δt| < nms_s) và 1 − cos(emb) ≥ min_dist. Thiếu thì chạy lại với cả hai ngưỡng
    ×0.5, ×0.25, ×0 và báo relaxed. Không so với tập đã keep (việc của novelty)."""

    name = "greedy_nms"
    version = "1"

    class Params(Selector.Params):
        field: str = "emb.dinov2_s"
        nms_s: float = Field(1.0, ge=0)
        min_dist: float = Field(0.1, ge=0)

    def pick(self, units, combined, mask, n, ctx) -> SelectResult:
        order = sorted(np.flatnonzero(mask), key=lambda i: (-combined[i], units[i].video_id, units[i].t0))
        k = min(n, len(order))
        if k == 0:
            return SelectResult([])
        emb = np.asarray(ctx.field(self.params.field, [units[i] for i in order]).values, np.float64)
        emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
        video = np.array([units[i].video_id for i in order])
        t = np.array([units[i].t0 for i in order])
        for f in RELAX:
            took: list[int] = []
            for j in range(len(order)):
                p = np.array(took, int)
                near_t = (video[p] == video[j]) & (np.abs(t[p] - t[j]) < self.params.nms_s * f)
                near_e = 1 - emb[p] @ emb[j] < self.params.min_dist * f
                if not (near_t | near_e).any():
                    took.append(j)
                    if len(took) == k:
                        return SelectResult([int(order[j]) for j in took], relaxed=f < 1)
        return SelectResult([int(i) for i in order[:k]], relaxed=True)
