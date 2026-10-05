"""`farthest_history`: greedy farthest-first có xét lịch sử (Algo 1 — docs/algo_1_semantic_retrieval_history_diversity.md §6)."""

from __future__ import annotations

import numpy as np

from ...core.registry import register
from ...core.types import SelectResult
from .base import Selector


def _unit(x) -> np.ndarray:
    x = np.asarray(x, np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


@register("selector")
class FarthestHistory(Selector):
    """Chọn n unit (mask True) lần lượt: mỗi bước lấy unit có δ = min d(x, y), y ∈ reference ∪ đã chọn trong batch, lớn nhất;
    d = (1 − cos(emb)) / 2 trên `field`. Hoà δ → combined cao hơn → (video_id, t0). Chưa có reference và batch rỗng → unit combined cao nhất.
    `history=False`: bỏ reference (chỉ diversity trong batch — đối chứng của Algo 1). Relevance là điều kiện qua mask (vd `text_match` top_k / tau),
    không cộng vào điểm chọn."""

    name = "farthest_history"
    version = "1"

    class Params(Selector.Params):
        field: str = "emb.clip_b32"
        history: bool = True

    def pick(self, units, combined, mask, n, ctx) -> SelectResult:
        cand = sorted(np.flatnonzero(mask), key=lambda i: (-combined[i], units[i].video_id, units[i].t0))
        k = min(n, len(cand))
        if k == 0:
            return SelectResult([])
        e = _unit(ctx.field(self.params.field, [units[i] for i in cand]).values)
        d = np.full(len(cand), np.inf)
        ref = ctx.reference() if self.params.history else []
        if ref:
            d = ((1 - e @ _unit(ctx.field(self.params.field, ref).values).T) / 2).min(axis=1)
        took: list[int] = []
        for _ in range(k):
            j = int(np.argmax(d))   # hoà → index nhỏ nhất = combined cao hơn (cand đã sắp)
            took.append(j)
            d = np.minimum(d, (1 - e @ e[j]) / 2)
            d[took] = -np.inf
        return SelectResult([int(cand[j]) for j in took])
