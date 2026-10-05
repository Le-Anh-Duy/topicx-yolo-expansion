"""`novelty`: càng khác tập đã keep càng cao; vòng đầu so với tâm pool. Công thức: CONTRACTS §9.5."""

from __future__ import annotations

import numpy as np

from ...core.registry import register
from ...core.types import Scores, Unit
from .base import Objective


def _unit(x) -> np.ndarray:
    x = np.asarray(x, np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


@register("objective")
class Novelty(Objective):
    """Mức khác biệt so với reference (tập đã keep): (1 − max cos(emb[u], emb[ref])) / 2.
    Reference rỗng: d = 1 − cos(emb[u], tâm pool), min-max về [0,1] trên các unit đang chấm (mọi d bằng nhau → 1).
    Cao = chưa có frame nào giống."""

    name = "novelty"
    version = "1"

    class Params(Objective.Params):
        field: str = "emb.dinov2_s"

    def score(self, units: list[Unit], ctx) -> Scores:
        ids = [u.id for u in units]
        if not units:
            return Scores(self.name, ids, np.zeros(0))
        emb = _unit(ctx.field(self.params.field, units).values)
        ref = ctx.reference()
        if ref:
            cos = emb @ _unit(ctx.field(self.params.field, ref).values).T
            values = (1 - cos.max(axis=1)) / 2
        else:
            d = 1 - emb @ _unit(np.asarray(ctx.field(self.params.field).values).mean(axis=0))
            span = d.max() - d.min()
            values = (d - d.min()) / span if span > 1e-12 else np.ones(len(d))
        return Scores(self.name, ids, np.clip(values, 0.0, 1.0))
