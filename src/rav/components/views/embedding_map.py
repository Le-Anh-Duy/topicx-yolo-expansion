"""`embedding_map`: scatter 2D (proj.pca2), màu theo pool / picked / kept / dropped. Chỉ để khám phá."""

from __future__ import annotations

from ...core.registry import register
from ...core.types import ViewSpec
from .base import View, picked_ids


@register("view")
class EmbeddingMap(View):
    """Mỗi unit của pool một điểm (x, y) = field 2D `field` (mặc định PCA của DINOv2-S). `status` ưu tiên kept > dropped > picked > pool;
    picked = pick của proposal `proposal_n` (mặc định mới nhất). Khoảng cách trên mặt phẳng 2D bị méo: chỉ để khám phá, không để đánh giá."""

    name = "embedding_map"
    version = "1"

    class Params(View.Params):
        field: str = "proj.pca2"
        proposal_n: int | None = None

    def build(self, ctx) -> ViewSpec:
        xy = ctx.field(self.params.field).values
        picked, state = picked_ids(ctx, self.params.proposal_n), ctx.state

        def status(uid: str) -> str:
            return ("kept" if uid in state.kept else "dropped" if uid in state.dropped
                    else "picked" if uid in picked else "pool")

        return ViewSpec(self.name, {"x": [float(v) for v in xy[:, 0]], "y": [float(v) for v in xy[:, 1]],
                                    "unit_id": [u.id for u in ctx.units], "status": [status(u.id) for u in ctx.units]})
