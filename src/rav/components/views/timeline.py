"""`timeline`: field theo t của một video, vị trí pick, kept / dropped."""

from __future__ import annotations

import numpy as np
from pydantic import Field

from ...core.registry import register
from ...core.types import ViewSpec
from .base import View, picked_ids


@register("view")
class Timeline(View):
    """Unit của video `video_id` (mặc định video đầu của pool) theo thời gian `t` (= t0). `series` = giá trị các field scalar theo tên;
    `picks` / `kept` / `dropped` = index vào `t` (picks của proposal `proposal_n`, mặc định mới nhất). Điểm objective FE chồng từ `ProposalDTO.scores`."""

    name = "timeline"
    version = "1"

    class Params(View.Params):
        video_id: str | None = None
        series: list[str] = Field(default_factory=list)
        proposal_n: int | None = None

    def build(self, ctx) -> ViewSpec:
        video_id = self.params.video_id or (ctx.units[0].video_id if ctx.units else None)
        units = [u for u in ctx.units if u.video_id == video_id]
        if self.params.video_id and not units:
            raise KeyError(f"video {video_id!r} không thuộc pool")
        series = {}
        for name in self.params.series:
            table = ctx.field(name, units)
            if table.kind != "scalar":
                raise ValueError(f"timeline chỉ vẽ field scalar; {name!r} là {table.kind}")
            series[name] = [float(v) for v in np.asarray(table.values, np.float64)]
        picked = picked_ids(ctx, self.params.proposal_n)
        ids = [u.id for u in units]
        return ViewSpec(self.name, {
            "video_id": video_id, "t": [u.t0 for u in units], "unit_id": ids, "series": series,
            "picks": [i for i, uid in enumerate(ids) if uid in picked],
            "kept": [i for i, uid in enumerate(ids) if uid in ctx.state.kept],
            "dropped": [i for i, uid in enumerate(ids) if uid in ctx.state.dropped]})
