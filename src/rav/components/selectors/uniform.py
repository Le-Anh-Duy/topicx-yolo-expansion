"""`uniform`: cách đều theo thời gian, allocation proportional/equal giữa video. Baseline."""

from __future__ import annotations

from typing import Literal

import numpy as np

from ...core.registry import register
from ...core.types import SelectResult
from .base import Selector


def allocate(sizes: dict[str, int], k: int, equal: bool) -> dict[str, int]:
    """Chia k suất (k ≤ tổng sizes) cho các video: theo tỉ lệ sizes hoặc chia đều, làm tròn theo phần dư lớn nhất
    (hoà → video_id); video hết chỗ thì phần còn thiếu chia lại cho video còn chỗ."""
    alloc = dict.fromkeys(sizes, 0)
    while (left := k - sum(alloc.values())) > 0:
        open_ = [v for v in sorted(sizes) if alloc[v] < sizes[v]]
        w = np.array([1.0 if equal else sizes[v] for v in open_])
        quota = left * w / w.sum()
        take = np.floor(quota).astype(int)
        for j in sorted(range(len(open_)), key=lambda j: (-(quota[j] - take[j]), open_[j]))[: left - take.sum()]:
            take[j] += 1
        for v, t in zip(open_, take):
            alloc[v] += min(int(t), sizes[v] - alloc[v])
    return alloc


@register("selector")
class Uniform(Selector):
    """Cách đều trên candidates (mask True) của từng video theo t0: n candidates, k suất → chỉ số floor((i + 0.5)·n/k).
    K chia giữa video theo `allocation` (proportional: theo số candidates; equal: chia đều), làm tròn phần dư lớn nhất.
    Không dùng seed, bỏ qua combined. Baseline."""

    name = "uniform"
    version = "1"

    class Params(Selector.Params):
        allocation: Literal["proportional", "equal"] = "proportional"

    def pick(self, units, combined, mask, n, ctx) -> SelectResult:
        by_video: dict[str, list[int]] = {}
        for i in np.flatnonzero(mask):
            by_video.setdefault(units[i].video_id, []).append(int(i))
        k = min(n, sum(map(len, by_video.values())))
        alloc = allocate({v: len(ix) for v, ix in by_video.items()}, k, self.params.allocation == "equal")
        out = []
        for v in sorted(by_video):
            ix = sorted(by_video[v], key=lambda i: units[i].t0)
            out += [ix[int((j + 0.5) * len(ix) / alloc[v])] for j in range(alloc[v])]
        return SelectResult(out)
