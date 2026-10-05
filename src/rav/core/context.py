"""`Context` — cổng giao tiếp duy nhất của component (CONTRACTS §4).

Component không gọi nhau: cần field thì hỏi `ctx.field(name)`; Context tìm feature `provides` field đó
trong registry, nhờ execution tính (hoặc lấy cache), rồi căn kết quả theo đúng thứ tự unit được hỏi.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Protocol

import numpy as np

from .component import Feature
from .registry import REGISTRY, Registry
from .types import FieldKey, FieldTable, Frame, Proposal, ReviewState, Unit, VideoRef


class Execution(Protocol):
    """Chính sách tính feature (CONTRACTS §5 dòng 5). Bản thật: `core.execution.Execution`."""

    def run(self, feature: Feature, units: list[Unit], ctx: Context) -> FieldTable: ...


class Context:
    def __init__(
        self,
        units: list[Unit],
        *,
        execution: Execution,
        videos: dict[str, VideoRef] | None = None,   # video của pool theo id, để pixels biết mở file nào (#37)
        state: ReviewState | None = None,
        proposals: list[Proposal] | None = None,
        seed: int = 0,
        reference=None,               # component kind `reference` của recipe; None nếu chưa có recipe
        decoder=None,                 # component kind `decoder` (có `read(video, frame)`)
        registry: Registry = REGISTRY,
        on_progress=None,             # on_progress(field, done, total): tiến độ tính field (P1 nền, #56); None = không báo
    ):
        self.units = units            # pool của phiên (≠ candidates, CONTRACTS §6)
        self.videos = videos or {}
        self.state = state or ReviewState()
        self.proposals = proposals or []
        self.rng = np.random.default_rng(seed)
        self._execution = execution
        self._reference = reference
        self._registry = registry
        self._read = lru_cache(maxsize=64)(lambda f: decoder.read(self.videos[f.video_id], f)) if decoder else None
        self._features: dict[str, Feature] = {}
        self._resolving: list[str] = []
        self._used: set[FieldKey] = set()
        self._on_progress = on_progress
        self._progress: list | None = None   # [field, done, total] khi execution đang tính một field

    def field(self, name: str, units: list[Unit] | None = None) -> FieldTable:
        """Field `name` căn theo `units` (mặc định cả pool)."""
        units = self.units if units is None else units
        feature = self._feature(name)
        if name in self._resolving:
            raise RuntimeError(f"vòng phụ thuộc field: {' → '.join([*self._resolving, name])}")
        self._resolving.append(name)
        try:
            table = self._execution.run(feature, units, self)
        finally:
            self._resolving.pop()
        self._used.add(table.key)
        return _align(table, [u.id for u in units])

    def cached(self, name: str, units: list[Unit]) -> FieldTable | None:
        """Field `name` căn theo `units` nếu đã có trong cache, **không tính** (bằng chứng hiển thị, CONTRACTS §8.0); thiếu phần nào → None."""
        table = self._execution.peek(self._feature(name), units, self)
        return None if table is None else _align(table, [u.id for u in units])

    def reference(self) -> list[Unit]:
        """Tập để so sánh, do component `reference` của recipe quyết định (MVP: tập đã keep)."""
        if self._reference is None:
            raise RuntimeError("Context chưa có reference (recipe chưa gắn)")
        return self._reference.select(self)

    def set_reference(self, ref) -> None:
        """Pipeline gắn component `reference` của recipe trước khi chấm (recipe đổi được giữa vòng, CONTRACTS §10)."""
        self._reference = ref

    def pixels(self, frame: Frame) -> np.ndarray:
        if self._read is None:
            raise RuntimeError("Context chưa có decoder")
        p = self._progress
        if p is not None:                    # tiến độ = số frame feature đã đọc (#56); báo thưa để không tốn
            p[1] += 1
            if p[1] % 8 == 0 or p[1] >= p[2]:
                self._on_progress(p[0], min(p[1], p[2]), p[2])
        return self._read(frame)

    def _start_progress(self, field: str, total: int) -> None:
        """Execution gọi trước khi tính `field` trên `total` unit (hạ tầng, không phải API của component)."""
        if self._on_progress is not None:
            self._progress = [field, 0, max(total, 1)]
            self._on_progress(field, 0, total)

    def _end_progress(self) -> None:
        if self._progress is not None:
            self._on_progress(self._progress[0], self._progress[2], self._progress[2])
            self._progress = None

    def fields_used(self) -> set[FieldKey]:
        """Mọi field đã đọc — để metric kiểm tra không tự chấm bằng field của selector (§12.5)."""
        return set(self._used)

    def _feature(self, name: str) -> Feature:
        if name not in self._features:
            cls = self._registry.feature_for(name)
            if cls.per_sample:
                # Luật phụ thuộc (CONTRACTS §3): cache theo video sẽ sai nếu dựa vào field tính trên cả pool.
                for dep in cls.requires:
                    if not self._registry.feature_for(dep).per_sample:
                        raise TypeError(f"{cls.name}: per_sample=True nhưng requires {dep!r} có per_sample=False")
            self._features[name] = cls()   # Params mặc định; biến thể = feature khác, field khác (CONTRACTS §3)
        return self._features[name]


def _align(table: FieldTable, ids: list[str]) -> FieldTable:
    """Chọn và sắp lại các hàng của `table` theo đúng thứ tự `ids`."""
    if table.unit_ids == ids:
        return table
    pos = {uid: i for i, uid in enumerate(table.unit_ids)}
    missing = [uid for uid in ids if uid not in pos]
    if missing:
        raise KeyError(f"{table.key}: thiếu {len(missing)} unit, vd {missing[:3]}")
    rows = [pos[uid] for uid in ids]
    values = table.values[rows] if isinstance(table.values, np.ndarray) else [table.values[i] for i in rows]
    return FieldTable(table.key, list(ids), values, table.kind)
