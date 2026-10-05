"""Tính và cache feature — hạ tầng, không phải kind (CONTRACTS §3, §5 dòng 5).

Lazy: chỉ tính khi `ctx.field` hỏi tới lần đầu; kết quả lưu đĩa qua `FieldStore`, lần sau đọc lại (mmap).
- `per_sample=True`: đơn vị cache = cả video. Video nào chưa có file thì `compute` trên **mọi unit của các video đó** trong pool
  (một lần), rồi tách và lưu mỗi video một file theo `(FieldKey, (video_id, spec_hash))`.
- `per_sample=False`: `compute` trên cả pool (`ctx.units`), lưu theo `(FieldKey, pool_hash)`.
Trả về FieldTable của các video / pool liên quan; `Context` tự cắt và căn theo `units` được hỏi.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import numpy as np

from .component import Component, Feature
from .types import FieldTable, Unit

if TYPE_CHECKING:
    from ..store.field_store import FieldStore
    from .context import Context


def spec_hash(decoder: Component, units: Component) -> str:
    """Hash cách tách frame (decoder + unit builder, gồm cả version và Params đã điền mặc định) — CONTRACTS §3."""
    spec = [[c.name, c.version, c.params.model_dump(mode="json")] for c in (decoder, units)]
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


def pool_hash(units: list[Unit]) -> str:
    """Hash danh sách `unit_id` theo thứ tự ổn định `(video_id, t0)` — CONTRACTS §3, §13.1.5."""
    ids = [u.id for u in sorted(units, key=lambda u: (u.video_id, u.t0))]
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()[:16]


class Execution:
    def __init__(self, store: FieldStore, spec_hash: str):
        self.store = store
        self.spec_hash = spec_hash

    def run(self, feature: Feature, units: list[Unit], ctx: Context) -> FieldTable:
        if not feature.per_sample:
            part = pool_hash(ctx.units)
            table = self.store.get(feature.key, part)
            if table is None:
                table = _compute(feature, ctx.units, ctx)
                self.store.put(table, part)
            return table

        video_ids = list(dict.fromkeys(u.video_id for u in units))
        tables = {v: self.store.get(feature.key, (v, self.spec_hash)) for v in video_ids}
        missing = {v for v, t in tables.items() if t is None}
        if missing:
            full = _compute(feature, [u for u in ctx.units if u.video_id in missing], ctx)
            for v in missing:
                tables[v] = _subset(full, [i for i, uid in enumerate(full.unit_ids) if uid.startswith(f"{v}:")])
                self.store.put(tables[v], (v, self.spec_hash))
        return _concat([tables[v] for v in video_ids])


    def peek(self, feature: Feature, units: list[Unit], ctx: Context) -> FieldTable | None:
        """Như `run` nhưng chỉ đọc cache: thiếu phần nào thì None, không tính."""
        if not feature.per_sample:
            return self.store.get(feature.key, pool_hash(ctx.units))
        tables = [self.store.get(feature.key, (v, self.spec_hash)) for v in dict.fromkeys(u.video_id for u in units)]
        return None if any(t is None for t in tables) else _concat(tables)


def _compute(feature: Feature, units: list[Unit], ctx: Context) -> FieldTable:
    ctx._start_progress(feature.provides, len(units))
    try:
        table = feature.compute(units, ctx)
    finally:
        ctx._end_progress()
    if table.key != feature.key or table.unit_ids != [u.id for u in units] or len(table.values) != len(units):
        raise ValueError(f"{feature.name}: FieldTable không khớp key / unit_ids được đưa vào")
    return table


def _subset(table: FieldTable, rows: list[int]) -> FieldTable:
    values = np.asarray(table.values)[rows] if table.kind != "structured" else [table.values[i] for i in rows]
    return FieldTable(table.key, [table.unit_ids[i] for i in rows], values, table.kind)


def _concat(tables: list[FieldTable]) -> FieldTable:
    if len(tables) == 1:
        return tables[0]
    first = tables[0]
    ids = [uid for t in tables for uid in t.unit_ids]
    if first.kind == "structured":
        values = [v for t in tables for v in t.values]
    else:
        values = np.concatenate([np.asarray(t.values) for t in tables])
    return FieldTable(first.key, ids, values, first.kind)
