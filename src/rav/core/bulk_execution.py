"""`BulkExecution`: như `Execution` (CONTRACTS §3) cho pool có rất nhiều video ngắn (vd BDD100K images: mỗi ảnh một video).

Khác duy nhất: sau khi tính feature `per_sample` cho các video thiếu cache, tách kết quả theo video **một lượt** (O(N)) thay vì
quét toàn bộ `unit_ids` cho từng video (O(N × số video) — 20k ảnh ≈ 4·10⁸ phép so chuỗi). Kết quả và cache giống hệt `Execution`.
Nên dùng cùng `PackedFieldStore` để không sinh mỗi video một file.
"""

from __future__ import annotations

from collections import defaultdict

from .execution import Execution, _compute, _concat, _subset


class BulkExecution(Execution):
    def run(self, feature, units, ctx):
        if not feature.per_sample:
            return super().run(feature, units, ctx)
        video_ids = list(dict.fromkeys(u.video_id for u in units))
        tables = {v: self.store.get(feature.key, (v, self.spec_hash)) for v in video_ids}
        missing = {v for v, t in tables.items() if t is None}
        if missing:
            todo = [u for u in ctx.units if u.video_id in missing]
            full = _compute(feature, todo, ctx)
            video_of = {u.id: u.video_id for u in todo}
            rows: dict[str, list[int]] = defaultdict(list)
            for i, uid in enumerate(full.unit_ids):
                rows[video_of[uid]].append(i)
            for v in missing:
                tables[v] = _subset(full, rows[v])
                self.store.put(tables[v], (v, self.spec_hash))
        return _concat([tables[v] for v in video_ids])
