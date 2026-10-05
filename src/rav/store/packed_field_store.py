"""`PackedFieldStore`: cùng interface `FieldStore` (get / put theo `(FieldKey, part)`), nhưng field `per_sample` của mọi video
nằm chung **một file** cho mỗi `(FieldKey, spec_hash)` thay vì mỗi video một file.

Dùng khi "video" rất nhiều và rất ngắn (vd BDD100K images: mỗi ảnh là một video một frame → 20k ảnh = 40k file / field với `FieldStore`).
`put` chỉ ghi vào bộ nhớ; gọi `flush()` để ghi đĩa (`<root>/packed/<spec_hash>/<stem>.npz` + `.json` ghi sau cùng = đánh dấu xong).
Field theo pool (`part` là str) và field `structured` giữ nguyên cách lưu của `FieldStore`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..core.types import FieldKey, FieldTable
from .field_store import FieldStore, Part


class PackedFieldStore(FieldStore):
    def __init__(self, root: Path):
        super().__init__(root)
        self._packs: dict[tuple[FieldKey, str], dict[str, tuple[list[str], np.ndarray]]] = {}
        self._dirty: set[tuple[FieldKey, str]] = set()

    def _pack_stem(self, key: FieldKey, spec: str) -> Path:
        return self.root / "packed" / spec / f"{key.name}--{key.producer}@{key.version}-{key.params_hash}"

    def _pack(self, key: FieldKey, spec: str) -> dict:
        pk = (key, spec)
        if pk not in self._packs:
            stem = self._pack_stem(key, spec)
            pack: dict = {}
            if Path(f"{stem}.json").is_file():
                meta = json.loads(Path(f"{stem}.json").read_text(encoding="utf-8"))
                values = np.load(Path(f"{stem}.npz"))["values"]
                start = 0
                for video, ids in zip(meta["videos"], meta["unit_ids"]):
                    pack[video] = (ids, values[start:start + len(ids)])
                    start += len(ids)
            self._packs[pk] = pack
        return self._packs[pk]

    def get(self, key: FieldKey, part: Part) -> FieldTable | None:
        if not isinstance(part, tuple):
            return super().get(key, part)
        video, spec = part
        hit = self._pack(key, spec).get(video)
        if hit is None:
            return super().get(key, part)   # cache cũ dạng mỗi video một file (nếu có)
        return FieldTable(key, list(hit[0]), hit[1], "vector" if hit[1].ndim > 1 else "scalar")

    def put(self, table: FieldTable, part: Part) -> None:
        if not isinstance(part, tuple) or table.kind == "structured":
            return super().put(table, part)
        video, spec = part
        self._pack(table.key, spec)[video] = (list(table.unit_ids), np.asarray(table.values))
        self._dirty.add((table.key, spec))

    def flush(self) -> None:
        for key, spec in sorted(self._dirty, key=str):
            pack = self._packs[(key, spec)]
            videos = sorted(pack)
            stem = self._pack_stem(key, spec)
            stem.parent.mkdir(parents=True, exist_ok=True)
            np.savez(Path(f"{stem}.npz"), values=np.concatenate([pack[v][1] for v in videos]))
            tmp = Path(f"{stem}.json.tmp")
            tmp.write_text(json.dumps({"videos": videos, "unit_ids": [pack[v][0] for v in videos]}), encoding="utf-8")
            tmp.replace(Path(f"{stem}.json"))
        self._dirty.clear()
