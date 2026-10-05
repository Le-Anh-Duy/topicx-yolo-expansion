"""Đọc/ghi field theo `(FieldKey, part)` (CONTRACTS §5 dòng 13, §3).

`part` = `(video_id, spec_hash)` cho feature `per_sample=True`, hoặc `pool_hash` (str) cho `per_sample=False`.
Mỗi phần hai file: `<stem>.npy` (giá trị, đọc lại bằng mmap) + `<stem>.json` (`unit_ids`, `kind`; ghi sau cùng = đánh dấu xong).
Field `structured` (list[dict]) nằm luôn trong `.json`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..core.types import FieldKey, FieldTable

Part = tuple[str, str] | str


class FieldStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    def path(self, key: FieldKey, part: Part) -> Path:
        """Đường dẫn chung của hai file, chưa có đuôi (tên field có dấu chấm → không dùng `with_suffix`)."""
        folder = self.root.joinpath(*part) if isinstance(part, tuple) else self.root / "pool" / part
        return folder / f"{key.name}--{key.producer}@{key.version}-{key.params_hash}"

    def get(self, key: FieldKey, part: Part) -> FieldTable | None:
        stem = self.path(key, part)
        meta_path = Path(f"{stem}.json")
        if not meta_path.is_file():
            return None
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        values = meta["values"] if meta["kind"] == "structured" else np.load(Path(f"{stem}.npy"), mmap_mode="r")
        return FieldTable(key, meta["unit_ids"], values, meta["kind"])

    def put(self, table: FieldTable, part: Part) -> None:
        stem = self.path(table.key, part)
        stem.parent.mkdir(parents=True, exist_ok=True)
        meta = {"unit_ids": list(table.unit_ids), "kind": table.kind}
        if table.kind == "structured":
            meta["values"] = list(table.values)
        else:
            np.save(Path(f"{stem}.npy"), np.asarray(table.values))
        tmp = Path(f"{stem}.json.tmp")
        tmp.write_text(json.dumps(meta), encoding="utf-8")
        tmp.replace(Path(f"{stem}.json"))
