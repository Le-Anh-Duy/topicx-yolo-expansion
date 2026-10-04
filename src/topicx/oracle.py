"""Kho nhãn ẩn của candidate pool (mô phỏng annotator). Mọi truy cập được ghi log.

- `reveal(manifest)`: chỉ mở nhãn của ảnh trong manifest đã chốt (simulated annotation từ ground truth có sẵn).
- `privileged_*`: chỉ dùng cho nhánh ORACLE_POSITIVE / RETRIEVAL_MATCHED và đánh giá retrieval.
- `no_oracle_access()`: khi bật, mọi open/listdir tới đường dẫn chứa 'oracle' bị chặn (audit hook) — bọc bước embed + selection.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

from .common import read_frozen, read_ids

_GUARD = {"on": False}
_EVENTS = {"open", "os.listdir", "os.scandir", "glob.glob"}
_installed = False


def _hook(event, args):
    if _GUARD["on"] and event in _EVENTS and args and "oracle" in str(args[0]).lower():
        raise PermissionError(f"truy cập oracle trong lúc selection: {args[0]}")


class no_oracle_access:
    def __enter__(self):
        global _installed
        if not _installed:
            sys.addaudithook(_hook)  # không gỡ được; bật/tắt bằng cờ
            _installed = True
        _GUARD["on"] = True
        return self

    def __exit__(self, *exc):
        _GUARD["on"] = False


class OracleStore:
    def __init__(self, oracle_dir: Path, log_path: Path):
        oracle_dir = Path(oracle_dir)
        self._boxes = pd.read_csv(oracle_dir / "pool_boxes.csv")
        self._images = pd.read_csv(oracle_dir / "pool_images.csv")
        self._pool = set(read_ids(oracle_dir / "pool_ids.txt"))
        self._log = Path(log_path)
        self._log.parent.mkdir(parents=True, exist_ok=True)

    def _note(self, what: str, **kw):
        with open(self._log, "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.time(), "what": what, **kw}) + "\n")

    def reveal(self, manifest_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        """(boxes đầy đủ base + novel, thuộc tính ảnh) cho đúng các ảnh pool trong manifest."""
        m = read_frozen(manifest_path)
        ids = set(m["ids"])
        assert ids <= self._pool, "manifest chứa ảnh ngoài pool"
        self._note("reveal", manifest=str(manifest_path), branch=m.get("branch"), n=len(ids))
        return self._boxes[self._boxes.image.isin(ids)].copy(), self._images[self._images.image.isin(ids)].copy()

    def privileged_positive_ids(self, cls: str, reason: str) -> list[str]:
        self._note("privileged_positive_ids", cls=cls, reason=reason)
        return sorted(set(self._boxes.image[self._boxes.cls == cls]))

    def privileged_instance_counts(self, ids: list[str], cls: str, reason: str) -> pd.Series:
        """Số instance `cls` theo ảnh, đúng thứ tự `ids`."""
        self._note("privileged_instance_counts", cls=cls, reason=reason, n=len(ids))
        c = self._boxes[self._boxes.cls == cls].groupby("image").size()
        return c.reindex(ids, fill_value=0)
