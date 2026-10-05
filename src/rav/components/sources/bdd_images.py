"""`bdd_images`: mỗi ảnh BDD100K (keyframe của một video BDD) là một VideoRef một frame; danh sách đọc từ file index không có nhãn."""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path

from ...core.registry import register
from ...core.types import VideoRef
from .base import Source


@register("source")
class BddImages(Source):
    """Mỗi dòng của `index` (CSV cột `image`, `path`) thành một VideoRef: id = tên ảnh bỏ đuôi (= tên video BDD `<ride>-<clip>`),
    uri = đường dẫn ảnh, fps = 1, duration_s = 1 (một frame). Index chỉ chứa ảnh và đường dẫn — không có nhãn hay thuộc tính cảnh,
    nên proposer chạy trên source này không thấy ground truth."""

    name = "bdd_images"
    version = "1"

    class Params(Source.Params):
        index: str = "pool_index.csv"

    def iter_videos(self) -> Iterator[VideoRef]:
        with open(self.params.index, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                yield VideoRef(Path(row["image"]).stem, row["path"], 1.0, 1.0, {"name": row["image"]})
