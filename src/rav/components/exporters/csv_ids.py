"""`csv_ids`: như `csv_images` nhưng không ghi ảnh — CSV cùng cột, `file` = đường dẫn gốc của video/ảnh."""

from __future__ import annotations

import csv
from pathlib import Path

from ...core.registry import register
from ...core.types import Unit
from .base import Exporter


@register("exporter")
class CsvIds(Exporter):
    """`export.csv` một dòng mỗi unit, cột như `csv_images` (unit_id, video_id, t0, t1, frame_idx, file) với `file` = `ctx.videos[video_id].uri`.
    Không decode, không copy ảnh: dùng khi dataset đã có file (vd BDD100K) và bên nhận (harness đánh giá) chỉ cần id."""

    name = "csv_ids"
    version = "1"

    def export(self, units: list[Unit], dest: Path, ctx) -> Path:
        path = Path(dest) / "export.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["unit_id", "video_id", "t0", "t1", "frame_idx", "file"])
            for u in units:
                video = ctx.videos.get(u.video_id)
                w.writerow([u.id, u.video_id, f"{u.t0:.3f}", f"{u.t1:.3f}", u.anchor.idx, video.uri if video else ""])
        return path
