"""`csv_images`: CSV (unit_id, video_id, t0, t1, frame_idx, file) + thư mục ảnh cho CVAT / Label Studio."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2

from ...core.registry import register
from ...core.types import Unit
from .base import Exporter


@register("exporter")
class CsvImages(Exporter):
    """`export.csv` một dòng mỗi unit (unit_id, video_id, t0, t1, frame_idx, file) + `images/<video_id>_<t0>.jpg`
    là frame đại diện (`anchor`) của unit, JPEG chất lượng `quality`."""

    name = "csv_images"
    version = "1"

    class Params(Exporter.Params):
        quality: int = 95

    def export(self, units: list[Unit], dest: Path, ctx) -> Path:
        images = dest / "images"
        images.mkdir(parents=True, exist_ok=True)
        path = dest / "export.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["unit_id", "video_id", "t0", "t1", "frame_idx", "file"])
            for u in units:
                file = f"images/{u.video_id}_{u.t0:.3f}.jpg"
                ok, buf = cv2.imencode(".jpg", cv2.cvtColor(ctx.pixels(u.anchor), cv2.COLOR_RGB2BGR),
                                       [cv2.IMWRITE_JPEG_QUALITY, self.params.quality])
                if not ok:
                    raise RuntimeError(f"{u.id}: không mã hoá được JPEG")
                (dest / file).write_bytes(buf.tobytes())   # cv2.imwrite không ghi được đường dẫn có dấu
                w.writerow([u.id, u.video_id, f"{u.t0:.3f}", f"{u.t1:.3f}", u.anchor.idx, file])
        return path
