"""`per_frame`: mỗi frame một Unit (kind="frame", t0 == t1)."""

from __future__ import annotations

from ...core.registry import register
from ...core.types import Frame, Unit, VideoRef, unit_id
from .base import UnitBuilder


@register("unit")
class PerFrame(UnitBuilder):
    """Mỗi frame lấy mẫu thành một Unit: kind="frame", t0 == t1 == frame.t, anchor = frame, frames = (frame,);
    id = "{video_id}:{t:.3f}-{t:.3f}"."""

    name = "per_frame"
    version = "1"

    def build(self, video: VideoRef, frames: list[Frame]) -> list[Unit]:
        return [Unit(unit_id(video.id, f.t, f.t), video.id, f.t, f.t, "frame", f, (f,)) for f in frames]
