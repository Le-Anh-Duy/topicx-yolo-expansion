"""`pyav`: lấy mẫu `fps` frame/giây, decode pixel theo yêu cầu bằng PyAV (ffmpeg); không ghi file ảnh (quyết định #37)."""

from __future__ import annotations

import math
import threading
from typing import ClassVar

import av
import numpy as np
from pydantic import Field

from ...core.registry import register
from ...core.types import Frame, VideoRef
from .base import Decoder

SEEK_GAP_S = 2.0   # đọc tiếp khi frame cần nằm trước mặt ≤ 2 s; xa hơn hoặc lùi lại thì seek


class _Handle:
    """File video đang mở + vị trí decode hiện tại."""

    def __init__(self, uri: str):
        self.container = av.open(uri)
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = "AUTO"
        self.start = float(self.stream.start_time * self.stream.time_base) if self.stream.start_time else 0.0
        self.frames = self.container.decode(self.stream)
        self.last: tuple[int, np.ndarray] | None = None
        self.lock = threading.Lock()   # route đồng bộ chạy song song trong threadpool (#49)

    def seek(self, t: float) -> None:
        self.container.seek(int((t + self.start) / self.stream.time_base), stream=self.stream, backward=True)
        self.frames = self.container.decode(self.stream)
        self.last = None


@register("decoder")
class PyAV(Decoder):
    """Frame lấy mẫu thứ k ứng với frame gốc idx = round(k × fps_gốc / fps), t = idx / fps_gốc, tới hết video.
    `read` giữ file mở: frame cần nằm ngay phía trước thì decode tiếp, còn lại seek về keyframe trước đó rồi decode tới idx."""

    name = "pyav"
    version = "1"

    class Params(Decoder.Params):
        fps: float = Field(4.0, gt=0)

    # local-only: một handle mỗi video dùng chung trong tiến trình, request cùng video chờ nhau (LOCAL_SINGLE_USER_LIMITS #13)
    _handles: ClassVar[dict[str, _Handle]] = {}
    _open_lock: ClassVar[threading.Lock] = threading.Lock()

    def frames(self, video: VideoRef) -> list[Frame]:
        n = int(round(video.duration_s * video.fps))
        step = max(video.fps / self.params.fps, 1.0)
        idxs = sorted({round(k * step) for k in range(math.ceil(n / step))})
        return [Frame(video.id, i, i / video.fps) for i in idxs if i < n]

    def read(self, video: VideoRef, frame: Frame) -> np.ndarray:
        with self._open_lock:
            h = self._handles.get(video.uri) or self._handles.setdefault(video.uri, _Handle(video.uri))
        with h.lock:
            return self._read(h, video, frame)

    def _read(self, h: _Handle, video: VideoRef, frame: Frame) -> np.ndarray:
        target = frame.idx
        if h.last is not None and h.last[0] == target:
            return h.last[1]
        if h.last is None or target < h.last[0] or target - h.last[0] > SEEK_GAP_S * video.fps:
            h.seek(target / video.fps)
        for f in h.frames:
            idx = round((float(f.time) - h.start) * video.fps)
            if idx >= target:
                h.last = (idx, f.to_ndarray(format="rgb24"))
                return h.last[1]
        h.last = None
        raise IndexError(f"{video.id}: không có frame {target}")
