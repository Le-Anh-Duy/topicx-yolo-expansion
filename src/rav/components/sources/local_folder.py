"""`local_folder`: video trong một thư mục (mặc định `data/videos/`; upload cũng lưu vào đây); telemetry từ file kèm `<video>.telemetry.json`."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

import av

from ...core.registry import register
from ...core.types import VideoRef
from .base import Source

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


@register("source")
class LocalFolder(Source):
    """Mọi file video (mp4, mov, mkv, avi, webm) trong `root`, sắp theo tên. `id` = 16 ký tự hex đầu của sha256 nội dung file
    (tải lại / đổi tên cùng file vẫn cùng id); `fps`, thời lượng, kích thước khung đọc từ metadata bằng PyAV; file không mở được thì bỏ qua.
    Telemetry (tuỳ chọn): file JSON cạnh video tên `<tên video>.telemetry.json` = danh sách `{t, speed_kmh?, brake?, steer_deg?, lat?, lon?}`."""

    name = "local_folder"
    version = "1"

    class Params(Source.Params):
        root: str = "data/videos"

    def iter_videos(self) -> Iterator[VideoRef]:
        root = Path(self.params.root)
        if not root.is_dir():
            return
        for path in sorted(root.iterdir()):
            if path.suffix.lower() in VIDEO_EXTS and path.is_file():
                stat = path.stat()
                try:
                    yield _probe(str(path.resolve()), stat.st_size, stat.st_mtime_ns)
                except (av.FFmpegError, IndexError, ValueError, ZeroDivisionError):
                    continue

    def telemetry(self, video: VideoRef) -> list[dict] | None:
        path = Path(video.uri + ".telemetry.json")
        if not path.is_file():
            return None
        return sorted(json.loads(path.read_text(encoding="utf-8")), key=lambda s: s["t"])


@lru_cache(maxsize=256)
def _probe(uri: str, size: int, mtime_ns: int) -> VideoRef:
    """Metadata + id nội dung; cache theo (đường dẫn, kích thước, mtime) để không hash lại file mỗi lần liệt kê."""
    digest = hashlib.sha256()
    with open(uri, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    with av.open(uri) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or stream.guessed_rate)
        width, height = stream.codec_context.width, stream.codec_context.height
        n = stream.frames
        if not n:
            seconds = float(stream.duration * stream.time_base) if stream.duration else container.duration / av.time_base
            n = int(round(seconds * fps))
    if fps <= 0 or n <= 0:
        raise ValueError(f"{uri}: không có frame")
    return VideoRef(digest.hexdigest()[:16], uri, fps, n / fps, {"name": Path(uri).name, "n_frames": n, "width": width, "height": height,
                                                                 "size_bytes": size, "mtime_ns": mtime_ns})
