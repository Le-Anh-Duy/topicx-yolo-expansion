"""Base của kind `source` (CONTRACTS §5 dòng 1, §13)."""

from __future__ import annotations

from collections.abc import Iterable
from typing import ClassVar

from ...core.component import Component
from ...core.types import VideoRef


class Source(Component):
    """Liệt kê video trong catalog. Không decode, không tính feature."""

    kind: ClassVar[str] = "source"

    def iter_videos(self) -> Iterable[VideoRef]:
        raise NotImplementedError

    def telemetry(self, video: VideoRef) -> list[dict] | None:
        """Telemetry đồng bộ theo giây video (`{t, speed_kmh?, brake?, steer_deg?, lat?, lon?}`, CONTRACTS §8.0); None = nguồn không có."""
        return None
