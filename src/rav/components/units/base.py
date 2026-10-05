"""Base của kind `unit` (CONTRACTS §2 Unit, §5 dòng 3)."""

from __future__ import annotations

from typing import ClassVar

from ...core.component import Component
from ...core.types import Frame, Unit, VideoRef


class UnitBuilder(Component):
    """Gom frame của một video thành các Unit (thứ được chấm / chọn / duyệt / export)."""

    kind: ClassVar[str] = "unit"

    def build(self, video: VideoRef, frames: list[Frame]) -> list[Unit]:
        raise NotImplementedError
