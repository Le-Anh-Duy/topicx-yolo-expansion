"""Base class của kind `exporter` (chữ ký: CONTRACTS §5 dòng 12, #48)."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from ...core.component import Component
from ...core.types import Unit


class Exporter(Component):
    """Ghi các unit đã keep vào thư mục `dest` (đã tồn tại, rỗng); ảnh lấy qua `ctx.pixels`. Trả file chính (vd CSV)."""

    kind: ClassVar[str] = "exporter"

    def export(self, units: list[Unit], dest: Path, ctx) -> Path:
        raise NotImplementedError
