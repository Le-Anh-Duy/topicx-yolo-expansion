"""`session_kept`: các unit đã keep trong phiên (đọc ctx.state)."""

from __future__ import annotations

from ...core.registry import register
from ...core.types import Unit
from .base import Reference


@register("reference")
class SessionKept(Reference):
    """Các unit của pool đã keep trong phiên (`ctx.state.kept`), theo thứ tự pool. Vòng đầu rỗng."""

    name = "session_kept"
    version = "1"

    def select(self, ctx) -> list[Unit]:
        return [u for u in ctx.units if u.id in ctx.state.kept]
