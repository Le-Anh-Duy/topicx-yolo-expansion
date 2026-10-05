"""Base class của kind `view` (chữ ký: CONTRACTS §5 dòng 14, architecture §7; #50)."""

from __future__ import annotations

from typing import ClassVar, get_origin

from pydantic import model_validator

from ...core.component import Component
from ...core.types import Proposal, ViewSpec


class View(Component):
    """Dựng dữ liệu cho một biểu đồ từ ctx (field, `ctx.state`, `ctx.proposals`); tham số trong `self.params`. Không trả ảnh."""

    kind: ClassVar[str] = "view"

    class Params(Component.Params):
        @model_validator(mode="before")
        @classmethod
        def _scalar_to_list(cls, data):
            """Query `?distances=8` (một giá trị) → `[8]` cho trường kiểu list; khoá lặp đã là list sẵn."""
            if isinstance(data, dict):
                for k, f in cls.model_fields.items():
                    if k in data and get_origin(f.annotation) is list and not isinstance(data[k], list):
                        data = {**data, k: [data[k]]}
            return data

    def build(self, ctx) -> ViewSpec:
        raise NotImplementedError


def proposal(ctx, n: int | None) -> Proposal | None:
    """Proposal thứ `n` của phiên (None = mới nhất); phiên chưa có proposal → None."""
    if not ctx.proposals:
        return None
    if n is None:
        return ctx.proposals[-1]
    found = [p for p in ctx.proposals if p.n == n]
    if not found:
        raise KeyError(f"không có proposal {n}")
    return found[0]


def picked_ids(ctx, n: int | None) -> set[str]:
    p = proposal(ctx, n)
    return {pick.unit.id for pick in p.picks} if p else set()
