"""`keep_drop`: replay Decision (keep / drop / clear) thành ReviewState."""

from __future__ import annotations

from ...core.registry import register
from ...core.types import Decision, ReviewState
from .base import Feedback


@register("feedback")
class KeepDrop(Feedback):
    """Áp quyết định theo thứ tự: `keep` / `drop` đưa unit vào tập tương ứng (bỏ khỏi tập kia), `clear` bỏ quyết định trước đó.
    `clear` cho unit chưa có quyết định thì không làm gì và không ghi vào `history`. Unit hiện mà chưa phản hồi vẫn là candidate (§9.3)."""

    name = "keep_drop"
    version = "1"

    def apply(self, decisions: list[Decision], state: ReviewState) -> ReviewState:
        kept, dropped, history = set(state.kept), set(state.dropped), list(state.history)
        for d in decisions:
            if d.action == "clear" and d.unit_id not in kept | dropped:
                continue
            kept.discard(d.unit_id)
            dropped.discard(d.unit_id)
            if d.action == "keep":
                kept.add(d.unit_id)
            elif d.action == "drop":
                dropped.add(d.unit_id)
            history.append(d)
        return ReviewState(kept, dropped, history)
