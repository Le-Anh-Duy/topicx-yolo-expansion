"""`accept_all`: judge mô phỏng — keep mọi frame được đề xuất (CONTRACTS §12.4, chế độ "Mô phỏng")."""

from __future__ import annotations

from datetime import datetime, timezone

from ...core.registry import register
from ...core.types import Decision, JudgeRequest
from .base import Judge


@register("judge")
class AcceptAll(Judge):
    """Trả `keep` cho mọi pick của proposal, actor `algo:accept_all@1`, reason ghi rõ là mô phỏng. Không xem ảnh, field hay nhãn:
    dùng để đánh giá recipe không cần người (mục tiêu (a) §12) — tập keep = đúng những gì proposer chọn."""

    name = "accept_all"
    version = "1"

    def judge(self, req: JudgeRequest, ctx) -> list[Decision]:
        now = datetime.now(timezone.utc)
        actor = f"algo:{self.name}@{self.version}"
        return [Decision(req.proposal.session_id or "simulation", p.unit.id, "keep", actor, req.goal, req.proposal.n, now, now,
                         reason="mô phỏng: chấp nhận mọi frame được đề xuất") for p in req.proposal.picks]
