"""`fixed_budget`: router cố định — một recipe, batch cố định, dừng khi đủ ngân sách (CONTRACTS §12.4, router `fixed`)."""

from __future__ import annotations

from pydantic import Field

from ...core.registry import register
from ...core.types import Action, History, Proposal, Propose, RecipeConfig, Stop
from .base import Router


@register("router")
class FixedBudget(Router):
    """Mỗi vòng `Propose(recipe, config, min(batch_size, target_k − |kept|))`; `Stop` khi |kept| ≥ target_k, khi proposal trước rỗng
    (hết ứng viên — không ép mẫu kém để đủ K) hoặc khi đã `max_rounds` proposal. `config` (tuỳ chọn) = RecipeConfig đầy đủ thay cho
    recipe đăng ký sẵn (như command propose, #45), vd recipe text_match với câu topic riêng."""

    name = "fixed_budget"
    version = "1"

    class Params(Router.Params):
        recipe: str
        config: dict | None = None
        batch_size: int = Field(16, ge=1)
        target_k: int = Field(200, ge=1)
        max_rounds: int = Field(10_000, ge=1)

    def next(self, history: History, ctx) -> Action:
        kept = len(ctx.state.kept)
        props = [h for h in history if isinstance(h, Proposal)]
        if kept >= self.params.target_k:
            return Stop(f"đủ target_k={self.params.target_k}")
        if props and not props[-1].picks:
            return Stop("hết ứng viên")
        if len(props) >= self.params.max_rounds:
            return Stop(f"đủ max_rounds={self.params.max_rounds}")
        config = RecipeConfig.model_validate(self.params.config) if self.params.config else None
        return Propose(self.params.recipe, config, min(self.params.batch_size, self.params.target_k - kept))
