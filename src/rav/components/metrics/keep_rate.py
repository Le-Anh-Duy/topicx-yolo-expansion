"""`keep_rate_by_recipe`: keep rate theo recipe (CONTRACTS §10; #50)."""

from __future__ import annotations

from collections import Counter

from ...core.registry import register
from .base import Metric


@register("metric")
class KeepRateByRecipe(Metric):
    """Keep rate = kept / (kept + dropped) trên **mọi quyết định của phiên** (không phụ thuộc `unit_ids`: chấm tập kept thì luôn 100%).
    Mỗi unit đang kept / dropped (`ctx.state`) tính theo quyết định keep/drop cuối cùng của nó, quy về recipe của proposal `proposal_n`;
    quyết định ngoài proposal không tính. `recipe/<tên>` = từng recipe (chưa có quyết định thì không có khoá). Không đọc field → luôn độc lập."""

    name = "keep_rate_by_recipe"
    version = "1"

    def evaluate(self, unit_ids, ctx) -> dict[str, float]:
        recipe_of = {p.n: p.recipe for p in ctx.proposals}
        last = {}
        for d in ctx.state.history:
            if d.action in ("keep", "drop"):
                last[d.unit_id] = d
        kept, total = Counter(), Counter()
        for uid in ctx.state.kept | ctx.state.dropped:
            d = last.get(uid)
            if d is None or d.proposal_n not in recipe_of:
                continue
            total[recipe_of[d.proposal_n]] += 1
            kept[recipe_of[d.proposal_n]] += uid in ctx.state.kept
        overall = sum(kept.values()) / sum(total.values()) if total else 0.0
        return {self.name: overall, **{f"recipe/{r}": kept[r] / total[r] for r in sorted(total)}}
