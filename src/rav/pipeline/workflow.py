"""Chế độ tự động / mô phỏng — `run_workflow(router, judge, feedback, goal, seed, ctx) -> History` (CONTRACTS §12.4).

Vòng: router.next → propose (đọc ctx.state) → judge → feedback.apply → lặp tới khi router trả Stop. Thuần: không DB, không UI;
mọi component lấy qua registry (truyền vào đã build). Proposal được gắn vào `ctx.proposals` để `n`, view và metric đọc như trong phiên thật.
`seed` của vòng n = seed * 100_000 + n → cùng input + seed → cùng History (trừ thời điểm trong Decision).
"""

from __future__ import annotations

import dataclasses

from ..core.context import Context
from ..core.registry import REGISTRY, discover
from ..core.types import Decision, History, JudgeRequest, Proposal, Stop
from .propose import propose


def run_workflow(router, judge, feedback, goal: str, seed: int, ctx: Context, session_id: str | None = None) -> tuple[History, Stop]:
    discover()
    history: History = []
    while not isinstance(action := router.next(history, ctx), Stop):
        recipe = action.config or REGISTRY.recipe(action.recipe)
        proposal = propose(recipe, action.batch_size, seed * 100_000 + len(ctx.proposals) + 1, ctx)
        proposal.session_id = session_id
        ctx.proposals.append(proposal)
        decisions = judge.judge(JudgeRequest(proposal, goal), ctx)
        ctx.state = feedback.apply(decisions, ctx.state)
        history += [proposal, *decisions]
    return history, action


def history_records(history: History) -> list[dict]:
    """History → dict JSON được (một dòng mỗi Proposal / Decision), để lưu `history.jsonl` và đọc lại ở nơi khác."""
    out = []
    for h in history:
        if isinstance(h, Proposal):
            out.append({"type": "proposal", "session_id": h.session_id, "n": h.n, "round": h.round, "recipe": h.recipe,
                        "config_hash": h.config_hash, "relaxed": h.relaxed, "fields_used": [str(k) for k in h.fields_used],
                        "picks": [{"unit_id": p.unit.id, "rank": p.rank, "combined": p.combined, "breakdown": p.breakdown} for p in h.picks]})
        elif isinstance(h, Decision):
            d = dataclasses.asdict(h)
            out.append({"type": "decision", **{k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in d.items()}})
    return out
