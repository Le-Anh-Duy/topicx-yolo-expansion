"""P2 — `propose(recipe, batch_size, seed, ctx) -> Proposal` (architecture §4.4, CONTRACTS §6, §7).

candidates = pool − kept − dropped → objectives → combiner (`get_score_candidates`) → selector.
Thuần: không đọc DB, state lấy từ `ctx.state`; `seed` là nguồn ngẫu nhiên duy nhất → cùng input + seed → cùng output.
Component lấy qua registry theo tên trong recipe (không import `components/`).
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from ..core.context import Context
from ..core.registry import REGISTRY, discover
from ..core.types import ComponentConfig, Pick, Proposal, RecipeConfig, Scores, Unit


def get_score_candidates(recipe: RecipeConfig, ctx: Context) -> tuple[list[Unit], list[Scores], np.ndarray, np.ndarray]:
    """candidates, điểm từng objective, combined, mask — chỉ lấy điểm, không chọn."""
    discover()
    ctx.set_reference(REGISTRY.build("reference", recipe.reference))
    done = ctx.state.kept | ctx.state.dropped
    candidates = [u for u in ctx.units if u.id not in done]
    scores = [REGISTRY.build("objective", o).score(candidates, ctx) for o in recipe.objectives]
    combined, mask = REGISTRY.build("combiner", recipe.combiner).combine(
        [(s, o.strength) for s, o in zip(scores, recipe.objectives)], [u.id for u in candidates])
    return candidates, scores, combined, mask


def propose(recipe: RecipeConfig, batch_size: int, seed: int, ctx: Context) -> Proposal:
    ctx.rng = np.random.default_rng(seed)
    candidates, scores, combined, mask = get_score_candidates(recipe, ctx)
    sel = REGISTRY.build("selector", recipe.selector).pick(candidates, combined, mask, batch_size, ctx)
    picks = [Pick(candidates[i], float(combined[i]), {s.name: float(s.values[i]) for s in scores}, rank)
             for rank, i in enumerate(sel.indices, 1)]
    n = len(ctx.proposals) + 1
    return Proposal(None, n, n, recipe.name, config_hash(recipe), picks, sel.relaxed, sorted(ctx.fields_used()))


def config_hash(recipe: RecipeConfig) -> str:
    """sha256 (16 hex) của recipe đã resolve: mỗi component [type, version, params điền mặc định], objective kèm strength."""
    discover()

    def resolve(kind: str, c: ComponentConfig) -> list:
        cls = REGISTRY.get(kind, c.type)
        return [c.type, cls.version, cls.Params.model_validate(c.params).model_dump(mode="json")]

    doc = {"reference": resolve("reference", recipe.reference),
           "objectives": [[*resolve("objective", o), o.strength] for o in recipe.objectives],
           "combiner": resolve("combiner", recipe.combiner),
           "selector": resolve("selector", recipe.selector)}
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()[:16]
