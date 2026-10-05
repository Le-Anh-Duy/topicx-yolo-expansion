"""L3 — P2: reference, objective novelty, combiner, selector, recipe qua registry, get_score_candidates + propose.

Pool 2 video, `emb.dinov2_s` do execution giả trả về (không chạy DINOv2).
"""

import numpy as np
import pytest

from src.rav.core.context import Context
from src.rav.core.registry import REGISTRY, discover
from src.rav.core.types import FieldTable, Frame, RecipeConfig, ReviewState, Scores, Unit, unit_id
from src.rav.pipeline.propose import config_hash, get_score_candidates, propose

discover()


def make_pool(sizes=(("a", 32), ("b", 20)), fps=4, seed=0):
    """Unit mỗi 1/fps giây; embedding = vector cảnh (đổi mỗi 2 s) + nhiễu nhỏ → frame liền nhau gần trùng."""
    rng = np.random.default_rng(seed)
    units, emb = [], {}
    for v, n in sizes:
        scenes = rng.normal(size=(n // (2 * fps) + 1, 16))
        for i in range(n):
            t = i / fps
            u = Unit(unit_id(v, t, t), v, t, t, "frame", Frame(v, i, t), (Frame(v, i, t),))
            units.append(u)
            emb[u.id] = scenes[int(t // 2)] + 0.05 * rng.normal(size=16)
    return units, emb


class FakeExecution:
    """Trả embedding dựng sẵn cho `emb.dinov2_s`, căn theo units được hỏi."""

    def __init__(self, emb):
        self.emb = emb

    def run(self, feature, units, ctx):
        assert feature.provides == "emb.dinov2_s"
        return FieldTable(feature.key, [u.id for u in units],
                          np.stack([self.emb[u.id] for u in units]).astype(np.float32), "vector")


def make_ctx(kept=(), dropped=(), **kw):
    units, emb = make_pool(**kw)
    return Context(units, execution=FakeExecution(emb), state=ReviewState(set(kept), set(dropped)))


def recipe(name):
    return REGISTRY.recipe(name)


def test_recipes_registered_and_components_resolve():
    assert REGISTRY.recipe_names() == ["diversity", "random", "uniform"]
    assert [o.type for o in recipe("diversity").objectives] == ["novelty"]
    assert recipe("random").objectives == [] and recipe("uniform").objectives == []
    with pytest.raises(KeyError):
        REGISTRY.recipe("nope")


@pytest.mark.parametrize("name", ["diversity", "random", "uniform"])
@pytest.mark.parametrize("batch", [1, 8, 100])
def test_propose_invariants_for_every_recipe(name, batch):
    ids = [u.id for u in make_pool()[0]]
    kept, dropped = set(ids[:3]), {ids[40]}
    ctx = make_ctx(kept, dropped)
    p = propose(recipe(name), batch, seed=7, ctx=ctx)
    got = [pk.unit.id for pk in p.picks]
    assert len(got) == min(batch, len(ids) - 4)
    assert len(set(got)) == len(got) and not set(got) & (kept | dropped)
    assert [pk.rank for pk in p.picks] == list(range(1, len(got) + 1))
    assert (p.session_id, p.n, p.round, p.recipe, p.config_hash) == (None, 1, 1, name, config_hash(recipe(name)))
    again = propose(recipe(name), batch, seed=7, ctx=make_ctx(kept, dropped))
    assert [pk.unit.id for pk in again.picks] == got and again.relaxed == p.relaxed


def test_random_depends_on_seed_only():
    a = [p.unit.id for p in propose(recipe("random"), 10, 1, make_ctx()).picks]
    ctx = make_ctx()
    ctx.rng.random(5)                                             # rng cũ bị dùng dở không ảnh hưởng: propose seed lại
    assert [p.unit.id for p in propose(recipe("random"), 10, 1, ctx).picks] == a
    assert [p.unit.id for p in propose(recipe("random"), 10, 2, make_ctx()).picks] != a


def test_breakdown_is_per_objective_score():
    ctx = make_ctx()
    cands, scores, combined, _ = get_score_candidates(recipe("diversity"), ctx)
    by_id = dict(zip([u.id for u in cands], scores[0].values))
    p = propose(recipe("diversity"), 5, 0, ctx)
    for pk in p.picks:
        assert pk.breakdown == {"novelty": pytest.approx(by_id[pk.unit.id])}
        assert pk.combined == pytest.approx(by_id[pk.unit.id])
    assert all(pk.breakdown == {} for pk in propose(recipe("uniform"), 5, 0, make_ctx()).picks)


def test_novelty_range_empty_reference_and_duplicate_of_kept():
    units, _ = make_pool()
    ctx = make_ctx()
    cands, (nov,), _, _ = get_score_candidates(recipe("diversity"), ctx)   # reference rỗng: so với tâm pool
    assert nov.name == "novelty" and nov.unit_ids == [u.id for u in cands]
    assert nov.values.min() == 0 and nov.values.max() == 1

    ctx = make_ctx(kept={units[0].id})
    _, (nov,), _, _ = get_score_candidates(recipe("diversity"), ctx)
    assert [u.id for u in ctx.reference()] == [units[0].id]              # session_kept qua set_reference
    assert ((nov.values >= 0) & (nov.values <= 1)).all()
    same_scene = nov.values[:7]                                           # unit 1..7: cùng cảnh với unit đã keep
    assert same_scene.max() < 0.05 < nov.values[7:].min()


def test_novelty_all_equal_gives_one():
    units, emb = make_pool()
    emb = {k: np.ones(16) for k in emb}
    ctx = Context(units, execution=FakeExecution(emb))
    _, (nov,), _, _ = get_score_candidates(recipe("diversity"), ctx)
    assert np.all(nov.values == 1.0)


def test_combiner_product_and_empty():
    comb = REGISTRY.build("combiner", {"type": "product_strength"})
    combined, mask = comb.combine([], ["x", "y", "z"])
    assert combined.tolist() == [1, 1, 1] and mask.tolist() == [True] * 3
    s1 = Scores("a", ["x", "y"], np.array([0.5, 1.0]))
    s2 = Scores("b", ["x", "y"], np.array([0.25, 0.5]), np.array([True, False]))
    combined, mask = comb.combine([(s1, 1.0), (s2, 2.0)], ["x", "y"])
    assert combined == pytest.approx([0.5 * 0.0625, 0.25]) and mask.tolist() == [True, False]
    with pytest.raises(ValueError):
        comb.combine([(s1, 1.0)], ["y", "x"])


def test_greedy_nms_relaxes_when_short():
    ctx = make_ctx()
    p = propose(recipe("diversity"), 5, 0, ctx)                    # 8 cảnh → đủ 5 không cần nới
    assert not p.relaxed
    ts = [(pk.unit.video_id, pk.unit.t0) for pk in p.picks]
    assert all(abs(t1 - t2) >= 1.0 for i, (v1, t1) in enumerate(ts) for v2, t2 in ts[i + 1:] if v1 == v2)
    p = propose(recipe("diversity"), 30, 0, make_ctx())            # 30 > số cảnh → phải nới
    assert p.relaxed and len(p.picks) == 30


def test_uniform_spacing_and_allocation():
    ctx = make_ctx(sizes=(("a", 30), ("b", 10)))
    p = propose(recipe("uniform"), 4, 0, ctx)
    assert [(pk.unit.video_id, pk.unit.anchor.idx) for pk in p.picks] == [("a", 5), ("a", 15), ("a", 25), ("b", 5)]
    equal = RecipeConfig.model_validate({**recipe("uniform").model_dump(),
                                         "selector": {"type": "uniform", "params": {"allocation": "equal"}}})
    p = propose(equal, 4, 0, make_ctx(sizes=(("a", 30), ("b", 1))))
    assert [pk.unit.video_id for pk in p.picks] == ["a", "a", "a", "b"]   # b thiếu → phần dư sang a


def test_config_hash_resolves_defaults():
    r = recipe("diversity").model_dump()
    short = {**r, "selector": {"type": "greedy_nms"}, "objectives": [{"type": "novelty"}]}
    assert config_hash(RecipeConfig.model_validate(short)) == config_hash(recipe("diversity"))
    changed = {**short, "selector": {"type": "greedy_nms", "params": {"nms_s": 2.0}}}
    assert config_hash(RecipeConfig.model_validate(changed)) != config_hash(recipe("diversity"))
    strong = {**short, "objectives": [{"type": "novelty", "strength": 2.0}]}
    assert config_hash(RecipeConfig.model_validate(strong)) != config_hash(recipe("diversity"))
    with pytest.raises(Exception):
        config_hash(RecipeConfig.model_validate({**short, "selector": {"type": "greedy_nms", "params": {"nms": 1}}}))


def test_proposal_n_follows_history():
    ctx = make_ctx()
    ctx.proposals = [propose(recipe("random"), 3, 0, ctx)]
    assert propose(recipe("random"), 3, 0, ctx).n == 2
