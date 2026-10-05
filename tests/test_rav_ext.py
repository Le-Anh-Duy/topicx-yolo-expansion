"""Component mới theo contract P-026 (src/rav) và cầu nối block 1 → 2 → 3. Dữ liệu giả, CPU, không tải model."""

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.rav.core.bulk_execution import BulkExecution
from src.rav.core.context import Context
from src.rav.core.execution import Execution
from src.rav.core.registry import REGISTRY, Registry, discover
from src.rav.core.component import Feature
from src.rav.core.types import FieldTable, Frame, ReviewState, Stop, Unit, unit_id
from src.rav.pipeline.workflow import history_records, run_workflow
from src.rav.store.field_store import FieldStore
from src.rav.store.packed_field_store import PackedFieldStore
from topicx import common

discover()


def _units(n, prefix="x"):
    return [Unit(unit_id(f"{prefix}{i}", 0.0, 0.0), f"{prefix}{i}", 0.0, 0.0, "frame", Frame(f"{prefix}{i}", 0, 0.0),
                 (Frame(f"{prefix}{i}", 0, 0.0),)) for i in range(n)]


class FakeExecution:
    def __init__(self, fields):
        self.fields = fields   # {field: {unit_id: vector}}

    def run(self, feature, units, ctx):
        vals = self.fields[feature.provides]
        return FieldTable(feature.key, [u.id for u in units], np.stack([vals[u.id] for u in units]).astype(np.float32), "vector")


def _toy_ctx(kept=()):
    # A, B, C, D trực giao; A', B' gần trùng A, B. Thứ tự x0..x5 = A, B, A', B', C, D
    e = np.eye(6)
    emb = [e[0], e[1], e[0] + 0.01 * e[4], e[1] + 0.01 * e[5], e[2], e[3]]
    units = _units(6)
    ctx = Context(units, execution=FakeExecution({"emb.clip_b32": {u.id: v for u, v in zip(units, emb)}}),
                  state=ReviewState(set(kept), set()))
    ctx.set_reference(REGISTRY.build("reference", {"type": "session_kept"}))
    return ctx, units


@pytest.mark.parametrize("history,expect", [(True, ["x4", "x5"]), (False, ["x2", "x3"])])
def test_farthest_history_avoids_kept_only_when_history(history, expect):
    ctx, units = _toy_ctx(kept={unit_id("x0", 0, 0), unit_id("x1", 0, 0)})   # lịch sử = A, B
    cand = units[2:]
    sel = REGISTRY.build("selector", {"type": "farthest_history", "params": {"history": history}})
    got = sel.pick(cand, np.array([0.9, 0.8, 0.7, 0.6]), np.ones(4, bool), 2, ctx)
    assert [cand[i].video_id for i in got.indices] == expect


def test_farthest_history_first_pick_is_most_relevant_and_respects_mask():
    ctx, units = _toy_ctx()
    sel = REGISTRY.build("selector", {"type": "farthest_history"})
    mask = np.array([False, True, True, True, True, True])
    got = sel.pick(units, np.array([0.99, 0.9, 0.8, 0.7, 0.6, 0.5]), mask, 3, ctx)
    assert got.indices[0] == 1 and 0 not in got.indices and len(set(got.indices)) == 3


def test_text_match_gate_and_scores(monkeypatch):
    from src.rav.components.objectives import text_match as tm
    monkeypatch.setattr(tm, "_encode", lambda texts: np.array([[1.0, 0.0]] * len(texts)))
    units = _units(4)
    crops = {u.id: np.array([[c, np.sqrt(1 - c * c)]] * 3) for u, c in zip(units, (0.9, 0.5, 0.7, 0.1))}
    ctx = Context(units, execution=FakeExecution({"emb.clip_b32x3": crops}))
    s = REGISTRY.build("objective", {"type": "text_match", "params": {"texts": ["bus"], "top_k": 2, "tau": 0.6}}).score(units, ctx)
    assert s.mask.tolist() == [True, False, True, False]
    assert s.values[0] == 1.0 and s.values[3] == 0.0 and 0 < s.values[2] < 1


def test_workflow_fixed_budget_accept_all_is_deterministic():
    units = _units(10)

    def run(seed):
        ctx = Context(units, execution=FakeExecution({}))
        router = REGISTRY.build("router", {"type": "fixed_budget", "params": {"recipe": "random", "batch_size": 3, "target_k": 7}})
        return run_workflow(router, REGISTRY.build("judge", {"type": "accept_all"}), REGISTRY.build("feedback", {"type": "keep_drop"}),
                            "goal", seed, ctx), ctx

    (history, stop), ctx = run(1)
    props = [h for h in history if hasattr(h, "picks")]
    assert [len(p.picks) for p in props] == [3, 3, 1] and len(ctx.state.kept) == 7 and isinstance(stop, Stop)
    assert all(d.actor == "algo:accept_all@1" and d.reason for d in history if hasattr(d, "actor"))
    assert [r["type"] for r in history_records(history)].count("decision") == 7
    (again, _), _ = run(1)
    assert [p.unit.id for h in again if hasattr(h, "picks") for p in h.picks] == [p.unit.id for h in props for p in h.picks]


def test_fixed_budget_stops_when_pool_exhausted():
    ctx = Context(_units(4), execution=FakeExecution({}))
    router = REGISTRY.build("router", {"type": "fixed_budget", "params": {"recipe": "random", "batch_size": 3, "target_k": 10}})
    history, stop = run_workflow(router, REGISTRY.build("judge", {"type": "accept_all"}),
                                 REGISTRY.build("feedback", {"type": "keep_drop"}), "", 0, ctx)
    assert len(ctx.state.kept) == 4 and stop.reason == "hết ứng viên"


def test_packed_store_and_bulk_execution_match_reference(tmp_path):
    reg = Registry()
    calls = []

    @reg.register("feature")
    class Fake(Feature):
        """Vector giả theo tên video (để test cache)."""
        name, version, provides, per_sample = "fake", "1", "emb.fake", True

        def compute(self, units, ctx):
            calls.append(len(units))
            return FieldTable(self.key, [u.id for u in units], np.array([[float(u.video_id[1:]), 1.0] for u in units]), "vector")

    units = _units(50)
    ref = Context(units, execution=Execution(FieldStore(tmp_path / "a"), "spec"), registry=reg).field("emb.fake")
    store = PackedFieldStore(tmp_path / "b")
    got = Context(units, execution=BulkExecution(store, "spec"), registry=reg).field("emb.fake")
    store.flush()
    assert got.unit_ids == ref.unit_ids and np.array_equal(got.values, ref.values)
    again = Context(units[::-1], execution=BulkExecution(PackedFieldStore(tmp_path / "b"), "spec"), registry=reg).field("emb.fake")
    assert calls == [50, 50] and again.unit_ids == [u.id for u in units[::-1]]   # lần 3 đọc cache, không tính
    assert len(list((tmp_path / "b").rglob("*.npz"))) == 1


@pytest.mark.parametrize("smoke", [False, True])
def test_proposal_config_recipes_are_valid_p026_recipes(smoke):
    from src.rav.core.types import RecipeConfig
    from src.rav.pipeline.propose import config_hash
    from topicx import proposals as PR
    pcfg = PR.load_proposal_config({"smoke": smoke})
    assert all(k % pcfg["batch_size"] == 0 for k in pcfg["ks"])
    for name, r in pcfg["recipes"].items():
        r = PR.resolve_recipe(r, ["a bus"], 0.25)
        recipe = RecipeConfig.model_validate(r) if isinstance(r, dict) else REGISTRY.recipe(r)
        assert config_hash(recipe)   # validate Params của từng component theo registry
        assert "$" not in str(r), name


# ---------- cầu nối block 1 → 2 → 3 trên BDD giả ----------

def _fake_bdd(tmp_path, monkeypatch):
    root = tmp_path / "bdd"
    for s in ("train", "val"):
        (root / "images" / "100k" / s).mkdir(parents=True)
    (root / "labels").mkdir()
    for s in ("train", "val"):
        (root / "labels" / f"det_{s}.json").write_text("[]")
    pool = [f"r{i:03d}-c{i:03d}.jpg" for i in range(12)]
    control = [f"b{i:03d}-c000.jpg" for i in range(6)]
    for name in pool + control:
        Image.new("RGB", (32, 18)).save(root / "images" / "100k" / "train" / name)
    monkeypatch.setattr(common, "WORK", tmp_path / "work")
    monkeypatch.setattr(common, "INPUT", tmp_path / "input")
    cfg = {"art": "art", "bdd_roots": [str(root)], "bdd_paths": {
        "labels_train": "labels/det_train.json", "labels_val": "labels/det_val.json",
        "images_train": "images/100k/train", "images_val": "images/100k/val"}}
    art = tmp_path / "work" / "art"
    common.write_ids(pool, art / "splits" / "pool_ids.txt")
    common.write_ids(control, art / "splits" / "control_ids.txt")
    pd.DataFrame({"image": pool + control, "src": "train"}).to_csv(art / "splits" / "image_src.csv", index=False)
    return cfg, pool, control, art


def test_run_recipe_export_then_import_manifest(tmp_path, monkeypatch):
    from topicx import pipeline as P
    from topicx import proposals as PR
    cfg, pool, control, art = _fake_bdd(tmp_path, monkeypatch)
    dst = PR.run_recipe(cfg, "random", "random", target_k=5, batch_size=2, seed=0)
    ex = pd.read_csv(dst / "export.csv")
    assert list(ex.columns) == ["unit_id", "video_id", "t0", "t1", "frame_idx", "file"] and len(ex) == 5
    assert common.load(dst / "meta.json")["n_rounds"] == 3
    # thêm một export ngoài (vd từ P-026) chỉ có 3 frame cho K = 5 → bù 2 ảnh control
    ext = art / "proposals" / "p026_algo" / "s0_k5"
    ext.mkdir(parents=True)
    pd.DataFrame({"unit_id": [f"{n[:-4]}:0.000-0.000" for n in pool[:3]]}).to_csv(ext / "export.csv", index=False)
    rep = P.import_exports(cfg).set_index("branch")
    assert rep.loc["random", "n"] == 5 and rep.loc["p026_algo", "shortfall"] == 2
    m = common.read_frozen(art / "selections" / "s0_k5_p026_algo.json")
    assert m["ids"] == pool[:3] and m["pad_ids"] == control[:2]


def test_import_rejects_ids_outside_pool_and_reserved_names(tmp_path, monkeypatch):
    from topicx import pipeline as P
    cfg, pool, control, art = _fake_bdd(tmp_path, monkeypatch)
    bad = art / "proposals" / "leaky" / "s0_k2"
    bad.mkdir(parents=True)
    pd.DataFrame({"image": [pool[0], control[0]]}).to_csv(bad / "export.csv", index=False)
    with pytest.raises(AssertionError, match="không thuộc pool"):
        P.import_exports(cfg)
    (bad / "export.csv").unlink()
    res = art / "proposals" / "REPLAY_ONLY" / "s0_k1"
    res.mkdir(parents=True)
    pd.DataFrame({"image": [pool[0]]}).to_csv(res / "export.csv", index=False)
    with pytest.raises(AssertionError, match="dành riêng"):
        P.import_exports(cfg)
