"""L1 — core: registry, Component + Params, Context với feature giả (không cần video thật)."""

import numpy as np
import pytest
from pydantic import ValidationError

from src.rav.core.component import Component, Feature
from src.rav.core.context import Context
from src.rav.core.registry import Registry, discover
from src.rav.core.types import FieldTable, Unit, unit_id


def make_units(video_id: str, n: int) -> list[Unit]:
    return [Unit(unit_id(video_id, t, t), video_id, t, t, "frame", None, ()) for t in range(n)]


class DirectExecution:
    """Execution giả: gọi compute thẳng, ghi lại tập unit được đưa vào. Bản thật (cache) ở L2."""

    def __init__(self):
        self.calls: list[tuple[str, list[str]]] = []

    def run(self, feature, units, ctx):
        units = units if feature.per_sample else ctx.units   # per_sample=False: tính trên cả pool (CONTRACTS §3)
        self.calls.append((feature.name, [u.id for u in units]))
        return feature.compute(units, ctx)


def fake_registry() -> Registry:
    reg = Registry()

    @reg.register("feature")
    class Index(Feature):
        """Số thứ tự t của unit."""
        name, version, provides, per_sample = "index", "1", "fake.t", True

        def compute(self, units, ctx):
            return FieldTable(self.key, [u.id for u in units], np.array([u.t0 for u in units]), "scalar")

    @reg.register("feature")
    class Share(Feature):
        """t chia tổng t của cả tập — phụ thuộc cả tập nên per_sample=False."""
        name, version, provides, requires = "share", "1", "fake.share", ("fake.t",)

        def compute(self, units, ctx):
            t = ctx.field("fake.t", units).values
            return FieldTable(self.key, [u.id for u in units], t / t.sum(), "scalar")

    return reg


def test_unit_id_format():
    assert unit_id("vid", 1.5, 1.5) == "vid:1.500-1.500"


def test_registry_rejects_missing_docstring_name_version_and_duplicates():
    reg = Registry()
    with pytest.raises(TypeError, match="docstring"):
        @reg.register("objective")
        class NoDoc(Component):
            name, version = "x", "1"
    with pytest.raises(TypeError, match="version"):
        @reg.register("objective")
        class NoVersion(Component):
            """có doc"""
            name = "x"

    @reg.register("objective")
    class Ok(Component):
        """có doc"""
        name, version = "ok", "1"
    with pytest.raises(ValueError, match="trùng"):
        @reg.register("objective")
        class Again(Component):
            """có doc"""
            name, version = "ok", "2"
    with pytest.raises(TypeError, match="kind"):
        reg.register("objective")(type("F", (Feature,), {"__doc__": "d", "name": "f", "version": "1"}))

    assert reg.get("objective", "ok") is Ok
    with pytest.raises(KeyError, match="đang có"):
        reg.get("objective", "nope")
    assert reg.describe("objective", "ok")["doc"] == "có doc"


def test_params_validate_and_hash():
    class Thing(Component):
        """d"""
        name, version = "thing", "1"

        class Params(Component.Params):
            alpha: float = 0.5

    assert Thing().params.alpha == 0.5
    assert Thing({"alpha": 0.5}).params_hash == Thing().params_hash != Thing({"alpha": 0.7}).params_hash
    with pytest.raises(ValidationError):
        Thing({"aplha": 0.7})    # gõ sai tên → lỗi, không âm thầm bỏ qua


def test_discover_imports_every_component_module():
    discover()   # mọi file trong components/ import được


def test_context_field_aligns_to_requested_units_and_records_fields_used():
    pool = make_units("a", 3) + make_units("b", 2)
    exe = DirectExecution()
    ctx = Context(pool, execution=exe, registry=fake_registry())

    subset = [pool[4], pool[0]]
    table = ctx.field("fake.t", subset)
    assert table.unit_ids == [pool[4].id, pool[0].id]
    assert table.values.tolist() == [1.0, 0.0]

    share = ctx.field("fake.share", subset)          # per_sample=False: tính trên cả pool rồi cắt
    total = sum(u.t0 for u in pool)
    assert share.values.tolist() == [1.0 / total, 0.0]
    assert ("share", [u.id for u in pool]) in exe.calls
    assert {str(k) for k in ctx.fields_used()} == {"fake.t@1", "fake.share@1"}


def test_context_detects_dependency_cycle():
    reg = Registry()
    for name, needs in (("p", "fake.q"), ("q", "fake.p")):
        def compute(self, units, ctx, needs=needs):
            return ctx.field(needs, units)
        reg.register("feature")(type(name, (Feature,), {
            "__doc__": "vòng", "name": name, "version": "1", "provides": f"fake.{name}",
            "requires": (needs,), "compute": compute}))
    ctx = Context(make_units("a", 2), execution=DirectExecution(), registry=reg)
    with pytest.raises(RuntimeError, match="vòng"):
        ctx.field("fake.p")


def test_per_sample_feature_cannot_require_pool_level_field():
    reg = fake_registry()

    @reg.register("feature")
    class Bad(Feature):
        """per_sample=True nhưng dựa vào field tính trên cả pool → cache theo video sẽ sai."""
        name, version, provides, per_sample, requires = "bad", "1", "fake.bad", True, ("fake.share",)

    ctx = Context(make_units("a", 2), execution=DirectExecution(), registry=reg)
    with pytest.raises(TypeError, match="per_sample"):
        ctx.field("fake.bad")


def test_context_rng_reproducible_and_missing_reference_is_explicit():
    units = make_units("a", 2)
    a = Context(units, execution=DirectExecution(), seed=7).rng.random(3)
    b = Context(units, execution=DirectExecution(), seed=7).rng.random(3)
    assert np.array_equal(a, b)
    with pytest.raises(RuntimeError, match="reference"):
        Context(units, execution=DirectExecution()).reference()
