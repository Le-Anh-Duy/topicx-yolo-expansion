"""Registry: tìm implementation theo tên (CONTRACTS §3).

`@register(kind)` đăng ký class; từ chối class thiếu docstring, name, version hoặc trùng tên.
`discover()` import mọi file trong `components/` để các `@register` chạy (auto-discover).
Recipe (`RecipeConfig`) đăng ký bằng `add_recipe` trong `components/recipes.py`; tầng trên đọc bằng `recipe(name)`.
"""

from __future__ import annotations

import importlib
import pkgutil

from pydantic import BaseModel

from .component import Component, Feature
from .types import RecipeConfig


class Registry:
    def __init__(self) -> None:
        self._by_kind: dict[str, dict[str, type[Component]]] = {}
        self._recipes: dict[str, RecipeConfig] = {}

    def register(self, kind: str):
        def decorator(cls: type[Component]) -> type[Component]:
            if not issubclass(cls, Component):
                raise TypeError(f"{cls.__name__}: phải kế thừa Component")
            if not (cls.__dict__.get("__doc__") or "").strip():
                raise TypeError(f"{cls.__name__}: thiếu docstring (mô tả làm gì + tính thế nào)")
            for attr in ("name", "version"):
                if not isinstance(cls.__dict__.get(attr), str):
                    raise TypeError(f"{cls.__name__}: thiếu `{attr}`")
            if getattr(cls, "kind", kind) != kind:
                raise TypeError(f"{cls.__name__}: kind {cls.kind!r} khác {kind!r}")
            names = self._by_kind.setdefault(kind, {})
            if cls.name in names:
                raise ValueError(f"trùng tên {kind} {cls.name!r}")
            cls.kind = kind
            names[cls.name] = cls
            return cls

        return decorator

    def get(self, kind: str, name: str) -> type[Component]:
        try:
            return self._by_kind[kind][name]
        except KeyError:
            raise KeyError(f"không có {kind} {name!r}; đang có: {self.names(kind)}") from None

    def build(self, kind: str, config: dict | BaseModel) -> Component:
        """Tạo component từ config `{"type": name, "params": {...}}` (session spec / recipe, CONTRACTS §7)."""
        if isinstance(config, BaseModel):
            config = config.model_dump()
        return self.get(kind, config["type"])(config.get("params") or {})

    def names(self, kind: str) -> list[str]:
        return sorted(self._by_kind.get(kind, {}))

    def describe(self, kind: str, name: str) -> dict:
        """Cho API / agent: docstring + JSON schema của Params (form ở UI)."""
        cls = self.get(kind, name)
        return {"kind": kind, "name": name, "version": cls.version,
                "doc": cls.__doc__.strip(), "params_schema": cls.Params.model_json_schema()}

    def add_recipe(self, recipe: RecipeConfig) -> None:
        if recipe.name in self._recipes:
            raise ValueError(f"trùng tên recipe {recipe.name!r}")
        self._recipes[recipe.name] = recipe

    def recipe(self, name: str) -> RecipeConfig:
        try:
            return self._recipes[name]
        except KeyError:
            raise KeyError(f"không có recipe {name!r}; đang có: {self.recipe_names()}") from None

    def recipe_names(self) -> list[str]:
        return sorted(self._recipes)

    def feature_for(self, field: str) -> type[Feature]:
        """Feature nào `provides` field này. Đúng một feature cho mỗi tên field."""
        found = [c for c in self._by_kind.get("feature", {}).values() if c.provides == field]
        if len(found) != 1:
            raise KeyError(f"field {field!r}: cần đúng 1 feature provides, thấy {[c.name for c in found]}")
        return found[0]


REGISTRY = Registry()
register = REGISTRY.register

_discovered = False


def discover() -> None:
    """Import mọi module trong `components/` (một lần) để các implementation tự đăng ký."""
    global _discovered
    if _discovered:
        return
    package = importlib.import_module(__name__.rsplit(".core", 1)[0] + ".components")
    for mod in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        importlib.import_module(mod.name)
    _discovered = True
