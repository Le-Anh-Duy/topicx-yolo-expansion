"""Base `Component` và `Feature` (CONTRACTS §3).

Mọi implementation là class kế thừa `Component`, có `Params` Pydantic lồng trong class (§9.9)
và docstring mô tả làm gì + tính thế nào (registry từ chối class không có docstring).
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, ClassVar

from pydantic import BaseModel, ConfigDict

from .types import FieldKey, FieldTable, Unit

if TYPE_CHECKING:
    from .context import Context


class Component:
    kind: ClassVar[str]                        # do @register(kind) gán
    name: ClassVar[str]                        # id trong registry, dùng trong config
    version: ClassVar[str]                     # đổi khi đổi logic → vô hiệu cache

    class Params(BaseModel):
        model_config = ConfigDict(extra="forbid")   # gõ sai tên tham số → lỗi ngay, không âm thầm bỏ qua

    requires: ClassVar[tuple[str, ...]] = ()   # tên field cần đọc

    def __init__(self, params: BaseModel | dict | None = None):
        self.params = self.Params.model_validate(params or {})

    @property
    def params_hash(self) -> str:
        return hashlib.sha256(self.params.model_dump_json().encode()).hexdigest()[:16]


class Feature(Component):
    """Base của kind `feature`: tính một field cho một tập unit."""

    kind: ClassVar[str] = "feature"
    provides: ClassVar[str]                    # tên field, vd "emb.dinov2_s"
    per_sample: ClassVar[bool] = False
    # True : giá trị của mỗi mẫu chỉ tính từ chính mẫu đó → execution cache theo video
    # False: giá trị phụ thuộc cả tập mẫu tính cùng → execution tính trên cả pool

    @property
    def key(self) -> FieldKey:
        return FieldKey(self.provides, self.name, self.version, self.params_hash)

    def compute(self, units: list[Unit], ctx: Context) -> FieldTable:
        raise NotImplementedError
