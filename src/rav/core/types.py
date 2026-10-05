"""Kiểu dữ liệu dùng chung (CONTRACTS §2, §12.3, §13.3).

Chỉ chứa dữ liệu, không có logic. Mọi component, pipeline, service trao đổi bằng các kiểu này.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


@dataclass(frozen=True)
class VideoRef:
    id: str
    uri: str
    fps: float
    duration_s: float
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Frame:
    """Địa chỉ một frame, không chứa pixel (pixel lấy qua `ctx.pixels`)."""

    video_id: str
    idx: int
    t: float


def unit_id(video_id: str, t0: float, t1: float) -> str:
    """Id toàn cục, deterministic: `"{video_id}:{t0:.3f}-{t1:.3f}"` (CONTRACTS §2, §13.1)."""
    return f"{video_id}:{t0:.3f}-{t1:.3f}"


@dataclass(frozen=True)
class Unit:
    """Mẫu — thứ DUY NHẤT được chấm / chọn / duyệt / export. MVP: frame (t0 == t1); sau: span."""

    id: str
    video_id: str
    t0: float
    t1: float
    kind: Literal["frame", "span"]
    anchor: Frame | None
    frames: tuple[Frame, ...]


@dataclass(frozen=True)
class FieldKey:
    """Định danh một kết quả tính toán, cũng là cache key."""

    name: str
    producer: str
    version: str
    params_hash: str

    def __str__(self) -> str:
        return f"{self.name}@{self.version}"


@dataclass
class FieldTable:
    """Kết quả của một Feature trên một tập unit, căn theo `unit_ids`."""

    key: FieldKey
    unit_ids: list[str]
    values: np.ndarray | list
    kind: Literal["vector", "scalar", "structured"]


@dataclass
class Scores:
    """Output của Objective: [0,1], CAO = NÊN CHỌN; `mask` False = loại cứng."""

    name: str
    unit_ids: list[str]
    values: np.ndarray
    mask: np.ndarray | None = None


@dataclass
class Pick:
    unit: Unit
    combined: float
    breakdown: dict[str, float]
    rank: int


@dataclass
class Proposal:
    session_id: str | None        # None khi benchmark
    n: int                        # thứ tự proposal trong phiên, từ 1; Decision.proposal_n trỏ tới đây
    round: int                    # vòng duyệt; MVP: round == n
    recipe: str
    config_hash: str
    picks: list[Pick]
    relaxed: bool = False
    fields_used: list[FieldKey] = field(default_factory=list)   # field recipe đã đọc; metric kiểm độc lập với nó (#44)


@dataclass
class SelectResult:
    """Output của Selector: index vào `units` đưa cho selector, theo thứ hạng."""

    indices: list[int]
    relaxed: bool = False


@dataclass(frozen=True)
class Decision:
    """Một dòng log quyết định (CONTRACTS §12.3). Kiểu duy nhất cho keep / drop / clear."""

    session_id: str
    unit_id: str
    action: Literal["keep", "drop", "clear"]
    actor: str                    # "human:<tên>" | "algo:<name>@<version>" | "ai:<model>"
    goal: str
    proposal_n: int | None
    decided_at: datetime
    shown_at: datetime | None
    reason: str | None = None
    confidence: float | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class ReviewState:
    """Kết quả duyệt của phiên = replay(history). Chỉ service / feedback được ghi."""

    kept: set[str] = field(default_factory=set)
    dropped: set[str] = field(default_factory=set)
    history: list[Decision] = field(default_factory=list)


@dataclass
class ViewSpec:
    """Output của View (CONTRACTS §5 dòng 14): chỉ dữ liệu JSON, FE vẽ theo `type`."""

    type: str
    data: dict


class PoolFilter(BaseModel):
    """Pool của phiên = bộ lọc trên bảng unit toàn cục (CONTRACTS §13.3). None = mọi video."""

    video_ids: list[str] | None = None


# ---- recipe (CONTRACTS §7, quyết định #39)


class ComponentConfig(BaseModel):
    """Một component trong config: `{"type": <name>, "params": {...}}`; `params` validate bằng `Params` của class."""

    model_config = ConfigDict(extra="forbid")
    type: str
    params: dict = Field(default_factory=dict)


class ObjectiveConfig(ComponentConfig):
    strength: float = 1.0         # số mũ trong combiner, không thuộc `params` của objective


class RecipeConfig(BaseModel):
    """Công thức ghép reference + objectives + combiner + selector. Khai báo ở `components/recipes.py`."""

    model_config = ConfigDict(extra="forbid")
    name: str
    description: str = ""         # cho người / agent (RecipeDTO), không vào config_hash (#48)
    reference: ComponentConfig
    objectives: list[ObjectiveConfig] = Field(default_factory=list)
    combiner: ComponentConfig
    selector: ComponentConfig


# ---- vai trò Judge / Router (CONTRACTS §12.3) — MVP chỉ có contract


@dataclass
class JudgeRequest:
    proposal: Proposal
    goal: str
    context_s: float = 2.0


@dataclass
class Propose:
    recipe: str
    config: RecipeConfig | None   # bản đầy đủ đã chỉnh; None = recipe có sẵn (#45)
    batch_size: int


@dataclass
class Stop:
    reason: str


Action = Propose | Stop
History = list[Proposal | Decision]   # theo thời gian: mỗi proposal, rồi các quyết định cho nó
