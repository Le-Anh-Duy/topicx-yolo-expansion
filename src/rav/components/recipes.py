"""Recipe = công thức ghép reference + objectives + combiner + selector (CONTRACTS §7).
MVP: `diversity` (novelty + greedy_nms), `random`, `uniform` (baseline: objectives rỗng, §9.8).
Đăng ký vào registry khi `discover()`; tầng trên đọc bằng `REGISTRY.recipe(name)` (không import file này)."""

from ..core.registry import REGISTRY
from ..core.types import RecipeConfig

_BASE = {"reference": {"type": "session_kept"}, "combiner": {"type": "product_strength"}}

RECIPES = [
    {"name": "diversity", **_BASE,
     "description": "Ưu tiên frame khác tập đã keep (novelty trên DINOv2-S), chọn batch bằng greedy + NMS thời gian để không trùng nhau.",
     "objectives": [{"type": "novelty", "strength": 1.0, "params": {"field": "emb.dinov2_s"}}],
     "selector": {"type": "greedy_nms", "params": {"field": "emb.dinov2_s", "nms_s": 1.0, "min_dist": 0.1}}},
    {"name": "random", **_BASE, "description": "Baseline: chọn ngẫu nhiên đều trong candidates.",
     "selector": {"type": "random"}},
    {"name": "uniform", **_BASE, "description": "Baseline: rải đều theo thời gian trong candidates, chia K theo tỉ lệ số unit mỗi video.",
     "selector": {"type": "uniform", "params": {"allocation": "proportional"}}},
]

for _r in RECIPES:
    REGISTRY.add_recipe(RecipeConfig.model_validate(_r))
