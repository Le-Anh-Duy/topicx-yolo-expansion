"""`text_match`: mức khớp giữa ảnh và mô tả text (CONTRACTS §8 "Chọn theo mô tả text", #38). Có thể làm cổng lọc (top_k / tau)."""

from __future__ import annotations

import numpy as np
from pydantic import Field

from ...core.registry import register
from ...core.types import Scores, Unit
from .base import Objective


def text_relevance(img_emb: np.ndarray, pos: np.ndarray, neg: np.ndarray | None = None) -> np.ndarray:
    """cos(ảnh, trung bình text dương) − max cos(ảnh, text âm); ảnh (N, D) hoặc (N, C, D) → lấy max theo crop C. Cosine, không phải xác suất."""
    def cos(t):
        t = t / np.linalg.norm(t)
        s = np.asarray(img_emb, np.float32) @ t
        return s.max(axis=1) if s.ndim == 2 else s
    r = cos(pos.mean(axis=0))
    if neg is not None and len(neg):
        r = r - np.max(np.stack([cos(n) for n in neg]), axis=0)
    return r


@register("objective")
class TextMatch(Objective):
    """Relevance r = cos(emb ảnh, trung bình embedding các câu `texts`) − max cos với từng câu `negative`; field nhiều crop thì lấy max theo crop.
    Text mã hoá bằng cùng model với field (OpenCLIP ViT-B/32). Điểm = min-max của r trên các unit đang chấm (mọi r bằng nhau → 1).
    Cổng lọc (mask False): `tau` = loại r < tau (trên r thô, không phải điểm đã chuẩn hoá); `top_k` = chỉ giữ top_k theo r trong các unit đang chấm
    (hoà theo (video_id, t0)). Không đặt cả hai thì không lọc."""

    name = "text_match"
    version = "1"
    uses_goal = True

    class Params(Objective.Params):
        field: str = "emb.clip_b32x3"
        texts: list[str] = Field(min_length=1)
        negative: list[str] = []
        top_k: int | None = Field(None, ge=1)
        tau: float | None = None

    def score(self, units: list[Unit], ctx) -> Scores:
        ids = [u.id for u in units]
        if not units:
            return Scores(self.name, ids, np.zeros(0), np.zeros(0, bool))
        pos, neg = _encode(self.params.texts), (_encode(self.params.negative) if self.params.negative else None)
        r = text_relevance(ctx.field(self.params.field, units).values, pos, neg)
        mask = np.ones(len(units), bool)
        if self.params.tau is not None:
            mask &= r >= self.params.tau
        if self.params.top_k is not None:
            order = sorted(range(len(units)), key=lambda i: (-r[i], units[i].video_id, units[i].t0))
            top = np.zeros(len(units), bool)
            top[order[:self.params.top_k]] = True
            mask &= top
        span = r.max() - r.min()
        values = (r - r.min()) / span if span > 1e-12 else np.ones(len(r))
        return Scores(self.name, ids, np.clip(values, 0.0, 1.0), mask)


def _encode(texts: list[str]) -> np.ndarray:
    import torch

    from ..features.clip_b32 import clip_model
    m, _, tok, device = clip_model()
    with torch.inference_mode():
        return torch.nn.functional.normalize(m.encode_text(tok(texts).to(device)).float(), dim=-1).cpu().numpy()
