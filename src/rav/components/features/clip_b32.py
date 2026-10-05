"""`emb.clip_b32x3` / `emb.clip_b32`: embedding ảnh OpenCLIP ViT-B/32 (cùng không gian với text, CONTRACTS §8 "Chọn theo mô tả text").

Ảnh dashcam 16:9 cắt 3 crop vuông trái/giữa/phải thay vì center-crop mặc định (mất hai bên ảnh, vật thể nhỏ ở rìa).
Model nạp lazy, chạy GPU nếu có. Checkpoint: open_clip `ViT-B-32` / `laion2b_s34b_b79k`.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from PIL import Image

from ...core.component import Feature
from ...core.registry import register
from ...core.types import FieldTable

BATCH = 64   # cấu hình chạy, không vào params_hash (CONTRACTS §3)
MODEL, PRETRAINED = "ViT-B-32", "laion2b_s34b_b79k"


@lru_cache(maxsize=2)
def clip_model(model: str = MODEL, pretrained: str = PRETRAINED):
    """(model, preprocess, tokenizer, device) — dùng chung với objective `text_match` để ảnh và text cùng model."""
    import open_clip
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    m, _, pre = open_clip.create_model_and_transforms(model, pretrained=pretrained, device=device)
    return m.eval(), pre, open_clip.get_tokenizer(model), device


def square_crops(img: Image.Image) -> list[Image.Image]:
    w, h = img.size
    s = min(w, h)
    if w >= h:
        return [img.crop((x, 0, x + s, s)) for x in (0, (w - s) // 2, w - s)]
    return [img.crop((0, y, s, y + s)) for y in (0, (h - s) // 2, h - s)]


@register("feature")
class ClipB32x3(Feature):
    """Embedding OpenCLIP ViT-B/32 của 3 crop vuông (trái/giữa/phải) của frame anchor, mỗi crop qua preprocess của checkpoint;
    (N, 3, 512) float32, chuẩn hoá L2 từng crop. Objective `text_match` lấy max cosine theo crop."""

    name = "clip_b32x3"
    version = "1"
    provides = "emb.clip_b32x3"
    per_sample = True

    def compute(self, units, ctx) -> FieldTable:
        import torch

        m, pre, _, device = clip_model()
        out = []
        for start in range(0, len(units), BATCH):
            x = torch.stack([pre(c) for u in units[start:start + BATCH]
                             for c in square_crops(Image.fromarray(ctx.pixels(u.anchor)))]).to(device)
            with torch.inference_mode(), torch.autocast("cuda", enabled=device == "cuda"):
                e = m.encode_image(x).float()
            e = torch.nn.functional.normalize(e, dim=-1).view(-1, 3, e.shape[-1])
            out.append(e.cpu().numpy())
        values = np.concatenate(out).astype(np.float32) if out else np.zeros((0, 3, 512), np.float32)
        return FieldTable(self.key, [u.id for u in units], values, "vector")


@register("feature")
class ClipB32(Feature):
    """Embedding cả ảnh = trung bình 3 crop của `emb.clip_b32x3` rồi chuẩn hoá L2; (N, 512). Dùng cho khoảng cách thị giác (diversity)."""

    name = "clip_b32"
    version = "1"
    provides = "emb.clip_b32"
    per_sample = True
    requires = ("emb.clip_b32x3",)

    def compute(self, units, ctx) -> FieldTable:
        e = np.asarray(ctx.field("emb.clip_b32x3", units).values, np.float32).mean(axis=1)
        e /= np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
        return FieldTable(self.key, [u.id for u in units], e, "vector")
