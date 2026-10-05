"""`emb.dinov2_s`: embedding DINOv2 ViT-S/14, (N, 384) float32 chuẩn hoá L2. per_sample=True.

Model nạp từ cache `torch.hub` (`<hub_dir>/facebookresearch_dinov2_main`) nếu có, không thì tải từ GitHub.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from ...core.component import Feature
from ...core.registry import register
from ...core.types import FieldTable

BATCH = 8   # cấu hình chạy, không vào params_hash (CONTRACTS §3)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


@lru_cache(maxsize=1)
def _model():
    import torch  # nạp lazy: import package không kéo theo torch

    hub = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
    if hub.is_dir():
        return torch.hub.load(str(hub), "dinov2_vits14", source="local").eval()
    return torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").eval()


@register("feature")
class DinoV2S(Feature):
    """Embedding DINOv2 ViT-S/14: ảnh RGB resize về image_size × image_size, chuẩn hoá mean/std ImageNet, lấy vector CLS
    384 chiều rồi chuẩn hoá L2 (cos = tích vô hướng). Unit nhiều frame: trung bình embedding các frame rồi chuẩn hoá lại."""

    name = "dinov2_s"
    version = "1"
    provides = "emb.dinov2_s"
    per_sample = True

    class Params(Feature.Params):
        image_size: int = 224   # bội của 14 (patch)

    def compute(self, units, ctx) -> FieldTable:
        import torch

        frames = [f for u in units for f in (u.frames or (u.anchor,))]
        size = self.params.image_size
        vectors = []
        for start in range(0, len(frames), BATCH):
            batch = []
            for f in frames[start:start + BATCH]:
                rgb = cv2.resize(ctx.pixels(f), (size, size), interpolation=cv2.INTER_AREA)
                batch.append(((rgb.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1))
            with torch.inference_mode():
                vectors.append(_model()(torch.from_numpy(np.stack(batch))).float().numpy())
        emb = np.concatenate(vectors) if vectors else np.zeros((0, 384), np.float32)
        out, i = [], 0
        for u in units:
            k = len(u.frames) or 1
            out.append(emb[i:i + k].mean(axis=0))
            i += k
        values = np.asarray(out, np.float32).reshape(len(units), -1)
        values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
        return FieldTable(self.key, [u.id for u in units], values, "vector")
