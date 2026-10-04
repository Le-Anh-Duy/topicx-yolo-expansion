"""Embedding không dùng nhãn.

- OpenCLIP: mỗi ảnh 1280×720 cắt 3 crop vuông (trái/giữa/phải) thay vì center-crop mặc định (mất hai bên ảnh).
  Điểm topic = max theo crop của cosine(ảnh, trung bình text embedding các prompt). Cosine KHÔNG phải xác suất.
- DINOv2-S: cả khung, resize 224×392 (bội 14, giữ ~16:9) — embedding cho nhánh DIVERSITY như P-026.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def square_crops(img: Image.Image) -> list[Image.Image]:
    w, h = img.size
    s = min(w, h)
    if w >= h:
        return [img.crop((x, 0, x + s, s)) for x in (0, (w - s) // 2, w - s)]
    return [img.crop((0, y, s, y + s)) for y in (0, (h - s) // 2, h - s)]


class _Images(torch.utils.data.Dataset):
    def __init__(self, paths, tf):
        self.paths, self.tf = [str(p) for p in paths], tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB"))


@torch.no_grad()
def _run(paths, tf, fn, batch: int, workers: int = 4) -> np.ndarray:
    dl = torch.utils.data.DataLoader(_Images(paths, tf), batch_size=batch, num_workers=workers)
    dev = device()
    out = []
    for x in tqdm(dl, desc="embed"):
        with torch.autocast("cuda", enabled=dev == "cuda"):
            out.append(fn(x.to(dev)).float().cpu().numpy())
    return np.concatenate(out)


def load_clip(model: str, pretrained: str):
    import open_clip
    m, _, pre = open_clip.create_model_and_transforms(model, pretrained=pretrained, device=device())
    return m.eval(), pre, open_clip.get_tokenizer(model)


def clip_images(paths, m, pre, batch: int) -> np.ndarray:
    """(N, 3 crop, D) float16, đã chuẩn hoá L2."""
    def fn(x):
        b, c = x.shape[:2]
        return F.normalize(m.encode_image(x.flatten(0, 1)), dim=-1).view(b, c, -1)
    return _run(paths, lambda img: torch.stack([pre(c) for c in square_crops(img)]), fn, batch).astype(np.float16)


@torch.no_grad()
def clip_texts(prompts: list[str], m, tok) -> np.ndarray:
    return F.normalize(m.encode_text(tok(prompts).to(device())).float(), dim=-1).cpu().numpy()


def topic_scores(img_emb: np.ndarray, text_emb: np.ndarray) -> np.ndarray:
    """Ensemble prompt = trung bình text embedding rồi chuẩn hoá; điểm ảnh = max cosine qua các crop."""
    t = text_emb.mean(0)
    t /= np.linalg.norm(t)
    return (img_emb.astype(np.float32) @ t).max(1)


def image_level(img_emb: np.ndarray) -> np.ndarray:
    e = img_emb.astype(np.float32).mean(1)
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def dinov2_images(paths, name: str, batch: int) -> np.ndarray:
    import torchvision.transforms as T
    m = torch.hub.load("facebookresearch/dinov2", name).to(device()).eval()
    tf = T.Compose([T.Resize((224, 392)), T.ToTensor(), T.Normalize(*IMAGENET)])
    return _run(paths, tf, lambda x: F.normalize(m(x), dim=-1), batch).astype(np.float16)
