"""Selector không dùng nhãn: chỉ nhận id, điểm, embedding. Module này không được import oracle/data (test kiểm)."""

from __future__ import annotations

import numpy as np

RELAX = (1.0, 0.5, 0.25)  # như P-026 greedy_nms: thiếu thì nới min_dist, hết mức thì lấy đầu thứ tự


def random_k(ids: list[str], k: int, seed: int) -> list[str]:
    ids = sorted(ids)
    pick = np.random.default_rng(seed).choice(len(ids), min(k, len(ids)), replace=False)
    return [ids[i] for i in pick]


def rank(ids: list[str], scores) -> list[int]:
    """Index theo điểm giảm dần; hoà thì theo id (tất định)."""
    return sorted(range(len(ids)), key=lambda i: (-float(scores[i]), ids[i]))


def topk(ids: list[str], scores, k: int) -> list[str]:
    return [ids[i] for i in rank(ids, scores)[:k]]


def greedy_diverse(ids: list[str], order: list[int], emb: np.ndarray, k: int, min_dist: float) -> tuple[list[str], float]:
    """Port P-026 `greedy_nms` (bỏ NMS thời gian: BDD100K mỗi video một ảnh). Duyệt `order`, nhận ảnh nếu
    1 − cos ≥ min_dist với mọi ảnh đã nhận; thiếu thì nới ×0.5, ×0.25, rồi lấy đầu `order`. Trả (ids, hệ số nới)."""
    order = list(order)
    k = min(k, len(order))
    e = np.asarray(emb, np.float32)[order]
    e /= np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
    for f in RELAX:
        took = np.empty((k, e.shape[1]), np.float32)
        idx: list[int] = []
        for j in range(len(order)):
            if not idx or (1 - took[: len(idx)] @ e[j]).min() >= min_dist * f:
                took[len(idx)] = e[j]
                idx.append(j)
                if len(idx) == k:
                    return [ids[order[j]] for j in idx], f
    return [ids[i] for i in order[:k]], 0.0
