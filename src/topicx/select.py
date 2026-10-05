"""Chọn ngẫu nhiên tất định cho các nhánh đối chứng của harness (ORACLE_POSITIVE). Logic chọn mẫu thật nằm ở src/rav."""

from __future__ import annotations

import numpy as np


def random_k(ids: list[str], k: int, seed: int) -> list[str]:
    ids = sorted(ids)
    pick = np.random.default_rng(seed).choice(len(ids), min(k, len(ids)), replace=False)
    return [ids[i] for i in pick]
