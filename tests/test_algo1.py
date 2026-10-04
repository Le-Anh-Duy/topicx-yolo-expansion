"""Algo 1 trên dữ liệu tổng hợp."""

import numpy as np
import pytest

from topicx.algo1 import InsufficientCandidates, algo1_select
from topicx.metrics import tau_dev_f1


def _toy():
    # A, B, C, D trực giao; A', B' gần trùng A, B. Relevance giảm dần theo thứ tự x0..x5.
    e = np.eye(6)
    emb = np.stack([e[0], e[1], e[0] + 0.01 * e[4], e[1] + 0.01 * e[5], e[2], e[3]])
    ids = [f"x{i}" for i in range(6)]
    return ids, np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4]), emb


def test_history_avoids_previous_rounds():
    ids, rel, emb = _toy()
    hist, meta = algo1_select(ids, rel, emb, budget=4, K=10, B=2, mode="history")
    batch, _ = algo1_select(ids, rel, emb, budget=4, K=10, B=2, mode="batch")
    topb, _ = algo1_select(ids, rel, emb, budget=4, K=10, B=2, mode="topb")
    assert hist == ["x0", "x1", "x4", "x5"]   # vòng 2 tránh A', B' vì gần ảnh đã lấy ở vòng 1
    assert batch == ["x0", "x1", "x2", "x3"]  # chỉ xét trong batch -> lấy lại bản gần trùng
    assert topb == ["x0", "x1", "x2", "x3"]
    assert len(meta["rounds"]) == 2 and meta["rounds"][1]["min_hist_d"] > 0.4


def test_tau_filter_and_prefix_property():
    rng = np.random.default_rng(0)
    ids = [f"i{i:03d}" for i in range(300)]
    rel, emb = rng.random(300), rng.normal(size=(300, 8))
    for mode in ("history", "batch", "topb"):
        small, _ = algo1_select(ids, rel, emb, budget=32, K=50, B=16, tau=0.3, mode=mode)
        big, _ = algo1_select(ids, rel, emb, budget=64, K=50, B=16, tau=0.3, mode=mode)
        assert big[:32] == small and len(set(big)) == 64
        assert min(rel[int(i[1:])] for i in big) >= 0.3


def test_insufficient_candidates_does_not_lower_tau():
    ids, rel, emb = _toy()
    with pytest.raises(InsufficientCandidates):
        algo1_select(ids, rel, emb, budget=4, K=10, B=2, tau=0.75)
    with pytest.raises(AssertionError, match="bội số"):
        algo1_select(ids, rel, emb, budget=3, K=10, B=2)


def test_k_expands_when_filter_leaves_too_few():
    ids, rel, emb = _toy()
    _, meta = algo1_select(ids, rel, emb, budget=2, K=1, B=2, tau=0.0)
    assert meta["rounds"][0]["k_used"] == 2


def test_tau_dev_f1():
    r = tau_dev_f1([0.9, 0.8, 0.1], [1, 1, 0])
    assert r["tau"] == 0.8 and r["dev_f1"] == 1.0
