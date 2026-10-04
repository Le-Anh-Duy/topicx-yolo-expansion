"""Algo 1 — Semantic Retrieval + History-aware Visual Diversity (docs/algo_1_semantic_retrieval_history_diversity.md).

Không dùng nhãn: chỉ nhận id, relevance r(x,q) và visual embedding v_x. Module này không được import oracle/data (test kiểm).

Mỗi vòng t:  C_t = {x ∈ TopK_{U_t} r(x,q) : r ≥ τ}  ->  greedy chọn B ảnh tối đa δ_t(x,S) = min_{y ∈ H_t ∪ S} d(x,y),
d(x,y) = (1 − cos(v_x, v_y)) / 2. Hoà distance -> relevance cao hơn -> id. Mô phỏng: mọi batch được chấp nhận,
H_{t+1} = H_t ∪ A_t, A_t bị loại khỏi pool.

mode:
  history  Algo 1 (xét H_t ∪ S)
  batch    đối chứng: diversity chỉ trong batch (bỏ H_t)
  topb     đối chứng: top-B relevance trong C_t
"""

from __future__ import annotations

import numpy as np

from .select import rank


class InsufficientCandidates(RuntimeError):
    pass


def _dist(v: np.ndarray, rows, j: int) -> np.ndarray:
    return (1.0 - v[rows] @ v[j]) / 2.0


def algo1_select(ids: list[str], rel: np.ndarray, emb: np.ndarray, budget: int, K: int = 500, B: int = 16,
                 tau: float | None = None, mode: str = "history", history_ids=()) -> tuple[list[str], dict]:
    """Trả (ids theo thứ tự chọn, meta từng vòng). `budget` phải là bội số của B.
    Thiếu ứng viên: nhân đôi K (luật cố định) tới cả pool; vẫn thiếu thì InsufficientCandidates (không hạ τ)."""
    assert mode in ("history", "batch", "topb"), mode
    assert budget % B == 0, f"budget {budget} phải là bội số của B={B}"
    rel = np.asarray(rel, np.float64)
    v = np.asarray(emb, np.float64)
    v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
    n = len(ids)
    order = rank(ids, rel)  # ranking nền cố định: encoder và topic không đổi qua các vòng
    pos = {i: j for j, i in enumerate(ids)}
    hist = [pos[h] for h in history_ids]
    d_hist = np.full(n, np.inf)  # khoảng cách gần nhất tới H_t, cập nhật tăng dần
    for h in hist:
        d_hist = np.minimum(d_hist, _dist(v, slice(None), h))
    used = np.zeros(n, bool)
    used[hist] = True
    picked: list[int] = []
    rounds = []
    for t in range(budget // B):
        k_used = K
        while True:
            top = [j for j in order if not used[j]][:k_used]
            cand = np.array([j for j in top if tau is None or rel[j] >= tau], int)
            if len(cand) >= B or k_used >= n:
                break
            k_used *= 2
        if len(cand) < B:
            raise InsufficientCandidates(f"vòng {t}: chỉ {len(cand)} ứng viên có r ≥ τ={tau}, cần B={B}")
        if mode == "topb":
            chosen = list(range(B))  # cand đã theo thứ tự relevance giảm dần, hoà theo id
        else:
            d = d_hist[cand].copy() if mode == "history" else np.full(len(cand), np.inf)
            chosen = []
            for _ in range(B):
                # argmax trả index đầu tiên khi hoà; cand theo relevance giảm dần -> hoà ưu tiên relevance rồi id.
                # Chưa có lịch sử và S rỗng: mọi d = inf -> lấy ảnh relevance cao nhất.
                c = int(np.argmax(d))
                chosen.append(c)
                d = np.minimum(d, _dist(v, cand, cand[c]))
                d[chosen] = -np.inf
        batch = cand[chosen]
        # tóm tắt giải thích (tính cho mọi mode để so được)
        pair = (1.0 - v[batch] @ v[batch].T) / 2.0
        np.fill_diagonal(pair, np.inf)
        rounds.append({"round": t, "k_used": k_used, "n_candidates": int(len(cand)),
                       "min_pair_d": float(pair.min()), "min_hist_d": float(d_hist[batch].min()) if np.isfinite(d_hist[batch]).any() else None,
                       "rel_mean": float(rel[batch].mean()), "rel_min": float(rel[batch].min())})
        for j in batch:
            d_hist = np.minimum(d_hist, _dist(v, slice(None), j))
        used[batch] = True
        picked += batch.tolist()
    return [ids[j] for j in picked], {"mode": mode, "K": K, "B": B, "tau": tau, "n_history": len(hist), "rounds": rounds}
