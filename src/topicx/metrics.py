"""Chỉ số retrieval và tổng hợp kết quả theo seed."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import size_bucket


def average_precision(scores, is_pos) -> float:
    """AP của một ranking (dùng chọn prompt trên dev)."""
    order = np.argsort(-np.asarray(scores, float), kind="stable")
    pos = np.asarray(is_pos, bool)[order]
    if not pos.any():
        return float("nan")
    return float((np.cumsum(pos)[pos] / (np.flatnonzero(pos) + 1)).mean())


def batch_redundancy(emb: np.ndarray) -> dict:
    """Mức trùng lặp trong batch: cosine với láng giềng gần nhất (trung bình) và tỉ lệ ảnh có láng giềng cos > 0.95."""
    e = np.asarray(emb, np.float32)
    e = e / np.linalg.norm(e, axis=1, keepdims=True)
    s = e @ e.T
    np.fill_diagonal(s, -1)
    nn = s.max(1)
    return {"nn_cos_mean": float(nn.mean()), "frac_nn_cos_gt_0.95": float((nn > 0.95).mean())}


def retrieval_report(ids: list[str], boxes: pd.DataFrame, images: pd.DataFrame, novel: str, n_pos_pool: int) -> dict:
    """`boxes`, `images` là nhãn đã reveal của đúng các ảnh `ids`."""
    nb = boxes[boxes.cls == novel]
    n_pos = nb.image.nunique()
    r = {"k": len(ids), "n_pos_images": n_pos, "precision_at_k": n_pos / len(ids), "recall_at_k": n_pos / n_pos_pool,
         "novel_instances": len(nb), "base_instances": int((boxes.cls != novel).sum())}
    r.update({f"novel_{k}": v for k, v in size_bucket(nb).value_counts().items()})
    r.update({f"tod_{k}": v for k, v in images.timeofday.value_counts().items()})
    return r


def paired_summary(df: pd.DataFrame, metrics: list[str], ref: str = "RANDOM") -> pd.DataFrame:
    """mean/std theo (k, branch) và hiệu từng cặp so với nhánh `ref` cùng (seed, k)."""
    agg = df.groupby(["k", "branch"])[metrics].agg(["mean", "std"])
    base = df[df.branch == ref].set_index(["seed", "k"])[metrics]
    d = df.set_index(["seed", "k"])
    diff = (d[metrics] - base.reindex(d.index).to_numpy()).assign(branch=d.branch.to_numpy()).reset_index()
    dagg = diff.groupby(["k", "branch"])[metrics].agg(["mean", "std"])
    dagg.columns = [f"d_{a}_vs_{ref}_{b}" for a, b in dagg.columns]
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg.join(dagg)
