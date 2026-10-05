"""Block 4 — hiển thị và lọc: frame theo tiêu chí, lỗi của model (mẫu đang sai), frame mà mỗi cách proposal đã chọn.

- Tập công khai (dev / test / base): xem thẳng nhãn và thuộc tính.
- Pool: chỉ xem nhãn của ảnh **đã được chọn và chốt manifest** (qua `OracleStore.reveal`, có log) — không duyệt nhãn pool tự do.
- Lỗi model trên dev là chẩn đoán (chọn điểm yếu); trên final test chỉ nên xem SAU khi đã chốt mọi thứ (đánh giá xong).
Lọc bằng `DataFrame.query`, ví dụ: `df.query("timeofday == 'night' and n_pedestrian >= 3")`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import common as C
from . import data as D
from .oracle import OracleStore

PUBLIC = ("dev", "test", "base")


def _art(cfg, rel) -> Path:
    return C.WORK / cfg["art"] / rel


def _frames(images: pd.DataFrame, boxes: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    b = boxes.assign(size=D.size_bucket(boxes))
    counts = pd.crosstab(b.image, b.cls).reindex(columns=classes, fill_value=0).add_prefix("n_")
    small = b[b["size"] == "small"].groupby("image").size().rename("n_small")
    df = images.set_index("image")[["timeofday", "weather", "scene"]].join(counts).join(small).fillna(0)
    num = [c for c in df if c.startswith("n_")]
    df[num] = df[num].astype(int)
    df["n_objects"] = df[[f"n_{c}" for c in classes if f"n_{c}" in df]].sum(axis=1)
    return df.reset_index()


def frame_table(cfg, split: str = "dev") -> pd.DataFrame:
    """Mỗi ảnh một dòng: thuộc tính cảnh + số box từng class + số box nhỏ. Chỉ tập công khai."""
    assert split in PUBLIC, f"{split}: chỉ xem tập công khai {PUBLIC}; ảnh pool xem qua selection_frames sau khi chốt manifest"
    ids = set(C.read_ids(_art(cfg, f"splits/{split}_ids.txt")))
    images = pd.read_csv(_art(cfg, "splits/public_images.csv"))
    boxes = pd.read_csv(_art(cfg, "splits/public_boxes.csv"))
    return _frames(images[images.image.isin(ids)], boxes[boxes.image.isin(ids)], cfg["classes"]).assign(split=split)


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lt, rb = np.maximum(a[:, None, :2], b[None, :, :2]), np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


def _match(gt: np.ndarray, pred: np.ndarray, conf: np.ndarray, thr: float) -> tuple[np.ndarray, np.ndarray]:
    """Greedy theo conf giảm dần, một–một, cùng class (gọi theo từng class). Trả (gt_hit, pred_matched)."""
    hit, used = np.zeros(len(gt), bool), np.zeros(len(pred), bool)
    if len(gt) and len(pred):
        iou = _iou(pred, gt)
        for p in np.argsort(-conf):
            row = np.where(hit, -1, iou[p])
            j = int(row.argmax())
            if row[j] >= thr:
                hit[j], used[p] = True, True
    return hit, used


def model_errors(cfg, weights: Path, split: str = "dev", tag: str = "base", conf: float = 0.25, iou: float = 0.5,
                 batch: int = 32) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chạy `weights` trên tập công khai `split`, khớp với ground truth ở conf / IoU cố định (P-026 focus §4C).
    Trả (gt, fp): mỗi GT box có cột `hit` (False = bỏ sót); mỗi prediction không khớp có `kind` ∈ {confusion:<class GT>, duplicate,
    background}. Cache ở `<art>/explore/<tag>_<split>_{gt,fp}.csv`."""
    from ultralytics import YOLO
    assert split in PUBLIC, split
    gp, fp_p = _art(cfg, f"explore/{tag}_{split}_gt.csv"), _art(cfg, f"explore/{tag}_{split}_fp.csv")
    if gp.exists() and fp_p.exists():
        return pd.read_csv(gp), pd.read_csv(fp_p)
    ids = C.read_ids(_art(cfg, f"splits/{split}_ids.txt"))
    paths = D.split_paths(cfg)
    m = YOLO(str(weights))
    names = list(m.names.values())
    boxes = pd.read_csv(_art(cfg, "splits/public_boxes.csv"))
    # GT của mọi class trong taxonomy: class model không có (vd novel ở kịch bản missing) luôn là bỏ sót
    by = {k: g for k, g in boxes[boxes.image.isin(set(ids)) & boxes.cls.isin(cfg["classes"])].groupby("image")}
    gts, fps = [], []
    from tqdm.auto import tqdm
    for s in tqdm(range(0, len(ids), batch), desc=f"lỗi {tag} trên {split}"):
        chunk = ids[s:s + batch]
        for img, r in zip(chunk, m.predict([str(paths[i]) for i in chunk], conf=conf, imgsz=cfg["yolo"]["imgsz"], verbose=False)):
            g = by.get(img, pd.DataFrame(columns=D.BOX_COLS))
            gb = g[["x1", "y1", "x2", "y2"]].to_numpy(float)
            pb, pc, ps = r.boxes.xyxy.cpu().numpy(), [names[int(c)] for c in r.boxes.cls.cpu().numpy()], r.boxes.conf.cpu().numpy()
            hit, used = np.zeros(len(g), bool), np.zeros(len(pb), bool)
            for c in set(g.cls) | set(pc):
                gi, pi = np.flatnonzero(g.cls.to_numpy() == c), np.flatnonzero(np.array(pc) == c)
                h, u = _match(gb[gi], pb[pi], ps[pi], iou)
                hit[gi], used[pi] = h, u
            gts.append(g.assign(hit=hit))
            for p in np.flatnonzero(~used):
                ov = _iou(pb[p:p + 1], gb)[0] if len(gb) else np.zeros(0)
                other = [j for j in np.flatnonzero(ov >= iou) if g.cls.iloc[j] != pc[p]]
                kind = f"confusion:{g.cls.iloc[other[0]]}" if other else ("duplicate" if (ov >= iou).any() else "background")
                fps.append({"image": img, "cls": pc[p], "conf": float(ps[p]), "kind": kind,
                            **dict(zip(("x1", "y1", "x2", "y2"), map(float, pb[p])))})
    gt, fp = pd.concat(gts, ignore_index=True), pd.DataFrame(fps, columns=["image", "cls", "conf", "kind", "x1", "y1", "x2", "y2"])
    gt["size"] = D.size_bucket(gt).to_numpy()
    gt.to_csv(C.out(cfg, f"explore/{tag}_{split}_gt.csv"), index=False)
    fp.to_csv(C.out(cfg, f"explore/{tag}_{split}_fp.csv"), index=False)
    return gt, fp


def error_frames(cfg, gt: pd.DataFrame, fp: pd.DataFrame, split: str = "dev") -> pd.DataFrame:
    """frame_table + số bỏ sót (`miss_<class>`) và số FP (`fp_<class>`) mỗi ảnh — lọc ra "mẫu đang sai"."""
    df = frame_table(cfg, split).set_index("image")
    miss = pd.crosstab(gt[~gt.hit].image, gt[~gt.hit].cls).add_prefix("miss_")
    fpc = pd.crosstab(fp.image, fp.cls).add_prefix("fp_")
    df = df.join(miss).join(fpc).fillna(0)
    cols = [c for c in df if c.startswith(("miss_", "fp_"))]
    df[cols] = df[cols].astype(int)
    df["n_miss"], df["n_fp"] = df[[c for c in cols if c.startswith("miss_")]].sum(axis=1), df[[c for c in cols if c.startswith("fp_")]].sum(axis=1)
    return df.reset_index()


def recall_table(gt: pd.DataFrame, frames: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Recall (ở conf / IoU của model_errors) theo class × các cột của frame (vd timeofday, weather) và kích thước."""
    g = gt.merge(frames[["image"] + [c for c in by if c in frames]], on="image", how="left")
    return g.groupby(["cls"] + by).hit.agg(recall="mean", n_gt="size").reset_index()


def selection_frames(cfg, name: str, seed: int, k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Frame một cách proposal đã chọn (manifest đã chốt), kèm vòng / thứ hạng / điểm từng tiêu chí (history.jsonl của block 2, nếu có)
    và nhãn + thuộc tính sau khi mở (simulated annotation, có log). Trả (frames, boxes)."""
    man = _art(cfg, f"selections/s{seed}_k{k}_{name}.json")
    m = C.read_frozen(man)
    oracle = OracleStore(_art(cfg, "splits/oracle"), C.out(cfg, "explore/oracle_log.jsonl"))
    bx, im = oracle.reveal(man)
    df = _frames(im, bx, cfg["classes"]).set_index("image").reindex(m["ids"]).reset_index()
    hist = _art(cfg, f"proposals/{name}/s{seed}_k{k}/history.jsonl")
    if hist.exists():
        rows = {}
        for line in hist.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["type"] == "proposal":
                for p in r["picks"]:
                    rows[p["unit_id"].rsplit(":", 1)[0] + ".jpg"] = {"round": r["n"], "rank": p["rank"], "combined": p["combined"],
                                                                     **{f"score_{a}": v for a, v in p["breakdown"].items()}}
        df = df.join(pd.DataFrame([rows.get(i, {}) for i in df.image]))
    return df, bx


def show(cfg, images: list[str], gt: pd.DataFrame | None = None, fp: pd.DataFrame | None = None, title_cols=(), frames=None,
         n: int = 12, cols: int = 4, width: float = 4.5):
    """Lưới ảnh: GT xanh lá (bỏ sót = vàng nếu `gt` có cột `hit`), prediction sai = đỏ. Tiêu đề = tên ảnh + các cột `title_cols` của `frames`."""
    import matplotlib.patches as mp
    import matplotlib.pyplot as plt
    from PIL import Image
    images = list(images)[:n]
    if not images:
        print("không có ảnh")
        return
    paths = D.split_paths(cfg)
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * width, rows * width * 0.6))
    axes = np.atleast_1d(axes).ravel()
    info = frames.set_index("image") if frames is not None else None
    for ax, img in zip(axes, images):
        ax.imshow(Image.open(paths[img]))
        ax.axis("off")
        for df, default in ((gt, "lime"), (fp, "red")):
            if df is None:
                continue
            for r in df[df.image == img].itertuples():
                color = "yellow" if not getattr(r, "hit", True) else default   # np.bool_ → không dùng `is False`
                ax.add_patch(mp.Rectangle((r.x1, r.y1), r.x2 - r.x1, r.y2 - r.y1, fill=False, color=color, lw=1.5))
                ax.text(r.x1, r.y1, r.cls if default == "lime" else f"{r.cls} {getattr(r, 'kind', '')}", color=color, fontsize=6)
        extra = " ".join(f"{c}={info.at[img, c]}" for c in title_cols if info is not None and c in info and img in info.index)
        ax.set_title(f"{img}\n{extra}", fontsize=7)
    for ax in axes[len(images):]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def compare_errors(gt_a: pd.DataFrame, gt_b: pd.DataFrame) -> pd.DataFrame:
    """So hai model trên cùng tập (vd base vs sau finetune): theo class, số GT được sửa (a bỏ sót → b bắt được), mới hỏng, recall a / b."""
    k = ["image", "cls", "x1", "y1", "x2", "y2"]
    m = gt_a[k + ["hit"]].merge(gt_b[k + ["hit"]], on=k, suffixes=("_a", "_b"))
    return m.groupby("cls").agg(n_gt=("hit_a", "size"), recall_a=("hit_a", "mean"), recall_b=("hit_b", "mean"),
                                fixed=("hit_b", lambda s: int((s & ~m.loc[s.index, "hit_a"]).sum())),
                                broken=("hit_a", lambda s: int((s & ~m.loc[s.index, "hit_b"]).sum())))
