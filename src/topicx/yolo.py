"""Ultralytics YOLO: mở rộng detection head nc -> nc+1, train (bỏ qua nếu đã xong), đánh giá AP theo tên class, recall ở conf cố định.

Mở rộng head: khi đổi nc, Ultralytics nạp weights qua so khớp shape nên conv cls cuối của Detect (`cv3[i][-1]`) bị bỏ
và khởi tạo lại TOÀN BỘ, kể cả class cũ. `expand_head` tự copy: mọi tensor cùng shape giữ nguyên; tensor có chiều 0 = nc
copy hàng class cũ; hàng class mới giữ khởi tạo mặc định của DetectionModel (seed cố định, bias theo `bias_init`).
"""

from __future__ import annotations

import copy
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def expand_head(base_pt: Path, out_pt: Path, names: list[str], seed: int) -> dict:
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel

    y = YOLO(str(base_pt))
    old = y.model.float()
    nc_old, nc_new = old.yaml["nc"], len(names)
    assert nc_new == nc_old + 1, (nc_old, nc_new)
    cfg = copy.deepcopy(old.yaml)
    cfg["nc"] = nc_new
    torch.manual_seed(seed)
    new = DetectionModel(cfg, nc=nc_new, verbose=False)
    so, sn = old.state_dict(), new.state_dict()
    copied, partial, fresh = [], [], []
    for k, v in sn.items():
        o = so.get(k)
        if o is None:
            fresh.append(k)
        elif o.shape == v.shape:
            sn[k] = o.clone()
            copied.append(k)
        elif v.shape[0] == nc_new and o.shape[0] == nc_old and v.shape[1:] == o.shape[1:]:
            sn[k][:nc_old] = o
            partial.append(k)
        else:
            raise RuntimeError(f"{k}: shape {tuple(o.shape)} -> {tuple(v.shape)} không copy được (kiến trúc head đổi theo nc?)")
    assert partial and all("cv3" in k for k in partial), f"tensor theo class không như mong đợi: {partial}"
    assert not fresh, f"tensor mới không có trong base: {fresh}"
    new.load_state_dict(sn)
    new.names = dict(enumerate(names))
    new.args = getattr(old, "args", {})
    y.model = new
    y.save(str(out_pt))
    return {"nc_old": nc_old, "nc_new": nc_new, "seed": seed, "n_copied": len(copied),
            "partial_rows_copied": partial, "new_class_init": "DetectionModel default init + bias_init, torch.manual_seed(seed)"}


@torch.no_grad()
def check_expansion(base_pt: Path, exp_pt: Path, imgsz: int = 320) -> float:
    """Đầu ra box và điểm class cũ của model mở rộng phải trùng model base. Trả sai khác lớn nhất."""
    from ultralytics import YOLO
    a, b = YOLO(str(base_pt)).model.float().eval(), YOLO(str(exp_pt)).model.float().eval()
    x = torch.rand(1, 3, imgsz, imgsz, generator=torch.Generator().manual_seed(0))
    ya, yb = a(x), b(x)
    ya, yb = (ya[0] if isinstance(ya, (list, tuple)) else ya), (yb[0] if isinstance(yb, (list, tuple)) else yb)
    n = ya.shape[1]
    assert yb.shape[1] == n + 1, (ya.shape, yb.shape)
    diff = float((yb[:, :n] - ya).abs().max())
    assert diff < 1e-4, f"model mở rộng lệch base: {diff}"
    return diff


def train(init_pt: Path, data_yaml: Path, out_dir: Path, args: dict, seed: int, imgsz: int) -> dict:
    """Train trong SCRATCH, copy weights/last.pt, best.pt, results.csv sang out_dir. Có out_dir/DONE thì bỏ qua."""
    from ultralytics import YOLO
    out_dir = Path(out_dir)
    if (out_dir / "DONE").exists():
        return {"skipped": True}
    from .common import SCRATCH
    project = SCRATCH / "runs"
    t0 = time.time()
    YOLO(str(init_pt)).train(data=str(data_yaml), project=str(project), name=out_dir.name, exist_ok=True,
                             seed=seed, deterministic=True, imgsz=imgsz, **args)
    secs = time.time() - t0
    run = project / out_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in ("weights/last.pt", "weights/best.pt", "results.csv", "args.yaml"):
        if (run / f).exists():
            shutil.copy(run / f, out_dir / Path(f).name)
    (out_dir / "DONE").write_text(str(secs))
    return {"seconds": secs}


def eval_ap(weights: Path, data_yaml: Path, split: str, imgsz: int, batch: int) -> dict:
    """AP theo TÊN class (không theo index) để so được giữa taxonomy base và base+novel."""
    from ultralytics import YOLO
    r = YOLO(str(weights)).val(data=str(data_yaml), split=split, imgsz=imgsz, batch=batch, plots=False, verbose=False)
    names = r.names
    return {names[int(c)]: {"ap50_95": float(r.box.ap[j]), "ap50": float(r.box.ap50[j])}
            for j, c in enumerate(r.box.ap_class_index)}


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lt, rb = np.maximum(a[:, None, :2], b[None, :, :2]), np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


def match_gt(gt: np.ndarray, pred: np.ndarray, pred_conf: np.ndarray, iou_thr: float) -> np.ndarray:
    """Greedy theo conf giảm dần: mỗi prediction khớp tối đa một GT (IoU cao nhất, chưa khớp). Trả cờ hit cho từng GT."""
    hit = np.zeros(len(gt), bool)
    if not len(gt) or not len(pred):
        return hit
    iou = _iou(pred[np.argsort(-pred_conf)], gt)
    for row in iou:
        row = np.where(hit, -1, row)
        j = int(row.argmax())
        if row[j] >= iou_thr:
            hit[j] = True
    return hit


def recall_at_conf(weights: Path, ids: list[str], paths: dict, gt: pd.DataFrame, cls: str,
                   conf: float, iou_thr: float, imgsz: int, batch: int) -> pd.DataFrame:
    """Recall của class `cls` ở ngưỡng conf cố định. Trả GT boxes của `cls` kèm cột `hit`."""
    from ultralytics import YOLO
    m = YOLO(str(weights))
    cid = {v: k for k, v in m.names.items()}[cls]
    g = gt[gt.cls == cls]
    by = {k: v for k, v in g.groupby("image")}
    rows = []
    want = [i for i in ids if i in by]
    for s in range(0, len(want), batch):
        chunk = want[s:s + batch]
        for i, r in zip(chunk, m.predict([str(paths[i]) for i in chunk], conf=conf, imgsz=imgsz, verbose=False)):
            keep = r.boxes.cls.cpu().numpy().astype(int) == cid
            b = by[i].copy()
            b["hit"] = match_gt(b[["x1", "y1", "x2", "y2"]].to_numpy(float), r.boxes.xyxy.cpu().numpy()[keep],
                                r.boxes.conf.cpu().numpy()[keep], iou_thr)
            rows.append(b)
    return pd.concat(rows) if rows else g.assign(hit=False)
