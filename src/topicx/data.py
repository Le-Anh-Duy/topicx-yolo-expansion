"""BDD100K: tìm file trên Kaggle, đọc nhãn, EDA, chia tập không giao nhau, kiểm tra, xuất định dạng YOLO."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

BOX_COLS = ["image", "cls", "x1", "y1", "x2", "y2"]
SPLITS = ("test", "dev", "pool", "base")


# ---------- đọc dữ liệu ----------

LABEL_NAMES = ("bdd100k_labels_images_{s}.json", "det_{s}.json", "det_v2_{s}_release.json")


def locate_bdd(root: Path, max_depth: int = 9) -> tuple[dict, dict]:
    """Tìm file nhãn {train,val} và thư mục ảnh 100k/{train,val}; đi theo symlink (Kaggle mount), không đi sâu vào thư mục ảnh."""
    labels, images = {}, {}
    root = Path(root)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        for f in filenames:
            for s in ("train", "val"):
                if f in [n.format(s=s) for n in LABEL_NAMES]:
                    labels.setdefault(s, Path(dirpath) / f)
        if len(filenames) > 2000:
            name = Path(dirpath).name
            if name in ("train", "val") and "100k" in dirpath and any(f.endswith(".jpg") for f in filenames[:50]):
                images.setdefault(name, Path(dirpath))
            dirnames[:] = []
        if len(Path(dirpath).relative_to(root).parts) >= max_depth:
            dirnames[:] = []
    return labels, images


def load_labels(path: Path, src: str, alias: dict, keep: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Đọc JSON nhãn BDD (định dạng 2018 hoặc det_20). Trả (images, boxes); chỉ giữ box2d của class trong `keep`."""
    with open(path, encoding="utf-8") as f:
        recs = json.load(f)
    keep = set(keep)
    imgs, boxes = [], []
    for r in recs:
        a = r.get("attributes") or {}
        imgs.append((r["name"], src, a.get("timeofday", "undefined"), a.get("weather", "undefined"), a.get("scene", "undefined")))
        for lab in r.get("labels") or []:
            b, c = lab.get("box2d"), alias.get(lab.get("category"), lab.get("category"))
            if b and c in keep:
                boxes.append((r["name"], c, b["x1"], b["y1"], b["x2"], b["y2"]))
    images = pd.DataFrame(imgs, columns=["image", "src", "timeofday", "weather", "scene"])
    # Tên ảnh BDD = tên video "<ride>-<clip>"; các clip cùng phần đầu thường từ cùng một lần ghi -> dùng làm group.
    images["group"] = images.image.str.split("-").str[0]
    return images, pd.DataFrame(boxes, columns=BOX_COLS)


def image_paths(images: pd.DataFrame, image_dirs: dict) -> dict[str, Path]:
    return {i: Path(image_dirs[s]) / i for i, s in zip(images.image, images.src)}


# ---------- EDA ----------

def size_bucket(boxes: pd.DataFrame) -> pd.Series:
    """COCO: small < 32², medium < 96², large còn lại (theo pixel ảnh gốc)."""
    area = (boxes.x2 - boxes.x1) * (boxes.y2 - boxes.y1)
    return pd.cut(area, [-np.inf, 32 ** 2, 96 ** 2, np.inf], labels=["small", "medium", "large"]).astype(str)


def class_stats(images: pd.DataFrame, boxes: pd.DataFrame) -> pd.DataFrame:
    b = boxes.assign(size=size_bucket(boxes)).merge(images[["image", "timeofday"]], on="image")
    g = b.groupby("cls")
    t = pd.DataFrame({"n_images": g.image.nunique(), "n_instances": g.size()})
    t = t.join(pd.crosstab(b.cls, b["size"], normalize="index").add_prefix("frac_"))
    t["frac_night"] = g.timeofday.apply(lambda s: (s == "night").mean())
    return t.sort_values("n_instances")


def removal_cost(boxes: pd.DataFrame, candidates: list[str]) -> pd.DataFrame:
    """Tỉ lệ instance của mỗi class bị mất khỏi base train nếu loại mọi ảnh chứa novel candidate (co-occurrence)."""
    total = boxes.groupby("cls").size()
    rows = {}
    for c in candidates:
        has = set(boxes.image[boxes.cls == c])
        rows[c] = boxes[boxes.image.isin(has)].groupby("cls").size().reindex(total.index, fill_value=0) / total
    return pd.DataFrame(rows).T.round(3)


def class_names(classes: list[str], novel: str) -> tuple[list[str], list[str]]:
    """Base giữ thứ tự config (id 0..B-1); novel luôn là id cuối B."""
    assert novel in classes, novel
    base = [c for c in classes if c != novel]
    return base, base + [novel]


# ---------- chia tập ----------

def make_splits(images: pd.DataFrame, has_novel: set[str], sizes: dict, seed: int,
                test_source: str | None = None, n_weak: int = 0) -> dict[str, list[str]]:
    """Chia theo group (không group nào nằm ở 2 tập). test/dev/pool lấy nguyên group (giữ cả ảnh positive và negative);
    base lấy từ các group còn lại nhưng bỏ ảnh có novel class.
    `test_source="val"`: test chỉ lấy group toàn ảnh BDD val; dev/pool/base chỉ lấy group không có ảnh val.
    `n_weak` > 0 (kịch bản weak): base giữ đúng n_weak ảnh có novel đầu tiên gặp, bỏ các ảnh novel còn lại."""
    groups = images.groupby("group").image.apply(sorted)
    order = list(np.random.default_rng(seed).permutation(sorted(groups.index)))
    if test_source:
        srcs = images.groupby("group").src.agg(set)
        test_order = [g for g in order if srcs[g] == {test_source}]
        order = [g for g in order if test_source not in srcs[g]]
    else:
        test_order = order
    used: set = set()

    def take(cands, n, keep=lambda i: True):
        got = []
        for g in cands:
            if len(got) >= n:
                break
            if g not in used:
                used.add(g)
                got += [i for i in groups[g] if keep(i)]
        assert len(got) >= n, f"không đủ ảnh: cần {n}, có {len(got)}"
        return got

    out = {"test": take(test_order, sizes["test"])}
    out["dev"] = take(order, sizes["dev"])
    out["pool"] = take(order, sizes["pool"])
    quota = [n_weak]

    def keep_base(i):
        if i not in has_novel:
            return True
        quota[0] -= 1
        return quota[0] >= 0
    out["base"] = take(order, sizes["base"], keep=keep_base)
    return out


def check_splits(splits: dict, images: pd.DataFrame, has_novel: set[str], n_weak: int = 0) -> dict:
    """Kiểm tra không giao nhau theo ảnh và group; base có đúng n_weak ảnh novel (missing: 0); pool có cả positive và negative."""
    group = dict(zip(images.image, images.group))
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not set(splits[a]) & set(splits[b]), f"ảnh trùng giữa {a} và {b}"
            ga, gb = {group[x] for x in splits[a]}, {group[x] for x in splits[b]}
            assert not ga & gb, f"group trùng giữa {a} và {b}"
    n_base_novel = len(set(splits["base"]) & has_novel)
    assert n_base_novel == n_weak, f"base train có {n_base_novel} ảnh novel, cần đúng {n_weak}"
    n_pos = len(set(splits["pool"]) & has_novel)
    assert 0 < n_pos < len(splits["pool"]), "pool phải có cả ảnh positive và negative"
    return {s: {"n_images": len(v), "n_novel_images": len(set(v) & has_novel)} for s, v in splits.items()}


def md5_duplicates(splits: dict, paths: dict[str, Path]) -> list[tuple]:
    """Exact duplicate (md5 nội dung file) xuất hiện ở hơn một tập."""
    seen = {}
    for s, ids in splits.items():
        for i in ids:
            seen.setdefault(hashlib.md5(Path(paths[i]).read_bytes()).hexdigest(), []).append((s, i))
    return [v for v in seen.values() if len({s for s, _ in v}) > 1]


# ---------- YOLO ----------

def yolo_line(c: int, x1, y1, x2, y2, w: int, h: int) -> str | None:
    x1, x2 = np.clip([x1, x2], 0, w)
    y1, y2 = np.clip([y1, y2], 0, h)
    if x2 - x1 <= 1 or y2 - y1 <= 1:
        return None
    return f"{c} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"


def write_yolo(dst: Path, ids: list[str], boxes: pd.DataFrame, names: list[str], paths: dict, img_wh) -> Path:
    """Thư mục YOLO: images/ là symlink tới /kaggle/input (read-only), labels/ ghi theo `names`.
    Box của class ngoài `names` bị bỏ (vd. novel khi xuất theo base taxonomy). Xoá thư mục cũ trước."""
    dst = Path(dst)
    shutil.rmtree(dst, ignore_errors=True)
    (dst / "images").mkdir(parents=True)
    (dst / "labels").mkdir()
    idx = {n: i for i, n in enumerate(names)}
    b = boxes[boxes.image.isin(set(ids)) & boxes.cls.isin(idx)]
    by = {k: g for k, g in b.groupby("image")}
    w, h = img_wh
    for i in ids:
        os.symlink(paths[i], dst / "images" / i)
        g = by.get(i)
        lines = [] if g is None else [yolo_line(idx[r.cls], r.x1, r.y1, r.x2, r.y2, w, h) for r in g.itertuples()]
        (dst / "labels" / (Path(i).stem + ".txt")).write_text("\n".join(x for x in lines if x))
    return dst


def write_data_yaml(path: Path, names: list[str], train: Path, val: Path, test: Path | None = None) -> Path:
    d = {"train": str(Path(train) / "images"), "val": str(Path(val) / "images"), "names": dict(enumerate(names))}
    if test is not None:
        d["test"] = str(Path(test) / "images")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(d, allow_unicode=True), encoding="utf-8")
    return path


def describe_tree(root: Path, depth: int = 5, max_entries: int = 12) -> str:
    """Cây thư mục rút gọn (đếm file, liệt kê .json) để chẩn đoán khi không tìm thấy BDD."""
    lines = []
    root = Path(root)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        rel = Path(dirpath).relative_to(root)
        if len(rel.parts) > depth:
            dirnames[:] = []
            continue
        js = [f for f in filenames if f.endswith(".json")][:max_entries]
        lines.append(f"{'  ' * len(rel.parts)}{rel.name or str(root)}/  ({len(filenames)} file{', json: ' + ', '.join(js) if js else ''})")
        if len(filenames) > 2000:
            dirnames[:] = []
        dirnames[:] = sorted(dirnames)[:max_entries]
    return "\n".join(lines[:300])


def bdd_locations(cfg: dict) -> tuple[dict, dict]:
    """(file nhãn {train,val}, thư mục ảnh {train,val}): lấy từ cfg["bdd_paths"] nếu khai báo, không thì tự dò /kaggle/input."""
    from .common import INPUT
    over = {}
    for k, p in (cfg.get("bdd_paths") or {}).items():
        if p and Path(p).exists():
            over[k] = p
        elif p:
            print(f"bdd_paths.{k} = {p} không tồn tại -> tự dò")
    labels, dirs = locate_bdd(INPUT) if len(over) < 4 else ({}, {})
    slug = cfg.get("bdd_dataset")
    if (set(labels) != {"train", "val"} or set(dirs) != {"train", "val"}) and slug:
        # Trong Kaggle notebook, kagglehub tự gắn dataset vào notebook (panel Input) và đọc từ cache dùng chung.
        import kagglehub
        root = Path(kagglehub.dataset_download(slug))
        print(f"kagglehub: {slug} -> {root}")
        found_l, found_d = locate_bdd(root)
        labels, dirs = {**found_l, **labels}, {**found_d, **dirs}
    for s in ("train", "val"):
        if over.get(f"labels_{s}"):
            labels[s] = Path(over[f"labels_{s}"])
        if over.get(f"images_{s}"):
            dirs[s] = Path(over[f"images_{s}"])
    if set(labels) != {"train", "val"} or set(dirs) != {"train", "val"}:
        print(describe_tree(INPUT))
        raise FileNotFoundError(
            f"không tìm đủ BDD100K: nhãn={labels} ảnh={dirs}. Cần ảnh images/100k/{{train,val}} và nhãn detection "
            f"({', '.join(n.format(s='{train,val}') for n in LABEL_NAMES)}). Gắn đúng dataset, hoặc khai báo bdd_paths "
            f"trong configs/exp.yaml / gán cfg['bdd_paths'] trong notebook. Cây /kaggle/input in ở trên.")
    return labels, dirs


def split_paths(cfg: dict) -> dict[str, Path]:
    """Đường dẫn ảnh của mọi ảnh trong các tập (từ splits/image_src.csv + thư mục BDD đang gắn)."""
    from .common import find
    src = pd.read_csv(find(cfg, "splits/image_src.csv"))
    return image_paths(src, bdd_locations(cfg)[1])
