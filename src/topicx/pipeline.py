"""Harness đánh giá (block 1 phần chia tập, block 3 finetune + đánh giá). Không chứa logic chọn mẫu: tập frame id đến từ
block 2 (`topicx.proposals` + `src.rav`) hoặc từ P-026 / nơi khác, dạng `proposals/<tên>/s<seed>_k<K>/export.csv`.
Bước nào đã có kết quả (trong /kaggle/working sau `common.sync_inputs`) thì bỏ qua → chạy lại notebook = chạy tiếp.

    splits -> base model -> [export.csv của từng cách chọn] -> chốt manifest -> oracle trả nhãn BDD
    -> (mở rộng head) + finetune (replay + K) -> đánh giá trên final test -> so với base model, `random` cùng seed / K, REPLAY_ONLY

Kịch bản (`cfg["scenario"]`):
  - missing: base train không có ảnh novel; base model chỉ biết base classes; mở rộng head trước khi finetune.
  - weak: base train có đúng `weak_novel_images` ảnh novel; base model biết đủ class (novel yếu); finetune thẳng từ base.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from . import common as C
from . import data as D
from . import metrics as M
from . import select as S
from .oracle import OracleStore

CONTROL = "REPLAY_ONLY"
PRIVILEGED = ("ORACLE_POSITIVE",)


def _art(cfg, rel) -> Path:
    return C.WORK / cfg["art"] / rel


def _sel(cfg, s, k, b) -> Path:
    return C.out(cfg, f"selections/s{s}_k{k}_{b}.json")


def _meta(cfg) -> dict:
    return C.load(_art(cfg, "splits/meta.json"))


def _weak(cfg) -> bool:
    return cfg["scenario"] == "weak"


def _n_weak(cfg) -> int:
    return cfg["weak_novel_images"] if _weak(cfg) else 0


def _base_taxonomy(meta: dict) -> list[str]:
    """Class mà base model được train: missing -> base classes; weak -> đủ base + novel."""
    return meta["names"] if meta["scenario"] == "weak" else meta["base_names"]


def _oracle(cfg, stage) -> OracleStore:
    return OracleStore(_art(cfg, "splits/oracle"), C.out(cfg, f"{stage}/oracle_log.jsonl"))


# ---------- 1. dữ liệu + chia tập ----------

def load_bdd(cfg):
    t0 = time.time()
    labels, dirs = D.bdd_locations(cfg)
    print("nhãn:", labels, "\nảnh:", dirs)
    parts = []
    for s in ("train", "val"):
        parts.append(D.load_labels(labels[s], s, cfg["alias"], cfg["classes"]))
        print(f"[{time.time() - t0:.0f}s] đọc nhãn {s}: {len(parts[-1][0])} ảnh")
    images = pd.concat([p[0] for p in parts], ignore_index=True)
    boxes = pd.concat([p[1] for p in parts], ignore_index=True)
    assert images.image.is_unique
    paths = D.image_paths(images, dirs)
    missing = set(images.image) - set(paths)  # image_paths chỉ chứa ảnh có file (duyệt đệ quy thư mục ảnh)
    print(f"[{time.time() - t0:.0f}s] kiểm file ảnh xong")
    assert len(missing) <= 0.01 * len(images), \
        f"{len(missing)}/{len(images)} ảnh có nhãn không có file trong {dirs} — sai thư mục ảnh (vd. bdd100k_seg thay vì images/100k)?"
    print(f"{len(images)} ảnh có nhãn, {len(missing)} không có file ảnh -> bỏ")
    images = images[~images.image.isin(missing)].reset_index(drop=True)
    boxes = boxes[boxes.image.isin(set(images.image))].reset_index(drop=True)
    from PIL import Image
    sizes = {Image.open(paths[i]).size for i in images.image.sample(min(200, len(images)), random_state=0)}
    assert sizes == {tuple(cfg["img_wh"])}, sizes
    return images, boxes, paths


def _diagnose_images(dirs: dict, labels: dict) -> None:
    """In thư mục con của thư mục ảnh và dò vị trí thật của ảnh trong record đầu file nhãn train."""
    import json
    for s, d in dirs.items():
        subs = [(e.name, sum(1 for f in os.scandir(e.path) if f.name.endswith(".jpg"))) for e in os.scandir(d) if e.is_dir()]
        print(f"[chẩn đoán] {d}: thư mục con {subs[:20] or 'không có'}")
    with open(labels["train"], "rb") as f:
        head = f.read(8_000_000).decode("utf-8", errors="ignore")
    name = json.JSONDecoder().raw_decode(head, head.index("{"))[0]["name"]
    print(f"[chẩn đoán] dò '{name}' trong {C.INPUT} ...", flush=True)
    t0, hits, jpg_dirs = time.time(), [], []
    for dirpath, dirnames, filenames in os.walk(C.INPUT, followlinks=True):
        if name in filenames:
            hits.append(dirpath)
        n = sum(f.endswith(".jpg") for f in filenames)
        if n > 500:
            jpg_dirs.append((dirpath, n))
        if time.time() - t0 > 300:
            print("[chẩn đoán] dừng sau 5 phút")
            break
    print(f"[chẩn đoán] '{name}' nằm ở: {hits or 'KHÔNG TÌM THẤY trong mọi input'}")
    print("[chẩn đoán] mọi thư mục có > 500 jpg:")
    for p, n in sorted(jpg_dirs):
        print(f"    {n:>7}  {p}")


def health_check(cfg, n_sample: int = 20):
    """Kiểm nhanh (vài giây, không đọc cả file nhãn) xem đang đọc đúng BDD100K. Trả (bảng kiểm, ảnh mẫu có vẽ box).
    Lỗi nghiêm trọng -> AssertionError kèm bảng đã in."""
    import json
    from PIL import Image, ImageDraw
    roots = [Path(r) for r in cfg.get("bdd_roots") or [] if Path(r).exists()]
    root = roots[0] if roots else C.INPUT
    print(f"Cây thư mục {root}:")
    print(D.describe_tree(root), "\n", flush=True)
    labels, dirs = D.bdd_locations(cfg)
    rows, sample = [], None
    expect_imgs = {"train": 70000, "val": 10000}
    for s in ("train", "val"):
        names = sorted(D.index_images(dirs[s]))  # đệ quy: tính cả ảnh trong thư mục con
        rows.append({"kiểm": f"ảnh {s}", "giá trị": f"{len(names)} jpg trong {dirs[s]}", "ok": len(names) >= 0.9 * expect_imgs[s],
                     "kỳ vọng": f"~{expect_imgs[s]}"})
        size_gb = labels[s].stat().st_size / 1e9
        rows.append({"kiểm": f"file nhãn {s}", "giá trị": f"{labels[s].name}, {size_gb:.2f} GB", "ok": size_gb > 0.01,
                     "kỳ vọng": "JSON nhãn gốc BDD (hàng trăm MB)"})
        # đọc record đầu tiên mà không load cả file
        with open(labels[s], "rb") as f:
            head = f.read(8_000_000).decode("utf-8", errors="ignore")
        try:
            rec, _ = json.JSONDecoder().raw_decode(head, head.index("{"))
            box_labels = [lab for lab in rec.get("labels") or [] if lab.get("box2d")]
            fmt_ok = isinstance(rec.get("name"), str) and "attributes" in rec and "labels" in rec
        except (ValueError, KeyError) as e:
            rec, box_labels, fmt_ok = None, [], False
            print(f"không đọc được record đầu của {labels[s]}: {e}")
        rows.append({"kiểm": f"định dạng nhãn {s}", "giá trị": f"record đầu: {rec.get('name') if rec else None}, "
                     f"{len(box_labels)} box2d, category: {sorted({lab['category'] for lab in box_labels})[:6]}",
                     "ok": fmt_ok, "kỳ vọng": "name + attributes + labels[].box2d"})
        known = set(cfg["classes"]) | set(cfg["alias"]) | {"train"}
        cats = {lab["category"] for lab in box_labels}
        rows.append({"kiểm": f"category {s}", "giá trị": f"lạ: {sorted(cats - known)}", "ok": not cats - known - {"other vehicle", "other person", "trailer"},
                     "kỳ vọng": "nằm trong classes/alias của config"})
        if rec:
            p = D.index_images(dirs[s]).get(rec["name"], Path(dirs[s]) / rec["name"])
            ok = p.exists()
            wh = Image.open(p).size if ok else None
            rows.append({"kiểm": f"ảnh của record đầu {s}", "giá trị": f"{p.name} tồn tại={ok}, size={wh}", "ok": ok and wh == tuple(cfg["img_wh"]),
                         "kỳ vọng": f"tồn tại, {tuple(cfg['img_wh'])}"})
            if ok and sample is None:
                img = Image.open(p).convert("RGB")
                dr = ImageDraw.Draw(img)
                for lab in box_labels:
                    b = lab["box2d"]
                    dr.rectangle([b["x1"], b["y1"], b["x2"], b["y2"]], outline=(255, 0, 0), width=3)
                    dr.text((b["x1"] + 3, b["y1"] + 2), lab["category"], fill=(255, 255, 0))
                sample = img
        # tên ảnh ngẫu nhiên đúng dạng <ride>-<clip>.jpg (dùng làm group)
        pick = names[:: max(1, len(names) // n_sample)][:n_sample]
        good = sum(len(n[:-4].split("-")) == 2 for n in pick)
        rows.append({"kiểm": f"tên ảnh {s}", "giá trị": f"{good}/{len(pick)} dạng <ride>-<clip>.jpg, vd {pick[:2]}", "ok": good == len(pick),
                     "kỳ vọng": "group theo ride chia tập được"})
    rep = pd.DataFrame(rows)
    with pd.option_context("display.max_colwidth", 200):
        print(rep.to_string(index=False))
    if not rep.ok.all():
        _diagnose_images(dirs, labels)
    assert rep.ok.all(), "health check có mục FAIL (xem bảng trên)"
    print("health check OK: đúng BDD100K 100k + nhãn JSON gốc")
    return rep, sample


def eda(cfg, images, boxes) -> dict:
    stats = D.class_stats(images, boxes)
    n = len(images)
    proj = pd.DataFrame({c: {"pool_pos_images": stats.n_images[c] / n * cfg["sizes"]["pool"],
                             "test_instances": stats.n_instances[c] / n * cfg["sizes"]["test"],
                             **{f"random_K{k}_pos_images": stats.n_images[c] / n * k for k in cfg["ks"]}}
                         for c in cfg["novel_candidates"]}).T.round(1)
    return {"class_stats": stats, "removal_cost": D.removal_cost(boxes, cfg["novel_candidates"]),
            "timeofday": images.timeofday.value_counts(), "projection (ước tính theo tỉ lệ toàn tập)": proj}


def prepare_splits(cfg, novel: str, reason: str, images=None, boxes=None, paths=None) -> dict:
    mp = _art(cfg, "splits/meta.json")
    if mp.exists():
        meta = C.load(mp)
        assert meta["novel"] == novel, f"splits đã có với novel={meta['novel']}; bỏ input đó hoặc đặt NOVEL khớp"
        assert meta["scenario"] == cfg["scenario"], f"splits trong input thuộc kịch bản {meta['scenario']}"
        return meta
    assert novel in cfg["novel_candidates"] and reason, "đặt NOVEL và REASON"
    if images is None:
        images, boxes, paths = load_bdd(cfg)
    has_novel = set(boxes.image[boxes.cls == novel])
    splits = D.make_splits(images, has_novel, cfg["sizes"], cfg["seed_split"], cfg.get("test_source"), _n_weak(cfg))
    report = D.check_splits(splits, images, has_novel, _n_weak(cfg))
    if cfg["md5_check"]:
        print("kiểm md5 exact duplicate giữa các tập (đọc toàn bộ ảnh, có thể mất hàng chục phút)...")
    dups = D.md5_duplicates(splits, paths) if cfg["md5_check"] else None
    assert not dups, dups[:5]
    base_names, names = D.class_names(cfg["classes"], novel)
    base_sorted = sorted(splits["base"])
    perm = np.random.default_rng(cfg["seed_split"]).permutation(len(base_sorted))
    replay = [base_sorted[i] for i in perm[:cfg["replay"]]]
    control = [base_sorted[i] for i in perm[cfg["replay"]:cfg["replay"] + max(cfg["ks"])]]
    assert len(control) == max(cfg["ks"]), "base train quá nhỏ cho replay + control"
    test_novel = int(((boxes.cls == novel) & boxes.image.isin(set(splits["test"]))).sum())
    for s, ids in splits.items():
        C.write_ids(ids, C.out(cfg, f"splits/{s}_ids.txt"))
    C.write_ids(replay, C.out(cfg, "splits/replay_ids.txt"))
    C.write_ids(control, C.out(cfg, "splits/control_ids.txt"))
    all_ids = {i for v in splits.values() for i in v}
    pool = set(splits["pool"])
    pub = all_ids - pool
    images[images.image.isin(all_ids)][["image", "src"]].to_csv(C.out(cfg, "splits/image_src.csv"), index=False)
    images[images.image.isin(pub)].to_csv(C.out(cfg, "splits/public_images.csv"), index=False)
    boxes[boxes.image.isin(pub)].to_csv(C.out(cfg, "splits/public_boxes.csv"), index=False)
    # Nhãn + thuộc tính ảnh của pool chỉ nằm trong oracle/
    images[images.image.isin(pool)].to_csv(C.out(cfg, "splits/oracle/pool_images.csv"), index=False)
    boxes[boxes.image.isin(pool)].to_csv(C.out(cfg, "splits/oracle/pool_boxes.csv"), index=False)
    C.write_ids(splits["pool"], C.out(cfg, "splits/oracle/pool_ids.txt"))
    meta = {"novel": novel, "reason": reason, "scenario": cfg["scenario"], "n_weak": _n_weak(cfg),
            "base_novel_instances": int(((boxes.cls == novel) & boxes.image.isin(set(splits["base"]))).sum()),
            "base_names": base_names, "names": names, "sizes": cfg["sizes"],
            "test_source": cfg.get("test_source"), "seed_split": cfg["seed_split"], "report": report,
            "test_novel_instances": test_novel, "md5_checked": cfg["md5_check"]}
    C.dump(meta, mp)
    if test_novel < 300 and not cfg["smoke"]:
        print("CẢNH BÁO: < 300 novel instances ở test -> AP novel dao động mạnh giữa seed")
    return meta


# ---------- 2. base model ----------

def train_base(cfg) -> tuple[Path, dict]:
    from . import yolo as Y
    run, mp = _art(cfg, "base/run"), _art(cfg, "base/meta.json")
    W = run / f"{cfg['yolo']['weights']}.pt"
    if mp.exists():
        return W, C.load(mp)
    meta = _meta(cfg)
    base_names = _base_taxonomy(meta)  # weak: đủ class
    paths = D.split_paths(cfg)
    pub = pd.read_csv(_art(cfg, "splits/public_boxes.csv"))
    ids = {s: C.read_ids(_art(cfg, f"splits/{s}_ids.txt")) for s in ("base", "dev")}
    n_nov = len(set(pub.image[pub.cls == meta["novel"]]) & set(ids["base"]))
    assert n_nov == meta["n_weak"], f"base train có {n_nov} ảnh novel, cần {meta['n_weak']}"
    yd = C.SCRATCH / "yolo"
    tr = D.write_yolo(yd / "base_train", ids["base"], pub, base_names, paths, cfg["img_wh"])
    dv = D.write_yolo(yd / "dev_base", ids["dev"], pub, base_names, paths, cfg["img_wh"])
    yml = D.write_data_yaml(yd / "base.yaml", base_names, tr, dv)
    Y.train(cfg["yolo"]["init"], yml, run, cfg["yolo"]["base"], seed=cfg["seed_split"], imgsz=cfg["yolo"]["imgsz"])
    from ultralytics import YOLO
    assert list(YOLO(str(W)).names.values()) == base_names
    secs = float((run / "DONE").read_text())
    bm = {"seconds": secs, "sec_per_img_epoch": secs / (len(ids["base"]) * cfg["yolo"]["base"]["epochs"]),
          "dev_ap": Y.eval_ap(W, yml, "val", cfg["yolo"]["imgsz"], cfg["eval"]["batch"]), "weights": W.name}
    C.dump(bm, mp)
    return W, bm


def estimate_hours(cfg, n_branches: int, seeds, ks) -> float:
    per = C.load(_art(cfg, "base/meta.json"))["sec_per_img_epoch"]
    return 1.3 * sum(per * (cfg["replay"] + k) * cfg["yolo"]["finetune"]["epochs"] for k in ks) * n_branches * len(seeds) / 3600


# ---------- 3. nhận kết quả proposal (block 2 / P-026 / nơi khác) ----------

def import_exports(cfg, names=None, seeds=None, ks=None) -> pd.DataFrame:
    """Mỗi `proposals/<tên>/s<seed>_k<K>/export.csv` (cột `unit_id` kiểu P-026, hoặc `image`) → manifest đã chốt
    `selections/s<seed>_k<K>_<tên>.json`. Kiểm: id thuộc pool, không trùng, ≤ K. Ít hơn K → bù ảnh control từ base train
    (`pad_ids`, giữ cùng số ảnh train / số update) và ghi `shortfall`. Manifest đã có thì giữ (đã chốt)."""
    from .proposals import image_of, list_exports
    pool = set(C.read_ids(_art(cfg, "splits/pool_ids.txt")))
    control = C.read_ids(_art(cfg, "splits/control_ids.txt"))
    ex = list_exports(cfg)
    if ex.empty:
        raise FileNotFoundError(f"không có {cfg['art']}/proposals/*/s*_k*/export.csv — chạy block 2 hoặc gắn output chứa nó")
    rows = []
    for r in ex.itertuples():
        if (names and r.name not in names) or (seeds and r.seed not in seeds) or (ks and r.k not in ks):
            continue
        assert r.name != CONTROL and r.name not in PRIVILEGED, f"tên nhánh {r.name} dành riêng"
        df = pd.read_csv(r.path)
        ids = [image_of(u) for u in df.unit_id] if "unit_id" in df else df.image.astype(str).tolist()
        assert len(ids) == len(set(ids)), f"{r.path}: id trùng"
        assert len(ids) <= r.k, f"{r.path}: {len(ids)} id > K={r.k}"
        bad = set(ids) - pool
        assert not bad, f"{r.path}: {len(bad)} id không thuộc pool, vd {sorted(bad)[:3]}"
        p = _sel(cfg, r.seed, r.k, r.name)
        if not p.exists():
            C.freeze(p, ids, branch=r.name, seed=r.seed, k=r.k, source=str(r.path), shortfall=r.k - len(ids),
                     pad_ids=control[:r.k - len(ids)])
        rows.append({"branch": r.name, "seed": r.seed, "k": r.k, "n": len(ids), "shortfall": r.k - len(ids)})
    return pd.DataFrame(rows)


def oracle_positive(cfg, seeds, ks) -> None:
    """Nhánh tham chiếu ORACLE_POSITIVE: ngẫu nhiên trong ảnh pool có novel (dùng nhãn ẩn, ghi log); thiếu thì bù negative ngẫu nhiên."""
    oracle = _oracle(cfg, "selection")
    novel = _meta(cfg)["novel"]
    pool = C.read_ids(_art(cfg, "splits/pool_ids.txt"))
    pos = oracle.privileged_positive_ids(novel, reason="ORACLE_POSITIVE")
    neg = sorted(set(pool) - set(pos))
    for s in seeds:
        for k in ks:
            ids = S.random_k(pos, k, 1000 + s)
            pad = k - len(ids)
            C.freeze(_sel(cfg, s, k, "ORACLE_POSITIVE"), ids + (S.random_k(neg, pad, 2000 + s) if pad else []),
                     branch="ORACLE_POSITIVE", seed=s, k=k, privileged=True, padded_negatives=pad)


def selection_eval(cfg) -> pd.DataFrame:
    """Simulated annotation + thống kê MỌI manifest đã chốt (mở nhãn sau khi chốt): precision@K, recall@K, instance novel,
    kích thước, timeofday; độ trùng (DINOv2) nếu field đã có trong cache của block 1."""
    oracle = _oracle(cfg, "selection")
    novel = _meta(cfg)["novel"]
    n_pos = len(oracle.privileged_positive_ids(novel, reason="recall@K denominator"))
    emb = None
    try:
        from .proposals import image_of, make_ctx
        ctx = make_ctx(cfg, "pool")
        t = ctx.cached("emb.dinov2_s", ctx.units)
        if t is not None:
            emb = dict(zip((image_of(u) for u in t.unit_ids), np.asarray(t.values)))
    except Exception as e:  # không có cache field -> bỏ cột độ trùng
        print("bỏ qua độ trùng:", e)
    rows = []
    for p in sorted(_art(cfg, "selections").glob("*.json")):
        m = C.read_frozen(p)
        bx, im = oracle.reveal(p)
        row = {"seed": m["seed"], "k": m["k"], "branch": m["branch"], "shortfall": m.get("shortfall", 0),
               **M.retrieval_report(m["ids"], bx, im, novel, n_pos)}
        if emb is not None and m["ids"]:
            row.update(M.batch_redundancy(np.stack([emb[i] for i in m["ids"]])))
        rows.append(row)
    df = pd.DataFrame(rows).fillna(0)
    df.to_csv(C.out(cfg, "selection/selection.csv"), index=False)
    return df



# ---------- 4. mở rộng head + finetune ----------

def train_branches(cfg, branches, seeds, ks) -> list[str]:
    """Train mọi (seed, K, nhánh) chưa có. Không bắt đầu run nếu ước tính vượt ngân sách session. Trả các run còn thiếu."""
    from . import yolo as Y
    meta = _meta(cfg)
    novel, names = meta["novel"], meta["names"]
    paths = D.split_paths(cfg)
    pub = pd.read_csv(_art(cfg, "splits/public_boxes.csv"))
    replay = C.read_ids(_art(cfg, "splits/replay_ids.txt"))
    control = C.read_ids(_art(cfg, "splits/control_ids.txt"))
    yd = C.SCRATCH / "yolo"
    dev_full = D.write_yolo(yd / "dev_full", C.read_ids(_art(cfg, "splits/dev_ids.txt")), pub, names, paths, cfg["img_wh"])
    base_w = _art(cfg, f"base/run/{cfg['yolo']['weights']}.pt")
    per = C.load(_art(cfg, "base/meta.json"))["sec_per_img_epoch"]
    ft, imgsz = cfg["yolo"]["finetune"], cfg["yolo"]["imgsz"]
    deadline = cfg["t0"] + cfg["kaggle"]["time_budget_h"] * 3600
    oracle = _oracle(cfg, "train")
    todo = []
    for s in seeds:
        exp = base_w if _weak(cfg) else C.out(cfg, f"expand/s{s}/expanded.pt")  # weak: finetune thẳng từ base
        if not exp.exists():
            log = Y.expand_head(base_w, exp, names, seed=s)
            log["max_diff_vs_base"] = Y.check_expansion(base_w, exp)
            C.dump(log, exp.parent / "expand_log.json")
        exp_md5 = hashlib.md5(exp.read_bytes()).hexdigest()
        for k in ks:
            for b in branches:
                run = f"s{s}_k{k}_{b}"
                dst = _art(cfg, f"runs/{run}")
                if (dst / "result.json").exists():
                    continue
                if todo or time.time() + 1.3 * per * (len(replay) + k) * ft["epochs"] > deadline:
                    todo.append(run)
                    continue
                if b == CONTROL:
                    new_ids, new_boxes = control[:k], pub[pub.image.isin(set(control[:k]))]
                else:
                    man = _sel(cfg, s, k, b)
                    assert man.exists(), f"chưa có manifest {man.name} — chạy bước proposal trước"
                    m = C.read_frozen(man)
                    bx, _ = oracle.reveal(man)  # nhãn đầy đủ base + novel của ảnh pool đã chọn
                    pad = m.get("pad_ids", [])
                    new_ids, new_boxes = m["ids"] + pad, pd.concat([bx, pub[pub.image.isin(set(pad))]])
                ids = replay + new_ids
                assert len(ids) == len(replay) + k and len(set(ids)) == len(ids), run
                done = sum((_art(cfg, f"runs/{f's{a}_k{b_}_{c}'}") / "result.json").exists() for a in seeds for b_ in ks for c in branches)
                print(f"=== [{done + 1}/{len(seeds) * len(ks) * len(branches)}] train {run} "
                      f"(ước tính {1.3 * per * (len(replay) + k) * ft['epochs'] / 60:.0f} phút) ===", flush=True)
                tr = D.write_yolo(yd / run, ids, pd.concat([pub[pub.image.isin(set(replay))], new_boxes]), names, paths, cfg["img_wh"])
                t = Y.train(exp, D.write_data_yaml(yd / f"{run}.yaml", names, tr, dev_full), dst, ft, seed=s, imgsz=imgsz)
                nb = new_boxes[new_boxes.image.isin(set(new_ids))]
                C.dump({"seed": s, "k": k, "branch": b, "n_images": len(ids), "n_new": len(new_ids),
                        "n_novel_instances": int((nb.cls == novel).sum()), "n_novel_images": int(nb.image[nb.cls == novel].nunique()),
                        "n_base_instances_new": int((nb.cls != novel).sum()), "expanded_md5": exp_md5,
                        "replay_sha": C.ids_sha(replay), **t}, dst / "result.json")
                shutil.rmtree(C.SCRATCH / "runs" / run, ignore_errors=True)
                shutil.rmtree(yd / run, ignore_errors=True)
    return todo


# ---------- 5. đánh giá ----------

def evaluate(cfg, branches, seeds, ks, ref: str | None = None) -> dict:
    """Đánh giá MỌI run đã train trên final test (cache theo run). So base classes theo TÊN trên cùng ảnh:
    base model với taxonomy base, model mở rộng với base + novel. Uplift = hiệu so với nhánh `ref` cùng (seed, K);
    mặc định `random` (recipe của P-026), hoặc `RANDOM` (kết quả cũ)."""
    from . import yolo as Y
    meta = _meta(cfg)
    novel, names, base_names = meta["novel"], meta["names"], meta["base_names"]
    paths = D.split_paths(cfg)
    pub = pd.read_csv(_art(cfg, "splits/public_boxes.csv"))
    pi = pd.read_csv(_art(cfg, "splits/public_images.csv"))
    tod = dict(zip(pi.image, pi.timeofday))
    test = C.read_ids(_art(cfg, "splits/test_ids.txt"))
    E, imgsz, W = cfg["eval"], cfg["yolo"]["imgsz"], cfg["yolo"]["weights"]
    yd = C.SCRATCH / "yolo"

    def ydir(name, ids, nm):
        d = D.write_yolo(yd / name, ids, pub, nm, paths, cfg["img_wh"])
        return D.write_data_yaml(yd / f"{name}.yaml", nm, d, d, d)

    y_full = ydir("test_full", test, names)
    subsets = {}
    for t in ("daytime", "night"):
        ids_t = [i for i in test if tod.get(i) == t]
        if ((pub.cls == novel) & pub.image.isin(set(ids_t))).sum() >= E["min_instances_subset"]:
            subsets[t] = ydir(f"test_{t}", ids_t, names)
    bp = _art(cfg, "eval/base_model.json")
    if not bp.exists():
        C.dump(Y.eval_ap(_art(cfg, f"base/run/{W}.pt"), ydir("test_base", test, _base_taxonomy(meta)), "test", imgsz, E["batch"]),
               C.out(cfg, "eval/base_model.json"))
    base_ap = C.load(bp)
    base_novel = base_ap.get(novel, {}).get("ap50_95", 0.0)  # missing: base không có class novel -> 0
    mean_ap = lambda ap, cls: float(np.nanmean([ap.get(c, {}).get("ap50_95", np.nan) for c in cls]))
    base_map = mean_ap(base_ap, base_names)
    rc = f"novel_recall@{E['conf']}"
    rows = []
    for res in tqdm(sorted(_art(cfg, "runs").glob("*/result.json")), desc="đánh giá run"):
        r, run = C.load(res), res.parent.name
        ep = C.out(cfg, f"eval/{run}.json")
        if not ep.exists():
            w = res.parent / f"{W}.pt"
            hits = Y.recall_at_conf(w, test, paths, pub, novel, E["conf"], E["iou"], imgsz, E["batch"])
            hits["size"], hits["tod"] = D.size_bucket(hits).to_numpy(), hits.image.map(tod)
            C.dump({"ap": Y.eval_ap(w, y_full, "test", imgsz, E["batch"]), "recall": float(hits.hit.mean()),
                    "recall_by_size": hits.groupby("size").hit.mean().to_dict(), "recall_by_tod": hits.groupby("tod").hit.mean().to_dict(),
                    "novel_ap_by_tod": {t: Y.eval_ap(w, y, "test", imgsz, E["batch"]).get(novel, {}).get("ap50_95") for t, y in subsets.items()}}, ep)
        ev = C.load(ep)
        ap = ev["ap"]
        rows.append({**r, "novel_ap50_95": ap.get(novel, {}).get("ap50_95", 0.0), "novel_ap50": ap.get(novel, {}).get("ap50", 0.0),
                     "uplift_vs_base_model": ap.get(novel, {}).get("ap50_95", 0.0) - base_novel,
                     "base_map": mean_ap(ap, base_names), "base_map_delta": mean_ap(ap, base_names) - base_map,
                     "all_map": mean_ap(ap, names), rc: ev["recall"],
                     **{f"recall_{a}": v for a, v in ev["recall_by_size"].items()},
                     **{f"novel_ap_{a}": v for a, v in ev["novel_ap_by_tod"].items()}})
    assert rows, "chưa có run nào train xong (runs/*/result.json)"
    df = pd.DataFrame(rows)
    df.to_csv(C.out(cfg, "eval/runs.csv"), index=False)
    # mọi nhánh trong một seed dùng chung expanded init, replay set và số ảnh train
    assert (df.groupby("seed").expanded_md5.nunique() == 1).all(), "expanded init khác nhau trong một seed"
    assert df.replay_sha.nunique() == 1, "replay set khác nhau"
    assert (df.groupby(["seed", "k"]).n_images.nunique() == 1).all(), "số ảnh train khác nhau giữa các nhánh"
    plan = {f"s{s}_k{k}_{b}" for s in seeds for k in ks for b in branches}
    missing = sorted(plan - {f"s{a}_k{b}_{c}" for a, b, c in zip(df.seed, df.k, df.branch)})
    metrics = ["novel_ap50_95", "novel_ap50", "uplift_vs_base_model", rc, "base_map", "base_map_delta", "all_map",
               "n_novel_images", "n_novel_instances", "n_base_instances_new"]
    ref = ref or next((b for b in ("random", "RANDOM") if b in set(df.branch)), None)
    summ = M.paired_summary(df, metrics, ref) if ref in set(df.branch) else df.groupby(["k", "branch"])[metrics].agg(["mean", "std"])
    uplift = None
    if ref in set(df.branch):
        rv = df[df.branch == ref].set_index(["seed", "k"]).novel_ap50_95
        uplift = df.assign(uplift_novel_ap50_95=df.novel_ap50_95 - rv.reindex(pd.MultiIndex.from_arrays([df.seed, df.k])).to_numpy()) \
            .pivot_table(index=["k", "branch"], columns="seed", values="uplift_novel_ap50_95")
    # AP50-95 từng class (thang 0–100, P-026 focus §4B); dòng BASE_MODEL để so trực tiếp
    pcs = []
    for res in sorted(_art(cfg, "runs").glob("*/result.json")):
        r, ap = C.load(res), C.load(_art(cfg, f"eval/{res.parent.name}.json"))["ap"]
        pcs.append({"k": r["k"], "branch": r["branch"], **{c: 100 * ap.get(c, {}).get("ap50_95", np.nan) for c in names}})
    per_class = pd.DataFrame(pcs).groupby(["k", "branch"])[names].mean()
    per_class.loc[(0, "BASE_MODEL"), :] = [100 * base_ap.get(c, {}).get("ap50_95", np.nan) for c in names]
    per_class = per_class.sort_index().round(2)
    ret_p = next((q for q in (_art(cfg, "selection/selection.csv"), _art(cfg, "retrieval/retrieval.csv")) if q.exists()), None)
    ret = None
    if ret_p is not None:
        rt = pd.read_csv(ret_p)
        ret = rt.groupby(["k", "branch"])[[c for c in ("precision_at_k", "recall_at_k", "novel_instances", "nn_cos_mean") if c in rt]].agg(["mean", "std"])

    def md(t):
        if t is None:
            return "(không có)"
        try:
            return t.to_markdown()
        except ImportError:
            return "```\n" + t.to_string() + "\n```"
    desc = (f"base train có {meta['n_weak']} ảnh / {meta['base_novel_instances']} instance {novel}" if _weak(cfg)
            else f"base model chưa có class {novel}")
    report = f"""# Kết quả uplift — novel = {novel}, kịch bản {cfg['scenario']} (smoke={cfg['smoke']})

Base model ({desc}): base mAP50-95 = {base_map:.4f}, AP50-95 {novel} = {base_novel:.4f} trên final test. Thiếu run: {missing or 'không'}.
`uplift_vs_base_model` = AP {novel} sau finetune − AP {novel} của base model.
Annotation là mô phỏng từ ground truth BDD. ORACLE_POSITIVE dùng nhãn ẩn (tham chiếu, không phải phương pháp thực tế). Nhánh tham chiếu: `{ref}`.

## 1. Tập dữ liệu do mỗi cách proposal tạo ra (pool, trước khi train)
{md(ret)}

## 2. Uplift AP50-95 novel so với `{ref}`, theo từng seed
{md(uplift)}

## 3. Detection trên final test: mean/std và hiệu từng cặp so với `{ref}`
{md(summ)}

## 4. AP50-95 từng class (0–100), trung bình theo seed; dòng BASE_MODEL = model ban đầu
{md(per_class)}

## 5. Từng run
{md(df[['seed', 'k', 'branch'] + metrics].sort_values(['k', 'branch', 'seed']).set_index(['k', 'branch', 'seed']))}
"""
    C.out(cfg, "eval/report.md").write_text(report, encoding="utf-8")
    return {"runs": df, "summary": summ, "uplift": uplift, "retrieval": ret, "per_class": per_class, "missing": missing,
            "base_map": base_map, "base_novel_ap": base_novel, "ref": ref}


# ---------- 6. gói kết quả ----------

BUNDLE_PATTERNS = [
    "*/env.json", "splits/meta.json", "base/meta.json", "base/run/results.csv", "base/run/args.yaml",
    "retrieval/*.json", "retrieval/*.jsonl", "retrieval/retrieval.csv", "selections/*.json",
    "expand/*/expand_log.json", "train/oracle_log.jsonl",
    "runs/*/result.json", "runs/*/results.csv", "runs/*/args.yaml",
    "eval/*.json", "eval/runs.csv", "eval/report.md",
]


def bundle(cfg, stage: str) -> Path:
    """Zip các file kết quả nhỏ (json/csv/md/yaml) vào /kaggle/working/<art>_<stage>_results.zip để tải ở tab Output.
    Không gồm weights .pt, embedding .npy, nhãn oracle của pool và bảng box công khai (lớn, tái tạo được từ splits)."""
    import zipfile
    art = C.WORK / cfg["art"]
    files = sorted({f for p in BUNDLE_PATTERNS for f in art.glob(p) if f.is_file()})
    dst = C.WORK / f"{cfg['art']}_{stage}_results.zip"
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f.relative_to(art).as_posix())
        z.writestr("MANIFEST.txt", "\n".join(f.relative_to(art).as_posix() for f in files))
    print(f"gói kết quả: {dst} ({len(files)} file, {dst.stat().st_size / 1e6:.2f} MB) -> tải ở tab Output")
    return dst
