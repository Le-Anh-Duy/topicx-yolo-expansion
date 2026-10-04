"""Các bước của pipeline uplift; notebook chỉ gọi các hàm ở đây. Bước nào đã có kết quả (trong /kaggle/working sau
`common.sync_inputs`) thì bỏ qua, nên chạy lại notebook = chạy tiếp phần còn thiếu.

    splits -> base model -> proposal (K ảnh từ pool, không nhãn) -> chốt manifest -> oracle trả nhãn BDD
    -> mở rộng head + finetune (replay + K) -> đánh giá trên final test -> uplift so với RANDOM cùng seed, cùng K

Kịch bản (`cfg["scenario"]`):
  - missing: base train không có ảnh novel; base model chỉ biết base classes; mở rộng head trước khi finetune.
  - weak: base train có đúng `weak_novel_images` ảnh novel; base model biết đủ class (novel yếu); finetune thẳng từ base.

Cắm thuật toán proposal mới:
  - hàm `fn(pool: Pool, k: int, seed: int) -> list[str]` (hoặc `(list[str], meta: dict)`), truyền vào `propose()`;
  - hoặc file xếp hạng (txt mỗi dòng một id ảnh pool, hoặc csv có cột `image`) -> `from_ranked_file()` / `discover_ranked_files()`.
Proposer chạy trong `no_oracle_access()` và chỉ thấy `Pool`: id ảnh, đường dẫn ảnh, topic, embedding. Không có nhãn pool.
"""

from __future__ import annotations

import hashlib
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import common as C
from . import data as D
from . import metrics as M
from . import select as S
from .oracle import OracleStore, no_oracle_access

CONTROL = "REPLAY_ONLY"
PRIVILEGED = ("ORACLE_POSITIVE", "RETRIEVAL_MATCHED")


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
    labels, dirs = D.bdd_locations(cfg)
    print("nhãn:", labels, "\nảnh:", dirs)
    parts = [D.load_labels(labels[s], s, cfg["alias"], cfg["classes"]) for s in ("train", "val")]
    images = pd.concat([p[0] for p in parts], ignore_index=True)
    boxes = pd.concat([p[1] for p in parts], ignore_index=True)
    assert images.image.is_unique
    paths = D.image_paths(images, dirs)
    missing = {i for i in images.image if not paths[i].exists()}
    print(f"{len(images)} ảnh có nhãn, {len(missing)} không có file ảnh -> bỏ")
    images = images[~images.image.isin(missing)].reset_index(drop=True)
    boxes = boxes[boxes.image.isin(set(images.image))].reset_index(drop=True)
    from PIL import Image
    sizes = {Image.open(paths[i]).size for i in images.image.sample(min(200, len(images)), random_state=0)}
    assert sizes == {tuple(cfg["img_wh"])}, sizes
    return images, boxes, paths


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


# ---------- 3. proposal ----------

class Pool:
    """Thứ proposer được thấy: `ids` (ảnh pool), `paths[id]`, `topic` + `prompts`, embedding tính lười (cache vào retrieval/).
    Không chứa nhãn pool."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.topic = _meta(cfg)["novel"]
        self.prompts = cfg["topic_prompts"][self.topic]
        self.ids = C.read_ids(_art(cfg, "splits/pool_ids.txt"))
        self.paths = D.split_paths(cfg)
        self._c: dict = {}

    def _cached(self, name, fn):
        if name not in self._c:
            p = C.out(self.cfg, f"retrieval/{name}.npy")
            if not p.exists():
                np.save(p, fn())
            self._c[name] = np.load(p)
        return self._c[name]

    def _clip_model(self):
        if "model" not in self._c:
            from . import embed as E
            self._c["model"] = E.load_clip(self.cfg["clip"]["model"], self.cfg["clip"]["pretrained"])
            import open_clip
            C.dump({"clip": self.cfg["clip"], "open_clip": open_clip.__version__, "preprocess": str(self._c["model"][1]),
                    "crops": "3 crop vuông trái/giữa/phải", "score": "max_crop cos(img, mean text emb)",
                    "dinov2": self.cfg["dinov2"], "dinov2_preprocess": "resize 224x392, ImageNet norm"},
                   C.out(self.cfg, "retrieval/embed_meta.json"))
        return self._c["model"]

    def clip(self) -> np.ndarray:
        """(N, 3 crop, D) CLIP image embedding, L2-normalized."""
        from . import embed as E
        return self._cached("clip_pool", lambda: E.clip_images([self.paths[i] for i in self.ids], *self._clip_model()[:2],
                                                              self.cfg["clip"]["batch"]))

    def dino(self) -> np.ndarray:
        """(N, D) DINOv2-S embedding, L2-normalized."""
        from . import embed as E
        return self._cached("dino_pool", lambda: E.dinov2_images([self.paths[i] for i in self.ids],
                                                                self.cfg["dinov2"]["model"], self.cfg["dinov2"]["batch"]))

    def topic_scores(self) -> np.ndarray:
        """Điểm CLIP với topic. Prompt chọn trên dev (nhãn dev công khai) theo luật cố định: AP ranking cao nhất."""
        from . import embed as E
        if "scores" not in self._c:
            m, pre, tok = self._clip_model()
            dev = C.read_ids(_art(self.cfg, "splits/dev_ids.txt"))
            clip_dev = self._cached("clip_dev", lambda: E.clip_images([self.paths[i] for i in dev], m, pre, self.cfg["clip"]["batch"]))
            pub = pd.read_csv(_art(self.cfg, "splits/public_boxes.csv"))
            dev_pos = set(pub.image[pub.cls == self.topic])
            y = [i in dev_pos for i in dev]
            T = E.clip_texts(self.prompts, m, tok)
            cands = {f"single:{p}": [j] for j, p in enumerate(self.prompts)} | {"ensemble:all": list(range(len(self.prompts)))}
            ap = {n: M.average_precision(E.topic_scores(clip_dev, T[idx]), y) for n, idx in cands.items()}
            best = max(ap, key=ap.get)
            C.dump({"rule": "max dev ranking AP", "dev_ap": ap, "best": best, "prompts": [self.prompts[j] for j in cands[best]],
                    "dev_positive_rate": float(np.mean(y))}, C.out(self.cfg, "retrieval/prompt_selection.json"))
            self._c["scores"] = E.topic_scores(self.clip(), T[cands[best]])
            np.save(C.out(self.cfg, "retrieval/pool_scores.npy"), self._c["scores"])
        return self._c["scores"]

    def order(self) -> list[int]:
        return S.rank(self.ids, self.topic_scores())


def p_random(pool: Pool, k: int, seed: int):
    return S.random_k(pool.ids, k, seed)


def p_diversity(pool: Pool, k: int, seed: int):
    """Recipe P-026 (không topic): greedy min_dist trên DINOv2-S theo thứ tự ngẫu nhiên."""
    md = pool.cfg["diverse"]["min_dist"]
    ids, f = S.greedy_diverse(pool.ids, np.random.default_rng(seed).permutation(len(pool.ids)), pool.dino(), k, md)
    return ids, {"relax": f, "emb": "dinov2", "min_dist": md}


def p_retrieval(pool: Pool, k: int, seed: int):
    return [pool.ids[i] for i in pool.order()[:k]]


def p_retrieval_diverse(pool: Pool, k: int, seed: int):
    from . import embed as E
    d = pool.cfg["diverse"]
    ids, f = S.greedy_diverse(pool.ids, pool.order()[:d["max_rank_mult"] * k], E.image_level(pool.clip()), k, d["min_dist"])
    return ids, {"relax": f, "emb": "clip", "min_dist": d["min_dist"]}


PROPOSERS = {"RANDOM": p_random, "DIVERSITY": p_diversity, "RETRIEVAL": p_retrieval, "RETRIEVAL_DIVERSE": p_retrieval_diverse}


def from_ranked_file(path: Path):
    """Proposer từ file xếp hạng ngoài (txt mỗi dòng một id, hoặc csv cột `image`): lấy k id đầu, bỏ qua seed.
    File phải được tạo KHÔNG dùng nhãn pool — pipeline chỉ kiểm được id thuộc pool."""
    path = Path(path)
    ids = pd.read_csv(path).image.astype(str).tolist() if path.suffix == ".csv" else C.read_ids(path)

    def fn(pool, k, seed):
        return ids[:k], {"source": str(path), "deterministic": True}
    return fn


def discover_ranked_files(root: Path = C.INPUT) -> dict:
    """Mọi file `proposals/*.txt|csv` trong input đã gắn -> {EXT_<tên file>: proposer}."""
    files = sorted({p for d in ("*", "*/*", "*/*/*") for p in root.glob(f"{d}/proposals/*") if p.suffix in (".txt", ".csv")})
    return {f"EXT_{p.stem}": from_ranked_file(p) for p in files}


def propose(cfg, proposers: dict, seeds, ks, pool: Pool) -> None:
    """Chạy từng proposer trong `no_oracle_access()`, kiểm đầu ra, chốt manifest. Manifest đã có thì bỏ qua."""
    pool_set = set(pool.ids)
    for name, fn in proposers.items():
        assert name != CONTROL and name not in PRIVILEGED, f"tên nhánh {name} dành riêng"
        for s in seeds:
            for k in ks:
                p = _sel(cfg, s, k, name)
                if p.exists():
                    continue
                t0 = time.time()
                with no_oracle_access():
                    r = fn(pool, k, 1000 + s)
                ids, meta = r if isinstance(r, tuple) else (r, {})
                ids = [str(i) for i in ids]
                assert len(ids) == k and len(set(ids)) == k, f"{name}: cần đúng {k} id khác nhau, nhận {len(ids)} ({len(set(ids))} khác nhau)"
                bad = set(ids) - pool_set
                assert not bad, f"{name}: {len(bad)} id không thuộc pool, vd {sorted(bad)[:3]}"
                C.freeze(p, ids, branch=name, seed=s, k=k, seconds=time.time() - t0, **meta)


def propose_privileged(cfg, branches, seeds, ks, pool: Pool) -> None:
    """Nhánh tham chiếu dùng nhãn ẩn (ghi log): ORACLE_POSITIVE, RETRIEVAL_MATCHED. Chạy SAU khi nhánh thường đã chốt."""
    if not set(branches) & set(PRIVILEGED):
        return
    oracle = _oracle(cfg, "retrieval")
    novel = pool.topic
    if "ORACLE_POSITIVE" in branches:
        pos = oracle.privileged_positive_ids(novel, reason="ORACLE_POSITIVE")
        neg = sorted(set(pool.ids) - set(pos))
        for s in seeds:
            for k in ks:
                ids = S.random_k(pos, k, 1000 + s)
                pad = k - len(ids)
                C.freeze(_sel(cfg, s, k, "ORACLE_POSITIVE"), ids + (S.random_k(neg, pad, 2000 + s) if pad else []),
                         branch="ORACLE_POSITIVE", seed=s, k=k, privileged=True, padded_negatives=pad)
    if "RETRIEVAL_MATCHED" in branches:
        # prefix ngắn nhất của ranking có >= số novel instance của RANDOM cùng (seed, K); bù ảnh control cho đủ K
        control = C.read_ids(_art(cfg, "splits/control_ids.txt"))
        ranked = [pool.ids[i] for i in pool.order()]
        for s in seeds:
            for k in ks:
                rb, _ = oracle.reveal(_sel(cfg, s, k, "RANDOM"))
                target = int((rb.cls == novel).sum())
                cum = oracle.privileged_instance_counts(ranked[:k], novel, reason="RETRIEVAL_MATCHED").cumsum().to_numpy()
                kp = min(int(np.searchsorted(cum, target)) + 1, k) if target else 0
                C.freeze(_sel(cfg, s, k, "RETRIEVAL_MATCHED"), ranked[:kp], branch="RETRIEVAL_MATCHED", seed=s, k=k,
                         privileged=True, pad_ids=control[:k - kp], target_novel_instances=target,
                         got_novel_instances=int(cum[kp - 1]) if kp else 0)


def retrieval_eval(cfg, pool: Pool) -> pd.DataFrame:
    """Simulated annotation + chỉ số retrieval cho MỌI manifest đã chốt."""
    oracle = _oracle(cfg, "retrieval")
    n_pos = len(oracle.privileged_positive_ids(pool.topic, reason="recall@K denominator"))
    dino, dix = pool.dino(), {i: j for j, i in enumerate(pool.ids)}
    rows = []
    for p in sorted(_art(cfg, "selections").glob("*.json")):
        m = C.read_frozen(p)
        bx, im = oracle.reveal(p)
        rows.append({"seed": m["seed"], "k": m["k"], "branch": m["branch"], **M.retrieval_report(m["ids"], bx, im, pool.topic, n_pos),
                     **M.batch_redundancy(dino[[dix[i] for i in m["ids"]]])})
    df = pd.DataFrame(rows).fillna(0)
    df.to_csv(C.out(cfg, "retrieval/retrieval.csv"), index=False)
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

def evaluate(cfg, branches, seeds, ks) -> dict:
    """Đánh giá MỌI run đã train trên final test (cache theo run). So base classes theo TÊN trên cùng ảnh:
    base model với taxonomy base, model mở rộng với base + novel. Uplift = hiệu so với RANDOM cùng (seed, K)."""
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
    for res in sorted(_art(cfg, "runs").glob("*/result.json")):
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
    summ = M.paired_summary(df, metrics) if "RANDOM" in set(df.branch) else df.groupby(["k", "branch"])[metrics].agg(["mean", "std"])
    uplift = None
    if "RANDOM" in set(df.branch):
        ref = df[df.branch == "RANDOM"].set_index(["seed", "k"]).novel_ap50_95
        uplift = df.assign(uplift_novel_ap50_95=df.novel_ap50_95 - ref.reindex(pd.MultiIndex.from_arrays([df.seed, df.k])).to_numpy()) \
            .pivot_table(index=["k", "branch"], columns="seed", values="uplift_novel_ap50_95")
    ret_p = _art(cfg, "retrieval/retrieval.csv")
    ret = pd.read_csv(ret_p).groupby(["k", "branch"])[["precision_at_k", "recall_at_k", "novel_instances", "nn_cos_mean"]].agg(["mean", "std"]) \
        if ret_p.exists() else None

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
Annotation là mô phỏng từ ground truth BDD. ORACLE_POSITIVE / RETRIEVAL_MATCHED dùng nhãn ẩn (tham chiếu, không phải phương pháp thực tế).

## 1. Tập dữ liệu do mỗi cách proposal tạo ra (pool, trước khi train)
{md(ret)}

## 2. Uplift AP50-95 novel so với RANDOM, theo từng seed
{md(uplift)}

## 3. Detection trên final test: mean/std và hiệu từng cặp so với RANDOM
{md(summ)}

## 4. Từng run
{md(df[['seed', 'k', 'branch'] + metrics].sort_values(['k', 'branch', 'seed']).set_index(['k', 'branch', 'seed']))}
"""
    C.out(cfg, "eval/report.md").write_text(report, encoding="utf-8")
    return {"runs": df, "summary": summ, "uplift": uplift, "retrieval": ret, "missing": missing, "base_map": base_map,
            "base_novel_ap": base_novel}
