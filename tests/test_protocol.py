"""Kiểm tra protocol bằng dữ liệu tổng hợp (CPU, không cần BDD/GPU). Chạy ở cell đầu mỗi notebook."""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from topicx import common, data, metrics, select
from topicx.oracle import OracleStore, no_oracle_access

SRC = Path(__file__).resolve().parents[1] / "src" / "topicx"


def _toy(n_groups=60, per=5, seed=0):
    rng = np.random.default_rng(seed)
    imgs = pd.DataFrame({"image": [f"g{g:03d}-{i}.jpg" for g in range(n_groups) for i in range(per)]})
    imgs["group"] = imgs.image.str.split("-").str[0]
    has_novel = {x for x in imgs.image if rng.random() < 0.3}
    return imgs, has_novel


def test_selector_modules_label_free():
    for f in ("select.py", "embed.py", "algo1.py"):
        tree = ast.parse((SRC / f).read_text(encoding="utf-8"))
        mods = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        mods |= {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        assert not {m for m in mods if "oracle" in m or m.endswith("data") or "pandas" in m}, (f, mods)


def test_splits_disjoint_base_clean_pool_mixed():
    imgs, has_novel = _toy()
    s = data.make_splits(imgs, has_novel, {"test": 40, "dev": 30, "pool": 80, "base": 60}, seed=1)
    rep = data.check_splits(s, imgs, has_novel)
    assert rep["pool"]["n_novel_images"] > 0 and rep["base"]["n_novel_images"] == 0
    assert s == data.make_splits(imgs, has_novel, {"test": 40, "dev": 30, "pool": 80, "base": 60}, seed=1)


def test_check_splits_catches_group_leak():
    imgs, has_novel = _toy()
    s = data.make_splits(imgs, has_novel, {"test": 40, "dev": 30, "pool": 80, "base": 60}, seed=1)
    extra = s["test"][0].split("-")[0] + "-99.jpg"  # ảnh mới cùng group với test
    imgs2 = pd.concat([imgs, pd.DataFrame({"image": [extra], "group": [extra.split("-")[0]]})])
    with pytest.raises(AssertionError, match="group"):
        data.check_splits(dict(s, dev=s["dev"] + [extra]), imgs2, has_novel)


def test_weak_base_has_exact_novel_quota():
    imgs, has_novel = _toy()
    s = data.make_splits(imgs, has_novel, {"test": 40, "dev": 30, "pool": 80, "base": 60}, seed=1, n_weak=3)
    assert data.check_splits(s, imgs, has_novel, n_weak=3)["base"]["n_novel_images"] == 3
    with pytest.raises(AssertionError, match="cần đúng 0"):
        data.check_splits(s, imgs, has_novel)


def test_scenario_artifact_dirs():
    assert common.load_config(False)["art"] == "art"
    assert common.load_config(True, "weak")["art"] == "art_smoke_weak"
    with pytest.raises(AssertionError):
        common.load_config(False, "other")


def test_test_source_val_only():
    imgs, has_novel = _toy()
    imgs["src"] = np.where(imgs.group < "g020", "val", "train")
    s = data.make_splits(imgs, has_novel, {"test": 40, "dev": 20, "pool": 60, "base": 40}, seed=1, test_source="val")
    src = dict(zip(imgs.image, imgs.src))
    assert {src[i] for i in s["test"]} == {"val"}
    assert {src[i] for k in ("dev", "pool", "base") for i in s[k]} == {"train"}


def test_propose_plugin_boundary(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from topicx import pipeline as P
    monkeypatch.setattr(common, "WORK", tmp_path)
    cfg = {"art": "art"}
    pool = SimpleNamespace(ids=[f"p{i}" for i in range(20)], topic="bus")
    P.propose(cfg, {"MINE": lambda pool, k, seed: (pool.ids[:k], {"note": "x"})}, [0], [5], pool)
    m = common.read_frozen(tmp_path / "art/selections/s0_k5_MINE.json")
    assert m["ids"] == pool.ids[:5] and m["note"] == "x"
    with pytest.raises(AssertionError, match="khác nhau"):
        P.propose(cfg, {"DUP": lambda pool, k, seed: [pool.ids[0]] * k}, [0], [5], pool)
    with pytest.raises(AssertionError, match="không thuộc pool"):
        P.propose(cfg, {"OUT": lambda pool, k, seed: [f"x{i}" for i in range(k)]}, [0], [5], pool)
    od = tmp_path / "splits_oracle_boxes.csv"
    od.write_text("image,cls")
    with pytest.raises(PermissionError):
        P.propose(cfg, {"PEEK": lambda pool, k, seed: open(od).read() and pool.ids[:k]}, [0], [5], pool)
    with pytest.raises(AssertionError, match="dành riêng"):
        P.propose(cfg, {"ORACLE_POSITIVE": lambda pool, k, seed: pool.ids[:k]}, [0], [5], pool)


def test_ranked_file_proposer(tmp_path):
    from topicx import pipeline as P
    f = tmp_path / "my-dataset" / "proposals" / "p026.txt"  # như /kaggle/input/<dataset>/proposals/
    common.write_ids(["b", "a", "c"], f)
    assert P.from_ranked_file(f)(None, 2, 0)[0] == ["b", "a"]
    assert list(P.discover_ranked_files(tmp_path)) == ["EXT_p026"]


def test_locate_bdd_through_symlink(tmp_path):
    real = tmp_path / "real" / "bdd100k"
    for s in ("train", "val"):
        d = real / "images" / "100k" / s
        d.mkdir(parents=True)
        for i in range(2001):
            (d / f"{i}.jpg").touch()
        (real / "labels").mkdir(exist_ok=True)
        (real / "labels" / f"det_{s}.json").write_text("[]")
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "bdd").symlink_to(real, target_is_directory=True)  # Kaggle mount kiểu symlink
    labels, dirs = data.locate_bdd(tmp_path / "input")
    assert set(labels) == set(dirs) == {"train", "val"}
    assert "json: det_train.json" in data.describe_tree(tmp_path / "input")


def test_class_mapping_novel_last():
    base, full = data.class_names(["a", "b", "c"], "b")
    assert base == ["a", "c"] and full == ["a", "c", "b"]


def test_yolo_line_normalized_and_clipped():
    assert data.yolo_line(2, 0, 0, 640, 360, 1280, 720) == "2 0.250000 0.250000 0.500000 0.500000"
    assert data.yolo_line(0, -10, 0, 1300, 720, 1280, 720).startswith("0 0.500000")
    assert data.yolo_line(0, 5, 5, 5.5, 50, 1280, 720) is None


def test_write_yolo_drops_classes_outside_taxonomy(tmp_path):
    img = tmp_path / "a.jpg"
    img.write_bytes(b"x")
    boxes = pd.DataFrame([("a.jpg", "car", 0, 0, 100, 100), ("a.jpg", "bus", 0, 0, 200, 200)], columns=data.BOX_COLS)
    d = data.write_yolo(tmp_path / "y", ["a.jpg"], boxes, ["car"], {"a.jpg": img}, (1280, 720))
    assert (d / "labels" / "a.txt").read_text().splitlines() == [data.yolo_line(0, 0, 0, 100, 100, 1280, 720)]


def test_freeze_is_immutable(tmp_path):
    p = tmp_path / "m.json"
    common.freeze(p, ["a", "b"], branch="X")
    common.freeze(p, ["a", "b"], branch="X")
    with pytest.raises(RuntimeError):
        common.freeze(p, ["a", "c"], branch="X")
    m = common.load(p)
    m["ids"] = ["z"]
    common.dump(m, p)
    with pytest.raises(RuntimeError, match="sha256"):
        common.read_frozen(p)


def test_reveal_only_manifest_and_guard(tmp_path):
    od = tmp_path / "oracle"  # tên test không được chứa "oracle": tmp_path lấy theo tên test
    od.mkdir()
    pd.DataFrame([("p1", "bus", 0, 0, 9, 9), ("p2", "car", 0, 0, 9, 9)], columns=data.BOX_COLS).to_csv(od / "pool_boxes.csv", index=False)
    pd.DataFrame({"image": ["p1", "p2"], "timeofday": ["night", "daytime"]}).to_csv(od / "pool_images.csv", index=False)
    common.write_ids(["p1", "p2"], od / "pool_ids.txt")
    o = OracleStore(od, tmp_path / "log.jsonl")
    common.freeze(tmp_path / "sel.json", ["p2"], branch="RANDOM")
    b, _ = o.reveal(tmp_path / "sel.json")
    assert list(b.image) == ["p2"]
    with no_oracle_access():
        with pytest.raises(PermissionError):
            open(od / "pool_boxes.csv").close()
        (tmp_path / "ok.txt").write_text("ok")  # đường dẫn khác vẫn được
    open(od / "pool_boxes.csv").close()


def test_random_and_topk_deterministic():
    ids = [f"i{i}" for i in range(50)]
    assert select.random_k(ids, 10, 3) == select.random_k(list(reversed(ids)), 10, 3)
    scores = np.arange(50)[::-1].astype(float)
    assert select.topk(ids, scores, 3) == ["i0", "i1", "i2"]


def test_greedy_diverse_skips_near_duplicates():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(5, 16))
    emb = np.repeat(base, 4, axis=0) + 1e-4 * rng.normal(size=(20, 16))  # 5 cụm, mỗi cụm 4 bản gần trùng
    ids = [f"i{i}" for i in range(20)]
    picked, f = select.greedy_diverse(ids, list(range(20)), emb, 5, min_dist=0.05)
    assert f == 1.0 and len({int(p[1:]) // 4 for p in picked}) == 5
    picked, f = select.greedy_diverse(ids, list(range(20)), emb, 8, min_dist=0.05)
    assert len(picked) == 8 and f == 0.0  # không đủ ảnh khác nhau -> nới hết, lấy đầu thứ tự


def test_average_precision():
    assert metrics.average_precision([3, 2, 1], [1, 0, 1]) == pytest.approx((1 + 2 / 3) / 2)


def test_match_gt_one_pred_per_gt():
    from topicx.yolo import match_gt
    gt = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], float)
    pred = np.array([[0, 0, 10, 10], [1, 1, 10, 10]], float)
    assert match_gt(gt, pred, np.array([0.9, 0.8]), 0.5).tolist() == [True, False]


def test_head_expansion_preserves_old_classes(tmp_path):
    pytest.importorskip("ultralytics")
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel

    from topicx.yolo import check_expansion, expand_head
    y = YOLO("yolo11n.yaml")
    y.model = DetectionModel("yolo11n.yaml", nc=3, verbose=False)
    y.model.names = {0: "a", 1: "b", 2: "c"}
    y.ckpt = y.ckpt or {}  # model build từ yaml chưa có ckpt
    y.save(str(tmp_path / "base.pt"))
    log = expand_head(tmp_path / "base.pt", tmp_path / "exp.pt", ["a", "b", "c", "d"], seed=0)
    assert log["partial_rows_copied"]
    assert check_expansion(tmp_path / "base.pt", tmp_path / "exp.pt") < 1e-4
    assert YOLO(str(tmp_path / "exp.pt")).names == {0: "a", 1: "b", 2: "c", 3: "d"}
