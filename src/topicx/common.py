"""Config, đường dẫn artifact trên Kaggle, manifest đã chốt, log môi trường."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
WORK = Path(os.environ.get("TOPICX_WORK", "/kaggle/working"))
INPUT = Path(os.environ.get("TOPICX_INPUT", "/kaggle/input"))
SCRATCH = Path(os.environ.get("TOPICX_SCRATCH", "/tmp/topicx"))  # thư mục YOLO tạm, không lưu vào output


def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(smoke: bool) -> dict:
    cfg = yaml.safe_load((REPO / "configs" / "exp.yaml").read_text(encoding="utf-8"))
    smoke_over = cfg.pop("smoke")
    if smoke:
        cfg = _merge(cfg, smoke_over)
    cfg["smoke"] = smoke
    cfg["art"] = "art_smoke" if smoke else "art"
    return cfg


def out(cfg: dict, rel: str) -> Path:
    """Đường dẫn ghi artifact trong /kaggle/working/<art>/rel (tạo thư mục cha)."""
    p = WORK / cfg["art"] / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def find(cfg: dict, rel: str, required: bool = True) -> Path | None:
    """Tìm artifact: /kaggle/working trước, rồi output notebook trước được gắn làm input (độ sâu ≤ 3)."""
    own = WORK / cfg["art"] / rel
    if own.exists():
        return own
    hits = sorted({p for d in ("*", "*/*", "*/*/*") for p in INPUT.glob(f"{d}/{cfg['art']}/{rel}")})
    if len(hits) > 1:
        raise RuntimeError(f"nhiều input chứa {rel}: {hits} — chỉ gắn một version")
    if not hits and required:
        raise FileNotFoundError(f"không thấy {cfg['art']}/{rel} trong {WORK} hay {INPUT} — đã gắn output notebook trước chưa?")
    return hits[0] if hits else None


def dump(obj, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path


def load(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_ids(path: Path) -> list[str]:
    return [x for x in Path(path).read_text(encoding="utf-8").split("\n") if x]


def write_ids(ids, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids), encoding="utf-8")
    return path


def ids_sha(ids) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def freeze(path: Path, ids, **meta) -> Path:
    """Chốt danh sách ảnh được chọn. Ghi lại cùng nội dung thì bỏ qua; khác nội dung thì lỗi (không sửa manifest đã chốt)."""
    ids = [str(i) for i in ids]
    payload = {"ids": ids, "sha256": ids_sha(ids), **meta}
    if path.exists():
        if load(path)["sha256"] != payload["sha256"]:
            raise RuntimeError(f"{path} đã chốt với danh sách khác")
        return path
    return dump(payload, path)


def read_frozen(path: Path) -> dict:
    m = load(path)
    if ids_sha(m["ids"]) != m["sha256"]:
        raise RuntimeError(f"{path}: sha256 không khớp — manifest bị sửa sau khi chốt")
    return m


def env_report(cfg: dict, stage: str) -> dict:
    """Ghi phiên bản thư viện, commit repo, GPU vào <art>/<stage>/env.json."""
    info = {"python": platform.python_version(), "smoke": cfg["smoke"]}
    try:
        info["commit"] = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                        capture_output=True, text=True, check=True).stdout.strip()
    except Exception as e:  # repo tải dạng zip / dataset
        info["commit"] = f"unknown ({e})"
    for mod in ("torch", "torchvision", "ultralytics", "open_clip", "numpy", "pandas"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None
    try:
        import torch
        info["cuda"] = torch.version.cuda
        info["gpu"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:
        pass
    dump(info, out(cfg, f"{stage}/env.json"))
    return info
