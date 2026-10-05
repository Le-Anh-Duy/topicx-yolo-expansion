"""Cầu nối harness ↔ pipeline proposal `src.rav` (copy từ P-026, xem src/rav/SYNC.md).

Block 1 — data management: index ảnh của một tập (không nhãn) + Context `rav` trên tập đó + cache field (emb.*, hash.*).
Block 2 — proposal nhiều vòng: `run_workflow` (CONTRACTS §12.4) với router `fixed_budget` + judge `accept_all` (không cần người),
ghi `proposals/<tên>/s<seed>_k<K>/{export.csv, history.jsonl, meta.json}` — export.csv cùng cột với export của P-026.

Module này không chứa logic chọn mẫu: recipe là config `{reference, objectives, combiner, selector}` của P-026, component nằm ở src/rav.
Proposer chạy trong `no_oracle_access()` và chỉ thấy ảnh + field tính từ ảnh.
"""

from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import common as C
from . import data as D
from .oracle import no_oracle_access

FIELDS = ("emb.dinov2_s", "emb.clip_b32x3", "emb.clip_b32")   # field block 1 tính sẵn (hash.phash của P-026 chỉ cần cho ND-Rate: thêm khi dùng)
NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _art(cfg, rel) -> Path:
    return C.WORK / cfg["art"] / rel


# ---------- block 1: data management ----------

def write_index(cfg, split: str = "pool") -> Path:
    """`pool/<split>_index.csv` (image, path) — chỉ tên ảnh và đường dẫn, không nhãn. Source `bdd_images` đọc file này."""
    p = C.out(cfg, f"pool/{split}_index.csv")
    ids = C.read_ids(_art(cfg, f"splits/{split}_ids.txt"))
    paths = D.split_paths(cfg)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "path"])
        for i in ids:
            w.writerow([i, str(paths[i])])
    return p


def spec(cfg, split: str = "pool") -> dict:
    """Session spec của P-026 (CONTRACTS §7) cho tập ảnh BDD: source `bdd_images` → decoder `image_file` → unit `per_frame`."""
    index = _art(cfg, f"pool/{split}_index.csv")
    if not index.exists():
        write_index(cfg, split)
    return {"source": {"type": "bdd_images", "params": {"index": str(index)}},
            "decoder": {"type": "image_file"}, "units": {"type": "per_frame"}}


def make_ctx(cfg, split: str = "pool", **kwargs):
    """Context `rav` trên mọi ảnh của `split`, cache field ở `<art>/fields` (PackedFieldStore + BulkExecution)."""
    from src.rav.core.bulk_execution import BulkExecution
    from src.rav.core.context import Context
    from src.rav.core.execution import spec_hash
    from src.rav.core.registry import REGISTRY, discover
    from src.rav.store.packed_field_store import PackedFieldStore
    discover()
    s = spec(cfg, split)
    videos = list(REGISTRY.build("source", s["source"]).iter_videos())
    decoder, builder = REGISTRY.build("decoder", s["decoder"]), REGISTRY.build("unit", s["units"])
    units = sorted((u for v in videos for u in builder.build(v, decoder.frames(v))), key=lambda u: (u.video_id, u.t0))
    store = PackedFieldStore(_art(cfg, "fields"))
    return Context(units, execution=BulkExecution(store, spec_hash(decoder, builder)), videos={v.id: v for v in videos},
                   decoder=decoder, **kwargs)


def prepare_fields(cfg, split: str = "pool", fields=FIELDS) -> dict:
    """Tính (hoặc đọc cache) các field cho `split`; trả {field: shape}. Chạy trong guard: không đọc được nhãn."""
    out = {}
    with no_oracle_access():
        ctx = make_ctx(cfg, split)
        for name in fields:
            t0 = time.time()
            out[name] = tuple(np.asarray(ctx.field(name).values).shape)
            ctx._execution.store.flush()
            print(f"{split} {name}: {out[name]} ({time.time() - t0:.0f}s)", flush=True)
    return out


def image_of(unit_id: str) -> str:
    """unit id P-026 `"<video_id>:<t0>-<t1>"` → tên ảnh BDD `<video_id>.jpg`."""
    return unit_id.rsplit(":", 1)[0] + ".jpg"


# ---------- block 2: proposal nhiều vòng (không người duyệt) ----------

def run_recipe(cfg, name: str, recipe: str | dict, target_k: int, batch_size: int, seed: int, goal: str = "",
               split: str = "pool") -> Path:
    """Chạy một recipe nhiều vòng tới khi đủ `target_k` (hoặc hết ứng viên). `recipe` = tên recipe đăng ký sẵn của P-026
    (random / uniform / diversity) hoặc RecipeConfig đầy đủ (dict). Đã có export.csv thì bỏ qua."""
    from src.rav.core.registry import REGISTRY, discover
    from src.rav.core.types import Decision, RecipeConfig
    from src.rav.pipeline.propose import config_hash
    from src.rav.pipeline.workflow import history_records, run_workflow
    assert NAME_RE.match(name), f"tên nhánh {name!r}: chỉ chữ, số, _ và -"
    dst = _art(cfg, f"proposals/{name}/s{seed}_k{target_k}")
    if (dst / "export.csv").exists():
        return dst
    discover()
    rcfg = recipe if isinstance(recipe, dict) else None
    rname = rcfg["name"] if rcfg else recipe
    chash = config_hash(RecipeConfig.model_validate(rcfg) if rcfg else REGISTRY.recipe(rname))
    t0 = time.time()
    with no_oracle_access():
        ctx = make_ctx(cfg, split)
        router = REGISTRY.build("router", {"type": "fixed_budget", "params": {
            "recipe": rname, "config": rcfg, "batch_size": batch_size, "target_k": target_k}})
        history, stop = run_workflow(router, REGISTRY.build("judge", {"type": "accept_all"}),
                                     REGISTRY.build("feedback", {"type": "keep_drop"}), goal, seed, ctx,
                                     session_id=f"{name}/s{seed}_k{target_k}")
        ctx._execution.store.flush()
        by_id = {u.id: u for u in ctx.units}
        kept = [d.unit_id for d in history if isinstance(d, Decision) and d.action == "keep"]
        dst.mkdir(parents=True, exist_ok=True)
        with (dst / "history.jsonl").open("w", encoding="utf-8") as f:
            for r in history_records(history):
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        C.dump({"name": name, "recipe": rname, "config": rcfg, "config_hash": chash, "seed": seed, "target_k": target_k,
                "batch_size": batch_size, "goal": goal, "n_kept": len(kept), "n_rounds": len(ctx.proposals),
                "stop": stop.reason, "seconds": time.time() - t0, "split": split}, dst / "meta.json")
        REGISTRY.build("exporter", {"type": "csv_ids"}).export([by_id[i] for i in kept], dst, ctx)   # ghi sau cùng = xong
    print(f"{name} seed={seed} K={target_k}: {len(kept)} frame, {len(ctx.proposals)} vòng, dừng: {stop.reason} ({time.time() - t0:.0f}s)")
    return dst


def list_exports(cfg) -> pd.DataFrame:
    """Mọi export của block 2 (hoặc của P-026 / nơi khác, cùng cấu trúc thư mục): name, seed, k, path, n."""
    rows = []
    for p in sorted(_art(cfg, "proposals").glob("*/s*_k*/export.csv")):
        m = re.match(r"s(\d+)_k(\d+)$", p.parent.name)
        if m:
            rows.append({"name": p.parent.parent.name, "seed": int(m[1]), "k": int(m[2]), "path": p,
                         "n": sum(1 for _ in open(p, encoding="utf-8")) - 1})
    return pd.DataFrame(rows)


# ---------- tinh chỉnh trên dev (nhãn dev công khai) ----------

def load_proposal_config(cfg) -> dict:
    """configs/proposals.yaml (+ override `smoke` khi SMOKE)."""
    import yaml
    p = yaml.safe_load((C.REPO / "configs" / "proposals.yaml").read_text(encoding="utf-8"))
    smoke = p.pop("smoke", {})
    return {**p, **smoke} if cfg["smoke"] else p


def resolve_recipe(recipe, texts: list[str], tau: float | None):
    """Thay `$texts` / `$tau` trong RecipeConfig (dict) bằng giá trị đã chọn trên dev; tên recipe đăng ký sẵn giữ nguyên."""
    if isinstance(recipe, dict):
        return {k: resolve_recipe(v, texts, tau) for k, v in recipe.items()}
    if isinstance(recipe, list):
        return [resolve_recipe(v, texts, tau) for v in recipe]
    return texts if recipe == "$texts" else tau if recipe == "$tau" else recipe


def choose_texts_and_tau(cfg, pcfg: dict) -> dict:
    """Chọn trên dev, luật khai báo trước: câu (từng câu hoặc ensemble tất cả) có AP ranking cao nhất; τ theo `tau_rule`
    (`dev_f1` | `dev_precision>=p` | null). Ghi `proposals/dev_choice.json`."""
    from .metrics import average_precision, tau_dev_f1
    cands = {f"single:{t}": [t] for t in pcfg["texts"]} | {"ensemble:all": list(pcfg["texts"])}
    scored = {name: dev_text_scores(cfg, texts) for name, texts in cands.items()}
    ap = {name: average_precision(d.relevance, d.is_novel) for name, d in scored.items()}
    best = max(ap, key=ap.get)
    d = scored[best]
    rule, tau, info = pcfg.get("tau_rule"), None, {}
    if rule == "dev_f1":
        info = tau_dev_f1(d.relevance, d.is_novel)
        tau = info["tau"]
    elif rule and rule.startswith("dev_precision>="):
        target = float(rule.split(">=")[1])
        s = d.sort_values("relevance", ascending=False)
        prec = s.is_novel.cumsum().to_numpy() / np.arange(1, len(s) + 1)
        ok = np.flatnonzero(prec >= target)
        tau = float(s.relevance.iloc[ok[-1]]) if len(ok) else float(s.relevance.iloc[0])
        info = {"target_precision": target, "dev_n_above": int(ok[-1] + 1) if len(ok) else 1}
    out = {"rule": "max dev ranking AP", "dev_ap": ap, "best": best, "texts": cands[best], "tau_rule": rule, "tau": tau,
           **info, "dev_positive_rate": float(d.is_novel.mean())}
    C.dump(out, C.out(cfg, "proposals/dev_choice.json"))
    return out


def dev_text_scores(cfg, texts: list[str], negative=()) -> pd.DataFrame:
    """Relevance của `text_match` (cùng công thức, cùng field) trên ảnh dev + cờ có novel — để chọn câu mô tả / τ trên dev
    theo luật khai báo trước, không đụng pool hay final test."""
    from src.rav.components.objectives.text_match import _encode, text_relevance
    ctx = make_ctx(cfg, "dev")
    emb = ctx.field("emb.clip_b32x3").values
    ctx._execution.store.flush()
    r = text_relevance(emb, _encode(list(texts)), _encode(list(negative)) if negative else None)
    meta = C.load(_art(cfg, "splits/meta.json"))
    pub = pd.read_csv(_art(cfg, "splits/public_boxes.csv"))
    pos = set(pub.image[pub.cls == meta["novel"]])
    images = [image_of(u.id) for u in ctx.units]
    return pd.DataFrame({"image": images, "relevance": r, "is_novel": [i in pos for i in images]})
