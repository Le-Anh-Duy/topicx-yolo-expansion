"""P1 — chuẩn bị video: source → decoder → unit builder → features (architecture §4.2). Nặng, 1 lần / video, cache.

Không đọc DB; component lấy qua registry theo tên trong spec (`{"type", "params"}`, CONTRACTS §7).
`spec` = phần session spec quyết định danh tính unit: `source`, `decoder`, `units`; mỗi mục `{"type", "params"}`.
Decoder serve frame theo yêu cầu nên P1 không ghi ảnh; chỉ feature được cache (qua `store`, do service truyền vào).
"""

from __future__ import annotations

from ..core.context import Context
from ..core.execution import Execution, spec_hash
from ..core.registry import REGISTRY, discover
from ..core.types import Frame, VideoRef

DEFAULT_SPEC = {
    "source": {"type": "local_folder", "params": {"root": "data/videos"}},
    "decoder": {"type": "pyav", "params": {"fps": 4}},
    "units": {"type": "per_frame"},
}
FIELDS = ("emb.dinov2_s", "hash.phash", "proj.pca2")   # field P1 tính sẵn (architecture §4.2)


def iter_videos(spec: dict):
    """Mọi file video của source, kể cả file trùng nội dung (cùng id)."""
    discover()
    return REGISTRY.build("source", spec["source"]).iter_videos()


def catalog(spec: dict) -> dict[str, VideoRef]:
    return {v.id: v for v in iter_videos(spec)}


def make_context(videos: list[VideoRef], spec: dict, store, **kwargs) -> Context:
    """Context trên pool = mọi unit của `videos`, thứ tự `(video_id, t0)` (CONTRACTS §13.1.5)."""
    discover()
    decoder = REGISTRY.build("decoder", spec["decoder"])
    builder = REGISTRY.build("unit", spec["units"])
    units = sorted((u for v in videos for u in builder.build(v, decoder.frames(v))), key=lambda u: (u.video_id, u.t0))
    return Context(units, execution=Execution(store, spec_hash(decoder, builder)),
                   videos={v.id: v for v in videos}, decoder=decoder, **kwargs)


def prepare(videos: list[VideoRef], spec: dict, store, **kwargs) -> Context:
    """Tính (hoặc đọc cache) mọi field P1 cho pool = `videos`; `kwargs` (vd `on_progress`) chuyển cho Context."""
    ctx = make_context(videos, spec, store, **kwargs)
    for name in FIELDS:
        ctx.field(name)
    return ctx


def is_prepared(video: VideoRef, spec: dict, store) -> bool:
    """Mọi field `per_sample=True` của P1 đã có cache cho video này (field theo pool tính khi tạo phiên, rất nhẹ)."""
    discover()
    part = (video.id, spec_hash(REGISTRY.build("decoder", spec["decoder"]), REGISTRY.build("unit", spec["units"])))
    features = [REGISTRY.feature_for(name)() for name in FIELDS]
    return all(store.get(f.key, part) is not None for f in features if f.per_sample)


def read_frame(video: VideoRef, idx: int, spec: dict):
    """Ảnh RGB của frame gốc `idx` (UI / kiểm tra upload), decode theo yêu cầu."""
    discover()
    return REGISTRY.build("decoder", spec["decoder"]).read(video, Frame(video.id, idx, idx / video.fps))
