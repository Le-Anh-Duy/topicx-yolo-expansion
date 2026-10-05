"""`duplicate_pairs`: cặp frame theo khoảng cách pHash, lấy mẫu theo từng mức để spot-check ngưỡng ND-Rate (CONTRACTS §9.10, #50)."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import Field

from ...core.registry import register
from ...core.types import ViewSpec
from .base import View, proposal

POPCOUNT = np.array([bin(i).count("1") for i in range(256)], np.uint8)


def hamming_matrix(h: np.ndarray) -> np.ndarray:
    """(N, N) khoảng cách Hamming giữa các hash 64-bit."""
    # ponytail: ma trận N×N×8 byte, ~11 MB với pool 1.200; pool vài chục nghìn unit thì tính theo khối
    x = (h[:, None] ^ h[None, :]).view(np.uint8).reshape(len(h), len(h), 8)
    return POPCOUNT[x].sum(-1, dtype=np.int64)


@register("view")
class DuplicatePairs(View):
    """Mọi cặp (a, b) trong tập `on` (pool / kept / picks của proposal `proposal_n`, mặc định mới nhất), khoảng cách = Hamming của field `field`
    (mặc định pHash). `hist[d]` = số cặp có khoảng cách đúng d (0..64). `pairs` = với mỗi d trong `distances`, tối đa `per_distance` cặp
    có khoảng cách đúng d, chọn ngẫu nhiên theo `seed` (tái lập). Người xem từng mức d để biết tới đâu thì còn là "gần trùng"."""

    name = "duplicate_pairs"
    version = "1"

    class Params(View.Params):
        field: str = "hash.phash"
        on: Literal["pool", "kept", "proposal"] = "pool"
        proposal_n: int | None = None
        distances: list[int] = Field(default_factory=lambda: [0, 2, 4, 6, 8, 10, 12, 14, 16])
        per_distance: int = Field(4, ge=1)
        seed: int = 0

    def build(self, ctx) -> ViewSpec:
        if self.params.on == "kept":
            units = [u for u in ctx.units if u.id in ctx.state.kept]
        elif self.params.on == "proposal":
            p = proposal(ctx, self.params.proposal_n)
            units = [pick.unit for pick in p.picks] if p else []
        else:
            units = list(ctx.units)
        hist = [0] * 65
        pairs = []
        if len(units) >= 2:
            h = np.asarray(ctx.field(self.params.field, units).values, dtype=np.uint64)
            ia, ib = np.triu_indices(len(units), k=1)
            dist = hamming_matrix(h)[ia, ib]
            hist = np.bincount(dist, minlength=65).tolist()
            rng = np.random.default_rng(self.params.seed)
            for d in self.params.distances:
                idx = np.flatnonzero(dist == d)
                for k in sorted(rng.choice(idx, min(len(idx), self.params.per_distance), replace=False)):
                    a, b = units[ia[k]], units[ib[k]]
                    pairs.append({"a": a.id, "b": b.id, "distance": int(d),
                                  "a_frame": [a.anchor.video_id, a.anchor.idx], "b_frame": [b.anchor.video_id, b.anchor.idx]})
        return ViewSpec(self.name, {"hist": hist, "pairs": pairs})
