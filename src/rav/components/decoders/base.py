"""Base của kind `decoder` (CONTRACTS §5 dòng 2, quyết định #37)."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from ...core.component import Component
from ...core.types import Frame, VideoRef


class Decoder(Component):
    """Serve frame theo yêu cầu: liệt kê frame lấy mẫu và decode pixel khi được hỏi; không ghi file ảnh."""

    kind: ClassVar[str] = "decoder"

    def frames(self, video: VideoRef) -> list[Frame]:
        """Frame lấy mẫu, tăng dần theo `t`; tất định từ metadata + Params (không cần decode)."""
        raise NotImplementedError

    def read(self, video: VideoRef, frame: Frame) -> np.ndarray:
        """Ảnh RGB (H, W, 3) uint8 của frame `frame.idx`."""
        raise NotImplementedError
