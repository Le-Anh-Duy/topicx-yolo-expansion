"""`image_file`: "video" là một file ảnh — một frame duy nhất, đọc pixel bằng PIL."""

from __future__ import annotations

import numpy as np
from PIL import Image

from ...core.registry import register
from ...core.types import Frame, VideoRef
from .base import Decoder


@register("decoder")
class ImageFile(Decoder):
    """Video một frame: `frames` trả đúng một Frame (idx = 0, t = 0); `read` mở `video.uri` bằng PIL, trả RGB (H, W, 3) uint8.
    Dùng cho dataset ảnh tĩnh (vd BDD100K images) với source `bdd_images`; unit id = "<video_id>:0.000-0.000"."""

    name = "image_file"
    version = "1"

    def frames(self, video: VideoRef) -> list[Frame]:
        return [Frame(video.id, 0, 0.0)]

    def read(self, video: VideoRef, frame: Frame) -> np.ndarray:
        with Image.open(video.uri) as im:
            return np.asarray(im.convert("RGB"))
