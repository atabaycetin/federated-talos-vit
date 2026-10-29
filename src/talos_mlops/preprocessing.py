"""Image preprocessing shared by training, export, benchmarking and serving.

Pure PIL + numpy so the serving image doesn't need torch. Training wraps these
functions instead of re-declaring transforms, which rules out the train/serve skew
the research loaders had (ImageNet stats for train, CIFAR stats for test).
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, UnidentifiedImageError

IMAGE_SIZE = 224
# DINO was pretrained with ImageNet statistics.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MAX_IMAGE_PIXELS = 4096 * 4096


class InvalidImageError(ValueError):
    pass


def decode_image(data: bytes) -> Image.Image:
    if not data:
        raise InvalidImageError("empty payload")
    try:
        img = Image.open(io.BytesIO(data))
        if img.width * img.height > MAX_IMAGE_PIXELS:
            raise InvalidImageError(f"image too large ({img.width}x{img.height})")
        img.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError(f"cannot decode image: {exc}") from exc
    return img


def resize_center_crop(img: Image.Image, size: int = IMAGE_SIZE) -> Image.Image:
    """Resize the shorter side to `size` (bicubic), then center-crop to a square."""
    img = img.convert("RGB")
    w, h = img.size
    scale = size / min(w, h)
    new_w, new_h = max(size, round(w * scale)), max(size, round(h * scale))
    img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
    left, top = (new_w - size) // 2, (new_h - size) // 2
    return img.crop((left, top, left + size, top + size))


def to_array(img: Image.Image) -> np.ndarray:
    """RGB PIL image of the final size -> normalized CHW float32."""
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return np.ascontiguousarray(arr.transpose(2, 0, 1))


def preprocess(img: Image.Image) -> np.ndarray:
    return to_array(resize_center_crop(img))


def preprocess_batch(images: list[Image.Image]) -> np.ndarray:
    return np.stack([preprocess(img) for img in images])


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)
