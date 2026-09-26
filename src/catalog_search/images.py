"""Bounded image decoding and identical catalog/query preprocessing."""

import io
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import MAX_IMAGE_BYTES, MAX_IMAGE_PIXELS

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


class ImageError(ValueError):
    """An invalid or oversized user image."""


@dataclass(frozen=True)
class Crop:
    """Coordinates in [0, 1], relative to the EXIF-oriented image."""

    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self):
        values = (self.left, self.top, self.right, self.bottom)
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in values):
            raise ImageError("Координаты области должны быть в пределах от 0 до 1")
        if self.left >= self.right or self.top >= self.bottom:
            raise ImageError("Область поиска должна иметь положительную площадь")


def decode_image(data: bytes, crop: Crop | None = None) -> Image.Image:
    if len(data) > MAX_IMAGE_BYTES:
        raise ImageError("Файл больше 20 МБ")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as source:
                if source.format not in {"JPEG", "PNG", "WEBP"}:
                    raise ImageError("Поддерживаются только JPEG, PNG и WebP")
                if source.width * source.height > MAX_IMAGE_PIXELS:
                    raise ImageError("Изображение больше 30 мегапикселей")
                oriented = ImageOps.exif_transpose(source)
                # Composite transparency instead of turning transparent pixels black.
                rgba = oriented.convert("RGBA")
                image = Image.new("RGBA", rgba.size, "white")
                image.alpha_composite(rgba)
                image = image.convert("RGB")
    except (
        UnidentifiedImageError,
        OSError,
        SyntaxError,
        Image.DecompressionBombWarning,
        Image.DecompressionBombError,
    ) as exc:
        raise ImageError("Не удалось прочитать изображение") from exc
    if crop:
        width, height = image.size
        box = (
            round(crop.left * width),
            round(crop.top * height),
            round(crop.right * width),
            round(crop.bottom * height),
        )
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ImageError("Выбранная область меньше одного пикселя")
        image = image.crop(box)
    return image


def read_image(path: Path, crop: Crop | None = None) -> Image.Image:
    with path.open("rb") as stream:
        return decode_image(stream.read(MAX_IMAGE_BYTES + 1), crop)


def prepare_tensor(image: Image.Image) -> np.ndarray:
    """Fit the full frame into 224px; no center crop of tall product photos."""
    fitted = ImageOps.contain(image, (224, 224), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (224, 224), (124, 116, 104))
    canvas.paste(fitted, ((224 - fitted.width) // 2, (224 - fitted.height) // 2))
    pixels = np.asarray(canvas, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return np.ascontiguousarray(((pixels - mean) / std).transpose(2, 0, 1))


def thumbnail_bytes(image: Image.Image) -> bytes:
    thumbnail = ImageOps.contain(image, (480, 480), Image.Resampling.LANCZOS)
    output = io.BytesIO()
    thumbnail.save(output, "JPEG", quality=85)
    return output.getvalue()
