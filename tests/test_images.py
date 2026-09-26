import io

import numpy as np
import pytest
from PIL import Image

from catalog_search.images import Crop, ImageError, decode_image, prepare_tensor


def png(image):
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


def test_letterbox_keeps_top_and_bottom_of_tall_photo():
    image = Image.new("RGB", (100, 400), "blue")
    image.paste("red", (0, 0, 100, 50))
    tensor = prepare_tensor(image)
    assert tensor.shape == (3, 224, 224)
    assert tensor[0, 5, 112] > tensor[2, 5, 112]
    assert tensor[2, 215, 112] > tensor[0, 215, 112]
    assert np.isfinite(tensor).all()


def test_transparency_and_crop():
    image = decode_image(png(Image.new("RGBA", (100, 200), (0, 0, 0, 0))), Crop(0, 0, 0.5, 0.5))
    assert image.size == (50, 100)
    assert image.getpixel((0, 0)) == (255, 255, 255)


def test_exif_rotation_happens_before_crop():
    image = Image.new("RGB", (40, 80), "red")
    exif = Image.Exif()
    exif[274] = 6
    data = io.BytesIO()
    image.save(data, "JPEG", exif=exif)
    assert decode_image(data.getvalue()).size == (80, 40)


@pytest.mark.parametrize("coords", [(0, 0, 0, 1), (-1, 0, 1, 1), (0, 0, float("nan"), 1)])
def test_invalid_crop(coords):
    with pytest.raises(ImageError):
        Crop(*coords)


def test_bad_image():
    with pytest.raises(ImageError):
        decode_image(b"not an image")
