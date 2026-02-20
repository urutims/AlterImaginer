"""I/O helpers with ICC-aware sRGB conversion."""

from __future__ import annotations

from typing import Union
from io import BytesIO

import numpy as np
from PIL import Image
from PIL import UnidentifiedImageError

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
    _HAS_HEIF = True
except Exception:
    _HAS_HEIF = False

try:
    from PIL import ImageCms

    _HAS_IMAGECMS = True
except Exception:
    ImageCms = None
    _HAS_IMAGECMS = False


SourceType = Union[str, "Image.Image"]


def _convert_to_srgb(image: Image.Image) -> Image.Image:
    icc_profile = image.info.get("icc_profile")
    if not icc_profile:
        return image.convert("RGB")

    if not _HAS_IMAGECMS:
        raise RuntimeError(
            "ICC profile present but Pillow ImageCms is unavailable")

    srgb_profile = ImageCms.createProfile("sRGB")
    src_profile = ImageCms.ImageCmsProfile(BytesIO(icc_profile))
    return ImageCms.profileToProfile(image, src_profile, srgb_profile, outputMode="RGB")


def load_image_rgb_u8(source: SourceType) -> np.ndarray:
    if isinstance(source, Image.Image):
        image = source
    else:
        try:
            image = Image.open(source)
        except UnidentifiedImageError as exc:
            source_str = str(source).lower()
            if source_str.endswith((".heic", ".heif")) and not _HAS_HEIF:
                raise RuntimeError(
                    "HEIC/HEIFの読み込みには pillow-heif が必要です"
                ) from exc
            raise

    image = _convert_to_srgb(image)
    return np.asarray(image, dtype=np.uint8)


def save_image_rgb_u8(path: str, img_rgb_u8: np.ndarray) -> None:
    image = Image.fromarray(img_rgb_u8, mode="RGB")
    image.save(path, format="JPEG")
