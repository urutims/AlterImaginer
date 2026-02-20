"""Core retouch processing pipeline."""

from __future__ import annotations

from typing import Dict

import numpy as np

from .color import clamp01, hsv_to_rgb, linear_to_srgb, rgb_to_hsv, srgb_to_linear
from .config import WB_COEFFS


def apply_retouch(img_rgb_u8: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    """Apply retouch parameters to a uint8 RGB image.

    Args:
        img_rgb_u8: RGB uint8 array with shape (H, W, 3).
        params: Dict with keys: exposure_ev, contrast, gamma, saturation, temp, tint.

    Returns:
        RGB uint8 array with the same shape.
    """

    if img_rgb_u8.dtype != np.uint8:
        raise ValueError("img_rgb_u8 must be uint8")
    if img_rgb_u8.ndim != 3 or img_rgb_u8.shape[2] != 3:
        raise ValueError("img_rgb_u8 must have shape (H, W, 3)")

    rgb = img_rgb_u8.astype(np.float32) / 255.0

    rgb = srgb_to_linear(rgb)

    temp = float(params["temp"])
    tint = float(params["tint"])

    gain_r = 1.0 + WB_COEFFS["temp_r"] * temp + WB_COEFFS["tint_r"] * tint
    gain_g = 1.0 + WB_COEFFS["tint_g"] * tint
    gain_b = 1.0 - WB_COEFFS["temp_b"] * temp + WB_COEFFS["tint_b"] * tint

    gains = np.array([gain_r, gain_g, gain_b], dtype=np.float32)
    rgb = rgb * gains

    exposure_ev = float(params["exposure_ev"])
    rgb = rgb * (2.0 ** exposure_ev)

    contrast = float(params["contrast"])
    luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    rgb = (rgb - luma[..., None]) * contrast + luma[..., None]

    rgb = np.clip(rgb, 0.0, None)

    gamma = float(params["gamma"])
    rgb = np.power(rgb, 1.0 / gamma)

    hsv = rgb_to_hsv(rgb)
    saturation = float(params["saturation"])
    hsv[..., 1] = clamp01(hsv[..., 1] * saturation)
    rgb = hsv_to_rgb(hsv)

    rgb = clamp01(rgb)
    rgb = linear_to_srgb(rgb)
    rgb = clamp01(rgb)

    out = (rgb * 255.0).round().astype(np.uint8)
    return out
