"""Color space conversions and helpers."""

from __future__ import annotations

import numpy as np


def clamp01(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0.0, 1.0)


def srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    threshold = 0.04045
    below = srgb <= threshold
    above = ~below
    out = np.empty_like(srgb)
    out[below] = srgb[below] / 12.92
    out[above] = ((srgb[above] + 0.055) / 1.055) ** 2.4
    return out


def linear_to_srgb(lin: np.ndarray) -> np.ndarray:
    threshold = 0.0031308
    below = lin <= threshold
    above = ~below
    out = np.empty_like(lin)
    out[below] = lin[below] * 12.92
    out[above] = 1.055 * (lin[above] ** (1.0 / 2.4)) - 0.055
    return out


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]

    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc
    delta = maxc - minc

    s = np.zeros_like(maxc)
    np.divide(delta, maxc, out=s, where=maxc != 0)

    h = np.zeros_like(maxc)
    mask = delta != 0

    r_mask = mask & (maxc == r)
    g_mask = mask & (maxc == g)
    b_mask = mask & (maxc == b)

    h[r_mask] = ((g[r_mask] - b[r_mask]) / delta[r_mask]) % 6.0
    h[g_mask] = ((b[g_mask] - r[g_mask]) / delta[g_mask]) + 2.0
    h[b_mask] = ((r[b_mask] - g[b_mask]) / delta[b_mask]) + 4.0

    h = h / 6.0
    hsv = np.stack([h, s, v], axis=-1)
    return hsv


def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    h = hsv[..., 0]
    s = hsv[..., 1]
    v = hsv[..., 2]

    h6 = h * 6.0
    c = v * s
    x = c * (1.0 - np.abs((h6 % 2.0) - 1.0))
    m = v - c

    z = np.zeros_like(h)

    conds = [
        (h6 >= 0) & (h6 < 1),
        (h6 >= 1) & (h6 < 2),
        (h6 >= 2) & (h6 < 3),
        (h6 >= 3) & (h6 < 4),
        (h6 >= 4) & (h6 < 5),
        (h6 >= 5) & (h6 < 6),
    ]

    rgb = np.stack([z, z, z], axis=-1)

    rgb[conds[0]] = np.stack([c[conds[0]], x[conds[0]], z[conds[0]]], axis=-1)
    rgb[conds[1]] = np.stack([x[conds[1]], c[conds[1]], z[conds[1]]], axis=-1)
    rgb[conds[2]] = np.stack([z[conds[2]], c[conds[2]], x[conds[2]]], axis=-1)
    rgb[conds[3]] = np.stack([z[conds[3]], x[conds[3]], c[conds[3]]], axis=-1)
    rgb[conds[4]] = np.stack([x[conds[4]], z[conds[4]], c[conds[4]]], axis=-1)
    rgb[conds[5]] = np.stack([c[conds[5]], z[conds[5]], x[conds[5]]], axis=-1)

    return rgb + m[..., None]
