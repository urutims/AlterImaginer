"""Parameter ranges and shared mapping utilities."""

from __future__ import annotations

from typing import Dict

PARAM_RANGES: Dict[str, tuple[float, float]] = {
    "exposure_ev": (-2.0, 2.0),
    "contrast": (0.70, 1.30),
    "gamma": (0.70, 1.30),
    "saturation": (0.50, 1.60),
    "temp": (-2.0, 2.0),
    "tint": (-2.0, 2.0),
}

WB_COEFFS = {
    "temp_r": 0.10,
    "temp_b": 0.10,
    "tint_r": -0.05,
    "tint_g": 0.10,
    "tint_b": -0.05,
}


def denorm_params(tanh_params: Dict[str, float]) -> Dict[str, float]:
    """Map tanh outputs in [-1, 1] to real parameter ranges.

    This function does not clamp to keep edge expressiveness.
    """

    real: Dict[str, float] = {}
    for key, (min_val, max_val) in PARAM_RANGES.items():
        value = tanh_params[key]
        real[key] = (value + 1.0) / 2.0 * (max_val - min_val) + min_val
    return real
