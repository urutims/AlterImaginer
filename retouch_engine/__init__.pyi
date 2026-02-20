from typing import Dict, Union

import numpy as np
from PIL import Image

PARAM_RANGES: Dict[str, tuple[float, float]]

SourceType = Union[str, Image.Image]


def denorm_params(tanh_params: Dict[str, float]) -> Dict[str, float]: ...


def apply_retouch(img_rgb_u8: np.ndarray,
                  params: Dict[str, float]) -> np.ndarray: ...


def load_image_rgb_u8(source: SourceType) -> np.ndarray: ...


def save_image_rgb_u8(path: str, img_rgb_u8: np.ndarray) -> None: ...
