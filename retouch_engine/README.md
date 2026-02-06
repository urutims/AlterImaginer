# retouch_engine

Shared image processing engine for collector, trainer, and infer apps.

## Quick usage
```python
from retouch_engine import apply_retouch, load_image_rgb_u8, save_image_rgb_u8

params = {
    "exposure_ev": 0.0,
    "contrast": 1.0,
    "gamma": 1.0,
    "saturation": 1.0,
    "temp": 0.0,
    "tint": 0.0,
}

img = load_image_rgb_u8("input.jpg")
out = apply_retouch(img, params)
save_image_rgb_u8("output.jpg", out)
```
