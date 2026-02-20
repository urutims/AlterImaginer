"""Streamlit inference app for Alter_Imagineer MVP."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

_retouch_engine = __import__(
    "retouch_engine",
    fromlist=["PARAM_RANGES", "apply_retouch",
              "load_image_rgb_u8", "save_image_rgb_u8"],
)
PARAM_RANGES = _retouch_engine.PARAM_RANGES
apply_retouch = _retouch_engine.apply_retouch
load_image_rgb_u8 = _retouch_engine.load_image_rgb_u8
save_image_rgb_u8 = _retouch_engine.save_image_rgb_u8

_trainer_model = __import__("trainer.model", fromlist=["build_model"])
build_model = _trainer_model.build_model


ARTIFACTS_DIR = ROOT_DIR / "artifacts"
DATASET_DIR = ROOT_DIR / "dataset"
AFTER_DIR = DATASET_DIR / "after"
OUTPUT_DIR = ROOT_DIR / "output"

DEFAULT_IMAGE_SIZE = 224


@st.cache_data
def load_config() -> Dict:
    config_path = ARTIFACTS_DIR / "config.json"
    if config_path.exists():
        return json.loads(config_path.read_text(encoding="utf-8"))
    return {
        "image_size": DEFAULT_IMAGE_SIZE,
        "param_order": list(PARAM_RANGES.keys()),
        "param_ranges": {k: [float(v[0]), float(v[1])] for k, v in PARAM_RANGES.items()},
        "preprocess": {
            "resize_short_side": DEFAULT_IMAGE_SIZE + 32,
            "center_crop": DEFAULT_IMAGE_SIZE,
            "normalize": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        },
    }


@st.cache_resource
def load_model() -> Tuple[torch.nn.Module, torch.device]:
    model_path = ARTIFACTS_DIR / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError("artifacts/model.pt not found")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model()
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, device


def build_transform(config: Dict) -> transforms.Compose:
    image_size = int(config.get("image_size", DEFAULT_IMAGE_SIZE))
    resize_size = int(config.get("preprocess", {}).get(
        "resize_short_side", image_size + 32))
    normalize = config.get("preprocess", {}).get("normalize", {})
    mean = normalize.get("mean", [0.485, 0.456, 0.406])
    std = normalize.get("std", [0.229, 0.224, 0.225])
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def denorm_params(raw: np.ndarray, param_order: List[str], param_ranges: Dict[str, List[float]]) -> Dict[str, float]:
    params: Dict[str, float] = {}
    for idx, key in enumerate(param_order):
        min_val, max_val = param_ranges[key]
        val = float(raw[idx])
        params[key] = (val + 1.0) / 2.0 * (max_val - min_val) + min_val
    return params


def maybe_find_gt_after(upload_name: str) -> Path | None:
    stem = Path(upload_name).stem
    if len(stem) == 4 and stem.isdigit():
        candidate = AFTER_DIR / f"{stem}.jpg"
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    st.set_page_config(page_title="Alter_Imagineer - Infer", layout="wide")
    st.title("Alter_Imagineer App")

    config = load_config()
    transform = build_transform(config)
    param_order = config.get("param_order", list(PARAM_RANGES.keys()))
    param_ranges = config.get("param_ranges", {k: [float(
        v[0]), float(v[1])] for k, v in PARAM_RANGES.items()})

    uploaded = st.file_uploader(
        "Upload JPG", type=["jpg", "jpeg", "png", "tif", "tiff"])
    if uploaded is None:
        st.info("Upload an image to run inference.")
        return

    before_pil = Image.open(uploaded).convert("RGB")
    before_np = load_image_rgb_u8(before_pil)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Before")
        st.image(before_np, width='content')

    run = st.button("Run Inference", type="primary")
    if not run:
        return

    with st.spinner("Running inference..."):
        model, device = load_model()
        input_tensor = transform(before_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            raw = model(input_tensor).cpu().numpy().squeeze(0)
        params = denorm_params(raw, param_order, param_ranges)
        after_np = apply_retouch(before_np, params)

    with col2:
        st.subheader("AI After")
        st.image(after_np, width='content')

    gt_path = maybe_find_gt_after(uploaded.name)
    if gt_path is not None:
        with col3:
            st.subheader("GT After")
            gt_np = load_image_rgb_u8(gt_path)
            st.image(gt_np, width='content')
    else:
        with col3:
            st.subheader("GT After")
            st.caption("No matching ID found in dataset/after.")

    st.subheader("Predicted Parameters")
    df = pd.DataFrame(
        [{"param": key, "value": float(params[key])} for key in param_order]
    )
    st.dataframe(df, width='content')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if st.button("Save AI After"):
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        out_path = OUTPUT_DIR / f"{timestamp}_after.jpg"
        save_image_rgb_u8(str(out_path), after_np)
        st.success(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
