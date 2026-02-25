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
def load_model(model_filename: str) -> Tuple[torch.nn.Module, torch.device]:
    model_path = ARTIFACTS_DIR / model_filename
    if not model_path.exists():
        raise FileNotFoundError(f"artifacts/{model_filename} not found")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model()
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, device


def list_available_models() -> List[str]:
    return sorted([path.name for path in ARTIFACTS_DIR.glob("*.pt")])


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


@st.cache_data
def get_landscape_pair_candidates(index_mtime_ns: int) -> List[Tuple[str, str]]:
    _ = index_mtime_ns
    index_path = DATASET_DIR / "index.csv"
    if not index_path.exists():
        return []

    df = pd.read_csv(index_path)
    candidates: List[Tuple[str, str]] = []
    for _, row in df.iterrows():
        before_path = DATASET_DIR / str(row["before_path"])
        after_path = DATASET_DIR / str(row["after_path"])
        if not before_path.exists() or not after_path.exists():
            continue
        try:
            with Image.open(before_path) as before_image:
                width, height = before_image.size
            if width > height:
                candidates.append((str(before_path), str(after_path)))
        except Exception:
            continue

    if not candidates:
        return []

    return list(dict.fromkeys(candidates))


def sample_landscape_pairs(max_count: int = 4) -> List[Tuple[Path, Path]]:
    index_path = DATASET_DIR / "index.csv"
    if not index_path.exists():
        return []

    index_mtime_ns = index_path.stat().st_mtime_ns
    candidate_strings = get_landscape_pair_candidates(index_mtime_ns)
    if not candidate_strings:
        return []

    unique_candidates = [(Path(before), Path(after))
                         for before, after in candidate_strings]
    if len(unique_candidates) <= max_count:
        return unique_candidates

    indices = np.random.choice(
        len(unique_candidates), size=max_count, replace=False)
    return [unique_candidates[int(i)] for i in indices]


def get_session_landscape_pairs(max_count: int = 4) -> List[Tuple[Path, Path]]:
    key = f"preview_landscape_pairs_{max_count}"
    if key not in st.session_state:
        st.session_state[key] = sample_landscape_pairs(max_count=max_count)
    return st.session_state[key]


@st.cache_data
def load_cached_preview_image(image_path: str, mtime_ns: int) -> np.ndarray:
    _ = mtime_ns
    return load_image_rgb_u8(Path(image_path))


def main() -> None:
    st.set_page_config(page_title="Alter_Imagineer - Infer", layout="wide")
    st.title("Alter_Imagineer App")

    config = load_config()
    transform = build_transform(config)
    param_order = config.get("param_order", list(PARAM_RANGES.keys()))
    param_ranges = config.get("param_ranges", {k: [float(
        v[0]), float(v[1])] for k, v in PARAM_RANGES.items()})

    model_options = list_available_models()
    if not model_options:
        st.error("artifacts 配下に .pt モデルが見つかりません。")
        st.stop()
    default_model = "wada_model.pt" if "wada_model.pt" in model_options else model_options[0]
    selected_model = st.selectbox(
        "使用モデル", options=model_options, index=model_options.index(default_model))

    st.subheader("加工後のレタッチの雰囲気")
    sampled_pairs = get_session_landscape_pairs(max_count=4)
    if sampled_pairs:
        preview_cols = st.columns(4)
        for index, (before_path, after_path) in enumerate(sampled_pairs):
            with preview_cols[index]:
                st.caption("Before")
                before_mtime = before_path.stat().st_mtime_ns
                st.image(load_cached_preview_image(str(before_path), before_mtime),
                         width='stretch')
                st.caption("After")
                after_mtime = after_path.stat().st_mtime_ns
                st.image(load_cached_preview_image(str(after_path), after_mtime),
                         width='stretch')
    else:
        st.caption("横長の Before→After ペアを表示できません。")

    st.subheader("画像をアップロードしてAIレタッチを試す")
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
        model, device = load_model(selected_model)
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
            st.subheader("真のAfter")
            gt_np = load_image_rgb_u8(gt_path)
            st.image(gt_np, width='content')
    else:
        with col3:
            st.subheader("真のAfter")
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
