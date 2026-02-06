"""Streamlit app for collecting retouch parameters and outputs."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import streamlit as st
from PIL import Image

try:
    from retouch_engine import PARAM_RANGES, apply_retouch, load_image_rgb_u8, save_image_rgb_u8
except ModuleNotFoundError:
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(PROJECT_ROOT))
    from retouch_engine import PARAM_RANGES, apply_retouch, load_image_rgb_u8, save_image_rgb_u8

ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT_DIR / "dataset"
BEFORE_DIR = DATASET_DIR / "before"
AFTER_DIR = DATASET_DIR / "after"
PARAMS_DIR = DATASET_DIR / "params"
INDEX_CSV = DATASET_DIR / "index.csv"

PARAM_ORDER = [
    "exposure_ev",
    "contrast",
    "gamma",
    "saturation",
    "temp",
    "tint",
]

DEFAULT_PARAMS = {
    "exposure_ev": 0.0,
    "contrast": 1.0,
    "gamma": 1.0,
    "saturation": 1.0,
    "temp": 0.0,
    "tint": 0.0,
}


def ensure_dataset_dirs() -> None:
    BEFORE_DIR.mkdir(parents=True, exist_ok=True)
    AFTER_DIR.mkdir(parents=True, exist_ok=True)
    PARAMS_DIR.mkdir(parents=True, exist_ok=True)


def _parse_id(name: str) -> int | None:
    stem = Path(name).stem
    if len(stem) != 4 or not stem.isdigit():
        return None
    return int(stem)


def get_next_id() -> str:
    ensure_dataset_dirs()
    max_id = 0
    for path in BEFORE_DIR.glob("*.jpg"):
        parsed = _parse_id(path.name)
        if parsed is not None:
            max_id = max(max_id, parsed)
    return f"{max_id + 1:04d}"


def write_index_row(sample_id: str, params: Dict[str, float]) -> None:
    header = [
        "id",
        "before_path",
        "after_path",
        *PARAM_ORDER,
    ]
    row = {
        "id": sample_id,
        "before_path": f"before/{sample_id}.jpg",
        "after_path": f"after/{sample_id}.jpg",
    }
    row.update({key: params[key] for key in PARAM_ORDER})

    file_exists = INDEX_CSV.exists()
    with INDEX_CSV.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def save_sample(
    sample_id: str,
    before_img: np.ndarray,
    after_img: np.ndarray,
    params: Dict[str, float],
) -> Tuple[Path, Path, Path]:
    ensure_dataset_dirs()
    before_path = BEFORE_DIR / f"{sample_id}.jpg"
    after_path = AFTER_DIR / f"{sample_id}.jpg"
    params_path = PARAMS_DIR / f"{sample_id}.json"

    tmp_suffix = f".tmp-{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"
    before_tmp = before_path.with_suffix(before_path.suffix + tmp_suffix)
    after_tmp = after_path.with_suffix(after_path.suffix + tmp_suffix)
    params_tmp = params_path.with_suffix(params_path.suffix + tmp_suffix)

    save_image_rgb_u8(str(before_tmp), before_img)
    save_image_rgb_u8(str(after_tmp), after_img)
    params_tmp.write_text(
        "{\n" + ",\n".join(
            [f"  \"{k}\": {params[k]:.6f}" for k in PARAM_ORDER]
        ) + "\n}",
        encoding="utf-8",
    )

    before_tmp.replace(before_path)
    after_tmp.replace(after_path)
    params_tmp.replace(params_path)

    return before_path, after_path, params_path


def build_params() -> Dict[str, float]:
    params = {}
    for key in PARAM_ORDER:
        min_val, max_val = PARAM_RANGES[key]
        default = DEFAULT_PARAMS[key]
        params[key] = st.slider(
            key,
            min_value=float(min_val),
            max_value=float(max_val),
            value=float(default),
            step=0.01,
        )
    return params


def main() -> None:
    st.set_page_config(page_title="Retouch Collector", layout="wide")
    st.title("Retouch Mimic - Collector")

    ensure_dataset_dirs()

    uploaded = st.file_uploader(
        "Upload JPG", type=["jpg", "jpeg", "png", "tif", "tiff"])
    if uploaded is None:
        st.info("Upload an image to start.")
        return

    image = Image.open(uploaded)
    before_img = load_image_rgb_u8(image)

    params = build_params()
    after_img = apply_retouch(before_img, params)

    col_before, col_after = st.columns(2)
    with col_before:
        st.subheader("Before")
        st.image(before_img, use_container_width=True)
    with col_after:
        st.subheader("After")
        st.image(after_img, use_container_width=True)

    if st.button("Save sample", type="primary"):
        sample_id = get_next_id()
        try:
            save_sample(sample_id, before_img, after_img, params)
            write_index_row(sample_id, params)
        except Exception as exc:
            st.error(f"Save failed: {exc}")
        else:
            st.success(f"Saved as ID {sample_id}")


if __name__ == "__main__":
    main()
