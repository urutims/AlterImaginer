"""Fast desktop collector app using Dear PyGui."""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image

try:
    from retouch_engine import PARAM_RANGES, apply_retouch, load_image_rgb_u8, save_image_rgb_u8
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(PROJECT_ROOT))
    from retouch_engine import PARAM_RANGES, apply_retouch, load_image_rgb_u8, save_image_rgb_u8

import dearpygui.dearpygui as dpg

ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT_DIR / "dataset"
BEFORE_DIR = DATASET_DIR / "before"
AFTER_DIR = DATASET_DIR / "after"
PARAMS_DIR = DATASET_DIR / "params"
INDEX_CSV = DATASET_DIR / "index.csv"
STATE_PATH = ROOT_DIR / ".collector_state.json"

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

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
CONTROLS_WIDTH = 360
MARGIN = 10
PREVIEW_WINDOW_WIDTH = WINDOW_WIDTH - CONTROLS_WIDTH - MARGIN * 3
PREVIEW_WINDOW_HEIGHT = WINDOW_HEIGHT - MARGIN * 2

STATE = {
    "before": None,
    "after": None,
    "preview_before": None,
    "path": None,
    "last_dir": None,
    "before_tex": None,
    "after_tex": None,
    "tex_counter": 0,
    "is_saving": False,
    "preview_w": 1,
    "preview_h": 1,
    "preview_max_w": 420,
    "preview_max_h": 640,
    "layout": "horizontal",
}


def ensure_dataset_dirs() -> None:
    BEFORE_DIR.mkdir(parents=True, exist_ok=True)
    AFTER_DIR.mkdir(parents=True, exist_ok=True)
    PARAMS_DIR.mkdir(parents=True, exist_ok=True)


def load_state() -> None:
    if not STATE_PATH.exists():
        return
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return
    last_dir = data.get("last_dir")
    if isinstance(last_dir, str):
        STATE["last_dir"] = last_dir


def save_state() -> None:
    payload = {"last_dir": STATE.get("last_dir")}
    STATE_PATH.write_text(json.dumps(payload), encoding="utf-8")


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
    header = ["id", "before_path", "after_path", *PARAM_ORDER]
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

    tmp_suffix = f".tmp-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
    before_tmp = before_path.with_suffix(before_path.suffix + tmp_suffix)
    after_tmp = after_path.with_suffix(after_path.suffix + tmp_suffix)
    params_tmp = params_path.with_suffix(params_path.suffix + tmp_suffix)

    save_image_rgb_u8(str(before_tmp), before_img)
    save_image_rgb_u8(str(after_tmp), after_img)
    params_tmp.write_text(
        "{\n" +
        ",\n".join(
            [f"  \"{k}\": {params[k]:.6f}" for k in PARAM_ORDER]) + "\n}",
        encoding="utf-8",
    )

    before_tmp.replace(before_path)
    after_tmp.replace(after_path)
    params_tmp.replace(params_path)

    return before_path, after_path, params_path


def params_from_ui() -> Dict[str, float]:
    return {key: float(dpg.get_value(f"param_{key}")) for key in PARAM_ORDER}


def u8_to_texture(img_u8: np.ndarray) -> Tuple[int, int, np.ndarray]:
    height, width = img_u8.shape[:2]
    if img_u8.shape[2] == 4:
        rgba = img_u8.astype(np.float32) / 255.0
    else:
        rgb = img_u8.astype(np.float32) / 255.0
        alpha = np.ones((height, width, 1), dtype=np.float32)
        rgba = np.concatenate([rgb, alpha], axis=2)
    return width, height, rgba.ravel()


def make_preview_rgb(img_u8: np.ndarray) -> np.ndarray:
    image = Image.fromarray(img_u8, mode="RGB")
    target_w = STATE["preview_max_w"]
    target_h = STATE["preview_max_h"]
    image.thumbnail((target_w, target_h), Image.LANCZOS)
    return np.asarray(image, dtype=np.uint8)


def make_preview_rgba(img_u8: np.ndarray) -> np.ndarray:
    image = Image.fromarray(img_u8, mode="RGB")
    target_w = STATE["preview_max_w"]
    target_h = STATE["preview_max_h"]
    image.thumbnail((target_w, target_h), Image.LANCZOS)
    return np.asarray(image.convert("RGBA"), dtype=np.uint8)


def update_texture(tag: str, img_u8: np.ndarray) -> None:
    preview = make_preview_rgba(img_u8)
    _, _, data = u8_to_texture(preview)
    dpg.set_value(tag, data)


def _new_texture_tag(prefix: str) -> str:
    STATE["tex_counter"] += 1
    return f"{prefix}_{STATE['tex_counter']}"


def rebuild_textures() -> None:
    width = STATE["preview_w"]
    height = STATE["preview_h"]
    blank = np.zeros((height, width, 4), dtype=np.float32)

    if not dpg.does_item_exist("texture_registry"):
        dpg.add_texture_registry(show=False, tag="texture_registry")

    old_before = STATE.get("before_tex")
    old_after = STATE.get("after_tex")
    new_before = _new_texture_tag("before_texture")
    new_after = _new_texture_tag("after_texture")

    dpg.add_dynamic_texture(
        width,
        height,
        blank.ravel(),
        tag=new_before,
        parent="texture_registry",
    )
    dpg.add_dynamic_texture(
        width,
        height,
        blank.ravel(),
        tag=new_after,
        parent="texture_registry",
    )

    STATE["before_tex"] = new_before
    STATE["after_tex"] = new_after

    if dpg.does_item_exist("before_image"):
        dpg.configure_item("before_image", texture_tag=new_before)
    if dpg.does_item_exist("after_image"):
        dpg.configure_item("after_image", texture_tag=new_after)

    for tag in (old_before, old_after):
        if tag and dpg.does_item_exist(tag):
            dpg.delete_item(tag)


def refresh_after() -> None:
    if STATE["preview_before"] is None:
        return
    params = params_from_ui()
    preview_after = apply_retouch(STATE["preview_before"], params)
    STATE["after"] = preview_after
    update_texture(STATE["after_tex"], preview_after)


def on_slider_change(sender, app_data, user_data) -> None:
    refresh_after()


def on_open_file(sender, app_data, user_data) -> None:
    selection = app_data.get("selections")
    if not selection:
        return
    path = next(iter(selection.values()))
    load_path(path)


def on_save(sender, app_data, user_data) -> None:
    if STATE["before"] is None or STATE["is_saving"]:
        return
    STATE["is_saving"] = True
    dpg.configure_item("save_button", enabled=False)
    params = params_from_ui()
    sample_id = get_next_id()
    try:
        full_after = apply_retouch(STATE["before"], params)
        save_sample(sample_id, STATE["before"], full_after, params)
        write_index_row(sample_id, params)
    except Exception as exc:
        dpg.set_value("status_text", f"Save failed: {exc}")
    else:
        dpg.set_value("status_text", f"Saved as ID {sample_id}")
        dpg.set_value("save_modal_text", f"Saved as ID {sample_id}")
        dpg.configure_item("save_modal", show=True)
    finally:
        STATE["is_saving"] = False
        dpg.configure_item("save_button", enabled=True)


def load_path(path: str) -> None:
    STATE["last_dir"] = str(Path(path).parent)
    save_state()
    STATE["path"] = path
    before = load_image_rgb_u8(path)
    STATE["before"] = before
    height, width = before.shape[:2]
    if width >= height:
        STATE["layout"] = "vertical"
        STATE["preview_max_w"] = PREVIEW_WINDOW_WIDTH
        STATE["preview_max_h"] = (PREVIEW_WINDOW_HEIGHT - MARGIN) // 2
    else:
        STATE["layout"] = "horizontal"
        STATE["preview_max_w"] = (PREVIEW_WINDOW_WIDTH - MARGIN) // 2
        STATE["preview_max_h"] = PREVIEW_WINDOW_HEIGHT

    STATE["preview_before"] = make_preview_rgb(before)
    STATE["preview_h"], STATE["preview_w"] = STATE["preview_before"].shape[:2]
    rebuild_textures()
    rebuild_preview_content()
    update_texture(STATE["before_tex"], before)
    dpg.set_value("path_text", f"Path: {path}")
    dpg.configure_item("save_button", enabled=True)
    refresh_after()


def on_open_explorer(sender, app_data, user_data) -> None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        dpg.set_value("status_text", f"Explorer dialog failed: {exc}")
        return

    root = tk.Tk()
    root.withdraw()
    root.wm_attributes("-topmost", 1)
    initial_dir = STATE.get("last_dir") or str(ROOT_DIR)
    path = filedialog.askopenfilename(
        title="Select image",
        initialdir=initial_dir,
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.tif *.tiff"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    if not path:
        return
    load_path(path)


def on_viewport_drop(sender, app_data) -> None:
    if not app_data:
        return
    path = app_data[0]
    if not Path(path).is_file():
        return
    load_path(path)


def rebuild_preview_content() -> None:
    if dpg.does_item_exist("preview_content"):
        dpg.delete_item("preview_content")

    horizontal = STATE["layout"] == "horizontal"
    with dpg.group(horizontal=horizontal, parent="preview_window", tag="preview_content"):
        with dpg.group():
            dpg.add_text("Before")
            dpg.add_image(STATE["before_tex"], tag="before_image")
        with dpg.group():
            dpg.add_text("After")
            dpg.add_image(STATE["after_tex"], tag="after_image")


def build_ui() -> None:
    dpg.create_context()
    dpg.create_viewport(
        title="Retouch Mimic - Collector (DPG)", width=WINDOW_WIDTH, height=WINDOW_HEIGHT)

    if not dpg.does_item_exist("texture_registry"):
        dpg.add_texture_registry(show=False, tag="texture_registry")
    rebuild_textures()

    default_path = STATE.get("last_dir") or str(ROOT_DIR)

    with dpg.file_dialog(
        directory_selector=False,
        show=False,
        callback=on_open_file,
        tag="file_dialog",
        file_count=1,
        default_path=default_path,
    ):
        dpg.add_file_extension(".jpg")
        dpg.add_file_extension(".jpeg")
        dpg.add_file_extension(".png")
        dpg.add_file_extension(".tif")
        dpg.add_file_extension(".tiff")

    with dpg.window(
        label="Controls",
        width=CONTROLS_WIDTH,
        height=WINDOW_HEIGHT - MARGIN * 2,
        pos=(MARGIN, MARGIN),
        tag="controls_window",
    ):
        dpg.add_button(label="Open Image",
                       callback=lambda: dpg.show_item("file_dialog"))
        dpg.add_button(label="Open via Explorer", callback=on_open_explorer)
        dpg.add_text("Path: ", tag="path_text")
        dpg.add_spacer(height=8)
        for key in PARAM_ORDER:
            min_val, max_val = PARAM_RANGES[key]
            dpg.add_slider_float(
                label=key,
                min_value=float(min_val),
                max_value=float(max_val),
                default_value=float(DEFAULT_PARAMS[key]),
                tag=f"param_{key}",
                callback=on_slider_change,
            )
        dpg.add_spacer(height=8)
        dpg.add_button(label="Save sample", tag="save_button",
                       callback=on_save, enabled=False)
        dpg.add_text("", tag="status_text")

    with dpg.window(
        label="Save",
        modal=True,
        show=False,
        no_title_bar=False,
        tag="save_modal",
        width=260,
        height=120,
    ):
        dpg.add_text("", tag="save_modal_text")
        dpg.add_spacer(height=8)
        dpg.add_button(
            label="OK",
            width=80,
            callback=lambda: dpg.configure_item("save_modal", show=False),
        )

    with dpg.window(
        label="Preview",
        width=PREVIEW_WINDOW_WIDTH,
        height=PREVIEW_WINDOW_HEIGHT,
        pos=(CONTROLS_WIDTH + MARGIN * 2, MARGIN),
        tag="preview_window",
    ):
        rebuild_preview_content()

    if hasattr(dpg, "set_viewport_drop_callback"):
        dpg.set_viewport_drop_callback(on_viewport_drop)

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()


def main() -> None:
    ensure_dataset_dirs()
    load_state()
    build_ui()


if __name__ == "__main__":
    main()
