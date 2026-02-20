"""Dataset utilities for parameter regression training."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

PARAM_ORDER = [
    "exposure_ev",
    "contrast",
    "gamma",
    "saturation",
    "temp",
    "tint",
]


def build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


class RetouchDataset(Dataset):
    def __init__(self, root_dir: Path, image_size: int) -> None:
        self.root_dir = root_dir
        self.index_path = root_dir / "index.csv"
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"index.csv not found at {self.index_path}")

        self.df = pd.read_csv(self.index_path)
        self.transform = build_transform(image_size)
        self.param_ranges = load_param_ranges(self.index_path)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        image_path = self.root_dir / row["before_path"]
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)

        target_raw = [float(row[key]) for key in PARAM_ORDER]
        target_norm = []
        for key, value in zip(PARAM_ORDER, target_raw):
            min_val, max_val = self.param_ranges[key]
            scaled = 2.0 * ((value - min_val) / (max_val - min_val)) - 1.0
            target_norm.append(float(max(-1.0, min(1.0, scaled))))
        target = torch.tensor(target_norm, dtype=torch.float32)
        return image_tensor, target


def load_param_ranges(_: Path) -> Dict[str, Tuple[float, float]]:
    from retouch_engine import PARAM_RANGES

    return {key: (float(v[0]), float(v[1])) for key, v in PARAM_RANGES.items()}
