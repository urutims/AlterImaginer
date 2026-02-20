"""Training entry point for parameter regression."""

from __future__ import annotations
from trainer.model import build_model
from trainer.dataset import RetouchDataset, PARAM_ORDER
from retouch_engine import PARAM_RANGES

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True,
                        help="Dataset root directory (absolute or project-relative)")
    parser.add_argument("--out", type=str, required=True,
                        help="Artifacts output directory (absolute or project-relative)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_cli_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def save_config(out_dir: Path, image_size: int) -> None:
    config = {
        "image_size": image_size,
        "param_order": PARAM_ORDER,
        "param_ranges": {k: [float(v[0]), float(v[1])] for k, v in PARAM_RANGES.items()},
        "preprocess": {
            "resize_short_side": image_size + 32,
            "center_crop": image_size,
            "normalize": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        },
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def train_one_epoch(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total = 0.0
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        preds = model(images)
        loss = loss_fn(preds, targets)
        loss.backward()
        optimizer.step()
        total += loss.item() * images.size(0)
    return total / max(len(loader.dataset), 1)


def eval_one_epoch(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            preds = model(images)
            loss = loss_fn(preds, targets)
            total += loss.item() * images.size(0)
    return total / max(len(loader.dataset), 1)


def main() -> None:
    args = parse_args()
    artifacts_dir = PROJECT_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    data_dir = resolve_cli_path(args.data)
    out_dir = resolve_cli_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = RetouchDataset(data_dir, args.image_size)
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(args.seed)
    train_set, val_set = random_split(
        dataset, [train_size, val_size], generator=generator)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=2, pin_memory=(device.type == "cuda"))
    model = build_model().to(device)

    loss_fn = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device)
        val_loss = eval_one_epoch(model, val_loader, loss_fn, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(
            f"Epoch {epoch + 1}/{args.epochs} train={train_loss:.6f} val={val_loss:.6f}")

    torch.save(model.state_dict(), out_dir / "model.pt")
    save_config(out_dir, args.image_size)
    (out_dir / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
