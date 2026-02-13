"""Model definition for parameter regression."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def build_model() -> nn.Module:
    backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
    backbone.fc = nn.Sequential(
        nn.Linear(512, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 6),
        nn.Tanh(),
    )
    return backbone
