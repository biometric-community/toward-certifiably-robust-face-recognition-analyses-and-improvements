"""Baseline and spectral-normalized (Lipschitz-aware) FR embedders."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class FaceEmbedder(nn.Module):
    def __init__(self, n_classes: int, emb_dim: int = 128, pretrained: bool = True, spectral: bool = False):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        net = models.resnet18(weights=weights)
        self.features = nn.Sequential(*list(net.children())[:-1])
        fc = nn.Linear(512, emb_dim)
        cls = nn.Linear(emb_dim, n_classes, bias=False)
        if spectral:
            fc = nn.utils.spectral_norm(fc)
            cls = nn.utils.spectral_norm(cls)
        self.fc = fc
        self.bn = nn.BatchNorm1d(emb_dim)
        self.classifier = cls
        self.spectral = spectral

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(self.features(x), 1)
        x = self.bn(self.fc(x))
        # LLN-style last-layer normalization (paper analyzes LLN headers)
        return F.normalize(x, p=2, dim=1)

    def forward(self, x: torch.Tensor):
        emb = self.embed(x)
        return self.classifier(emb), emb
