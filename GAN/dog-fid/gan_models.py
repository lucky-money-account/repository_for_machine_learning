"""DCGAN generator / discriminator for 64x64 dog images.

Standard DCGAN architecture (Radford et al. 2015) adapted to 64x64 RGB:

  Generator      z(100) -> 4x4 -> 8 -> 16 -> 32 -> 64   (ConvTranspose2d)
  Discriminator  64 -> 32 -> 16 -> 8 -> 4 -> FC -> sigmoid

The discriminator ends with a fully-connected layer + Sigmoid, so it returns a
**probability score in (0, 1)** per image. Train it with ``nn.BCELoss``
(do NOT use ``BCEWithLogitsLoss`` -- that would apply a second sigmoid).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, z_dim: int = 100, ngf: int = 64, nc: int = 3):
        super().__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Discriminator(nn.Module):
    """Discriminator returning a probability score in (0, 1).

    ``forward`` returns shape ``(N,)``: the probability that each image is real.
    """

    def __init__(self, ndf: int = 64, nc: int = 3):
        super().__init__()
        # 4 stride-2 convs: 64 -> 32 -> 16 -> 8 -> 4, channels -> ndf*8 (=512)
        self.features = nn.Sequential(
            SN(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            SN(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            SN(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            SN(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Flatten -> 512 * 4 * 4 = 8192 features -> FC -> probability score
        self.classifier = nn.Sequential(
            nn.Flatten(),
            SN(nn.Linear(ndf * 8 * 4 * 4, 1)),   # 全连接层，输出 1 个分数
            nn.Sigmoid(),                    # 压到 (0, 1) —— 概率分数
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x)).view(-1)


def weights_init(module: nn.Module) -> None:
    """DCGAN init: N(0, 0.02) for conv/linear, BN gamma ~ N(1, 0.02)."""
    classname = module.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif classname.find("Linear") != -1:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0.0)
