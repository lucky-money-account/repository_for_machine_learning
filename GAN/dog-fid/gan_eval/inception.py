"""InceptionV3 feature extractor used by all GAN metrics.

We deliberately use torchvision's ImageNet-pretrained InceptionV3
(weights hosted on download.pytorch.org, reliably reachable from mainland
China) instead of torch-fidelity's GitHub-release weights, because the
latter frequently times out behind the GFW.

The extractor returns two things in a single forward pass:

* the 2048-d pooled feature  -> used by FID / KID / MiFID / precision-recall
* the 1000-d class logits    -> used by Inception Score

Reference:
  - FID: Heusel et al. 2017, https://arxiv.org/abs/1706.08500
  - MiFID: Bai et al. 2021, https://arxiv.org/abs/2106.03062
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import Inception_V3_Weights, inception_v3

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class InceptionFeatureExtractor(nn.Module):
    """ImageNet InceptionV3 wrapper for GAN evaluation.

    Parameters
    ----------
    device:
        ``"cuda"`` or ``"cpu"``.
    resize:
        InceptionV3 was trained on 299x299 inputs; keep this at 299.
    """

    def __init__(self, device: str | torch.device = "cpu", resize: int = 299):
        super().__init__()
        self.device = torch.device(device)
        self.resize = resize

        self.model = inception_v3(
            weights=Inception_V3_Weights.IMAGENET1K_V1,
            transform_input=False,
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.to(self.device)

        self._feat: torch.Tensor | None = None
        self.model.avgpool.register_forward_hook(self._capture_feat)

        self.register_buffer(
            "_mean", torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "_std", torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        )

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _capture_feat(self, module, inputs, output):  # noqa: ARG002
        # output of AdaptiveAvgPool2d((1,1)) is (N, 2048, 1, 1)
        self._feat = torch.flatten(output, 1)

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """Accept uint8 [0,255] or float [0,1] ``(N,3,H,W)`` -> normalized 299x299."""
        x = images.to(self.device, non_blocking=True)
        if x.dtype == torch.uint8:
            x = x.float().div_(255.0)
        else:
            x = x.float()
            if x.max() > 1.5:  # tolerate [0,255] float inputs
                x = x.div_(255.0)

        if x.shape[-2] != self.resize or x.shape[-1] != self.resize:
            x = F.interpolate(
                x,
                size=(self.resize, self.resize),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        return (x - self._mean) / self._std

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def extract(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(features (N,2048), logits (N,1000))`` as float CPU tensors."""
        x = self.preprocess(images)
        logits = self.model(x)
        feats = self._feat
        self._feat = None
        if feats is None:  # pragma: no cover - hook should always fire
            raise RuntimeError("InceptionV3 avgpool hook did not capture features")
        return feats.detach().float().cpu(), logits.detach().float().cpu()
