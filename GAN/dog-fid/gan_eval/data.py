"""Image loading utilities for GAN evaluation."""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff"}


def list_images(folder: str | os.PathLike) -> list[str]:
    """Recursively collect image paths under ``folder`` (sorted, deterministic)."""
    folder = os.fspath(folder)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Image folder not found: {folder}")
    out: list[str] = []
    for root, _dirs, files in os.walk(folder):
        for name in files:
            if os.path.splitext(name)[1].lower() in IMG_EXTS:
                out.append(os.path.join(root, name))
    if not out:
        raise FileNotFoundError(f"No images found under: {folder}")
    return sorted(out)


class ImageFolderDataset(Dataset):
    """Loads images as ``uint8`` CHW tensors, optionally resized to ``size``."""

    def __init__(self, paths: Sequence[str], size: int | None = 64):
        self.paths = list(paths)
        self.size = size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as img:
            img = img.convert("RGB")
            if self.size is not None:
                img = img.resize((self.size, self.size), Image.BILINEAR)
            arr = np.array(img, dtype=np.uint8)  # writable HWC copy
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # CHW


def make_loader(
    paths: Sequence[str],
    size: int | None = 64,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    dataset = ImageFolderDataset(paths, size=size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
