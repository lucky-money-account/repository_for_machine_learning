"""Visualisation helpers: sample grids and nearest-neighbour memorization checks."""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torchvision.utils import make_grid, save_image

from .data import list_images


def save_sample_grid(images: torch.Tensor, path: str, nrow: int = 8, padding: int = 2) -> str:
    """Save a grid of uint8 ``(N,3,H,W)`` images to ``path``."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    grid = make_grid(images.float().div(255.0), nrow=nrow, padding=padding)
    save_image(grid, path)
    return path


def nearest_neighbour_indices(feats_fake, feats_real, chunk: int = 2048) -> np.ndarray:
    """For each fake feature, index of the most similar real feature (cosine)."""
    g = np.asarray(feats_fake, dtype=np.float64)
    r = np.asarray(feats_real, dtype=np.float64)
    g = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-12)
    r = r / (np.linalg.norm(r, axis=1, keepdims=True) + 1e-12)

    out = np.empty(g.shape[0], dtype=np.int64)
    for i in range(0, g.shape[0], chunk):
        sim = g[i : i + chunk] @ r.T
        out[i : i + chunk] = sim.argmax(axis=1)
    return out


def _load_resized(path: str, size: int) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB").resize((size, size), Image.BILINEAR)


def save_nearest_neighbour_grid(
    fake_paths: Sequence[str],
    real_paths: Sequence[str],
    nn_indices: np.ndarray,
    path: str,
    size: int = 64,
    n: int = 8,
) -> str:
    """Top row: generated images. Bottom row: their nearest real neighbours.

    If a generated image looks nearly identical to its neighbour, the model is
    memorising the training set (which MiFID punishes).
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    n = min(n, len(fake_paths))

    top = [_load_resized(fake_paths[i], size) for i in range(n)]
    bottom = [_load_resized(real_paths[int(nn_indices[i])], size) for i in range(n)]

    canvas = Image.new("RGB", (size * n, size * 2), color=(255, 255, 255))
    for col in range(n):
        canvas.paste(top[col], (col * size, 0))
        canvas.paste(bottom[col], (col * size, size))
    canvas.save(path)
    return path
