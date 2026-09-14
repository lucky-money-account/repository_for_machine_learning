"""High-level GAN evaluator: real-image caching + all metrics in one call."""

from __future__ import annotations

import os

import numpy as np
import torch
from tqdm.auto import tqdm

from .data import list_images, make_loader
from .inception import InceptionFeatureExtractor
from .metrics import (
    format_results,
    frechet_distance,
    improved_precision_recall,
    inception_score,
    kernel_inception_distance,
    memorization_informed_fid,
)
from .viz import nearest_neighbour_indices, save_nearest_neighbour_grid, save_sample_grid

ALL_METRICS = ("fid", "kid", "is", "mifid", "pr")


class GANEvaluator:
    """Evaluate generated images against a fixed real-image reference set.

    Parameters
    ----------
    real_dir:
        Folder with real images (recursively scanned). Features are computed
        once and cached, since the reference set never changes.
    image_size:
        Both real and generated images are resized to ``image_size`` before
        feature extraction, so they are compared at the same resolution.
        For the Kaggle dog competition use 64.
    batch_size, num_workers:
        DataLoader settings. Keep ``num_workers=0`` inside Jupyter on Windows.
    device:
        ``"cuda"`` / ``"cpu"``; auto-detected by default.
    """

    def __init__(
        self,
        real_dir: str | os.PathLike | None = None,
        image_size: int = 64,
        batch_size: int = 64,
        num_workers: int = 0,
        device: str | torch.device | None = None,
    ):
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.extractor = InceptionFeatureExtractor(self.device)

        self.real_dir: str | None = None
        self._real_feats: np.ndarray | None = None
        self._real_logits: np.ndarray | None = None
        self._real_paths: list[str] | None = None

        if real_dir is not None:
            self.set_real_dir(real_dir)

    # ------------------------------------------------------------------ #
    # feature extraction
    # ------------------------------------------------------------------ #
    def _extract_loader(self, loader, desc: str):
        feats, logits = [], []
        for batch in tqdm(loader, desc=desc, leave=False):
            f, l = self.extractor.extract(batch)
            feats.append(f)
            logits.append(l)
        return torch.cat(feats).numpy(), torch.cat(logits).numpy()

    def _extract_folder(self, folder: str | os.PathLike, desc: str):
        paths = list_images(folder)
        loader = make_loader(
            paths,
            size=self.image_size,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda",
        )
        return (*self._extract_loader(loader, desc), paths)

    def _extract_tensor(self, images: torch.Tensor, desc: str):
        feats, logits = [], []
        n = images.shape[0]
        for i in tqdm(range(0, n, self.batch_size), desc=desc, leave=False):
            f, l = self.extractor.extract(images[i : i + self.batch_size])
            feats.append(f)
            logits.append(l)
        return torch.cat(feats).numpy(), torch.cat(logits).numpy()

    # ------------------------------------------------------------------ #
    # real reference set
    # ------------------------------------------------------------------ #
    def set_real_dir(self, real_dir: str | os.PathLike) -> None:
        self.real_dir = os.fspath(real_dir)
        self._real_feats = None
        self._real_logits = None
        self._real_paths = None

    def _ensure_real(self):
        if self._real_feats is None:
            if self.real_dir is None:
                raise ValueError("No real_dir set; pass real_dir=... to GANEvaluator.")
            self._real_feats, self._real_logits, self._real_paths = self._extract_folder(
                self.real_dir, "real features"
            )
        return self._real_feats

    @property
    def real_features(self) -> np.ndarray:
        return self._ensure_real()

    @property
    def real_paths(self) -> list[str]:
        self._ensure_real()
        return self._real_paths  # type: ignore[return-value]

    # ------------------------------------------------------------------ #
    # fake / generated images
    # ------------------------------------------------------------------ #
    def extract_fake(self, fake):
        """``fake`` may be a folder path or a uint8 ``(N,3,H,W)`` tensor.

        Returns ``(features, logits, paths_or_None)``.
        """
        if isinstance(fake, (str, os.PathLike)):
            feats, logits, paths = self._extract_folder(fake, "fake features")
            return feats, logits, paths
        if isinstance(fake, torch.Tensor):
            feats, logits = self._extract_tensor(fake, "fake features")
            return feats, logits, None
        raise TypeError("fake must be a folder path or a uint8 torch tensor")

    def evaluate(self, fake, metrics: tuple[str, ...] = ALL_METRICS) -> dict:
        """Compute the requested metrics. Returns a dict of floats."""
        real_feats = self.real_features

        if isinstance(fake, (str, os.PathLike)):
            fake_feats, fake_logits, _ = self._extract_folder(fake, "fake features")
        elif isinstance(fake, torch.Tensor):
            fake_feats, fake_logits = self._extract_tensor(fake, "fake features")
        else:
            raise TypeError("fake must be a folder path or a uint8 torch tensor")

        results: dict[str, float] = {}
        if "fid" in metrics or "mifid" in metrics:
            fid = frechet_distance(real_feats, fake_feats)
            results["fid"] = fid
        if "mifid" in metrics:
            results.update(memorization_informed_fid(real_feats, fake_feats))
        if "kid" in metrics:
            kid, kid_std = kernel_inception_distance(real_feats, fake_feats, return_std=True)
            results["kid"] = kid
            results["kid_std"] = kid_std
        if "is" in metrics:
            is_mean, is_std = inception_score(fake_logits)
            results["is"] = is_mean
            results["is_std"] = is_std
        if "pr" in metrics:
            results.update(improved_precision_recall(real_feats, fake_feats))
        return results

    # ------------------------------------------------------------------ #
    # convenience
    # ------------------------------------------------------------------ #
    def save_grid(self, images: torch.Tensor, path: str, nrow: int = 8) -> str:
        return save_sample_grid(images, path, nrow=nrow)

    def nearest_neighbours(self, fake, path: str, n: int = 8, size: int | None = None) -> str:
        """Save a generated-vs-nearest-real grid to inspect memorization."""
        size = size or self.image_size
        if not isinstance(fake, (str, os.PathLike)):
            raise TypeError("nearest_neighbours needs a fake image folder (for file paths)")

        fake_paths = list_images(fake)
        fake_feats, _, _ = self._extract_folder(fake, "fake features")
        real_feats = self.real_features

        idx = nearest_neighbour_indices(fake_feats, real_feats)
        return save_nearest_neighbour_grid(
            fake_paths, self.real_paths, idx, path, size=size, n=n
        )

    def report(self, results: dict) -> str:
        return format_results(results)
