"""Training-time FID tracking: sample -> evaluate -> log -> plot the curve."""

from __future__ import annotations

import json
import os
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from .evaluator import GANEvaluator  # noqa: E402

SampleFn = Callable[[int, int], torch.Tensor]
"""``sample_fn(n, seed) -> uint8 tensor (n, 3, H, W)``; the generator's sampling."""


class FIDTracker:
    """Call :meth:`step` at the end of selected epochs to build an FID curve.

    Example
    -------
    >>> tracker = FIDTracker(evaluator, out_dir="outputs", every=5, num_samples=2000)
    >>> def sample_fn(n, seed):
    ...     g = torch.Generator(device="cpu").manual_seed(seed)
    ...     z = torch.randn(n, 100, generator=g)
    ...     with torch.no_grad():
    ...         imgs = generator(z)            # (n,3,64,64) in [-1,1]
    ...     return ((imgs.clamp(-1, 1) + 1) * 127.5).to(torch.uint8)
    >>> tracker.step(epoch=5, sample_fn=sample_fn)
    """

    def __init__(
        self,
        evaluator: GANEvaluator,
        out_dir: str = "outputs",
        every: int = 5,
        num_samples: int = 2000,
        seed: int = 0,
        metrics: tuple[str, ...] = ("fid", "mifid", "kid", "is"),
        save_grid: bool = True,
    ):
        self.evaluator = evaluator
        self.out_dir = out_dir
        self.every = every
        self.num_samples = num_samples
        self.seed = seed
        self.metrics = metrics
        self.save_grid = save_grid
        self.history: list[dict] = []
        os.makedirs(out_dir, exist_ok=True)

    def step(self, epoch: int, sample_fn: SampleFn) -> dict:
        """Generate ``num_samples`` images, evaluate, persist, and return metrics."""
        fake = sample_fn(self.num_samples, self.seed + epoch)
        if not isinstance(fake, torch.Tensor):
            raise TypeError("sample_fn must return a torch uint8 tensor (n,3,H,W)")

        if self.save_grid:
            grid_path = os.path.join(self.out_dir, f"samples_epoch_{epoch:04d}.png")
            self.evaluator.save_grid(fake, grid_path)

        results = self.evaluator.evaluate(fake, metrics=self.metrics)
        results["epoch"] = epoch
        self.history.append(results)
        self._save_json()
        print(f"[epoch {epoch:4d}] " + "  ".join(
            f"{k}={results[k]:.3f}" for k in ("fid", "mifid") if k in results
        ))
        return results

    # ------------------------------------------------------------------ #
    def _save_json(self) -> str:
        path = os.path.join(self.out_dir, "fid_history.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.history, fh, indent=2)
        return path

    def best_epoch(self, key: str = "fid") -> dict:
        """Return the history entry with the lowest ``key`` (e.g. FID/MiFID)."""
        candidates = [h for h in self.history if key in h]
        if not candidates:
            raise ValueError(f"No history entries contain metric {key!r}")
        return min(candidates, key=lambda h: h[key])

    def plot(self, path: str | None = None, keys: tuple[str, ...] = ("fid", "mifid")) -> str:
        """Plot metric curves (FID and MiFID by default). Lower is better."""
        path = path or os.path.join(self.out_dir, "fid_curve.png")
        fig, ax = plt.subplots(figsize=(8, 5))
        for key in keys:
            xs = [h["epoch"] for h in self.history if key in h]
            ys = [h[key] for h in self.history if key in h]
            if xs:
                ax.plot(xs, ys, marker="o", label=key.upper())
        ax.set_xlabel("epoch")
        ax.set_ylabel("score (lower is better)")
        ax.set_title("GAN evaluation over training")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path
