"""Self-test: run the full evaluation pipeline on synthetic images.

No dataset required -- this verifies that torchvision InceptionV3 loads and
that every metric computes. Run it once after setup:

    python selftest.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gan_eval import GANEvaluator, format_results  # noqa: E402


def _write_images(folder: str, n: int, size: int, seed: int, mode: str) -> None:
    os.makedirs(folder, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(n):
        if mode == "smooth":
            base = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
            img = np.asarray(
                Image.fromarray(base).resize((size, size), Image.BILINEAR)
            )
        elif mode == "noise":
            img = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
        else:
            raise ValueError(mode)
        Image.fromarray(img).save(os.path.join(folder, f"{i:04d}.png"))


def main() -> int:
    size, n = 64, 48
    with tempfile.TemporaryDirectory() as tmp:
        real_dir = os.path.join(tmp, "real")
        fake_dir = os.path.join(tmp, "fake")
        _write_images(real_dir, n, size, seed=0, mode="smooth")
        _write_images(fake_dir, n, size, seed=1, mode="smooth")

        print(f"device : {'cuda' if torch.cuda.is_available() else 'cpu'}")
        evaluator = GANEvaluator(real_dir=real_dir, image_size=size, batch_size=16)

        # folder path path
        results = evaluator.evaluate(fake_dir, metrics=("fid", "mifid", "kid", "is", "pr"))
        print("\n[folder input]")
        print(format_results(results))

        # tensor path
        fake_tensor = torch.randint(0, 256, (n, 3, size, size), dtype=torch.uint8)
        results_t = evaluator.evaluate(fake_tensor, metrics=("fid", "mifid"))
        print("\n[tensor input]")
        print(format_results(results_t))

        # nearest-neighbour grid
        nn_path = os.path.join(tmp, "nn.png")
        evaluator.nearest_neighbours(fake_dir, nn_path, n=6)
        print(f"\nnearest-neighbour grid written: {os.path.exists(nn_path)}")

    print("\nSELFTEST OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
