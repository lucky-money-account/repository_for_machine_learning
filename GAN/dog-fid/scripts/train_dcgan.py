"""DCGAN baseline for the Generative Dog Images project.

Pipeline
--------
    dog images -> 64x64 [-1,1] -> DCGAN -> sample grids + checkpoints
                                        -> FID / MiFID curve (via gan_eval.FIDTracker)

Usage
-----
    # quick smoke test on a small subset
    python train_dcgan.py --data data/stanford-dogs --limit 500 --epochs 2 --fid-every 1

    # real run
    python train_dcgan.py --data data/stanford-dogs --epochs 50 --fid-every 5

Everything is written under --out-dir (default outputs/dcgan):
    samples/   sample grids per epoch
    ckpt/      G.pt / D.pt
    fid_history.json + fid_curve.png
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from gan_eval import FIDTracker, GANEvaluator, list_images  # noqa: E402
from gan_models import Discriminator, Generator, weights_init  # noqa: E402


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
class DogDataset(Dataset):
    """Recursively loads dog images, resized to ``image_size`` and scaled to [-1,1]."""

    def __init__(self, root: str, image_size: int = 64, limit: int = 0):
        self.paths = list_images(root)
        if limit > 0:
            self.paths = self.paths[:limit]
        self.size = image_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as img:
            img = img.convert("RGB").resize((self.size, self.size), Image.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1) * 2.0 - 1.0


# --------------------------------------------------------------------------- #
# sampling helper (used by FIDTracker and for grid images)
# --------------------------------------------------------------------------- #
def make_sample_fn(generator: Generator, z_dim: int, device: torch.device, chunk: int = 256):
    """Return ``sample_fn(n, seed) -> uint8 (n,3,64,64)`` for FIDTracker."""

    @torch.no_grad()
    def sample_fn(n: int, seed: int) -> torch.Tensor:
        was_training = generator.training
        generator.eval()
        gen = torch.Generator(device=device).manual_seed(seed)
        out = []
        for i in range(0, n, chunk):
            m = min(chunk, n - i)
            z = torch.randn(m, z_dim, 1, 1, generator=gen, device=device)
            imgs = generator(z)
            imgs = ((imgs.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).cpu()
            out.append(imgs)
        if was_training:
            generator.train()
        return torch.cat(out)

    return sample_fn


# --------------------------------------------------------------------------- #
# args
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a DCGAN on dog images.")
    p.add_argument("--data", required=True, help="folder with dog images (recursive)")
    p.add_argument("--out-dir", default=os.path.join(_PROJECT_ROOT, "outputs", "dcgan"))
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--z-dim", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    p.add_argument("--workers", type=int, default=0, help="DataLoader workers (0 safest on Windows)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0, help="cap dataset size (0 = all), for smoke tests")
    p.add_argument("--fid-every", type=int, default=5, help="compute FID every N epochs")
    p.add_argument("--num-fid-samples", type=int, default=2000)
    p.add_argument("--no-fid", action="store_true", help="disable FID tracking")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    args = parse_args()
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    samples_dir = os.path.join(args.out_dir, "samples")
    ckpt_dir = os.path.join(args.out_dir, "ckpt")
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # ---- data ----
    dataset = DogDataset(args.data, image_size=args.image_size, limit=args.limit)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    print(f"dataset: {len(dataset)} images  |  device: {device}")

    # ---- models ----
    G = Generator(z_dim=args.z_dim).to(device)
    D = Discriminator().to(device)
    G.apply(weights_init)
    D.apply(weights_init)

    optG = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    # Discriminator outputs a probability score (Sigmoid) -> use BCELoss.
    # Do NOT use BCEWithLogitsLoss here (it would apply a second sigmoid).
    criterion = nn.BCELoss()

    fixed_z = torch.randn(64, args.z_dim, 1, 1, device=device)
    sample_fn = make_sample_fn(G, args.z_dim, device)

    # ---- FID tracking (optional) ----
    tracker = None
    if not args.no_fid:
        evaluator = GANEvaluator(
            real_dir=args.data, image_size=args.image_size,
            batch_size=64, device=str(device),
        )
        tracker = FIDTracker(
            evaluator, out_dir=args.out_dir, every=args.fid_every,
            num_samples=args.num_fid_samples, seed=0,
            metrics=("fid", "mifid", "kid", "is"),
        )

    # ---- training ----
    for epoch in range(1, args.epochs + 1):
        G.train()
        D.train()
        t0 = time.time()
        d_sum = g_sum = 0.0
        n_batches = 0

        for real in loader:
            real = real.to(device, non_blocking=True)
            b = real.size(0)
            ones = torch.ones(b, device=device)
            zeros = torch.zeros(b, device=device)

            # --- discriminator ---
            optD.zero_grad(set_to_none=True)
            z = torch.randn(b, args.z_dim, 1, 1, device=device)
            fake = G(z)
            lossD = 0.5 * (
                criterion(D(real), ones) + criterion(D(fake.detach()), zeros)
            )
            lossD.backward()
            optD.step()

            # --- generator ---
            optG.zero_grad(set_to_none=True)
            lossG = criterion(D(fake), ones)
            lossG.backward()
            optG.step()

            d_sum += lossD.item()
            g_sum += lossG.item()
            n_batches += 1

        # sample grid every epoch
        with torch.no_grad():
            grid_imgs = G(fixed_z)
            save_image(
                (grid_imgs.clamp(-1, 1) + 1) / 2,
                os.path.join(samples_dir, f"epoch_{epoch:04d}.png"),
                nrow=8,
            )

        print(
            f"[{epoch:3d}/{args.epochs}] D={d_sum / max(n_batches,1):.3f} "
            f"G={g_sum / max(n_batches,1):.3f}  {time.time() - t0:.1f}s"
        )

        # FID / MiFID curve
        if tracker is not None and epoch % args.fid_every == 0:
            tracker.step(epoch, sample_fn)

        # checkpoint
        torch.save(G.state_dict(), os.path.join(ckpt_dir, "G.pt"))
        torch.save(D.state_dict(), os.path.join(ckpt_dir, "D.pt"))

    if tracker is not None and tracker.history:
        print("\nFID curve ->", tracker.plot())
        print("best epoch (FID):", tracker.best_epoch("fid"))

    print("\nDone. Samples in", samples_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
