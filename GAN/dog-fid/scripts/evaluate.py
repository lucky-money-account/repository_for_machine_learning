"""Command-line entry point.

Usage
-----
    python evaluate.py --real data/all-dogs --fake samples/epoch_0100 --image-size 64
    python evaluate.py --real data/all-dogs --fake samples/epoch_0100 --nn-grid
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gan_eval import ALL_METRICS, GANEvaluator  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate generated images with FID/MiFID/KID/IS/PR.")
    p.add_argument("--real", required=True, help="folder of real reference images")
    p.add_argument("--fake", required=True, help="folder of generated images")
    p.add_argument("--image-size", type=int, default=64, help="resize both sets to this size")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    p.add_argument("--metrics", default=",".join(ALL_METRICS), help="comma-separated subset")
    p.add_argument("--out", default=None, help="optional path to write results JSON")
    p.add_argument("--nn-grid", action="store_true", help="also save a nearest-neighbour grid")
    p.add_argument("--nn-out", default=None, help="path for the nearest-neighbour grid")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    metrics = tuple(m.strip() for m in args.metrics.split(",") if m.strip())

    evaluator = GANEvaluator(
        real_dir=args.real,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )

    print(f"real : {args.real}")
    print(f"fake : {args.fake}")
    print(f"size : {args.image_size}   device: {evaluator.device}")
    print("-" * 60)

    results = evaluator.evaluate(args.fake, metrics=metrics)
    print(evaluator.report(results))

    if args.nn_grid:
        nn_out = args.nn_out or os.path.join(os.path.dirname(args.fake.rstrip("/\\")), "nearest_neighbours.png")
        path = evaluator.nearest_neighbours(args.fake, nn_out)
        print(f"\nnearest-neighbour grid -> {path}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nresults -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
