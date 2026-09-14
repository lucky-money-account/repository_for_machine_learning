"""Download the Stanford Dogs Dataset.

Why this exists
---------------
The Kaggle `generative-dog-images` competition images ARE the Stanford Dogs
Dataset (20,580 images, 120 breeds). Kaggle now gates dataset downloads behind
Persona identity verification, but Stanford serves the *identical* images with
no login and no verification, so you can skip the whole face-scan question.

Usage
-----
    python download_data.py
    python download_data.py --dest data/stanford-dogs
    python download_data.py --skip-annotations      # images only (~793 MB)
"""

from __future__ import annotations

import argparse
import os
import sys
import tarfile
import urllib.request

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE = "http://vision.stanford.edu/aditya86/ImageNetDogs"
FILES = {
    "images.tar": f"{BASE}/images.tar",           # ~793 MB, the actual dog photos
    "annotation.tar": f"{BASE}/annotation.tar",   # ~21 MB, bounding boxes
    "lists.tar": f"{BASE}/lists.tar",             # train/test split lists
}


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _progress(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, downloaded * 100.0 / total_size)
        sys.stdout.write(
            f"\r  {pct:5.1f}%  {_human(downloaded)} / {_human(total_size)}"
        )
    else:
        sys.stdout.write(f"\r  {_human(downloaded)}")
    sys.stdout.flush()


def download(url: str, dest: str) -> None:
    tmp = dest + ".part"
    urllib.request.urlretrieve(url, tmp, reporthook=_progress)
    sys.stdout.write("\n")
    os.replace(tmp, dest)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Stanford Dogs Dataset.")
    parser.add_argument("--dest", default=os.path.join(_PROJECT_ROOT, "data", "stanford-dogs"),
                        help="output folder")
    parser.add_argument("--skip-annotations", action="store_true",
                        help="download images only (annotations not needed for GAN)")
    args = parser.parse_args()

    dest = os.path.abspath(args.dest)
    os.makedirs(dest, exist_ok=True)
    print(f"destination: {dest}\n")

    wanted = ["images.tar"] if args.skip_annotations else list(FILES)

    for name in wanted:
        url = FILES[name]
        archive = os.path.join(dest, name)
        if os.path.exists(archive):
            print(f"[skip] {name} already downloaded")
        else:
            print(f"[get ] {name}  ({url})")
            download(url, archive)

        print(f"[unpack] {name}")
        with tarfile.open(archive) as tar:
            tar.extractall(dest)
        os.remove(archive)
        print(f"[done] {name}\n")

    print("Stanford Dogs is ready.")
    print(f"  real image folder -> {dest}")
    print("\nNext: point the evaluator at it, e.g.")
    print(f'  GANEvaluator(real_dir=r"{dest}", image_size=64)')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
