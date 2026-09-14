"""Training data interface for the dog GAN.

Two data sources, one API:

* **preprocessed ``.pt``** (recommended, fast) -- ``data/stanford_dogs_64.pt``
  stores ``{'X': uint8 (N,3,64,64), 'Y': int64, 'classes': list[str]}``.
* **image folder** (streaming fallback) -- any folder of images, scanned recursively.

Both yield **float tensors normalized to [-1, 1]** (matching the generator's
``tanh`` output), so the training loop needs no extra normalization.

Quick start
-----------
    from gan_data import dog_dataloader

    loader = dog_dataloader(batch_size=64)          # uses the project's .pt
    for real in loader:                             # real: (B,3,64,64) in [-1,1]
        ...

    # with a validation split
    train_loader, val_loader = dog_dataloader(batch_size=64, val_split=0.1)

    # explicit source
    from gan_data import make_dataloader
    loader = make_dataloader(folder=r"...\data\all-dogs")
"""

from __future__ import annotations

import os
from typing import Optional, Union

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PT = os.path.join(_PROJECT_ROOT, "data", "stanford_dogs_64.pt")
DEFAULT_FOLDER = os.path.join(_PROJECT_ROOT, "data", "all-dogs")


# --------------------------------------------------------------------------- #
# normalization
# --------------------------------------------------------------------------- #
def to_minus_one_one(x: torch.Tensor) -> torch.Tensor:
    """Map an image tensor to float in ``[-1, 1]``.

    uint8 ``[0,255]`` -> ``/127.5 - 1``;  float ``[0,1]`` -> ``*2 - 1``.
    """
    if x.dtype == torch.uint8:
        return x.float().div(127.5).sub(1.0)
    return x.float().mul(2.0).sub(1.0)


# --------------------------------------------------------------------------- #
# .pt loader
# --------------------------------------------------------------------------- #
def load_pt(pt_path: str) -> dict:
    """Load a preprocessed ``.pt`` dict (``X`` / ``Y`` / ``classes``)."""
    if not os.path.isfile(pt_path):
        raise FileNotFoundError(f"preprocessed .pt not found: {pt_path}")
    # weights_only=False: the dict holds a list[str] (`classes`), not just tensors.
    return torch.load(pt_path, weights_only=False)


# --------------------------------------------------------------------------- #
# datasets
# --------------------------------------------------------------------------- #
class PTImageDataset(Dataset):
    """In-memory dataset backed by a preprocessed ``.pt`` file.

    Parameters
    ----------
    pt_path: path to the ``.pt``.
    normalize: if True (default) return float ``[-1,1]``; else return raw ``uint8``.
    return_labels: if True ``__getitem__`` returns ``(image, label)``.
    """

    def __init__(self, pt_path: str, normalize: bool = True, return_labels: bool = False):
        data = load_pt(pt_path)
        self.X = data["X"]
        self.Y = data.get("Y")
        self.classes = data.get("classes")
        self.normalize = normalize
        self.return_labels = return_labels

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int):
        x = self.X[index]
        if self.normalize:
            x = to_minus_one_one(x)
        if self.return_labels:
            if self.Y is None:
                raise ValueError("This .pt has no labels ('Y') to return")
            return x, int(self.Y[index])
        return x


class FolderImageDataset(Dataset):
    """Streaming dataset over an image folder (recursive).

    Images are read on demand, resized to ``image_size`` and (by default)
    normalized to ``[-1,1]``.
    """

    def __init__(self, folder: str, image_size: int = 64,
                 normalize: bool = True, return_labels: bool = False):
        from PIL import Image  # local import: only needed for the streaming path

        from gan_eval import list_images

        self._Image = Image
        self.paths = list_images(folder)
        self.image_size = image_size
        self.normalize = normalize
        self.return_labels = return_labels

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        import numpy as np

        with self._Image.open(self.paths[index]) as img:
            img = img.convert("RGB").resize((self.image_size, self.image_size),
                                            self._Image.BILINEAR)
            arr = np.array(img, dtype=np.uint8)
        x = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        if self.normalize:
            x = to_minus_one_one(x)
        if self.return_labels:
            return x, 0
        return x


# --------------------------------------------------------------------------- #
# dataloader factory
# --------------------------------------------------------------------------- #
def make_dataloader(
    pt: Optional[str] = None,
    folder: Optional[str] = None,
    batch_size: int = 64,
    image_size: int = 64,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
    val_split: float = 0.0,
    seed: int = 42,
    normalize: bool = True,
    return_labels: bool = False,
) -> Union[DataLoader, tuple[DataLoader, DataLoader]]:
    """Build a ``DataLoader`` (or a train/val pair) for GAN training.

    ``pt`` takes priority over ``folder``. Images come out as float ``[-1,1]``.

    Notes
    -----
    * ``drop_last=True`` (default) is **required** for GAN training: a final
      batch of size 1 breaks BatchNorm. Set it False only if you know why.
    * ``num_workers=0`` is best when reading the in-memory ``.pt`` (no I/O to
      parallelize); for the streaming folder path, 2-4 helps on Linux/macOS.
    * When ``val_split > 0`` returns ``(train_loader, val_loader)`` where the
      val loader is ``shuffle=False, drop_last=False``.
    """
    if pt is None and folder is None:
        raise ValueError("provide either pt=... or folder=...")
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    if pt is not None:
        dataset: Dataset = PTImageDataset(pt, normalize=normalize,
                                          return_labels=return_labels)
    else:
        dataset = FolderImageDataset(folder, image_size=image_size,  # type: ignore[arg-type]
                                     normalize=normalize, return_labels=return_labels)

    def _loader(ds, train: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle and train,
            num_workers=num_workers,
            drop_last=drop_last and train,
            pin_memory=pin_memory,
        )

    if val_split > 0:
        n_val = int(len(dataset) * val_split)
        n_train = len(dataset) - n_val
        if n_train <= 0 or n_val <= 0:
            raise ValueError(f"val_split={val_split} leaves an empty split "
                             f"(dataset size {len(dataset)})")
        g = torch.Generator().manual_seed(seed)
        train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=g)
        return _loader(train_ds, True), _loader(val_ds, False)

    return _loader(dataset, True)


def dog_dataloader(**kwargs) -> Union[DataLoader, tuple[DataLoader, DataLoader]]:
    """Convenience wrapper using this project's default paths.

    Uses ``data/stanford_dogs_64.pt`` if it exists, else ``data/all-dogs``.
    Accepts the same keyword arguments as :func:`make_dataloader`.
    """
    if "pt" not in kwargs and "folder" not in kwargs:
        if os.path.isfile(DEFAULT_PT):
            kwargs["pt"] = DEFAULT_PT
        else:
            kwargs["folder"] = DEFAULT_FOLDER
    return make_dataloader(**kwargs)


# --------------------------------------------------------------------------- #
# self-test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("default .pt     :", DEFAULT_PT, "| exists:", os.path.isfile(DEFAULT_PT))
    print("default folder  :", DEFAULT_FOLDER)

    loader = dog_dataloader(batch_size=8, shuffle=True)
    batch = next(iter(loader))
    print("batch shape     :", tuple(batch.shape))
    print("batch dtype     :", batch.dtype)
    print("batch range     : %.2f ~ %.2f" % (batch.min().item(), batch.max().item()))

    train_loader, val_loader = dog_dataloader(batch_size=8, val_split=0.1)
    print("train batches   :", len(train_loader))
    print("val batches     :", len(val_loader))
    print("OK")
