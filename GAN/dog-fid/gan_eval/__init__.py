"""gan_eval -- self-contained GAN evaluation for the Kaggle dog-images project.

Main entry points::

    from gan_eval import GANEvaluator, FIDTracker

    ev = GANEvaluator(real_dir="data/all-dogs", image_size=64)
    scores = ev.evaluate("samples/epoch_0100", metrics=("fid", "mifid", "kid", "is", "pr"))
    print(ev.report(scores))
"""

from .data import ImageFolderDataset, list_images, make_loader
from .evaluator import ALL_METRICS, GANEvaluator
from .inception import InceptionFeatureExtractor
from .metrics import (
    format_results,
    frechet_distance,
    improved_precision_recall,
    inception_score,
    kernel_inception_distance,
    memorization_distance,
    memorization_informed_fid,
)
from .tracker import FIDTracker

__all__ = [
    "ALL_METRICS",
    "FIDTracker",
    "GANEvaluator",
    "ImageFolderDataset",
    "InceptionFeatureExtractor",
    "format_results",
    "frechet_distance",
    "improved_precision_recall",
    "inception_score",
    "kernel_inception_distance",
    "list_images",
    "make_loader",
    "memorization_distance",
    "memorization_informed_fid",
]
