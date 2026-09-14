"""GAN evaluation metrics, implemented from scratch (numpy + scipy only).

All functions take feature matrices of shape ``(N, D)`` (numpy arrays or torch
tensors) unless stated otherwise.

Metrics
-------
* ``frechet_distance``              -- FID (quality)
* ``kernel_inception_distance``     -- KID (unbiased, small-sample friendly)
* ``inception_score``               -- IS (needs logits, not features)
* ``memorization_informed_fid``     -- MiFID (the competition metric)
* ``improved_precision_recall``     -- Precision / Recall (quality vs coverage)
"""

from __future__ import annotations

import numpy as np
from scipy import linalg


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


# --------------------------------------------------------------------------- #
# FID
# --------------------------------------------------------------------------- #
def frechet_distance(feats_real, feats_fake, eps: float = 1e-6) -> float:
    """Frechet Inception Distance. Lower is better.

    FID = ||mu_r - mu_f||^2 + Tr(Sigma_r + Sigma_f - 2 (Sigma_r Sigma_f)^0.5)
    """
    a = to_numpy(feats_real)
    b = to_numpy(feats_fake)
    mu1, mu2 = a.mean(axis=0), b.mean(axis=0)
    sigma1 = np.cov(a, rowvar=False)
    sigma2 = np.cov(b, rowvar=False)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError(
                f"sqrtm produced a large imaginary component "
                f"({np.max(np.abs(covmean.imag)):.3e}); the covariance is ill-conditioned."
            )
        covmean = covmean.real

    trace = np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean)
    return float(diff.dot(diff) + trace)


# --------------------------------------------------------------------------- #
# KID
# --------------------------------------------------------------------------- #
def _poly_kernel(x, y, degree=3, coef=1.0, gamma=None):
    if gamma is None:
        gamma = 1.0 / x.shape[1]
    return (gamma * (x @ y.T) + coef) ** degree


def _mmd2(x, y, kernel) -> float:
    kxx, kyy, kxy = kernel(x, x), kernel(y, y), kernel(x, y)
    n, m = x.shape[0], y.shape[0]
    return float(
        (kxx.sum() - np.trace(kxx)) / (n * (n - 1))
        + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
        - 2.0 * kxy.mean()
    )


def kernel_inception_distance(
    feats_real,
    feats_fake,
    degree: int = 3,
    coef: float = 1.0,
    subsets: int = 50,
    subset_size: int = 1000,
    seed: int = 0,
    return_std: bool = False,
):
    """Kernel Inception Distance (MMD with a polynomial kernel). Lower is better.

    Unbiased and more stable than FID for small sample sizes. Values are tiny;
    multiply by 1000 for readability (``kid x 1000``).
    """
    a = to_numpy(feats_real)
    b = to_numpy(feats_fake)
    kernel = lambda x, y: _poly_kernel(x, y, degree, coef)  # noqa: E731

    n, m = a.shape[0], b.shape[0]
    if min(n, m) <= subset_size:
        val = _mmd2(a, b, kernel)
        return (val, 0.0) if return_std else val

    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(subsets):
        ia = rng.choice(n, subset_size, replace=False)
        ib = rng.choice(m, subset_size, replace=False)
        vals.append(_mmd2(a[ia], b[ib], kernel))
    vals = np.asarray(vals)
    return (float(vals.mean()), float(vals.std())) if return_std else float(vals.mean())


# --------------------------------------------------------------------------- #
# Inception Score
# --------------------------------------------------------------------------- #
def inception_score(logits, splits: int = 10):
    """Inception Score from classifier logits. Higher is better.

    Returns ``(mean, std)`` over ``splits`` chunks.
    """
    p = to_numpy(logits)
    p = p - p.max(axis=1, keepdims=True)
    p = np.exp(p)
    p = p / p.sum(axis=1, keepdims=True)

    scores = []
    for part in np.array_split(p, splits, axis=0):
        if part.shape[0] == 0:
            continue
        py = part.mean(axis=0, keepdims=True)
        kl = part * (np.log(part + 1e-10) - np.log(py + 1e-10))
        scores.append(np.exp(kl.sum(axis=1).mean()))
    scores = np.asarray(scores)
    return float(scores.mean()), float(scores.std())


# --------------------------------------------------------------------------- #
# MiFID (competition metric)
# --------------------------------------------------------------------------- #
def memorization_distance(feats_gen, feats_real, chunk: int = 2048) -> float:
    """Average minimum cosine distance of generated images to the real set.

    Smaller value  =>  more memorization (generated images too close to training).
    """
    g = to_numpy(feats_gen)
    r = to_numpy(feats_real)
    g = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-12)
    r = r / (np.linalg.norm(r, axis=1, keepdims=True) + 1e-12)

    mins = np.empty(g.shape[0], dtype=np.float64)
    for i in range(0, g.shape[0], chunk):
        sim = g[i : i + chunk] @ r.T  # (c, Nr)
        mins[i : i + chunk] = (1.0 - np.abs(sim)).min(axis=1)
    return float(mins.mean())


def memorization_informed_fid(feats_real, feats_fake, tau: float = 0.1, eps: float = 1e-8):
    """MiFID = penalty * FID. Lower is better.

    penalty = 1/(s+eps) if s < tau else 1, where ``s`` is the memorization
    distance. A model that simply copies training images gets a small ``s``
    and therefore a large penalty.

    Returns a dict with ``mifid``, ``fid``, ``memorization_distance``, ``penalty``.
    """
    fid = frechet_distance(feats_real, feats_fake)
    s = memorization_distance(feats_fake, feats_real)
    penalty = 1.0 / (s + eps) if s < tau else 1.0
    return {
        "mifid": float(penalty * fid),
        "fid": float(fid),
        "memorization_distance": float(s),
        "penalty": float(penalty),
    }


# --------------------------------------------------------------------------- #
# Improved Precision / Recall (Kynkaanniemi et al. 2019)
# --------------------------------------------------------------------------- #
def _kth_nn_distance(x, k: int, chunk: int = 1024) -> np.ndarray:
    x = to_numpy(x)
    n = x.shape[0]
    x2 = (x * x).sum(axis=1)
    kth = np.empty(n, dtype=np.float64)
    for i in range(0, n, chunk):
        xb = x[i : i + chunk]
        d2 = x2[i : i + chunk, None] + x2[None, :] - 2.0 * xb @ x.T
        np.maximum(d2, 0, out=d2)
        rows = np.arange(xb.shape[0])
        d2[rows, i + rows] = np.inf  # exclude self
        kth[i : i + chunk] = np.sqrt(np.partition(d2, k - 1, axis=1)[:, k - 1])
    return kth


def _in_manifold(x, ref, ref_radius, chunk: int = 1024) -> float:
    x = to_numpy(x)
    ref = to_numpy(ref)
    x2 = (x * x).sum(axis=1)
    r2 = (ref * ref).sum(axis=1)
    inside = np.empty(x.shape[0], dtype=bool)
    for i in range(0, x.shape[0], chunk):
        xb = x[i : i + chunk]
        d2 = x2[i : i + chunk, None] + r2[None, :] - 2.0 * xb @ ref.T
        np.maximum(d2, 0, out=d2)
        inside[i : i + chunk] = (np.sqrt(d2) <= ref_radius[None, :]).any(axis=1)
    return float(inside.mean())


def improved_precision_recall(feats_real, feats_fake, k: int = 3) -> dict:
    """Improved precision/recall. Precision=quality, Recall=coverage/diversity.

    A low recall with a decent FID is the classic signature of mode collapse.
    """
    rad_real = _kth_nn_distance(feats_real, k)
    rad_fake = _kth_nn_distance(feats_fake, k)
    return {
        "precision": _in_manifold(feats_fake, feats_real, rad_real),
        "recall": _in_manifold(feats_real, feats_fake, rad_fake),
    }


# --------------------------------------------------------------------------- #
# formatting
# --------------------------------------------------------------------------- #
def format_results(results: dict) -> str:
    """Pretty one-line-per-metric table."""
    order = [
        ("fid", "FID (lower better)", "{:.3f}"),
        ("mifid", "MiFID (lower better)", "{:.3f}"),
        ("memorization_distance", "Memorization distance", "{:.4f}"),
        ("penalty", "Memorization penalty", "{:.3f}"),
        ("kid", "KID x1000 (lower better)", "{:.4f}"),
        ("kid_std", "KID std", "{:.4f}"),
        ("is", "Inception Score (higher better)", "{:.3f}"),
        ("is_std", "IS std", "{:.3f}"),
        ("precision", "Precision (quality)", "{:.4f}"),
        ("recall", "Recall (diversity)", "{:.4f}"),
    ]
    lines = []
    for key, label, fmt in order:
        if key in results:
            val = results[key]
            if key == "kid":
                val = val * 1000.0
            lines.append(f"  {label:<34s}: {fmt.format(val)}")
    return "\n".join(lines)
