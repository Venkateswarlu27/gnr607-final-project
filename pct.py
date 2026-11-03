"""
Core Principal Component Transform (PCT) maths for GNR607.

Rules followed:
  * mean, covariance, PCT and inverse PCT are written by hand with NumPy
  * the ONLY linear-algebra built-in used is np.linalg.eigh (permitted)
  * no np.cov, no sklearn.PCA

This module has no web code, so it can be run on its own:
    python pct.py scene.tif --red 3 --nir 4 --k 2
"""
from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------
def load_multiband(source, max_size=1024):
    """Read a multiband TIFF (path or file-like) into an (H, W, N) float64 array.

    Handles both (N, H, W) and (H, W, N) layouts. Images larger than
    max_size are centre-cropped to keep the demo fast.
    """
    import tifffile

    arr = tifffile.imread(source)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    elif arr.shape[0] <= 32 and arr.shape[0] < min(arr.shape[1:3]):
        arr = np.transpose(arr, (1, 2, 0))  # (N, H, W) -> (H, W, N)

    H, W, _ = arr.shape
    if max(H, W) > max_size:
        h, w = min(H, max_size), min(W, max_size)
        y0, x0 = (H - h) // 2, (W - w) // 2
        arr = arr[y0:y0 + h, x0:x0 + w]
    return arr.astype(np.float64)


# --------------------------------------------------------------------------
# Forward PCT
# --------------------------------------------------------------------------
@dataclass
class PCTResult:
    mean: np.ndarray        # (N,)    mean of each band
    cov: np.ndarray         # (N, N)  covariance matrix
    eigvals: np.ndarray     # (N,)    sorted, largest first
    eigvecs: np.ndarray     # (N, N)  columns are eigenvectors, same order
    scores: np.ndarray      # (M, N)  PC values for every pixel
    shape: tuple            # (H, W, N)

    @property
    def variance_pct(self):
        return 100.0 * self.eigvals / self.eigvals.sum()


def compute_pct(img):
    """Mean -> covariance -> eigen-decomposition -> PC scores."""
    H, W, N = img.shape
    X = img.reshape(-1, N)                    # one row per pixel
    M = X.shape[0]

    mean = X.sum(axis=0) / M                  # step 1: mean vector
    Xc = X - mean                             # step 2: centre the data
    cov = (Xc.T @ Xc) / (M - 1)               # step 3: covariance (hand-written)

    vals, vecs = np.linalg.eigh(cov)          # step 4: eigen (allowed built-in)
    order = np.argsort(vals)[::-1]            # step 5: sort, largest first
    vals = np.clip(vals[order], 0.0, None)    # rounding can give tiny negatives
    vecs = vecs[:, order]

    scores = Xc @ vecs                        # step 6: forward PCT
    return PCTResult(mean, cov, vals, vecs, scores, (H, W, N))


# --------------------------------------------------------------------------
# Inverse PCT, MSE, NDVI
# --------------------------------------------------------------------------
def inverse_pct(res, k):
    """Rebuild the image from the top-k principal components."""
    k = int(np.clip(k, 1, res.shape[2]))
    Xhat = res.scores[:, :k] @ res.eigvecs[:, :k].T + res.mean
    return Xhat.reshape(res.shape)


def mse(original, recon):
    """Return (overall MSE, per-band MSE)."""
    d2 = (original - recon) ** 2
    return float(d2.mean()), d2.reshape(-1, d2.shape[2]).mean(axis=0)


def ndvi(img, red, nir, eps=1e-12):
    """NDVI = (NIR - Red) / (NIR + Red). red / nir are 0-based band indices."""
    r, n = img[..., red], img[..., nir]
    return (n - r) / (n + r + eps)


def ndvi_error(ndvi_orig, ndvi_rec, mask_thr=0.1):
    """Two ways to express 'NDVI differs by x %'. Both are returned.

    relative : mean(|dNDVI| / |NDVI_orig|) * 100, using only pixels with
               |NDVI_orig| > mask_thr (the ratio explodes near zero).
    range    : mean(|dNDVI|) / (NDVI range) * 100, uses every pixel.
    """
    diff = np.abs(ndvi_rec - ndvi_orig)
    mask = np.abs(ndvi_orig) > mask_thr
    rel = float((diff[mask] / np.abs(ndvi_orig[mask])).mean() * 100) if mask.any() else float("nan")
    rng = float(ndvi_orig.max() - ndvi_orig.min())
    rng_err = float(diff.mean() / rng * 100) if rng > 0 else float("nan")
    return rel, rng_err


def sweep(res, img, red, nir, mask_thr=0.1, tol=5.0):
    """Try k = 1..N and report MSE and NDVI error for each, plus minimum k."""
    N = res.shape[2]
    n0 = ndvi(img, red, nir)
    out = {"k": [], "mse": [], "ndvi_err_relative": [], "ndvi_err_range": []}
    for k in range(1, N + 1):
        rec = inverse_pct(res, k)
        rel, rng = ndvi_error(n0, ndvi(rec, red, nir), mask_thr)
        out["k"].append(k)
        out["mse"].append(mse(img, rec)[0])
        out["ndvi_err_relative"].append(rel)
        out["ndvi_err_range"].append(rng)
    out["min_k_relative"] = first_k_within(out["ndvi_err_relative"], tol)
    out["min_k_range"] = first_k_within(out["ndvi_err_range"], tol)
    return out


def first_k_within(errors, tol):
    """First k (1-based) whose error is <= tol, else None."""
    for i, e in enumerate(errors):
        if e <= tol:
            return i + 1
    return None


# --------------------------------------------------------------------------
# Stand-alone run (useful for the assignment submission)
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="PCT on a multiband TIFF")
    ap.add_argument("image")
    ap.add_argument("--red", type=int, required=True, help="1-based Red band")
    ap.add_argument("--nir", type=int, required=True, help="1-based NIR band")
    ap.add_argument("--k", type=int, default=2, help="user-specified number of PCs")
    ap.add_argument("--tol", type=float, default=5.0)
    a = ap.parse_args()

    img = load_multiband(a.image)
    res = compute_pct(img)
    print("Image shape:", img.shape)
    print("Variance explained (%):", np.round(res.variance_pct, 3))

    rec = inverse_pct(res, a.k)
    total, per_band = mse(img, rec)
    print(f"\nk = {a.k}: MSE = {total:.4f}   per band = {np.round(per_band, 4)}")

    s = sweep(res, img, a.red - 1, a.nir - 1, tol=a.tol)
    print(f"\n{'k':>3} {'MSE':>14} {'NDVI err rel %':>16} {'NDVI err range %':>18}")
    for k, m, r, g in zip(s["k"], s["mse"], s["ndvi_err_relative"], s["ndvi_err_range"]):
        print(f"{k:>3} {m:>14.4f} {r:>16.3f} {g:>18.3f}")
    print(f"\nMinimum k (relative metric, <= {a.tol}%): {s['min_k_relative']}")
    print(f"Minimum k (range metric,    <= {a.tol}%): {s['min_k_range']}")
