"""
FastAPI backend. All maths lives in pct.py; this file only handles HTTP,
caching and turning arrays into PNG previews.

Run:  uvicorn backend:app --reload --port 8000
Docs: http://127.0.0.1:8000/docs
"""
import base64
import io
import uuid

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from matplotlib import cm
from PIL import Image
from pydantic import BaseModel

import pct

app = FastAPI(title="PCT Backend")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

SESSIONS: dict = {}   # image_id -> cached arrays and settings (in memory)


# ----------------------------- helpers ------------------------------------
def to_png(arr_u8):
    buf = io.BytesIO()
    Image.fromarray(arr_u8).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def stretch(a, lo, hi):
    return (np.clip((a - lo) / (hi - lo + 1e-12), 0, 1) * 255).astype(np.uint8)


def gray_png(a):
    lo, hi = np.percentile(a, [2, 98])
    return to_png(stretch(a, lo, hi))


def composite_png(img, bands, limits):
    """False/true-colour preview using fixed stretch limits so the original
    and every reconstruction are displayed on the same scale."""
    chans = [stretch(img[..., b], *limits[i]) for i, b in enumerate(bands)]
    return to_png(np.dstack(chans))


def ndvi_png(nd):
    rgba = cm.RdYlGn((np.clip(nd, -1, 1) + 1) / 2)
    return to_png((rgba[..., :3] * 255).astype(np.uint8))


def get_session(image_id):
    s = SESSIONS.get(image_id)
    if s is None:
        raise HTTPException(404, "Unknown image_id. Upload the image again.")
    return s


def need_pct(s):
    if "res" not in s:
        raise HTTPException(400, "Run /pct first.")
    return s


# ----------------------------- schemas ------------------------------------
class PCTRequest(BaseModel):
    image_id: str
    red_band: int                         # 1-based
    nir_band: int                         # 1-based
    rgb_bands: list[int] = [3, 2, 1]      # 1-based, bands shown as R, G, B
    ndvi_mask: float = 0.1


class ReconRequest(BaseModel):
    image_id: str
    k: int


class SweepRequest(BaseModel):
    image_id: str
    tolerance: float = 5.0


# ----------------------------- endpoints ----------------------------------
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    data = await file.read()
    try:
        img = pct.load_multiband(io.BytesIO(data))
    except Exception as e:
        raise HTTPException(400, f"Could not read image as a multiband TIFF: {e}")
    if img.shape[2] < 3:
        raise HTTPException(400, f"Need at least 3 bands, found {img.shape[2]}.")

    image_id = uuid.uuid4().hex[:12]
    SESSIONS[image_id] = {"img": img}
    H, W, N = img.shape
    return {
        "image_id": image_id, "height": H, "width": W, "n_bands": N,
        "band_stats": [
            {"band": i + 1, "min": float(img[..., i].min()),
             "max": float(img[..., i].max()), "mean": float(img[..., i].mean())}
            for i in range(N)
        ],
    }


@app.post("/pct")
def run_pct(req: PCTRequest):
    s = get_session(req.image_id)
    img = s["img"]
    H, W, N = img.shape
    for name, b in [("red_band", req.red_band), ("nir_band", req.nir_band)] + \
                   [("rgb_bands", b) for b in req.rgb_bands]:
        if not 1 <= b <= N:
            raise HTTPException(400, f"{name}={b} is outside 1..{N}.")
    if req.red_band == req.nir_band:
        raise HTTPException(400, "Red and NIR must be different bands.")
    if len(req.rgb_bands) != 3:
        raise HTTPException(400, "rgb_bands needs exactly 3 band numbers.")

    res = pct.compute_pct(img)
    red, nir = req.red_band - 1, req.nir_band - 1
    rgb = [b - 1 for b in req.rgb_bands]
    limits = [tuple(np.percentile(img[..., b], [2, 98])) for b in rgb]
    nd0 = pct.ndvi(img, red, nir)

    s.update(res=res, red=red, nir=nir, rgb=rgb, limits=limits,
             nd0=nd0, mask=req.ndvi_mask)

    var = res.variance_pct
    pcs = [{
        "index": i + 1,
        "eigenvalue": float(res.eigvals[i]),
        "variance_pct": float(var[i]),
        "cumulative_pct": float(var[: i + 1].sum()),
        "png": gray_png(res.scores[:, i].reshape(H, W)),
    } for i in range(N)]

    return {
        "original_png": composite_png(img, rgb, limits),
        "ndvi_png": ndvi_png(nd0),
        "ndvi_min": float(nd0.min()), "ndvi_max": float(nd0.max()),
        "n_components": N,
        "pcs": pcs,
        "mean": res.mean.tolist(),
        "covariance": res.cov.tolist(),
        "eigenvalues": res.eigvals.tolist(),
        "eigenvectors": res.eigvecs.tolist(),
    }


@app.post("/reconstruct")
def reconstruct(req: ReconRequest):
    s = need_pct(get_session(req.image_id))
    N = s["res"].shape[2]
    if not 1 <= req.k <= N:
        raise HTTPException(400, f"k must be between 1 and {N}.")

    rec = pct.inverse_pct(s["res"], req.k)
    total, per_band = pct.mse(s["img"], rec)
    nd_k = pct.ndvi(rec, s["red"], s["nir"])
    rel, rng = pct.ndvi_error(s["nd0"], nd_k, s["mask"])
    return {
        "k": req.k,
        "recon_png": composite_png(rec, s["rgb"], s["limits"]),
        "ndvi_png": ndvi_png(nd_k),
        "mse": total,
        "mse_per_band": per_band.tolist(),
        "ndvi_err_relative": rel,
        "ndvi_err_range": rng,
    }


@app.post("/sweep")
def run_sweep(req: SweepRequest):
    s = need_pct(get_session(req.image_id))
    return pct.sweep(s["res"], s["img"], s["red"], s["nir"], s["mask"], req.tolerance)


@app.get("/health")
def health():
    return {"status": "ok", "sessions": len(SESSIONS)}
