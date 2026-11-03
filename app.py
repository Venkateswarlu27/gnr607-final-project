"""
Streamlit frontend. It only uploads, displays and controls; every calculation
happens in the FastAPI backend (backend.py -> pct.py).

Run:  streamlit run app.py
"""
import base64

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="PCT Explorer", layout="wide")

METRICS = {
    "Relative error (mean |ΔNDVI| / |NDVI|)": "ndvi_err_relative",
    "Range-normalised error (mean |ΔNDVI| / NDVI range)": "ndvi_err_range",
}


# ------------------------------ helpers -----------------------------------
def img_bytes(b64):
    return base64.b64decode(b64)


def api(method, path, **kwargs):
    try:
        r = requests.request(method, BACKEND + path, timeout=300, **kwargs)
    except requests.ConnectionError:
        st.error(f"Cannot reach the backend at {BACKEND}. "
                 "Start it with: uvicorn backend:app --port 8000")
        st.stop()
    if not r.ok:
        try:
            msg = r.json().get("detail", r.text)
        except ValueError:
            msg = r.text
        st.error(msg)
        st.stop()
    return r.json()


def first_k_within(errors, tol):
    return next((i + 1 for i, e in enumerate(errors) if e <= tol), None)


# ------------------------------ sidebar -----------------------------------
st.sidebar.header("Setup")
BACKEND = st.sidebar.text_input("Backend URL", "http://127.0.0.1:8000").rstrip("/")

upload = st.sidebar.file_uploader("Multiband image (.tif)", type=["tif", "tiff"])
if upload and st.sidebar.button("Load image", use_container_width=True):
    meta = api("POST", "/upload", files={"file": (upload.name, upload.getvalue())})
    st.session_state.update(meta=meta, pct=None, sweep=None, recon={})

meta = st.session_state.get("meta")
if meta:
    n = meta["n_bands"]
    st.sidebar.caption(f"{meta['width']} × {meta['height']} px, {n} bands")
    bands = list(range(1, n + 1))

    st.sidebar.subheader("Band numbers")
    red = st.sidebar.selectbox("Red", bands, index=min(2, n - 1))
    nir = st.sidebar.selectbox("NIR", bands, index=min(3, n - 1))
    c1, c2, c3 = st.sidebar.columns(3)
    dr = c1.selectbox("Show R", bands, index=min(2, n - 1))
    dg = c2.selectbox("Show G", bands, index=min(1, n - 1))
    db = c3.selectbox("Show B", bands, index=0)
    mask = st.sidebar.slider("Ignore pixels with |NDVI| below", 0.0, 0.3, 0.1, 0.01,
                             help="Relative error divides by NDVI, so pixels near zero "
                                  "would dominate. They are left out of the relative metric.")

    if st.sidebar.button("Run PCT", type="primary", use_container_width=True):
        with st.spinner("Computing covariance, eigenvectors and PC images…"):
            p = api("POST", "/pct", json={
                "image_id": meta["image_id"], "red_band": red, "nir_band": nir,
                "rgb_bands": [dr, dg, db], "ndvi_mask": mask})
            sw = api("POST", "/sweep", json={"image_id": meta["image_id"]})
        st.session_state.update(pct=p, sweep=sw, recon={})

st.sidebar.divider()
metric_label = st.sidebar.radio("NDVI difference measure", list(METRICS))
metric = METRICS[metric_label]
tol = st.sidebar.number_input("Allowed NDVI difference (%)", 0.1, 50.0, 5.0, 0.5)

# ------------------------------ main page ---------------------------------
st.title("Principal Component Transform on multispectral imagery")

pct_out = st.session_state.get("pct")
sweep = st.session_state.get("sweep")
if not pct_out:
    st.info("Upload a multiband GeoTIFF in the sidebar, choose the Red and NIR bands, "
            "then select **Run PCT**. No image yet? Run `python make_sample.py` to create one.")
    st.stop()

N = pct_out["n_components"]
tab_pc, tab_rec, tab_min, tab_math = st.tabs(
    ["Original and PCs", "Reconstruct with k PCs", "Minimum k for NDVI", "Matrices"])

# -- Tab 1: original + all PCs ---------------------------------------------
with tab_pc:
    a, b = st.columns(2)
    a.image(img_bytes(pct_out["original_png"]), caption="Input image", use_container_width=True)
    b.image(img_bytes(pct_out["ndvi_png"]),
            caption=f"NDVI of input (range {pct_out['ndvi_min']:.2f} to {pct_out['ndvi_max']:.2f})",
            use_container_width=True)

    st.subheader("All principal components")
    per_row = min(N, 4)
    for start in range(0, N, per_row):
        cols = st.columns(per_row)
        for col, pc in zip(cols, pct_out["pcs"][start:start + per_row]):
            col.image(img_bytes(pc["png"]), use_container_width=True,
                      caption=f"PC{pc['index']}: {pc['variance_pct']:.2f}% of variance "
                              f"(cumulative {pc['cumulative_pct']:.2f}%)")

    fig, ax = plt.subplots(figsize=(6, 2.6))
    ax.bar([p["index"] for p in pct_out["pcs"]], [p["variance_pct"] for p in pct_out["pcs"]])
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Variance (%)")
    ax.set_xticks(range(1, N + 1))
    st.pyplot(fig)

# -- Tab 2: user-specified k ----------------------------------------------
with tab_rec:
    st.write("Pick how many of the top principal components to keep. The image is "
             "rebuilt with the inverse PCT and compared with the original.")
    k = st.slider("Number of principal components (k)", 1, N, min(2, N))
    cache = st.session_state.setdefault("recon", {})
    if k not in cache:
        cache[k] = api("POST", "/reconstruct", json={"image_id": meta["image_id"], "k": k})
    r = cache[k]

    c1, c2, c3 = st.columns(3)
    c1.image(img_bytes(pct_out["original_png"]), caption="Original", use_container_width=True)
    c2.image(img_bytes(r["recon_png"]), caption=f"Reconstructed with k = {k}", use_container_width=True)
    c3.image(img_bytes(r["ndvi_png"]), caption=f"NDVI from reconstruction (k = {k})", use_container_width=True)

    err = r[metric]
    m1, m2, m3 = st.columns(3)
    m1.metric("Mean squared error", f"{r['mse']:.4f}")
    m2.metric("NDVI difference", f"{err:.3f}%")
    m3.metric(f"Within {tol:g}%?", "Yes" if err <= tol else "No")
    st.dataframe(pd.DataFrame({"Band": range(1, N + 1), "MSE": r["mse_per_band"]}),
                 hide_index=True, use_container_width=True)

# -- Tab 3: minimum k ------------------------------------------------------
with tab_min:
    errs = sweep[metric]
    min_k = first_k_within(errs, tol)
    if min_k:
        st.success(f"Minimum number of PCs so that NDVI differs by no more than {tol:g}%: "
                   f"**k = {min_k}** (of {N})")
    else:
        st.warning(f"No k up to {N} keeps the NDVI difference within {tol:g}%.")

    ks = sweep["k"]
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.plot(ks, errs, marker="o")
        ax.axhline(tol, linestyle="--", color="tab:red", label=f"{tol:g}% limit")
        if min_k:
            ax.axvline(min_k, linestyle=":", color="tab:green", label=f"minimum k = {min_k}")
        ax.set_xlabel("k (PCs kept)")
        ax.set_ylabel("NDVI difference (%)")
        ax.set_xticks(ks)
        ax.legend()
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.semilogy(ks, [max(v, 1e-12) for v in sweep["mse"]], marker="o")
        ax.set_xlabel("k (PCs kept)")
        ax.set_ylabel("MSE (log scale)")
        ax.set_xticks(ks)
        st.pyplot(fig)

    st.dataframe(pd.DataFrame({
        "k": ks, "MSE": sweep["mse"],
        "NDVI diff, relative (%)": sweep["ndvi_err_relative"],
        "NDVI diff, range-normalised (%)": sweep["ndvi_err_range"],
    }), hide_index=True, use_container_width=True)
    st.caption("The two measures can give different answers. State which one you used "
               "in your report and why.")

# -- Tab 4: matrices -------------------------------------------------------
with tab_math:
    labels = [f"Band {i}" for i in range(1, N + 1)]
    pcs = [f"PC{i}" for i in range(1, N + 1)]
    st.subheader("Mean of each band")
    st.dataframe(pd.DataFrame([pct_out["mean"]], columns=labels), hide_index=True)
    st.subheader("Covariance matrix")
    st.dataframe(pd.DataFrame(pct_out["covariance"], index=labels, columns=labels))
    st.subheader("Eigenvalues (sorted)")
    st.dataframe(pd.DataFrame([pct_out["eigenvalues"]], columns=pcs), hide_index=True)
    st.subheader("Eigenvectors (each column is one PC)")
    st.dataframe(pd.DataFrame(pct_out["eigenvectors"], index=labels, columns=pcs))
