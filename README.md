# GNR607 – PCT Explorer

pct.py          core maths (mean, covariance, eigen, PCT, inverse, MSE, NDVI)
backend.py      FastAPI wrapper around pct.py
app.py          Streamlit UI that calls the backend
make_sample.py  creates a synthetic 4-band test image

## Run
pip install -r requirements.txt
python make_sample.py                      # optional test image

# terminal 1
uvicorn backend:app --port 8000
# terminal 2
streamlit run app.py

Band order of sample_scene.tif: 1=Blue 2=Green 3=Red 4=NIR
(Sentinel-2 stack B2,B3,B4,B8 uses the same order.)

## Without the web app
python pct.py sample_scene.tif --red 3 --nir 4 --k 2
