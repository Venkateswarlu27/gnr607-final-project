"""Create sample_scene.tif: a synthetic 4-band (Blue, Green, Red, NIR) scene
with water, vegetation and bare soil, so you can test the app before
downloading real Sentinel-2 / Landsat data."""
import numpy as np
import tifffile

rng = np.random.default_rng(42)
H = W = 512


def smooth(sigma):
    """Smooth random field, zero mean, unit std (Gaussian filter in Fourier space)."""
    f = np.fft.fft2(rng.standard_normal((H, W)))
    fy = np.fft.fftfreq(H)[:, None]
    fx = np.fft.fftfreq(W)[None, :]
    f *= np.exp(-2 * np.pi**2 * sigma**2 * (fx**2 + fy**2))
    a = np.real(np.fft.ifft2(f))
    return (a - a.mean()) / a.std()


land = smooth(35)
water = land < -0.7
veg = land > 0.1
soil = ~water & ~veg

# reflectance per class: Blue, Green, Red, NIR
means = {
    "water": [0.06, 0.07, 0.05, 0.03],
    "veg":   [0.04, 0.08, 0.05, 0.45],
    "soil":  [0.15, 0.18, 0.22, 0.28],
}
bands = np.zeros((4, H, W))
for name, mask in (("water", water), ("veg", veg), ("soil", soil)):
    for b in range(4):
        bands[b][mask] = means[name][b]

light = 1 + 0.15 * smooth(80)          # illumination shared by all bands
bands = bands * light + rng.normal(0, 0.008, bands.shape)
bands = np.clip(bands, 0.001, 1) * 10000

tifffile.imwrite("sample_scene.tif", bands.astype(np.uint16),
                 photometric="minisblack", planarconfig="separate")
print("Wrote sample_scene.tif  (bands: 1=Blue 2=Green 3=Red 4=NIR)")
