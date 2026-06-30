"""
features.py
-----------
Handcrafted features for screen-recapture (photo-of-a-screen) detection.

The core idea: a recaptured image carries *capture artifacts* that a real photo
does not. The strongest, most physical one is MOIRE,  interference between the
camera sensor grid and the display's pixel grid,  which shows up as periodic
energy in the 2D FFT. We pair that with screen-door micro-texture (LBP), a
double-capture sharpness drop, and small colour/gamut shifts.

These features are deliberately cheap (pure numpy/scipy) so the whole thing is
small, fast, and runs on-device. Shared by train.py and predict.py so the exact
same vector is produced at train and inference time.
"""

import numpy as np
from PIL import Image

# Optional: transparently handle iPhone .HEIC if pillow-heif is installed.
# Not required (convert to .jpg first) -- this is just a convenience, no hard dep.
try:
    import pillow_heif  # type: ignore
    pillow_heif.register_heif_opener()
except Exception:
    pass

FFT_SIZE = 512    # resolution for frequency analysis (fixed -> consistent features)
TEX_SIZE = 256    # resolution for texture / colour (fixed -> scale-invariant)
N_RADIAL = 16     # number of radial frequency bands
N_LBP_BINS = 16   # texture histogram bins


# --------------------------------------------------------------------------- #
# loading helpers
# --------------------------------------------------------------------------- #
def _open_rgb(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.open(image).convert("RGB")


def _gray_array(img_rgb, size):
    g = img_rgb.convert("L").resize((size, size), Image.BILINEAR)
    return np.asarray(g, dtype=np.float32) / 255.0


# --------------------------------------------------------------------------- #
# 1. frequency-domain features (moire is the money signal)
# --------------------------------------------------------------------------- #
def _fft_features(gray):
    h, w = gray.shape
    # Hann window kills the bright edge-cross artifact so real peaks stand out
    win = np.hanning(h)[:, None] * np.hanning(w)[None, :]
    F = np.fft.fftshift(np.fft.fft2(gray * win))
    mag = np.abs(F)
    logmag = np.log1p(mag)

    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_norm = r / r.max()

    # (a) radial energy profile -> overall spectral shape, normalised to a
    #     distribution so absolute brightness doesn't matter
    radial = np.zeros(N_RADIAL, dtype=np.float32)
    for i in range(N_RADIAL):
        lo, hi = i / N_RADIAL, (i + 1) / N_RADIAL
        m = (r_norm >= lo) & (r_norm < hi)
        radial[i] = logmag[m].mean() if m.any() else 0.0
    radial = radial / (radial.sum() + 1e-8)

    # (b) mid/high-frequency "peakiness" -> moire creates sharp localised spikes
    hf = mag[r_norm > 0.5]
    hf_mean = hf.mean() + 1e-8
    peak_ratio = hf.max() / hf_mean                       # tallest spike vs avg
    spikiness = hf.std() / hf_mean                        # spread / burstiness
    thr = np.percentile(hf, 99.9)
    peak_energy_frac = hf[hf > thr].sum() / (hf.sum() + 1e-8)  # energy in spikes

    return np.concatenate([
        radial,
        [np.log1p(peak_ratio), spikiness, peak_energy_frac],
    ]).astype(np.float32)


# --------------------------------------------------------------------------- #
# 2. spatial / sharpness features (double capture tends to soften)
# --------------------------------------------------------------------------- #
def _spatial_features(gray):
    from scipy.ndimage import gaussian_filter, laplace
    lap = laplace(gray)
    sharpness = float(lap.var())              # variance-of-Laplacian focus measure
    residual = gray - gaussian_filter(gray, sigma=1.0)
    return np.array([sharpness, float(residual.std()),
                     float(np.abs(residual).mean())], dtype=np.float32)


# --------------------------------------------------------------------------- #
# 3. colour / gamut features (screens shift saturation & channel balance)
# --------------------------------------------------------------------------- #
def _color_features(img_rgb):
    rgb = np.asarray(img_rgb.resize((TEX_SIZE, TEX_SIZE)), dtype=np.float32) / 255.0
    feats = []
    for c in range(3):
        ch = rgb[..., c]
        feats += [float(ch.mean()), float(ch.std())]
    r, g, b = rgb[..., 0].ravel(), rgb[..., 1].ravel(), rgb[..., 2].ravel()
    feats += [float(np.corrcoef(r, g)[0, 1]),
              float(np.corrcoef(r, b)[0, 1]),
              float(np.corrcoef(g, b)[0, 1])]
    hsv = np.asarray(img_rgb.convert("HSV").resize((TEX_SIZE, TEX_SIZE)),
                     dtype=np.float32) / 255.0
    s = hsv[..., 1]
    feats += [float(s.mean()), float(s.std())]
    return np.nan_to_num(np.array(feats, dtype=np.float32))


# --------------------------------------------------------------------------- #
# 4. texture: simple 8-neighbour LBP histogram (screen-door micro-pattern)
# --------------------------------------------------------------------------- #
def _lbp_features(gray_tex):
    g = gray_tex
    c = g[1:-1, 1:-1]
    nb = [g[:-2, :-2], g[:-2, 1:-1], g[:-2, 2:],
          g[1:-1, 2:], g[2:, 2:], g[2:, 1:-1],
          g[2:, :-2], g[1:-1, :-2]]
    codes = np.zeros_like(c, dtype=np.uint8)
    for i, n in enumerate(nb):
        codes |= ((n >= c).astype(np.uint8) << i)
    hist, _ = np.histogram(codes, bins=N_LBP_BINS, range=(0, 256), density=True)
    return hist.astype(np.float32)


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def feature_names():
    names = [f"fft_radial_{i:02d}" for i in range(N_RADIAL)]
    names += ["fft_peak_ratio", "fft_spikiness", "fft_peak_energy_frac"]
    names += ["sharpness", "residual_std", "residual_mad"]
    for c in "RGB":
        names += [f"{c}_mean", f"{c}_std"]
    names += ["corr_rg", "corr_rb", "corr_gb", "sat_mean", "sat_std"]
    names += [f"lbp_{i:02d}" for i in range(N_LBP_BINS)]
    return names


def extract_features(image) -> np.ndarray:
    """image: a file path or a PIL.Image. Returns a 1-D float32 feature vector."""
    img = _open_rgb(image)
    gray_fft = _gray_array(img, FFT_SIZE)
    gray_tex = _gray_array(img, TEX_SIZE)
    return np.concatenate([
        _fft_features(gray_fft),
        _spatial_features(gray_fft),
        _color_features(img),
        _lbp_features(gray_tex),
    ]).astype(np.float32)


if __name__ == "__main__":
    import sys
    v = extract_features(sys.argv[1])
    print("feature vector length:", len(v))
    for n, x in zip(feature_names(), v):
        print(f"{n:24s} {x: .5f}")
