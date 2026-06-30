"""
diagnose.py -- run this locally to SEE whether moire is actually present in
your photos. Saves fft_compare.png you can open and look at.

  python diagnose.py data/real/IMG_2210.jpg data/screen/IMG_2309.jpg
"""
import sys
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def fft_mag(path, size=512):
    img = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
    g = np.asarray(img, dtype=np.float32) / 255.0
    win = np.hanning(size)[:, None] * np.hanning(size)[None, :]
    F = np.fft.fftshift(np.fft.fft2(g * win))
    return g, np.log1p(np.abs(F))

real_path, screen_path = sys.argv[1], sys.argv[2]
g1, m1 = fft_mag(real_path)
g2, m2 = fft_mag(screen_path)

fig, ax = plt.subplots(2, 2, figsize=(10, 10))
ax[0,0].imshow(g1, cmap="gray"); ax[0,0].set_title("REAL - image"); ax[0,0].axis("off")
ax[0,1].imshow(m1, cmap="inferno"); ax[0,1].set_title("REAL - log FFT magnitude"); ax[0,1].axis("off")
ax[1,0].imshow(g2, cmap="gray"); ax[1,0].set_title("SCREEN - image"); ax[1,0].axis("off")
ax[1,1].imshow(m2, cmap="inferno"); ax[1,1].set_title("SCREEN - log FFT magnitude"); ax[1,1].axis("off")
plt.tight_layout()
plt.savefig("fft_compare.png", dpi=130)
print("Saved fft_compare.png -- open it and compare the two FFT panels (right column).")
print("Look for: bright dots/spikes/grid pattern in SCREEN's FFT that REAL's FFT lacks.")
print("If both FFTs look like a similar plain blob/cross with no extra spikes,")
print("moire is weak/absent in your photos and the fix is more distance/angle variety")
print("or leaning on the other features instead.")
