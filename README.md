# Spot the Fake Photo

A small detector that tells a **real photo** apart from a **photo of a screen**
(someone re-photographing a phone/laptop instead of shooting the real thing).

Built as a take-home for SalesCode. The goal was a small, fast, cheap solution
rather than a big model — so this uses handcrafted image features fed into a
light scikit-learn classifier, no GPU and no deep net.

## How it works (short version)

`predict.py image.jpg` prints a number from 0 to 1 (0 = real, 1 = photo of a
screen). Under the hood:

- **FFT / frequency features** — re-photographing a screen can leave periodic
  moiré patterns; I look for that in the 2D FFT.
- **Texture (LBP)** — catches the fine "screen-door" pattern of a display.
- **Colour + sharpness** — screens shift colour slightly and a second capture
  tends to soften detail.

Those ~49 features go into a small voting ensemble (RandomForest +
GradientBoosting + SVM). Same feature code runs at training and prediction time.

Full write-up, accuracy, and what I'd do next are in **[NOTE.md](NOTE.md)**.

## Results

- **Accuracy:** 86.8% (5-fold cross-validation on 136 of my own photos)
- **Latency:** ~217 ms/image on an M-series MacBook Air (single CPU core)
- **Cost:** runs on-device, so effectively free; ~$1–2 per million on a small
  cloud CPU as a fallback

(Honest note: I aimed for 95% but found modern phone cameras suppress the moiré
signal — the FFT of my real vs recaptured pairs looked nearly identical. NOTE.md
explains the full diagnosis and why the ceiling sits around 87% for this
approach.)

## Run it

```bash
pip install -r requirements.txt

python train.py                     # trains on data/real and data/screen, saves model.pkl
python predict.py path/to/image.jpg # prints the 0–1 score
python bench.py                     # prints latency per image
```

Expected data layout (not included in this repo — they're my own photos):

```
data/
  real/      # normal photos
  screen/    # photos of a screen
```

## Files

| File | What it does |
|------|--------------|
| `predict.py` | The predictor SalesCode runs |
| `features.py` | Feature extraction (shared by train + predict) |
| `train.py` | Trains the model, reports accuracy + confusion matrix |
| `bench.py` | Measures latency per image |
| `diagnose.py` | Visualises the FFT of a real vs screen image (`fft_compare.png`) |
| `app.py` + `index.html` | Optional live webcam demo (Flask) |
| `NOTE.md` | Half-page write-up |

## Live demo (optional)

```bash
pip install flask
python app.py        # then open http://127.0.0.1:5000
```

Heads-up: the model is trained on phone photos, so the laptop webcam (different
sensor) skews its scores — it's a quick demo, not the graded path.
