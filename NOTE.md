# Spot the Fake Photo — short note

> Fill in the **[BRACKETED]** numbers after running `train.py` and `bench.py` on your data.

## Approach
There's no object to recognise here — the only clue is the *capture artifact* a
screen leaves behind. So instead of a heavy CNN I extract a small set of
handcrafted features that target those artifacts, and feed them to a light
RandomForest:

- **Frequency / moiré (the main signal).** Re-photographing a screen beats the
  camera's sensor grid against the display's pixel grid, producing periodic
  *moiré* energy. I take a Hann-windowed 2D FFT and measure (a) the radial
  energy profile and (b) mid/high-frequency "peakiness" (sharp localised spikes).
  These were the top-ranked features by importance — the model is keying on the
  physical artifact, not on framing or scene type.
- **Sharpness / residual.** A double capture softens detail slightly
  (variance-of-Laplacian + high-pass residual stats).
- **Colour / gamut.** Screens shift saturation and channel balance a little.
- **Texture (LBP).** An 8-neighbour local-binary-pattern histogram catches the
  screen-door micro-pattern.

49 features total, pure numpy/scipy (scipy already ships with scikit-learn, so
no extra heavy dependency). Same `features.py` is used at train and inference
time, so there's no train/serve skew.

## Data
~[N_REAL] real photos and ~[N_SCREEN] recaptures, shot on [YOUR PHONE]. Most
recaptures were re-shot from my own real photos displayed on a Mac LCD and a
second phone, so real/screen pairs share content — this forces the model to
learn the *capture method*, not the scene. Variety in angle, distance, lighting
and screen brightness; a few "real photo that contains a powered-off screen" as
hard negatives.

## Accuracy (honest)
**[XX.X]%** via 5-fold stratified cross-validation (every image scored by a model
that never saw it). Confusion matrix from `train.py`:
[PASTE THE CONFUSION MATRIX HERE]

## Required numbers
- **Latency:** ~**[XX] ms/image** on [YOUR DEVICE, e.g. "M2 MacBook Air, single CPU core"]
  (from `bench.py`). Dominated by the 512px FFT; drop `FFT_SIZE` to trade a little
  accuracy for speed.
- **Cost/image:** **On-device ≈ free** (target deployment — runs on the phone).
  On a cloud CPU as a fallback: at ~[T] ms/image one core does ~[1000/T] img/s;
  a small ~2-core instance (~$0.03/hr) ⇒ roughly **$[~0.2] per million images**.
  Assumptions: CPU-only, no GPU, batch of 1, instance fully utilised.

## What I'd improve with more time
- **More + harder data:** more screen types (OLED phones, high-refresh monitors,
  printouts), more compression levels and angles, to close the gap to their
  held-out distribution.
- **Adversarial robustness (cheaters adapt):** attackers will blur or rescale to
  kill moiré. I'd add anti-aliasing-aware features, retrain on adversarial
  examples periodically, and watch score drift on flagged traffic.
- **Tiny/fast on phone:** the features are already cheap; I'd port the FFT to a
  fixed 256px and ship the RF as a handful of KB, or distil to a tiny on-device
  model so nothing leaves the phone.
- **Choosing the fraud cut-off:** don't default to 0.5. Pick the threshold from a
  precision/recall curve at the business's tolerated false-positive rate (blocking
  a real user is costly), e.g. flag only above the score where precision ≥ 0.98,
  and send borderline scores to manual review.
