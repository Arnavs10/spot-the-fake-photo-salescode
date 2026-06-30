"""Fill this in. That's the whole interface.

Usage:
    python predict.py some_image.jpg
Prints ONE number from 0 to 1:
    0 = real photo,  1 = photo of a screen (recapture / fraud)
A hard 0 or 1 is fine if your method gives a yes/no answer.

Approach: handcrafted frequency/texture/colour features (moire in the FFT is the
main signal) -> a small RandomForest. See features.py and train.py. The model
file model.pkl must sit next to this script.
"""

import os
import sys
import joblib
from features import extract_features

_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        here = os.path.dirname(os.path.abspath(__file__))
        _MODEL = joblib.load(os.path.join(here, "model.pkl"))
    return _MODEL


def predict(image_path: str) -> float:
    feats = extract_features(image_path).reshape(1, -1)
    # probability that the image is class 1 (photo-of-a-screen)
    score = _model().predict_proba(feats)[0, 1]
    return float(score)


if __name__ == "__main__":
    print(predict(sys.argv[1]))
