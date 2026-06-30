"""
train.py
--------
Trains the screen-recapture detector.

  python train.py                 # uses ./data/real and ./data/screen
  python train.py path/to/data    # custom data dir with real/ and screen/ inside

Reports HONEST accuracy via 5-fold stratified cross-validation (every image is
scored by a model that never saw it), prints a confusion matrix + the most
useful features, then refits on all data and saves model.pkl.
"""

import os
import sys
import glob
import time
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from features import extract_features, feature_names

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".tif", ".tiff")


def load_folder(folder, label):
    X, paths = [], []
    for p in sorted(glob.glob(os.path.join(folder, "*"))):
        if p.lower().endswith(IMG_EXTS):
            try:
                X.append(extract_features(p))
                paths.append(p)
            except Exception as e:
                print(f"  [skip] {p}: {e}")
    return X, [label] * len(X), paths


def build_model():
    # Small dataset (~120 imgs), 49 features -> a soft-voting ensemble of three
    # different model families generalizes better than any single one here,
    # while staying tiny/fast (all three together still predict in a few ms).
    rf = RandomForestClassifier(
        n_estimators=400, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    gb = GradientBoostingClassifier(
        n_estimators=200, max_depth=2, learning_rate=0.05, random_state=42,
    )
    svm = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=4, gamma="scale", probability=True,
            class_weight="balanced", random_state=42),
    )
    return VotingClassifier(
        estimators=[("rf", rf), ("gb", gb), ("svm", svm)],
        voting="soft", n_jobs=-1,
    )


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    real_dir = os.path.join(data_dir, "real")
    screen_dir = os.path.join(data_dir, "screen")

    print(f"Loading from {real_dir} and {screen_dir} ...")
    t0 = time.time()
    Xr, yr, pr = load_folder(real_dir, 0)
    Xs, ys, ps = load_folder(screen_dir, 1)
    if not Xr or not Xs:
        sys.exit("ERROR: need images in BOTH data/real and data/screen")

    X = np.array(Xr + Xs, dtype=np.float32)
    y = np.array(yr + ys)
    print(f"  real={len(Xr)}  screen={len(Xs)}  "
          f"features={X.shape[1]}  ({time.time()-t0:.1f}s to extract)")

    # ---- honest cross-validated accuracy ----
    clf = build_model()
    n_splits = min(5, len(Xr), len(Xs))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_cv = cross_val_predict(clf, X, y, cv=skf, n_jobs=-1)

    acc = accuracy_score(y, y_cv)
    cm = confusion_matrix(y, y_cv)
    print("\n================ CROSS-VALIDATED RESULTS ================")
    print(f"Accuracy: {acc*100:.1f}%   ({n_splits}-fold CV, held-out)")
    print("\nConfusion matrix (rows=true, cols=pred):")
    print("              pred_real  pred_screen")
    print(f"  true_real      {cm[0,0]:4d}       {cm[0,1]:4d}")
    print(f"  true_screen    {cm[1,0]:4d}       {cm[1,1]:4d}")
    print("\n" + classification_report(y, y_cv, target_names=["real", "screen"]))

    # ---- refit on everything, save ----
    clf.fit(X, y)
    joblib.dump(clf, "model.pkl")
    print("Saved model.pkl")

    # ---- top features (for the write-up) ----
    # VotingClassifier has no feature_importances_ itself; report them from
    # the RandomForest sub-model, which is interpretable in the same way.
    names = feature_names()
    rf_fitted = clf.named_estimators_["rf"]
    imp = rf_fitted.feature_importances_
    order = np.argsort(imp)[::-1][:10]
    print("\nTop 10 features by importance (from the RF sub-model):")
    for i in order:
        print(f"  {names[i]:24s} {imp[i]:.4f}")


if __name__ == "__main__":
    main()
