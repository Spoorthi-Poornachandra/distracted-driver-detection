import json
import os

import joblib
import numpy as np
from sklearn.metrics import accuracy_score

PROC = os.path.join("data", "processed")
CLASSES = ["safe driving", "texting - right", "phone - right", "texting - left",
           "phone - left", "radio", "drinking", "reaching behind",
           "hair & makeup", "talking to passenger"]


def load_data():
    """Scale to [0,1] and subtract the training mean."""
    Xtr = np.load(os.path.join(PROC, "X_train.npy")).astype(np.float32) / 255.0
    Xva = np.load(os.path.join(PROC, "X_val.npy")).astype(np.float32) / 255.0
    ytr = np.load(os.path.join(PROC, "y_train.npy"))
    yva = np.load(os.path.join(PROC, "y_val.npy"))
    mean = Xtr.mean(axis=0)
    Xtr -= mean
    Xva -= mean
    os.makedirs("models", exist_ok=True)
    np.save(os.path.join("models", "mean.npy"), mean)
    return Xtr, ytr, Xva, yva


def report(name, model, Xtr, ytr, Xva, yva, extra=None):
    tr = accuracy_score(ytr, model.predict(Xtr))
    va = accuracy_score(yva, model.predict(Xva))
    print(f"{name}: train acc = {tr:.4f}, val acc = {va:.4f}")
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    joblib.dump(model, os.path.join("models", f"{name}.joblib"))
    out = {"model": name, "train_acc": tr, "val_acc": va}
    if extra:
        out.update(extra)
    with open(os.path.join("results", f"{name}.json"), "w") as f:
        json.dump(out, f, indent=2)