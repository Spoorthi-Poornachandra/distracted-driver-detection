import glob
import json
import os
import shutil
import sys

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, "src")
from utils import CLASSES

# Recreate the exact split from preprocess.py
df = pd.read_csv(os.path.join("data", "driver_imgs_list.csv"))
y = df["classname"].str[1:].astype(int).values
idx = np.arange(len(df))
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
_, va = next(gss.split(idx, y, groups=df["subject"].values))
val = df.iloc[va]

# Best model + preprocessing mean
rows = [json.load(open(p)) for p in glob.glob(os.path.join("results", "*.json"))]
best = max(rows, key=lambda r: r["val_acc"])["model"]
model = joblib.load(os.path.join("models", f"{best}.joblib"))
mean = np.load(os.path.join("models", "mean.npy"))
print("Model:", best, "\n")

out = os.path.join("demo", "val_samples")
os.makedirs(out, exist_ok=True)

correct = total = 0
for c in range(10):
    sub = val[val["classname"] == f"c{c}"].sample(3, random_state=1)
    for img in sub["img"]:
        src = os.path.join("data", "train", f"c{c}", img)
        im = cv2.cvtColor(cv2.imread(src), cv2.COLOR_BGR2RGB)
        x = cv2.resize(im, (64, 64), interpolation=cv2.INTER_AREA)
        x = (x.flatten().astype(np.float32) / 255.0 - mean).reshape(1, -1)
        pred = int(model.predict(x)[0])
        ok = pred == c
        correct += ok
        total += 1
        name = f"{'OK' if ok else 'WRONG'}_true-{CLASSES[c].replace(' ', '')}_pred-{CLASSES[pred].replace(' ', '')}_{img}"
        shutil.copy(src, os.path.join(out, name))
        print(f"{'OK   ' if ok else 'WRONG'} true={CLASSES[c]:22s} pred={CLASSES[pred]}")

print(f"\n{correct}/{total} correct on these held-out-driver images")