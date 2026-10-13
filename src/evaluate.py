import glob
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from utils import CLASSES, load_data

# 1. Results table from all saved json files
rows = [json.load(open(p)) for p in glob.glob(os.path.join("results", "*.json"))]
df = pd.DataFrame(rows)[["model", "train_acc", "val_acc"]].sort_values("val_acc", ascending=False)
df[["train_acc", "val_acc"]] = (df[["train_acc", "val_acc"]] * 100).round(2)
df.to_csv(os.path.join("results", "results_table.csv"), index=False)
print(df.to_string(index=False))

best = df.iloc[0]["model"]
print("\nBest model:", best)

# 2. Predictions of the best model on the validation set
_, _, Xva, yva = load_data()
model = joblib.load(os.path.join("models", f"{best}.joblib"))
pred = model.predict(Xva)

# 3. Per-class accuracy
cm = confusion_matrix(yva, pred, labels=range(10))
per_class = cm.diagonal() / cm.sum(axis=1)
pc = pd.DataFrame({"class": CLASSES, "accuracy": (per_class * 100).round(2)})
pc.to_csv(os.path.join("results", "per_class_accuracy.csv"), index=False)
print("\nPer-class accuracy (%):")
print(pc.to_string(index=False))

# 4. Confusion matrix heatmap (row-normalised)
plt.figure(figsize=(9, 7))
sns.heatmap(cm / cm.sum(axis=1, keepdims=True), annot=True, fmt=".2f",
            cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Confusion matrix - {best}")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join("results", "confusion_matrix.png"), dpi=150)

# 5. Per-class accuracy bar chart
plt.figure(figsize=(9, 4))
plt.bar(CLASSES, per_class * 100)
plt.ylabel("Accuracy (%)")
plt.xticks(rotation=45, ha="right")
plt.title(f"Per-class accuracy - {best}")
plt.tight_layout()
plt.savefig(os.path.join("results", "per_class_accuracy.png"), dpi=150)

# 6. Top confusions (for the report)
off = cm.copy()
np.fill_diagonal(off, 0)
print("\nTop 5 confusions:")
for idx in np.dstack(np.unravel_index(np.argsort(off.ravel())[::-1], off.shape))[0][:5]:
    t, p = idx
    print(f"  true '{CLASSES[t]}' predicted as '{CLASSES[p]}': {off[t, p]} images")