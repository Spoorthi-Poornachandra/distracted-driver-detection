import glob
import json
import os
import random
import sys

import cv2
import gradio as gr
import joblib
import numpy as np

sys.path.insert(0, "src")
from utils import CLASSES

rows = [json.load(open(p)) for p in glob.glob(os.path.join("results", "*.json"))]
best = max(rows, key=lambda r: r["val_acc"])["model"]
model = joblib.load(os.path.join("models", f"{best}.joblib"))
mean = np.load(os.path.join("models", "mean.npy"))
print("Loaded model:", best)


def predict(img):
    if img is None:
        return None
    im = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
    x = (im.flatten().astype(np.float32) / 255.0 - mean).reshape(1, -1)
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(x)[0]
    else:
        s = model.decision_function(x)[0]
        e = np.exp(s - s.max())
        p = e / e.sum()
    return {CLASSES[i]: float(p[i]) for i in range(10)}


examples = []
for c in range(10):
    files = glob.glob(os.path.join("data", "train", f"c{c}", "*.jpg"))
    if files:
        examples.append(random.choice(files))

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Driver image"),
    outputs=gr.Label(num_top_classes=5, label="Predicted activity"),
    title="Distracted Driver Detection",
    description=f"Upload a driver image to classify it into 10 activities. Model: {best}",
    examples=examples,
    cache_examples=False,
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()