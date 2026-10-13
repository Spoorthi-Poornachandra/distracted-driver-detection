"""Load State Farm images, resize to 64x64x3, flatten, split 80/20, save as .npy."""
import argparse
import os

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

DATA_DIR = "data"
OUT_DIR = os.path.join(DATA_DIR, "processed")
IMG_SIZE = 64


def load_images(df):
    X = np.zeros((len(df), IMG_SIZE * IMG_SIZE * 3), dtype=np.uint8)
    for i, (cls, img) in enumerate(zip(df["classname"], df["img"])):
        path = os.path.join(DATA_DIR, "train", cls, img)
        im = cv2.imread(path)
        if im is None:
            raise FileNotFoundError(path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        X[i] = im.flatten()
        if i % 2000 == 0:
            print(f"  loaded {i}/{len(df)}")
    y = df["classname"].str[1:].astype(int).values  # c0..c9 -> 0..9
    return X, y


def main(split):
    df = pd.read_csv(os.path.join(DATA_DIR, "driver_imgs_list.csv"))
    print(f"{len(df)} images, {df['subject'].nunique()} drivers")

    X, y = load_images(df)
    idx = np.arange(len(df))

    if split == "driver":
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        tr, va = next(gss.split(idx, y, groups=df["subject"].values))
    else:
        tr, va = train_test_split(idx, test_size=0.2, random_state=42, shuffle=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(os.path.join(OUT_DIR, "X_train.npy"), X[tr])
    np.save(os.path.join(OUT_DIR, "y_train.npy"), y[tr])
    np.save(os.path.join(OUT_DIR, "X_val.npy"), X[va])
    np.save(os.path.join(OUT_DIR, "y_val.npy"), y[va])

    print("Split:", split)
    print("X_train", X[tr].shape, "X_val", X[va].shape)
    print("Train class counts:", np.bincount(y[tr], minlength=10))
    print("Val class counts:  ", np.bincount(y[va], minlength=10))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["driver", "random"], default="driver")
    main(p.parse_args().split)