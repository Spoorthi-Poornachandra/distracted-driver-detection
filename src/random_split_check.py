import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

P = os.path.join("data", "processed")
X = np.concatenate([np.load(os.path.join(P, "X_train.npy")),
                    np.load(os.path.join(P, "X_val.npy"))]).astype(np.float32) / 255.0
y = np.concatenate([np.load(os.path.join(P, "y_train.npy")),
                    np.load(os.path.join(P, "y_val.npy"))])

Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
mean = Xtr.mean(axis=0)
Xtr -= mean
Xva -= mean

m = MLPClassifier(hidden_layer_sizes=(100,), activation="relu", alpha=1e-2,
                  max_iter=60, random_state=0).fit(Xtr, ytr)
line = f"Random split NN: train={m.score(Xtr, ytr):.4f} val={m.score(Xva, yva):.4f}"
print(line)
with open(os.path.join("results", "random_split_check.txt"), "w") as f:
    f.write(line + "\n")