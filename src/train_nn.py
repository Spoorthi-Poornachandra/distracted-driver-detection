from sklearn.neural_network import MLPClassifier
from utils import load_data, report

Xtr, ytr, Xva, yva = load_data()
# 12288 -> 100 (ReLU) -> 10 (softmax), as in the paper
m = MLPClassifier(hidden_layer_sizes=(100,), activation="relu", alpha=1e-2,
                  max_iter=60, random_state=0, verbose=True)
m.fit(Xtr, ytr)
report("two_layer_nn", m, Xtr, ytr, Xva, yva, {"hidden": 100, "alpha": 1e-2})