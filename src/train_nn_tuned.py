from sklearn.neural_network import MLPClassifier
from utils import load_data, report

Xtr, ytr, Xva, yva = load_data()

best, best_va, best_alpha = None, -1, None
for alpha in [0.1, 1.0, 3.0]:
    m = MLPClassifier(hidden_layer_sizes=(100,), activation="relu", alpha=alpha,
                      max_iter=30, random_state=0)
    m.fit(Xtr, ytr)
    va = m.score(Xva, yva)
    print(f"alpha={alpha}: val acc = {va:.4f}")
    if va > best_va:
        best, best_va, best_alpha = m, va, alpha

report("two_layer_nn_tuned", best, Xtr, ytr, Xva, yva,
       {"hidden": 100, "alpha": best_alpha})