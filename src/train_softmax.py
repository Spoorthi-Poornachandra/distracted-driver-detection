from sklearn.linear_model import LogisticRegression
from utils import load_data, report

Xtr, ytr, Xva, yva = load_data()

best, best_va, best_C = None, -1, None
for C in [1e-4, 1e-3, 1e-2]:  # inverse L2 regularization strength
    m = LogisticRegression(C=C, max_iter=200)  # multinomial softmax, lbfgs
    m.fit(Xtr, ytr)
    va = m.score(Xva, yva)
    print(f"C={C}: val acc = {va:.4f}")
    if va > best_va:
        best, best_va, best_C = m, va, C

report("softmax", best, Xtr, ytr, Xva, yva, {"C": best_C})