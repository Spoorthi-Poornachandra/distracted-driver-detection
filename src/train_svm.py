from sklearn.linear_model import SGDClassifier
from utils import load_data, report

Xtr, ytr, Xva, yva = load_data()

best, best_va, best_alpha = None, -1, None
for alpha in [1e-5, 1e-4, 1e-3, 1e-2]:  # L2 regularization strength
    m = SGDClassifier(loss="hinge", penalty="l2", alpha=alpha,
                      max_iter=30, tol=None, random_state=0, n_jobs=-1)
    m.fit(Xtr, ytr)
    va = m.score(Xva, yva)
    print(f"alpha={alpha}: val acc = {va:.4f}")
    if va > best_va:
        best, best_va, best_alpha = m, va, alpha

report("linear_svm", best, Xtr, ytr, Xva, yva, {"alpha": best_alpha})