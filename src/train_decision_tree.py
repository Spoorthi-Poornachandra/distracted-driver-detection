from sklearn.tree import DecisionTreeClassifier
from utils import load_data, report

Xtr, ytr, Xva, yva = load_data()
# max_features="sqrt" tries about 110 random pixels per split instead of all 12288,
# which makes training roughly 100x faster
m = DecisionTreeClassifier(max_depth=25, max_features="sqrt", random_state=0)
m.fit(Xtr, ytr)
report("decision_tree", m, Xtr, ytr, Xva, yva,
       {"max_depth": 25, "max_features": "sqrt"})