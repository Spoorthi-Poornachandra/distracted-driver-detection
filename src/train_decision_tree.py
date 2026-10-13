from sklearn.tree import DecisionTreeClassifier
from utils import load_data, report

Xtr, ytr, Xva, yva = load_data()
m = DecisionTreeClassifier(max_depth=25, random_state=0).fit(Xtr, ytr)
report("decision_tree", m, Xtr, ytr, Xva, yva, {"max_depth": 25})