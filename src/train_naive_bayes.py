from sklearn.naive_bayes import GaussianNB
from utils import load_data, report

Xtr, ytr, Xva, yva = load_data()
m = GaussianNB().fit(Xtr, ytr)
report("naive_bayes", m, Xtr, ytr, Xva, yva)