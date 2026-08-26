import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

K = 5
N_FEATURES = 512
TRIALS = 20

d = np.load("features.npz")
X_tr, y_tr = d["X_tr"], d["y_tr"]
X_te, y_te = d["X_te"], d["y_te"]

sc = StandardScaler().fit(X_tr)
X_tr = sc.transform(X_tr)
X_te = sc.transform(X_te)


def test_acc(mask):
    knn = KNeighborsClassifier(n_neighbors=K)
    knn.fit(X_tr[:, mask], y_tr)
    return knn.score(X_te[:, mask], y_te)


saved = np.load("early_stopping.npz")
chosen = saved["mask_stop"].astype(bool)
n = int(chosen.sum())

mine = test_acc(chosen)
baseline = test_acc(np.ones(N_FEATURES, dtype=bool))

print("selected solution:", n, "features, test", round(mine, 4))
print("baseline, 512 features: test", round(baseline, 4))
print()
print("comparing against", TRIALS, "random subsets of", n, "features")
print()

rng = np.random.RandomState(0)
scores = []

for i in range(TRIALS):
    m = np.zeros(N_FEATURES, dtype=bool)
    m[rng.choice(N_FEATURES, n, replace=False)] = True
    s = test_acc(m)
    scores.append(s)
    print("  trial", i + 1, ":", round(s, 4))

scores = np.array(scores)

print()
print("random subsets: mean", round(scores.mean(), 4),
      "| sd", round(scores.std(), 4),
      "| min", round(scores.min(), 4),
      "| max", round(scores.max(), 4))

diff = mine - scores.mean()
z = diff / scores.std() if scores.std() > 0 else 0.0

print()
print("advantage over random:", round(diff, 4))
print("z score:", round(z, 2))
print("random trials matching or beating it:",
      int((scores >= mine).sum()), "of", TRIALS)
print()

if z > 2:
    print("the search contributed something random selection does not reach")
elif z > 1:
    print("larger than random on average, but inside the spread of the draws")
else:
    print("no detectable advantage: the gain came from using fewer features,")
    print("not from which features were chosen")

print()
print("against the 512-feature baseline:")
print("  selected  ", round(mine - baseline, 4))
print("  random avg", round(scores.mean() - baseline, 4))
