import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

K = 5

data = np.load("features.npz")
X_tr, y_tr = data["X_tr"], data["y_tr"]
X_va, y_va = data["X_va"], data["y_va"]
X_te, y_te = data["X_te"], data["y_te"]

scaler = StandardScaler().fit(X_tr)
X_tr = scaler.transform(X_tr)
X_va = scaler.transform(X_va)
X_te = scaler.transform(X_te)

knn = KNeighborsClassifier(n_neighbors=K)
knn.fit(X_tr, y_tr)

val_acc = knn.score(X_va, y_va)
test_acc = knn.score(X_te, y_te)

print("baseline with all 512 features, k =", K)
print("  validation accuracy:", round(val_acc, 4))
print("  test accuracy:      ", round(test_acc, 4))

np.savez("baseline.npz", val=val_acc, test=test_acc, n_features=512)
