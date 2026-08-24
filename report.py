import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

K = 5

data = np.load("features.npz")
X_tr, y_tr = data["X_tr"], data["y_tr"]
X_te, y_te = data["X_te"], data["y_te"]

scaler = StandardScaler().fit(X_tr)
X_tr = scaler.transform(X_tr)
X_te = scaler.transform(X_te)

front = np.load("front.npz")
n_features = front["n_features"]
val_acc = front["val_acc"]
masks = front["masks"]

baseline = np.load("baseline.npz")
base_val = float(baseline["val"])
base_test = float(baseline["test"])


def test_accuracy(mask):
    knn = KNeighborsClassifier(n_neighbors=K)
    knn.fit(X_tr[:, mask], y_tr)
    return knn.score(X_te[:, mask], y_te)


best = int(np.argmax(val_acc))
best_test = test_accuracy(masks[best])

print("Pareto front")
for n, acc in zip(n_features, val_acc):
    print("  ", n, "features -> val acc", round(acc, 4))

print()
print("baseline, all 512 features")
print("   val ", round(base_val, 4), " test ", round(base_test, 4))

print()
print("best validation solution")
print("   features:", n_features[best])
print("   val ", round(val_acc[best], 4), " test ", round(best_test, 4))
print("   gap ", round(val_acc[best] - best_test, 4))

print()
print("features kept:", n_features[best], "of 512",
      "(" + str(round(100 * n_features[best] / 512, 1)) + "%)")

plt.figure(figsize=(7, 4.5))
plt.plot(n_features, val_acc, "o-", color="#2b6cb0", linewidth=1.4,
         markersize=5, label="Pareto front")
plt.axhline(base_val, linestyle="--", color="gray", linewidth=1,
            label="baseline, 512 features (" + str(round(base_val, 3)) + ")")
plt.scatter(n_features[best], val_acc[best], s=130, facecolors="none",
            edgecolors="#c53030", linewidths=2, zorder=5,
            label="best validation")
plt.xlabel("number of selected features")
plt.ylabel("validation accuracy")
plt.title("NSGA-II feature selection on ResNet-18 features (CIFAR-10)")
plt.grid(alpha=0.25)
plt.legend(frameon=False, fontsize=9)
plt.tight_layout()
plt.savefig("pareto_front.png", dpi=160)

print()
print("saved pareto_front.png")
