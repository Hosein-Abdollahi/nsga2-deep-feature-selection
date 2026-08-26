"""
Track validation and test accuracy every generation.

The test set is read here only to draw the curve. It is never part of the
fitness and never affects selection.
"""
import random

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from deap import base, creator, tools
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

SEED = 42
K = 5
N_FEATURES = 512
POP_SIZE = 100
N_GEN = 150
CX_PROB, MUT_PROB = 0.9, 0.1
WINDOW = 9

random.seed(SEED)
np.random.seed(SEED)

d = np.load("features.npz")
scaler = StandardScaler().fit(d["X_tr"])
X_tr, y_tr = scaler.transform(d["X_tr"]), d["y_tr"]
X_va, y_va = scaler.transform(d["X_va"]), d["y_va"]
X_te, y_te = scaler.transform(d["X_te"]), d["y_te"]

creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)


def score(mask, X, y):
    knn = KNeighborsClassifier(n_neighbors=K).fit(X_tr[:, mask], y_tr)
    return knn.score(X[:, mask], y)


def evaluate(ind):
    mask = np.array(ind, dtype=bool)
    n = int(mask.sum())
    return (0.0, N_FEATURES) if n == 0 else (score(mask, X_va, y_va), n)


toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_bool, N_FEATURES)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxUniform, indpb=0.5)
toolbox.register("mutate", tools.mutFlipBit, indpb=1.0 / N_FEATURES)
toolbox.register("select", tools.selNSGA2)

pop = toolbox.population(n=POP_SIZE)
for ind in pop:
    ind.fitness.values = toolbox.evaluate(ind)
pop = toolbox.select(pop, POP_SIZE)

val_curve, test_curve, size_curve = [], [], []

for gen in range(1, N_GEN + 1):
    kids = [toolbox.clone(p) for p in tools.selTournamentDCD(pop, POP_SIZE)]

    for a, b in zip(kids[::2], kids[1::2]):
        if random.random() < CX_PROB:
            toolbox.mate(a, b)
            del a.fitness.values, b.fitness.values
        if random.random() < MUT_PROB:
            toolbox.mutate(a)
            del a.fitness.values
        if random.random() < MUT_PROB:
            toolbox.mutate(b)
            del b.fitness.values

    for ind in kids:
        if not ind.fitness.valid:
            ind.fitness.values = toolbox.evaluate(ind)

    pop = toolbox.select(pop + kids, POP_SIZE)

    best = max(pop, key=lambda i: i.fitness.values[0])
    mask = np.array(best, dtype=bool)

    val_curve.append(best.fitness.values[0])
    test_curve.append(score(mask, X_te, y_te))
    size_curve.append(int(mask.sum()))

    print("gen", gen, "of", N_GEN, "| val", round(val_curve[-1], 4),
          "| test", round(test_curve[-1], 4), end="\r")

print()

val_curve = np.array(val_curve)
test_curve = np.array(test_curve)
size_curve = np.array(size_curve)

smooth = np.convolve(test_curve, np.ones(WINDOW) / WINDOW, mode="valid")
smooth_x = np.arange(WINDOW // 2 + 1, WINDOW // 2 + 1 + len(smooth))
peak = int(smooth_x[np.argmax(smooth)])

print()
print("validation  gen 1", round(val_curve[0], 4),
      "-> gen", N_GEN, round(val_curve[-1], 4))
print("test        gen 1", round(test_curve[0], 4),
      "-> gen", N_GEN, round(test_curve[-1], 4))
print()
print("best test", round(test_curve.max(), 4),
      "at generation", int(np.argmax(test_curve)) + 1)
print("smoothed peak at generation", peak)
print("lost from peak to end:", round(test_curve.max() - test_curve[-1], 4))

np.savez("overfitting_curve.npz", val=val_curve, test=test_curve,
         size=size_curve, peak=peak)

gens = np.arange(1, N_GEN + 1)

plt.figure(figsize=(8, 4.8))
plt.plot(gens, val_curve, color="#2b6cb0", lw=1.4,
         label="validation (the fitness)")
plt.plot(gens, test_curve, color="#c53030", lw=1.0, alpha=0.45, label="test")
plt.plot(smooth_x, smooth, color="#c53030", lw=2.0, ls="--",
         label="test, smoothed")
plt.axvline(peak, color="gray", ls=":", lw=1.2,
            label="test peak, gen " + str(peak))
plt.xlabel("generation")
plt.ylabel("accuracy")
plt.title("Validation keeps climbing, test does not")
plt.grid(alpha=0.25)
plt.legend(frameon=False, fontsize=9)
plt.tight_layout()
plt.savefig("overfitting_curve.png", dpi=160)

print("saved overfitting_curve.png")
