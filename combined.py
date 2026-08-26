"""
Both fixes at once: a cross validated fitness so the objective is less noisy,
and a stopping generation found by an inner cross validation.

Nothing here reads the test set until the final table.
"""

import random

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from deap import base, creator, tools
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

SEED = 42
K = 5
N_FEATURES = 512
POP_SIZE = 100
N_GEN = 40
INNER_GEN = 40
INNER_FOLDS = 2
CV_FOLDS = 3
CX_PROB, MUT_PROB = 0.9, 0.1
WINDOW = 5

d = np.load("features.npz")
X_dev_raw = np.vstack([d["X_tr"], d["X_va"]])
y_dev = np.concatenate([d["y_tr"], d["y_va"]])

scaler = StandardScaler().fit(X_dev_raw)
X_dev = scaler.transform(X_dev_raw)
X_te, y_te = scaler.transform(d["X_te"]), d["y_te"]

creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)


def knn_acc(mask, Xfit, yfit, Xeval, yeval):
    knn = KNeighborsClassifier(n_neighbors=K).fit(Xfit[:, mask], yfit)
    return knn.score(Xeval[:, mask], yeval)


def cv_fitness(X, y, folds):
    """Fitness is the mean accuracy over fixed CV folds, not a single split."""
    splits = list(StratifiedKFold(n_splits=folds, shuffle=True,
                                  random_state=0).split(X, y))

    def fn(ind):
        mask = np.array(ind, dtype=bool)
        n = int(mask.sum())
        if n == 0:
            return 0.0, N_FEATURES
        cols = np.where(mask)[0]
        out = []
        for tr, va in splits:
            knn = KNeighborsClassifier(n_neighbors=K)
            knn.fit(X[np.ix_(tr, cols)], y[tr])
            out.append(knn.score(X[np.ix_(va, cols)], y[va]))
        return float(np.mean(out)), n

    return fn


def build(fitness_fn):
    tb = base.Toolbox()
    tb.register("attr_bool", random.randint, 0, 1)
    tb.register("individual", tools.initRepeat, creator.Individual,
                tb.attr_bool, N_FEATURES)
    tb.register("population", tools.initRepeat, list, tb.individual)
    tb.register("evaluate", fitness_fn)
    tb.register("mate", tools.cxUniform, indpb=0.5)
    tb.register("mutate", tools.mutFlipBit, indpb=1.0 / N_FEATURES)
    tb.register("select", tools.selNSGA2)
    return tb


def evolve(toolbox, n_gen, hook=None, tag=""):
    pop = toolbox.population(n=POP_SIZE)
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)
    pop = toolbox.select(pop, POP_SIZE)

    fronts = []
    for gen in range(1, n_gen + 1):
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
        fronts.append([np.array(i, dtype=bool) for i in
                       tools.sortNondominated(pop, len(pop),
                                              first_front_only=True)[0]])
        if hook:
            hook(gen, best, np.array(best, dtype=bool))

        print(tag, "gen", gen, "of", n_gen,
              "| fitness", round(best.fitness.values[0], 4), end="\r")

    print()
    return fronts


def smooth(y, w=WINDOW):
    v = np.convolve(y, np.ones(w) / w, mode="valid")
    off = w // 2
    return v, np.arange(off + 1, off + 1 + len(v))


# ---- find the stopping generation -----------------------------------

print("inner CV:", INNER_FOLDS, "folds,", INNER_GEN, "generations,",
      CV_FOLDS, "fold CV fitness")
print()

skf = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED)
curves = []

for fold, (et_idx, e_idx) in enumerate(skf.split(X_dev_raw, y_dev), 1):
    X_et_raw, y_et = X_dev_raw[et_idx], y_dev[et_idx]
    X_e_raw, y_e = X_dev_raw[e_idx], y_dev[e_idx]

    s = StandardScaler().fit(X_et_raw)
    X_et, X_e = s.transform(X_et_raw), s.transform(X_e_raw)

    print("fold", fold, ": GA sees", len(X_et), "| held out", len(X_e))

    random.seed(SEED + fold)
    np.random.seed(SEED + fold)

    held = []

    def hook(gen, best, mask):
        held.append(knn_acc(mask, X_et, y_et, X_e, y_e))

    evolve(build(cv_fitness(X_et, y_et, CV_FOLDS)), INNER_GEN, hook=hook,
           tag="  fold " + str(fold) + ":")
    curves.append(held)

curves = np.array(curves)
mean_held = curves.mean(axis=0)
sm, sm_x = smooth(mean_held)
stop_gen = int(sm_x[np.argmax(sm)])

print()
print("held out: gen 1", round(mean_held[0], 4),
      "-> gen", INNER_GEN, round(mean_held[-1], 4))
print("stopping generation:", stop_gen)
print()

# ---- final run ------------------------------------------------------

print("final run with CV fitness on all", len(X_dev), "development images")
print()

random.seed(SEED)
np.random.seed(SEED)

fronts = evolve(build(cv_fitness(X_dev, y_dev, CV_FOLDS)), N_GEN, tag=" final:")
front = fronts[min(stop_gen, N_GEN) - 1]

fit_fn = cv_fitness(X_dev, y_dev, CV_FOLDS)

rows, seen = [], set()
for mask in front:
    n = int(mask.sum())
    if n == 0 or n in seen:
        continue
    seen.add(n)
    rows.append((n,
                 fit_fn(mask.astype(int).tolist())[0],
                 knn_acc(mask, X_dev, y_dev, X_te, y_te),
                 mask))
rows.sort(key=lambda r: r[0])

everything = np.ones(N_FEATURES, dtype=bool)
base_cv = fit_fn(everything.astype(int).tolist())[0]
base_test = knn_acc(everything, X_dev, y_dev, X_te, y_te)

print()
print("baseline, 512 features:  CV", round(base_cv, 4),
      " test", round(base_test, 4),
      " gap", round(base_cv - base_test, 4))
print()
print("front at generation", stop_gen)
print("  feats       CV     test      gap   vs baseline")
for n, cv, te, _ in rows:
    print("  {:5d}   {:.4f}   {:.4f}   {:+.4f}   {:+.4f}".format(
        n, cv, te, cv - te, te - base_test))

gaps = np.array([cv - te for _, cv, te, _ in rows])
tests = np.array([te for _, _, te, _ in rows])
ns = np.array([n for n, _, _, _ in rows])

print()
print("mean gap across the front:", round(gaps.mean(), 4))

ok = tests >= base_test
if ok.any():
    smallest = int(ns[ok].min())
    i = int(np.where(ns == smallest)[0][0])
    print()
    print("smallest subset matching or beating the 512 baseline:")
    print("  ", smallest, "features, test", round(rows[i][2], 4), "-",
          round(100 * smallest / N_FEATURES, 1), "% of the features")
else:
    print()
    print("no solution on this front matches the 512 baseline on test")

np.savez("combined.npz", stop_gen=stop_gen, curves=curves,
         n_features=ns, cv_acc=np.array([cv for _, cv, _, _ in rows]),
         test_acc=tests, masks=np.array([m for _, _, _, m in rows]))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

ax = axes[0]
g = np.arange(1, INNER_GEN + 1)
for i, c in enumerate(curves, 1):
    ax.plot(g, c, lw=0.9, alpha=0.4, label="fold " + str(i))
ax.plot(g, mean_held, color="#c53030", lw=1.2, alpha=0.6, label="mean")
ax.plot(sm_x, sm, color="#c53030", lw=2.0, ls="--", label="smoothed")
ax.axvline(stop_gen, color="gray", ls=":", lw=1.2,
           label="stop at gen " + str(stop_gen))
ax.set_xlabel("generation")
ax.set_ylabel("held-out accuracy")
ax.set_title("Choosing where to stop")
ax.grid(alpha=0.25)
ax.legend(frameon=False, fontsize=8)

ax = axes[1]
ax.plot(ns, [cv for _, cv, _, _ in rows], "o-", color="#2b6cb0", lw=1.4, ms=5,
        label="CV accuracy (the fitness)")
ax.plot(ns, tests, "s--", color="#c53030", lw=1.4, ms=5, label="test accuracy")
ax.axhline(base_test, color="gray", ls=":", lw=1.2,
           label="baseline, 512 features (" + str(round(base_test, 3)) + ")")
ax.set_xlabel("number of selected features")
ax.set_ylabel("accuracy")
ax.set_title("Trade-off at generation " + str(stop_gen))
ax.grid(alpha=0.25)
ax.legend(frameon=False, fontsize=8)

fig.tight_layout()
fig.savefig("combined.png", dpi=160)
print()
print("saved combined.png")
