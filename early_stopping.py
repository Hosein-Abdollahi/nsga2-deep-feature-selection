import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from deap import base, creator, tools
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split

SEED = 42
K = 5
N_FEATURES = 512
POP_SIZE = 100
N_GEN = 150
INNER_GEN = 150
N_FOLDS = 3
CX_PROB = 0.9
MUT_PROB = 0.1
WINDOW = 9

d = np.load("features.npz")
X_tr_raw, y_tr = d["X_tr"], d["y_tr"]
X_va_raw, y_va = d["X_va"], d["y_va"]
X_te_raw, y_te = d["X_te"], d["y_te"]

sc = StandardScaler().fit(X_tr_raw)
X_tr = sc.transform(X_tr_raw)
X_va = sc.transform(X_va_raw)
X_te = sc.transform(X_te_raw)

# training and validation together, used only to pick the stopping point
X_dev_raw = np.vstack([X_tr_raw, X_va_raw])
y_dev = np.concatenate([y_tr, y_va])

creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)


def knn_acc(mask, Xfit, yfit, Xeval, yeval):
    knn = KNeighborsClassifier(n_neighbors=K)
    knn.fit(Xfit[:, mask], yfit)
    return knn.score(Xeval[:, mask], yeval)


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

    masks = []
    for gen in range(1, n_gen + 1):
        kids = [toolbox.clone(p) for p in tools.selTournamentDCD(pop, POP_SIZE)]

        for a, b in zip(kids[::2], kids[1::2]):
            if random.random() < CX_PROB:
                toolbox.mate(a, b)
                del a.fitness.values
                del b.fitness.values
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
        m = np.array(best, dtype=bool)
        masks.append(m)

        if hook is not None:
            hook(gen, best, m)

        print(tag, "gen", gen, "of", n_gen,
              "| fitness", round(best.fitness.values[0], 4), end="\r")

    print()
    return masks


def smooth(y, w=WINDOW):
    v = np.convolve(y, np.ones(w) / w, mode="valid")
    off = w // 2
    return v, np.arange(off + 1, off + 1 + len(v))


# --------------------------------------------------------------------
# inner cross validation
#
# three levels: the classifier is fitted on one part, the fitness measured
# on a second, and the stopping point tracked on a third the search never
# sees. If the second and third are the same data the tracked curve is just
# the fitness again, and it can never turn over.
# --------------------------------------------------------------------

print("inner CV:", N_FOLDS, "folds,", INNER_GEN, "generations each")
print()

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
curves, fitness_curves = [], []

for fold, (et_idx, e_idx) in enumerate(skf.split(X_dev_raw, y_dev), 1):
    X_et, y_et = X_dev_raw[et_idx], y_dev[et_idx]
    X_e, y_e = X_dev_raw[e_idx], y_dev[e_idx]

    X_fit_raw, X_val_raw, y_fit, y_val = train_test_split(
        X_et, y_et, test_size=0.30, stratify=y_et, random_state=SEED)

    s = StandardScaler().fit(X_fit_raw)
    X_fit = s.transform(X_fit_raw)
    X_val = s.transform(X_val_raw)
    X_hold = s.transform(X_e)

    print("fold", fold, ": fit", len(X_fit), "| fitness", len(X_val),
          "| held out", len(X_hold))

    random.seed(SEED + fold)
    np.random.seed(SEED + fold)

    def fitness_fn(ind):
        mask = np.array(ind, dtype=bool)
        n = int(mask.sum())
        if n == 0:
            return 0.0, N_FEATURES
        return knn_acc(mask, X_fit, y_fit, X_val, y_val), n

    held, fits = [], []

    def hook(gen, best, mask):
        fits.append(best.fitness.values[0])
        held.append(knn_acc(mask, X_fit, y_fit, X_hold, y_e))

    evolve(build(fitness_fn), INNER_GEN, hook=hook,
           tag="  fold " + str(fold) + ":")
    curves.append(held)
    fitness_curves.append(fits)

curves = np.array(curves)
fitness_curves = np.array(fitness_curves)
mean_held = curves.mean(axis=0)
mean_fit = fitness_curves.mean(axis=0)

sm, sm_x = smooth(mean_held)
stop_gen = int(sm_x[np.argmax(sm)])

print()
print("mean fitness   gen 1", round(mean_fit[0], 4),
      "-> gen", INNER_GEN, round(mean_fit[-1], 4))
print("mean held out  gen 1", round(mean_held[0], 4),
      "-> gen", INNER_GEN, round(mean_held[-1], 4))
print("held out peaks at generation", int(np.argmax(mean_held)) + 1)
print("smoothed peak at generation", stop_gen)
print()

# --------------------------------------------------------------------
# apply it
# --------------------------------------------------------------------

print("final run, stopping point chosen without the test set")
print()

random.seed(SEED)
np.random.seed(SEED)


def fitness_final(ind):
    mask = np.array(ind, dtype=bool)
    n = int(mask.sum())
    if n == 0:
        return 0.0, N_FEATURES
    return knn_acc(mask, X_tr, y_tr, X_va, y_va), n


test_track = []


def hook_final(gen, best, mask):
    test_track.append(knn_acc(mask, X_tr, y_tr, X_te, y_te))


masks = evolve(build(fitness_final), N_GEN, hook=hook_final, tag=" final:")
test_track = np.array(test_track)

mask_stop = masks[min(stop_gen, N_GEN) - 1]
mask_full = masks[-1]
everything = np.ones(N_FEATURES, dtype=bool)

rows = [
    ("baseline, all 512", 512, everything),
    ("full run, gen " + str(N_GEN), int(mask_full.sum()), mask_full),
    ("early stop, gen " + str(stop_gen), int(mask_stop.sum()), mask_stop),
]

print()
print("                       feats      val     test      gap")
for name, n, m in rows:
    v = knn_acc(m, X_tr, y_tr, X_va, y_va)
    t = knn_acc(m, X_tr, y_tr, X_te, y_te)
    print("{:<20s} {:5d}   {:.4f}   {:.4f}   {:+.4f}".format(name, n, v, t, v - t))

best_gen = int(np.argmax(test_track)) + 1
t_stop = knn_acc(mask_stop, X_tr, y_tr, X_te, y_te)
t_full = knn_acc(mask_full, X_tr, y_tr, X_te, y_te)

print()
print("best test reachable:", round(test_track.max(), 4),
      "at generation", best_gen, "(not selectable without the test set)")
print("early stopping recovered", round(t_stop - t_full, 4),
      "of", round(test_track.max() - t_full, 4), "available")

np.savez("early_stopping.npz", fold_curves=curves,
         fold_fitness=fitness_curves, stop_gen=stop_gen,
         test_track=test_track, mask_stop=mask_stop, mask_full=mask_full)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

ax = axes[0]
g = np.arange(1, INNER_GEN + 1)
for i, c in enumerate(curves, 1):
    ax.plot(g, c, lw=0.8, alpha=0.35, label="fold " + str(i))
ax.plot(g, mean_held, color="#c53030", lw=1.0, alpha=0.5, label="mean held out")
ax.plot(sm_x, sm, color="#c53030", lw=2.0, ls="--", label="smoothed")
ax.plot(g, mean_fit, color="#2b6cb0", lw=1.6, label="mean fitness")
ax.axvline(stop_gen, color="gray", ls=":", lw=1.2,
           label="stop at gen " + str(stop_gen))
ax.set_xlabel("generation")
ax.set_ylabel("accuracy")
ax.set_title("Inner CV: fitness vs genuinely held out")
ax.grid(alpha=0.25)
ax.legend(frameon=False, fontsize=8)

ax = axes[1]
g = np.arange(1, N_GEN + 1)
sm_t, sm_tx = smooth(test_track)
ax.plot(g, test_track, color="#c53030", lw=0.9, alpha=0.4, label="test")
ax.plot(sm_tx, sm_t, color="#c53030", lw=2.0, ls="--", label="test, smoothed")
ax.axvline(stop_gen, color="gray", ls=":", lw=1.2,
           label="chosen stop, gen " + str(stop_gen))
ax.axvline(best_gen, color="black", ls="-.", lw=1.0,
           label="true peak, gen " + str(best_gen))
ax.set_xlabel("generation")
ax.set_ylabel("test accuracy")
ax.set_title("Where the stopping point landed")
ax.grid(alpha=0.25)
ax.legend(frameon=False, fontsize=8)

fig.tight_layout()
fig.savefig("early_stopping.png", dpi=160)
print()
print("saved early_stopping.png")
