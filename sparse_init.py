import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from deap import base, creator, tools
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# same as nsga2.py except for the initialisation rate
SEED = 42
K = 5
N_FEATURES = 512
POP_SIZE = 100
N_GEN = 50
CX_PROB = 0.9
MUT_PROB = 0.1
INIT_RATE = 0.05

random.seed(SEED)
np.random.seed(SEED)

d = np.load("features.npz")
X_tr, y_tr = d["X_tr"], d["y_tr"]
X_va, y_va = d["X_va"], d["y_va"]
X_te, y_te = d["X_te"], d["y_te"]

sc = StandardScaler().fit(X_tr)
X_tr = sc.transform(X_tr)
X_va = sc.transform(X_va)
X_te = sc.transform(X_te)


def acc(mask, X, y):
    knn = KNeighborsClassifier(n_neighbors=K)
    knn.fit(X_tr[:, mask], y_tr)
    return knn.score(X[:, mask], y)


def evaluate(ind):
    mask = np.array(ind, dtype=bool)
    n = int(mask.sum())
    if n == 0:
        return 0.0, N_FEATURES
    return acc(mask, X_va, y_va), n


def sparse_bit():
    return 1 if random.random() < INIT_RATE else 0


creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("attr_bool", sparse_bit)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_bool, N_FEATURES)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxUniform, indpb=0.5)
toolbox.register("mutate", tools.mutFlipBit, indpb=1.0 / N_FEATURES)
toolbox.register("select", tools.selNSGA2)


def run():
    pop = toolbox.population(n=POP_SIZE)

    sizes = [sum(i) for i in pop]
    print("starting sizes:", min(sizes), "to", max(sizes),
          "| mean", round(np.mean(sizes), 1))

    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)
    pop = toolbox.select(pop, POP_SIZE)

    for gen in range(1, N_GEN + 1):
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
        best = max(i.fitness.values[0] for i in pop)
        print("gen", gen, "of", N_GEN, "| best val acc", round(best, 4), end="\r")

    print()
    return pop


def front_of(pop):
    f = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
    seen = {}
    for ind in f:
        a, n = ind.fitness.values
        key = (int(n), round(a, 6))
        if key not in seen:
            seen[key] = np.array(ind, dtype=bool)
    keys = sorted(seen)
    return (np.array([k[0] for k in keys]),
            np.array([k[1] for k in keys]),
            np.array([seen[k] for k in keys]))


def random_same_size(n, trials=20):
    rng = np.random.RandomState(n)
    out = []
    for _ in range(trials):
        m = np.zeros(N_FEATURES, dtype=bool)
        m[rng.choice(N_FEATURES, n, replace=False)] = True
        out.append(acc(m, X_te, y_te))
    return np.array(out)


if __name__ == "__main__":
    pop = run()
    n_feat, val, masks = front_of(pop)
    test = np.array([acc(m, X_te, y_te) for m in masks])

    print()
    print("Pareto front:", len(n_feat), "solutions")
    print()
    print("feats     val    test   random mean    diff       z")

    zs = []
    for i in range(len(n_feat)):
        r = random_same_size(int(n_feat[i]))
        diff = test[i] - r.mean()
        z = diff / r.std() if r.std() > 0 else 0.0
        zs.append(z)
        print(" {:4d}  {:.4f}  {:.4f}      {:.4f}   {:+.4f}   {:+.2f}".format(
            n_feat[i], val[i], test[i], r.mean(), diff, z))

    zs = np.array(zs)
    print()
    print("mean z:", round(zs.mean(), 2))
    print("solutions with z > 2:", int((zs > 2).sum()), "of", len(zs))

    small = n_feat < 60
    if small.any() and (~small).any():
        print("under 60 features: mean z", round(zs[small].mean(), 2))
        print("60 or more:        mean z", round(zs[~small].mean(), 2))

    np.savez("front_sparse.npz", n_features=n_feat, val_acc=val,
             test_acc=test, masks=masks, z=zs)

    rnd = [random_same_size(int(n)).mean() for n in n_feat]

    plt.figure(figsize=(7.5, 4.8))
    plt.plot(n_feat, val, "o-", color="#2b6cb0", lw=1.4, ms=5,
             label="validation")
    plt.plot(n_feat, test, "s--", color="#c53030", lw=1.4, ms=5, label="test")
    plt.plot(n_feat, rnd, "^:", color="gray", lw=1.2, ms=5,
             label="random subset, same size")
    plt.xlabel("number of selected features")
    plt.ylabel("accuracy")
    plt.title("Sparse initialisation (" + str(INIT_RATE) + " per bit)")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig("pareto_front_sparse.png", dpi=160)

    print()
    print("saved front_sparse.npz and pareto_front_sparse.png")
