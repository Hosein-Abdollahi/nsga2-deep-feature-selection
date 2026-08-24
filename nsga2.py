import random
import numpy as np
from deap import base, creator, tools
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

SEED = 42
K = 5
N_FEATURES = 512
POP_SIZE = 100
N_GEN = 50
CX_PROB = 0.9
MUT_PROB = 0.1

random.seed(SEED)
np.random.seed(SEED)

data = np.load("features.npz")
X_tr, y_tr = data["X_tr"], data["y_tr"]
X_va, y_va = data["X_va"], data["y_va"]

scaler = StandardScaler().fit(X_tr)
X_tr = scaler.transform(X_tr)
X_va = scaler.transform(X_va)


def evaluate(individual):
    mask = np.array(individual, dtype=bool)
    n_selected = int(mask.sum())

    if n_selected == 0:
        return 0.0, N_FEATURES

    knn = KNeighborsClassifier(n_neighbors=K)
    knn.fit(X_tr[:, mask], y_tr)
    accuracy = knn.score(X_va[:, mask], y_va)

    return accuracy, n_selected


creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_bool, N_FEATURES)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxUniform, indpb=0.5)
toolbox.register("mutate", tools.mutFlipBit, indpb=1.0 / N_FEATURES)
toolbox.register("select", tools.selNSGA2)


def run():
    population = toolbox.population(n=POP_SIZE)

    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)

    population = toolbox.select(population, POP_SIZE)

    for gen in range(1, N_GEN + 1):
        parents = tools.selTournamentDCD(population, POP_SIZE)
        offspring = [toolbox.clone(p) for p in parents]

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CX_PROB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

            if random.random() < MUT_PROB:
                toolbox.mutate(child1)
                del child1.fitness.values

            if random.random() < MUT_PROB:
                toolbox.mutate(child2)
                del child2.fitness.values

        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind)

        population = toolbox.select(population + offspring, POP_SIZE)

        best = max(ind.fitness.values[0] for ind in population)
        print("gen", gen, "of", N_GEN, "| best val acc", round(best, 4), end="\r")

    print()
    return population


def get_front(population):
    fronts = tools.sortNondominated(population, len(population),
                                    first_front_only=True)
    front = fronts[0]

    solutions = {}
    for ind in front:
        acc, n = ind.fitness.values
        key = (int(n), round(acc, 6))
        if key not in solutions:
            solutions[key] = np.array(ind, dtype=bool)

    keys = sorted(solutions.keys())
    n_features = np.array([k[0] for k in keys])
    val_acc = np.array([k[1] for k in keys])
    masks = np.array([solutions[k] for k in keys])

    return n_features, val_acc, masks


if __name__ == "__main__":
    pop = run()
    n_features, val_acc, masks = get_front(pop)

    print()
    print("Pareto front:", len(n_features), "solutions")
    for n, acc in zip(n_features, val_acc):
        print("  ", n, "features -> val acc", round(acc, 4))

    np.savez("front.npz", n_features=n_features, val_acc=val_acc, masks=masks)
    print()
    print("saved front.npz")
