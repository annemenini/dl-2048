# from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing.pool import ThreadPool

import matplotlib.pyplot as plt
import numpy as np

from Grid import Grid


NUM_THREAD = 4


def initialize(pop_size, elem_size, mini, maxi):
    pool = ThreadPool(NUM_THREAD)
    pop = pool.map(lambda dummy: mini + (maxi - mini) * np.random.rand(elem_size), range(0, pop_size))
    return pop


def cross(elem0, elem1, rate=0.5):
    choice = np.random.rand(elem0.size)
    new_elem = np.array([(elem0[i] if choice[i] < rate else elem1[i]) for i in range(0, elem0.size)])
    return new_elem


def mutate(elem, mini, maxi, rate):
    choice = np.random.rand(elem.size)
    new_elem = np.array([(elem[i] if choice[i] < rate else np.random.uniform(mini[i], maxi[i]))
                         for i in range(0, elem.size)])
    return new_elem


def generate_elem(i, pop, breeding_rate, fitness, mutation_rate, mini, maxi, gene_mutation_rate):
    elem = pop[i]
    partner_choice = np.random.randint(0, len(pop), breeding_rate)
    mutation_choice = np.random.rand(breeding_rate)
    new_pop = []
    for breeding in range(0, breeding_rate):
        crossing_rate = fitness[i] / fitness[partner_choice[breeding]]
        new_elem = cross(elem, pop[partner_choice[breeding]], crossing_rate)
        if mutation_choice[breeding] < mutation_rate:
            new_elem = mutate(new_elem, mini, maxi, gene_mutation_rate)
            new_pop.append(new_elem)
    return new_pop


def generate(pop, fitness, breeding_rate, mutation_rate, mutation_margin, gene_mutation_rate):
    mini0 = np.min(pop, axis=1)
    maxi0 = np.max(pop, axis=1)
    mini = mini0 - mutation_margin * (maxi0 - mini0)
    maxi = maxi0 + mutation_margin * (maxi0 - mini0)
    pool = ThreadPool(NUM_THREAD)
    new_pop = pool.map(lambda i: generate_elem(i, pop, breeding_rate, fitness, mutation_rate, mini, maxi, gene_mutation_rate),
                       range(0, len(pop)))
    pool.close()
    pool.join()
    new_pop += pop
    return new_pop


def select(pop, fitness_fn, rate):
    pool = ThreadPool(NUM_THREAD)
    fitness = pool.map(fitness_fn, pop)
    index = np.argsort(-np.array(fitness))
    selected_pop = pool.map(lambda i: pop[index[i]], range(0, int(rate * len(pop))))
    selected_fitness = pool.map(lambda i: fitness[index[i]], range(0, int(rate * len(pop))))
    pool.close()
    pool.join()
    return selected_pop, selected_fitness


def evolve(pop, fitness, breeding_rate, mutation_rate, mutation_margin, gene_mutation_rate, fitness_fn):
    pop = generate(pop, fitness, breeding_rate, mutation_rate, mutation_margin, gene_mutation_rate)
    pop, fitness = select(pop, fitness_fn, 1 / (breeding_rate + 1))
    return pop, fitness


def select_best(pop, fitness):
    index = np.argsort(-np.array(fitness))
    best_elem = pop[index[0]]
    best_fitness = fitness[index[0]]
    return best_elem, best_fitness


def show(pop, fitness):
    plt.subplot(2, 1, 1)
    plt.hist(fitness, bins=32, log=True)
    plt.subplot(2, 1, 2)
    pool = ThreadPool(NUM_THREAD)
    gene0 = pool.map(lambda i: pop[i][0], range(0, len(pop)))
    pool.close()
    pool.join()
    plt.hist(gene0, bins=32)
    plt.pause(0.01)


def evolutionary_solver():
    # Initialization
    ns = [4 * 4, 64, 32, 16, 4]  # Network size
    i2 = 0

    delta_weight = 1
    delta_bias = 0.01

    mini = []
    maxi = []

    for i in range(1, len(ns)):
        i0 = i2
        i1 = i0 + ns[i - 1] * ns[i]
        i2 = i1 + ns[i]
        mini += [-delta_weight] * (i1 - i0)
        mini += [-delta_bias] * (i2 - i1)
        maxi += [delta_weight] * (i1 - i0)
        maxi += [delta_bias] * (i2 - i1)

    elem_size = i2
    pop_size = elem_size * 4
    pop = initialize(pop_size, elem_size, np.array(mini), np.array(maxi))
    grid = Grid()
    pool = ThreadPool(NUM_THREAD)
    fitness = pool.map(grid.play_until_end, pop)
    pool.close()
    pool.join()
    show(pop, fitness)

    # Evolve
    num_generation = 32

    for generation in range(0, num_generation):
        grid = Grid()
        pop, fitness = evolve(pop, fitness, 8, 0.05, 0.1, 0.01, lambda weights: grid.play_until_end(weights))
        show(pop, fitness)

    best_elem, best_fitness = select_best(pop, fitness)

    return best_elem, best_fitness


if __name__ == "__main__":
    best_elem, best_fitness = evolutionary_solver()
    print(best_fitness)
