from ga.crossover import crossover
from ga.mutation import mutate
from ga.parameters import *
from datasets.asia_test import get_dataset
from ga.fitness import get_fitness
import numpy as np
import copy
import datetime

from ga.population import Population
from ga.select import select
from ga.solution_writer import init_writer, write


def main(dataset):
    Population.init_state(POPULATION_SIZE, dataset.columns)
    generation = Population()
    generation.random()
    print('Population shape:', generation.shape)

    no_generation = 0
    log = open(LOG_FILE, 'a')
    init_writer()
    while no_generation < MAX_NO_GENERATION:
        print('Generation: ', no_generation)

        fitness = np.array(get_fitness(generation.population_as_nets))
        indexes_of_best_fitnesses = np.argsort(fitness)[-ELITIST:]
        new_generation = Population(
            generation.population_as_array[indexes_of_best_fitnesses],
            generation.population_as_nets[indexes_of_best_fitnesses]
        )

        log.write(str(np.max(fitness)) + '\n')
        log.flush()
        print(str(np.max(fitness)) + '\n')

        no_pairs = int((POPULATION_SIZE - ELITIST) / 2)
        for _ in range(no_pairs):
            index_p1, index_p2 = select(fitness)
            offsprings = crossover(generation.get(index_p1), generation.get(index_p2))
            new_generation.add_individs(offsprings)

        for i in range(POPULATION_SIZE):
            new_generation.add_individs([mutate(generation.get(i))])

        generation = copy.deepcopy(new_generation)

        no_generation += 1

        if no_generation % INTERMEDIAR_SAVE == 0:
            fitness = np.array(get_fitness(generation.population_as_nets))
            write(fitness, generation)


main(get_dataset())
