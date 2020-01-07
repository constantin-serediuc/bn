from ga.parameters import *
from datasets.asia_test import get_dataset
import ga.mapper as mapper
from ga.fitness import get_fitness
import numpy as np


def main(dataset):
    population_shape = (
        POPULATION_SIZE,
        mapper.get_array_representation_len(dataset.shape[1])
    )

    index_to_feature = {i:feature for (i,feature) in list(zip(range(dataset.shape[1]),dataset.columns))}

    population_as_array = np.random.randint(2, size=population_shape)
    population_as_nets = mapper.as_nets(population_as_array, index_to_feature)
    print('Population shape:',population_as_array.shape)

    no_generation = 0
    while no_generation < MAX_NO_GENERATION:
        print('Generation: ', no_generation)
        fitness = get_fitness(population_as_nets)
        new_generation = []



main(get_dataset())
