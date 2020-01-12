import os
import sys
sys.path.insert(0,
                os.path.abspath(__file__).rsplit(os.sep, 2)[0])
from ga.crossover import crossover
from ga.mutation import mutate
from ga.node_ordering import order
from ga.parameters import *
from datasets.datasets import get_dataset
from ga.fitness import get_fitness
import numpy as np
import copy
import time
from ga.population import Population
from ga.select import select
from ga.solution_writer import init_writer, write


def main(dataset):
    # node_ordering = order(dataset) #un dictionar de forma{i:nume variabila}. variabila[i] poate fi parinte pentru orice j>i
    # node_ordering ={0: 'asia', 1: 'tub', 2: 'smoke', 3: 'bronc', 4: 'lung', 5: 'either', 6: 'dysp', 7: 'xray'} #perfect node ordering
    node_ordering = {0: 'diabetes', 1: 'flatulence', 2: 'upper_pain', 3: 'hepatotoxic', 4: 'anorexia', 5: 'nausea', 6: 'hepatomegaly', 7: 'hepatalgia', 8: 'bleeding', 9: 'vh_amn', 10: 'hospital', 11: 'fat', 12: 'transfusion', 13: 'joints', 14: 'proteins', 15: 'hbsag', 16: 'density', 17: 'le_cells', 18: 'ascites', 19: 'skin', 20: 'alcoholism', 21: 'surgery', 22: 'spiders', 23: 'hcv_anti', 24: 'RHepatitis', 25: 'hbsag_anti', 26: 'choledocholithotomy', 27: 'fatigue', 28: 'hbc_anti', 29: 'THepatitis', 30: 'palms', 31: 'edema', 32: 'pain', 33: 'consciousness', 34: 'pain_ruq', 35: 'obesity', 36: 'alcohol', 37: 'edge', 38: 'jaundice', 39: 'itching', 40: 'gallstones', 41: 'injections', 42: 'spleen', 43: 'sex', 44: 'pressure_ruq', 45: 'Hyperbilirubinemia', 46: 'irregular_liver', 47: 'hbeag', 48: 'encephalopathy', 49: 'ama', 50: 'fibrosis', 51: 'amylase', 52: 'PBC', 53: 'Steatosis', 54: 'urea', 55: 'alt', 56: 'ESR', 57: 'triglycerides', 58: 'inr', 59: 'albumin', 60: 'ChHepatitis', 61: 'phosphatase', 62: 'cholesterol', 63: 'ast', 64: 'carcinoma', 65: 'platelet', 66: 'bilirubin', 67: 'Cirrhosis', 68: 'age', 69: 'ggtp'}
    Population.init_state(POPULATION_SIZE, dataset.columns, node_ordering)
    generation = Population()
    generation.random()
    print('Population shape:', generation.shape)

    no_generation = 0
    log = open(LOG_FILE, 'a')
    init_writer()
    no_same_score = 0
    old_fitness = 0

    while no_generation < MAX_NO_GENERATION and no_same_score < MAX_NO_GENERATION_SAME_SCORE:
        print('Generation: ', no_generation)
        t=time.time()
        fitness = np.array(get_fitness(generation.population_as_nets))
        print("               after fitness computation", time.time() - t)

        indexes_of_best_fitnesses = np.argsort(fitness)[-ELITIST:]
        new_generation = Population(
            generation.population_as_array[indexes_of_best_fitnesses],
            generation.population_as_nets[indexes_of_best_fitnesses]
        )
        print("               after elitism")

        log.write(str(np.max(fitness)) + '\n')
        log.flush()
        print(str(np.max(fitness)) + '\n')

        if np.max(fitness) == old_fitness:
            no_same_score+=1
        else:
            no_same_score = 0
        old_fitness = np.max(fitness)

        no_pairs = int((POPULATION_SIZE - ELITIST) / 2)
        for _ in range(no_pairs):
            index_p1, index_p2 = select(fitness)
            offsprings = crossover(generation.get(index_p1), generation.get(index_p2))
            new_generation.add_individs(offsprings)
        print("               after crossover")
        for i in range(POPULATION_SIZE):
            new_generation.add_individs([mutate(generation.get(i))])
        print("               after mutation")

        generation = copy.deepcopy(new_generation)

        no_generation += 1

        if no_generation % INTERMEDIAR_SAVE == 0:
            fitness = np.array(get_fitness(generation.population_as_nets))
            write(fitness, generation)
        print("              end generation")



main(get_dataset())
