from datasets.asia_test import get_dataset

POPULATION_SIZE=10
MAX_NO_GENERATION = 10

DATA = get_dataset()

ELITIST = 2 # POPULATION_SIZE-ELITISM trebuie sa fie numar par pentru a fi mai usor la crossover si a scuti o verificare in plus
CROSSOVER_RATE=0.8
MUTATION_RATE=0.08