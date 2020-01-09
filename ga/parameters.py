from datasets.asia_test import get_dataset

POPULATION_SIZE=30
MAX_NO_GENERATION = 100

DATA = get_dataset()

ELITIST = 4 # POPULATION_SIZE-ELITISM trebuie sa fie numar par pentru a fi mai usor la crossover si a scuti o verificare in plus
CROSSOVER_RATE=0.9
MUTATION_RATE=0.1

LOG_FILE = "log.txt"
SAVE_FILE = "save.txt"
INTERMEDIAR_SAVE = 5