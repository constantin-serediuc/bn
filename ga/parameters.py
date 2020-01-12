from datasets.datasets import get_dataset

POPULATION_SIZE=4
MAX_NO_GENERATION = 500
MAX_NO_GENERATION_SAME_SCORE = 20

DATA = get_dataset()

ELITIST = 2 # POPULATION_SIZE-ELITISM trebuie sa fie numar par pentru a fi mai usor la crossover si a scuti o verificare in plus
CROSSOVER_RATE=0.8
MUTATION_RATE=0.2

LOG_FILE = "log.txt"
SAVE_FILE = "save.txt"
INTERMEDIAR_SAVE = 20