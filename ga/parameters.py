from datasets.datasets import get_dataset

POPULATION_SIZE=50
MAX_NO_GENERATION = 1000
MAX_NO_GENERATION_SAME_SCORE = 100

DATA = get_dataset()

ELITIST = 4 # POPULATION_SIZE-ELITISM trebuie sa fie numar par pentru a fi mai usor la crossover si a scuti o verificare in plus
CROSSOVER_RATE=0.8
MUTATION_RATE=0.08

LOG_FILE = "log.txt"
SAVE_FILE = "save.txt"
INTERMEDIAR_SAVE = 20