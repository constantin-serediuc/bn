from datasets.datasets import get_dataset

POPULATION_SIZE=20
MAX_NO_GENERATION = 1400
MAX_NO_GENERATION_SAME_SCORE = 70

DATA = get_dataset()

ELITIST = 2 # POPULATION_SIZE-ELITISM trebuie sa fie numar par pentru a fi mai usor la crossover si a scuti o verificare in plus
CROSSOVER_RATE=0.8
MUTATION_RATE=0.001

LOG_FILE = "log.txt"
SAVE_FILE = "save.txt"
CHECKPOINT_FILE = "checkpoint__perfect_mi_mutate_worst.txt"
INTERMEDIAR_SAVE = 10