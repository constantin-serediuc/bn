from ga.parameters import SAVE_FILE
import numpy as np
import datetime
import json

save_file = None


def init_writer():
    global save_file
    save_file = open(SAVE_FILE, 'a')


def write(fitness, generation):
    global save_file
    argmax = np.argsort(fitness)[-1]
    as_array = generation.population_as_array[argmax]
    as_edges = list(generation.population_as_nets[argmax].graph.edges)
    save_file.write(
        f'{str(np.max(fitness))}|{as_array}|{json.dumps(as_edges)}|{datetime.datetime.now()}\n'
    )
