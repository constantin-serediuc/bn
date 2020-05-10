from ga.parameters import SAVE_FILE, CHECKPOINT_FILE
import numpy as np
import datetime
import json

save_file = None
generation_checkpoint_file = None

def init_writer():
    global save_file, generation_checkpoint_file
    save_file = open(SAVE_FILE, 'a')
    generation_checkpoint_file = open(CHECKPOINT_FILE,'w')


def write(fitness, generation):
    global save_file, generation_checkpoint_file
    argmax = np.argsort(fitness)[-1]
    as_array = generation.population_as_array[argmax]
    as_edges = list(generation.population_as_nets[argmax].graph.edges)
    save_file.write(
        f'{str(np.max(fitness))}|{json.dumps(as_array.tolist())}|{json.dumps(as_edges)}|{datetime.datetime.now()}\n'
    )
    save_file.flush()

def checkpoint(generation):
    generation_checkpoint_file.seek(0)
    generation_checkpoint_file.truncate()
    generation_checkpoint_file.write(json.dumps(np.array(generation.population_as_array).tolist()))
    generation_checkpoint_file.flush()
