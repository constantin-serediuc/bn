from net.net import Net

from ga.parameters import DATA


def bic_fitness_based(solution: Net):
    solution.compute_score_per_family(DATA)
    return solution.get_score()


CURRENT_FITNESS_FUNCTION = bic_fitness_based


def get_fitness(population_as_nets):
    global CURRENT_FITNESS_FUNCTION
    fitness = []
    for solution in population_as_nets:
        fitness.append(CURRENT_FITNESS_FUNCTION(solution))
    return fitness
