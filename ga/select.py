import numpy as np


def roulette_wheel(fitness):
    distribution = np.true_divide(fitness,np.sum(fitness))
    i1 = np.random.choice(len(fitness), p=distribution)

    fitness_2 = np.copy(fitness) #prevent selection same parent
    fitness_2[i1] = 0
    distribution = np.true_divide(fitness_2,np.sum(fitness_2))

    return i1, np.random.choice(len(fitness_2), p=distribution)




select_method = roulette_wheel


def select(fitness):
    return select_method(fitness)

# for i in range(10):
#     print(roulette_wheel(np.array([10,4])))