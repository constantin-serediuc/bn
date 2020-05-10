import numpy as np
import matplotlib.pyplot as plt

def get_loss_data(file):
    with open(f'../poze/{file}', 'r') as reader:
        r = reader.readlines()
        r_processed = []
        for i in r:
          try:
            r_processed.append(float(i.strip()))
          except:
            continue
    return r_processed
plt.rcParams.update({'font.size': 16})

y_1 = get_loss_data('alarm_perfect/log.txt')
y_2 = get_loss_data('alarm_perfect_mutate_plus/log.txt')
y_3 = get_loss_data('alarm_median_order/log.txt')
y_4 = get_loss_data('alarm_median_mutation_plus/log.txt')
# x = range(max([len(y_1),len(y_2)]))

# fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
# fig.suptitle('Horizontally stacked subplots')
plt.plot(range(len(y_1)), y_1,label='perfect ordering conf1', linewidth=2)
plt.plot(range(len(y_2)), y_2,label='perfect ordering conf2',linewidth=2)
plt.plot(range(len(y_3)), y_3,label='mi_median ordering conf1',linewidth=2)
plt.plot(range(len(y_4)), y_4,label='mi_median ordering conf2',linewidth=2)
plt.legend()
plt.title('Loss function dynamics for genetic algorithm run on alarm dataset')
plt.xlabel('number of generations')
plt.ylabel('generation\'s best fitness')
plt.show()