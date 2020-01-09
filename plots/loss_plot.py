import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def print_r(i):
    with open('/Users/constantin/Documents/bn/ga/log.txt', 'r') as reader:
        r = reader.readlines()
        r_processed = []
        for i in r:
          try:
            r_processed.append(float(i.strip()))
          except:
            continue

        plt.cla()
        plt.plot(range(len(r_processed)),r_processed)
        plt.tight_layout()


ani = FuncAnimation(plt.gcf(), print_r, interval=1000)
plt.tight_layout()
plt.show()
