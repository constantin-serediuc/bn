import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
def print_r(i):
    with open('../rl/rl_logs_add_remove/solutions/log_max_bic.txt', 'r') as reader:
        r = reader.readlines()
        r_processed = []
        for i in r:
          try:
            r_processed.append(float(i.strip()))
          except:
            continue
        r_processed = r_processed[-8000:]
        plt.cla()
        plt.plot(range(len(r_processed)),r_processed)
        plt.tight_layout()
        #plt.xticks(np.arange(-15, len(r_processed)+21, 10))
        #plt.yticks(np.arange(r_processed[0], r_processed[-1] + 200, 75))
        # plt.hlines(y=-32128.86, xmin=0, xmax=len(r_processed) + 10, linewidth=2, color='r')
        # plt.text(len(r_processed), r_processed[-1], str(r_processed[-1]),horizontalalignment="right")
        i=0
        for x, y in zip(range(len(r_processed)), r_processed):
            label = "{:.2f}".format(y)
            if i % 30 != 0:
                i += 1
                continue
            i+=1

            plt.annotate(label,  # this is the text
                         (x, y),  # this is the point to label
                         textcoords="offset points",  # how to position the text
                         xytext=(0, 10),  # distance from text to points (x,y)
                         ha='right')  # ho
        plt.annotate("{:.2f}".format(r_processed[-1]),  # this is the text
                     (len(r_processed), r_processed[-1]),  # this is the point to label
                     textcoords="offset points",  # how to position the text
                     xytext=(0, 10),  # distance from text to points (x,y)
                     ha='left')  # ho

ani = FuncAnimation(plt.gcf(), print_r, interval=1000)
plt.tight_layout()
plt.show()
