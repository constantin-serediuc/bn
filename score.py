import pandas as pd
import numpy as np
import math

from ga.parameters import DATA


def entropy(node, parents, data):
    df = data[[node] + parents]
    n = data.shape[0]

    if len(parents) == 0:
        count_by_family = df.groupby([node]).size().reset_index(name='count')
        p_joint = np.divide(count_by_family['count'].to_numpy(), n)

        return round(n*np.sum(np.multiply(p_joint, np.log2(p_joint))) - math.log2(n)/2,4)

    count_by_family = df.groupby([node] + parents).size().reset_index(name='count_by_family')
    count_by_parents = df.groupby(parents).size().reset_index(name='count_by_parents')
    counts_df = pd.merge(count_by_family, count_by_parents, how='left', left_on=parents, right_on=parents)

    joint_count = counts_df['count_by_family'].to_numpy()
    parents_count = counts_df['count_by_parents'].to_numpy()

    p_joint = np.divide(joint_count, n)
    log_p_joint = np.log2(np.divide(parents_count, joint_count))

    dimension = count_by_parents.shape[0]*max(data[node].nunique()-1,1) * math.log2(n) /2
    return round(-n*np.sum(np.multiply(p_joint, log_p_joint))-dimension, 4)


# l = [
#     (True, True, False, True),
#     (True, True, False, False),
#     (True, False, True, True),
#     (False, True, True, False),
#     (True, True, False, False)
# ]
#
# df = pd.DataFrame(l, columns=['A', 'B', 'C', 'D'])
# print(entropy('dysp', ['smoke'], DATA))
