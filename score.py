import pandas as pd
import numpy as np
import math

from ga.parameters import DATA


def bic_score_of_a_family(node, parents, data):
    df = data[[node] + parents]
    n = data.shape[0]

    if len(parents) == 0:
        count_by_family = df.groupby([node]).size().reset_index(name='count')
        p_joint = np.divide(count_by_family['count'].to_numpy(), n)

        return round(n * np.sum(np.multiply(p_joint, np.log2(p_joint))) - math.log2(n) / 2, 4)

    count_by_family = df.groupby([node] + parents).size().reset_index(name='count_by_family')
    count_by_parents = df.groupby(parents).size().reset_index(name='count_by_parents')
    counts_df = pd.merge(count_by_family, count_by_parents, how='left', left_on=parents, right_on=parents)

    joint_count = counts_df['count_by_family'].to_numpy()
    parents_count = counts_df['count_by_parents'].to_numpy()

    p_joint = np.divide(joint_count, n)
    log_p_joint = np.log2(np.divide(parents_count, joint_count))

    dimension = count_by_parents.shape[0] * max(data[node].nunique() - 1, 1) * math.log2(n) / 2
    return round(-n * np.sum(np.multiply(p_joint, log_p_joint)) - dimension, 4)


def mi(a, b, data):
    df = data[[a, b]]
    n = data.shape[0]

    count_b = df.groupby(b).size().reset_index(name='count_b')
    count_a = df.groupby(a).size().reset_index(name='count_a')
    count_ab = df.groupby([a, b]).size().reset_index(name='count_joint')

    counts_df = pd.merge(count_ab, count_a, how='left', left_on=a, right_on=a)
    counts_df = pd.merge(counts_df, count_b, how='left', left_on=b, right_on=b)

    pr_a_b = np.divide(counts_df['count_joint'].to_numpy(), n)
    pr_a = np.divide(counts_df['count_a'].to_numpy(), n)
    pr_b = np.divide(counts_df['count_b'].to_numpy(), n)
    return round(np.sum(
        pr_a_b * np.log2(pr_a_b / (pr_a * pr_b))
    ), 4)

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
