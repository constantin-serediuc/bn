from ga.parameters import DATA
import numpy as np
import pandas as pd


def order_based_on_mi_sum(data):
    index_to_feature = {i: feature for (i, feature) in list(zip(range(len(data.columns)), data.columns))}

    mi = get_mi_matrix(data, index_to_feature)
    order = []
    for i in range(mi.shape[0]):
        sum_mi = np.sum(mi, axis=1)
        node_index = list(filter(lambda x: x not in order, np.argwhere(sum_mi == np.amax(sum_mi)).flatten()))[0]
        order.append(node_index)
        mi[node_index] = 0
        mi[:,node_index] = 0
        t = 0
    a=0

def order(data):
    return order_based_on_median(data)


def order_based_on_median(data):
    index_to_feature = {i: feature for (i, feature) in list(zip(range(len(data.columns)), data.columns))}
    mi = get_mi_matrix(data, index_to_feature)
    index_ordering = np.argsort(pd.DataFrame(mi).median().to_numpy())
    return {i: data.columns[index_ordering[i]] for i in range(len(index_ordering))}


def get_mi_matrix(data, index_to_feature):
    individual_probs = {}
    for i in data.columns:
        individual_probs[i] = data.groupby([i]).size().reset_index(name='count' + i)
    mi = np.zeros((len(index_to_feature), len(index_to_feature)))
    n = data.shape[0]
    no_columns = len(data.columns)
    for i in range(no_columns):
        for j in range(i + 1, no_columns):
            if i == j:
                continue
            i_label = index_to_feature[i]
            j_label = index_to_feature[j]

            probs = data.groupby([i_label, j_label]).size().reset_index(name='marginal')
            probs = pd.merge(probs, individual_probs[i_label], left_on=i_label, right_on=i_label)
            probs = pd.merge(probs, individual_probs[j_label], left_on=j_label, right_on=j_label)
            p_xy = probs['marginal'].to_numpy() / n
            p_x = probs['count' + i_label].to_numpy() / n
            p_y = probs['count' + j_label].to_numpy() / n
            mi_ij = np.sum(np.multiply(p_xy, np.log2(p_xy / np.multiply(p_x, p_y))))
            mi[i][j] = mi_ij
            mi[j][i] = mi_ij
    return mi


# order(DATA)
