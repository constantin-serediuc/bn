from ga.parameters import DATA
import numpy as np
import pandas as pd
import random

def order_based_on_mi_sum_with_chosen_var_elimination(data):
    index_to_feature = {i: feature for (i, feature) in list(zip(range(len(data.columns)), data.columns))}
    get_ce_matrix(data, index_to_feature)
    mi = get_mi_matrix(data, index_to_feature)
    order = []
    for i in range(mi.shape[0]):
        sum_mi = np.sum(mi, axis=1)
        node_index = list(filter(lambda x: x not in order, np.argwhere(sum_mi == np.amin(sum_mi)).flatten()))[0]
        order.append(node_index)
        mi[node_index] = 2
        mi[:, node_index] = 2
    # return {i: data.columns[order[i]] for i in range(len(order))}
    return [data.columns[order[i]] for i in range(len(order))]


def order_based_on_mi_sum_without_eliminating_chosen_vars(data):
    index_to_feature = {i: feature for (i, feature) in list(zip(range(len(data.columns)), data.columns))}

    mi = get_mi_matrix(data, index_to_feature)
    order = np.argsort(pd.DataFrame(mi).sum().to_numpy())

    # return {i: data.columns[order[i]] for i in range(len(order))}
    return [data.columns[order[i]] for i in range(len(order))]


def order_based_on_median(data):
    index_to_feature = {i: feature for (i, feature) in list(zip(range(len(data.columns)), data.columns))}
    mi = get_mi_matrix(data, index_to_feature)
    index_ordering = np.argsort(pd.DataFrame(mi).median().to_numpy())
    return {i: data.columns[index_ordering[i]] for i in range(len(index_ordering))}
    # return [data.columns[index_ordering[i]] for i in range(len(index_ordering))]

def order_random(data):
    node_order = data.columns.to_list()
    random.shuffle(node_order)
    return node_order

def order_based_on_conditional_entropy_sum_on_line(data):
    '''sum(H(X|A))'''
    index_to_feature = {i: feature for (i, feature) in list(zip(range(len(data.columns)), data.columns))}
    ce = get_ce_matrix(data, index_to_feature)
    node_ordering =  ce.sum(axis='index').sort_values().index.to_list()
    node_ordering.reverse()
    # return node_ordering
    return {i:node_ordering[i] for i in range(len(node_ordering))}


def order_based_on_conditional_entropy_sum_on_columns(data):
    '''sum(H(A|X))'''
    index_to_feature = {i: feature for (i, feature) in list(zip(range(len(data.columns)), data.columns))}
    ce = get_ce_matrix(data, index_to_feature)

    return ce.sum(axis='columns').sort_values().index.to_list()


def order_based_by_intrinsic_entropy(data):
    """
    individual entropy: cata entropie e 'independenta' raportat la cata entropie ramane daca se stie variabila
    daca avem aceiasi entropie independenta intre A si B dar exista mai multa entropie in sistem stiind A decat
    stiind B raportul va da A < B ceea ce inseamna ca B ar fi mai inspre mijlocul retelei
    """
    index_to_feature = {i: feature for (i, feature) in list(zip(range(len(data.columns)), data.columns))}

    independent_entropy = get_independent_entropy(data, index_to_feature)
    ce = get_ce_matrix(data, index_to_feature)
    entropy_given_x = ce.sum(axis='index') # am obtinut rezultate mai proasta facand independent_entropy.iloc[0]/entropy_x
    return (independent_entropy.iloc[0]).sort_values().index.to_list()
    # asc_order = (independent_entropy.iloc[0]).sort_values().index.to_list()
    # third = len(asc_order)//3
    # return asc_order[third:2*third] + asc_order[:third] + asc_order[2*third:]



def order(data):
    return order_based_on_median(data)


def get_mi_matrix(data, index_to_feature):
    individual_probs = get_individual_probs(data)
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


def get_individual_probs(data):
    individual_probs = {}
    for i in data.columns:
        individual_probs[i] = data.groupby([i]).size().reset_index(name='count' + i)
    return individual_probs


def get_ce_matrix(data, index_to_feature):
    individual_probs = get_individual_probs(data)
    ce = np.zeros((len(index_to_feature), len(index_to_feature)))
    n = data.shape[0]
    no_columns = len(data.columns)
    for i in range(no_columns):
        for j in range(no_columns):
            if i == j:
                continue
            i_label = index_to_feature[i]
            j_label = index_to_feature[j]

            probs = data.groupby([i_label, j_label]).size().reset_index(name='marginal')
            probs = pd.merge(probs, individual_probs[i_label], left_on=i_label, right_on=i_label)
            probs = pd.merge(probs, individual_probs[j_label], left_on=j_label, right_on=j_label)

            p_xy = probs['marginal'].to_numpy() / n
            p_x = probs['count' + j_label].to_numpy() / n
            # H(Y|X)= - sum(p(x,y)log2(p(x,y)/p(x))
            # -> pe linii e H(A|X) pe coloana e H(X|A)
            ce[i][j] = -1 * np.sum(np.multiply(p_xy, np.log2(p_xy / p_x)))

    return pd.DataFrame(ce).rename(index_to_feature).rename(index_to_feature, axis=1)


def get_independent_entropy(data, index_to_feature):
    """ entropy of X - sum of mutual information MI(X,other) .
     entropy that cannot disappear if other variables are known
     """
    n = data.shape[0]
    individual_probs = get_individual_probs(data)
    entropy = {}
    for i in range(len(data.columns)):
        i_label = index_to_feature[i]
        p_x = individual_probs[i_label]['count' + i_label].to_numpy() / n
        # H(X)=-sum(p(x)*log2(p(x))
        entropy[i_label] = -1 * np.sum(np.multiply(p_x, np.log2(p_x)))

    mi = get_mi_matrix(data, index_to_feature)
    mi_sum = pd.DataFrame(mi).rename(index_to_feature).rename(index_to_feature, axis=1).sum(axis='index')
    result = {}
    for k in entropy.keys():
        result[k] = [entropy[k] - mi_sum[k]]

    return pd.DataFrame.from_dict(result)

# order(DATA)

def perfect_order_asia():
    reference_possible_parents = {'asia': [], 'smoke': [], "tub": ['asia'], "lung": ["smoke"], 'bronc': ["smoke"],
                              "either": ['tub', 'lung'], "xray": ["either"], "dysp": ["either", "bronc"]}
    return order_perfect(reference_possible_parents)


def order_perfect(reference_possible_parents):
    order_best = []
    for i in reference_possible_parents:
        for j in reference_possible_parents[i]:
            if j not in order_best:
                order_best.append(j)
    for i in reference_possible_parents.keys():
        if i not in order_best:
            order_best.append(i)
    return order_best


def perfect_order_alarm():
    reference_possible_parents = {'LVFAILURE': [], 'HISTORY': ['LVFAILURE'], 'LVEDVOLUME': ['HYPOVOLEMIA', 'LVFAILURE'], 'CVP': ['LVEDVOLUME'], 'PCWP': ['LVEDVOLUME'], 'HYPOVOLEMIA': [], 'STROKEVOLUME': ['HYPOVOLEMIA', 'LVFAILURE'], 'ERRLOWOUTPUT': [], 'HRBP': ['ERRLOWOUTPUT', 'HR'], 'HR': ['CATECHOL'], 'ERRCAUTER': [], 'HREKG': ['ERRCAUTER', 'HR'], 'HRSAT': ['ERRCAUTER', 'HR'], 'ANAPHYLAXIS': [], 'TPR': ['ANAPHYLAXIS'], 'ARTCO2': ['VENTALV'], 'EXPCO2': ['ARTCO2', 'VENTLUNG'], 'VENTLUNG': ['INTUBATION', 'KINKEDTUBE', 'VENTTUBE'], 'INTUBATION': [], 'MINVOL': ['INTUBATION', 'VENTLUNG'], 'FIO2': [], 'PVSAT': ['FIO2', 'VENTALV'], 'VENTALV': ['INTUBATION', 'VENTLUNG'], 'SAO2': ['PVSAT', 'SHUNT'], 'SHUNT': ['INTUBATION', 'PULMEMBOLUS'], 'PULMEMBOLUS': [], 'PAP': ['PULMEMBOLUS'], 'PRESS': ['INTUBATION', 'KINKEDTUBE', 'VENTTUBE'], 'KINKEDTUBE': [], 'VENTTUBE': ['DISCONNECT', 'VENTMACH'], 'MINVOLSET': [], 'VENTMACH': ['MINVOLSET'], 'DISCONNECT': [], 'CATECHOL': ['ARTCO2', 'INSUFFANESTH', 'SAO2', 'TPR'], 'INSUFFANESTH': [], 'CO': ['HR', 'STROKEVOLUME'], 'BP': ['CO', 'TPR']}
    order_asia_as_list = order_perfect(reference_possible_parents)
    return {i:order_asia_as_list[i] for i in range(len(order_asia_as_list))}


def perfect_order_hepar():
    reference_possible_parents = {'hepatotoxic': [], 'THepatitis': ['hepatotoxic', 'alcoholism'], 'alcoholism': [], 'gallstones': [], 'choledocholithotomy': ['gallstones'], 'hospital': [], 'injections': ['hospital', 'surgery', 'choledocholithotomy'], 'surgery': [], 'transfusion': ['hospital', 'surgery', 'choledocholithotomy'], 'ChHepatitis': ['transfusion', 'vh_amn', 'injections'], 'vh_amn': [], 'sex': [], 'PBC': ['sex', 'age'], 'age': [], 'fibrosis': ['ChHepatitis'], 'diabetes': [], 'obesity': ['diabetes'], 'Steatosis': ['obesity', 'alcoholism'], 'Cirrhosis': ['fibrosis', 'Steatosis'], 'Hyperbilirubinemia': ['age', 'sex'], 'triglycerides': ['Steatosis'], 'RHepatitis': ['hepatotoxic'], 'fatigue': ['ChHepatitis', 'THepatitis', 'RHepatitis'], 'bilirubin': ['Hyperbilirubinemia', 'PBC', 'Cirrhosis', 'gallstones', 'ChHepatitis'], 'itching': ['bilirubin'], 'upper_pain': ['gallstones'], 'fat': ['gallstones'], 'pain_ruq': ['Steatosis', 'Hyperbilirubinemia'], 'pressure_ruq': ['gallstones', 'PBC', 'ChHepatitis'], 'phosphatase': ['RHepatitis', 'THepatitis', 'Cirrhosis', 'ChHepatitis'], 'skin': ['bilirubin'], 'ama': ['PBC'], 'le_cells': ['PBC'], 'joints': ['PBC'], 'pain': ['PBC', 'joints'], 'proteins': ['Cirrhosis'], 'edema': ['Cirrhosis'], 'platelet': ['Cirrhosis', 'PBC'], 'inr': ['ChHepatitis', 'Cirrhosis', 'THepatitis', 'Hyperbilirubinemia'], 'bleeding': ['platelet', 'inr'], 'flatulence': ['gallstones'], 'alcohol': ['Cirrhosis'], 'encephalopathy': ['Cirrhosis', 'PBC'], 'urea': ['encephalopathy'], 'ascites': ['proteins'], 'hepatomegaly': ['RHepatitis', 'THepatitis', 'Steatosis', 'Hyperbilirubinemia'], 'hepatalgia': ['hepatomegaly'], 'density': ['encephalopathy'], 'ESR': ['PBC', 'ChHepatitis', 'Steatosis', 'Hyperbilirubinemia'], 'alt': ['ChHepatitis', 'RHepatitis', 'THepatitis', 'Steatosis', 'Cirrhosis'], 'ast': ['ChHepatitis', 'RHepatitis', 'THepatitis', 'Steatosis', 'Cirrhosis'], 'amylase': ['gallstones'], 'ggtp': ['PBC', 'THepatitis', 'RHepatitis', 'Steatosis', 'ChHepatitis', 'Hyperbilirubinemia'], 'cholesterol': ['PBC', 'Steatosis', 'ChHepatitis'], 'hbsag': ['vh_amn', 'ChHepatitis'], 'hbsag_anti': ['vh_amn', 'ChHepatitis', 'hbsag'], 'anorexia': ['RHepatitis', 'THepatitis'], 'nausea': ['RHepatitis', 'THepatitis'], 'spleen': ['Cirrhosis', 'RHepatitis', 'THepatitis'], 'consciousness': ['encephalopathy'], 'spiders': ['Cirrhosis'], 'jaundice': ['bilirubin'], 'albumin': ['Cirrhosis'], 'edge': ['Cirrhosis'], 'irregular_liver': ['Cirrhosis'], 'hbc_anti': ['vh_amn', 'ChHepatitis'], 'hcv_anti': ['vh_amn', 'ChHepatitis'], 'palms': ['Cirrhosis'], 'hbeag': ['vh_amn', 'ChHepatitis'], 'carcinoma': ['Cirrhosis', 'PBC']}

    return order_perfect(reference_possible_parents)
