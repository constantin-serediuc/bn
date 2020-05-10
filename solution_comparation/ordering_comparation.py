from datasets.datasets import get_dataset, get_asia_dataset, get_hepar_dataset, get_alarm_dataset
from ga.node_ordering import *

reference_possible_parents = {'asia': [], 'smoke': [], "tub": ['asia'], "lung": ["smoke"], 'bronc': ["smoke"],
                              "either": ['tub', 'lung'], "xray": ["either"], "dysp": ["either", "bronc"]}
# reference_possible_parents = {'LVFAILURE': [], 'HISTORY': ['LVFAILURE'], 'LVEDVOLUME': ['HYPOVOLEMIA', 'LVFAILURE'], 'CVP': ['LVEDVOLUME'], 'PCWP': ['LVEDVOLUME'], 'HYPOVOLEMIA': [], 'STROKEVOLUME': ['HYPOVOLEMIA', 'LVFAILURE'], 'ERRLOWOUTPUT': [], 'HRBP': ['ERRLOWOUTPUT', 'HR'], 'HR': ['CATECHOL'], 'ERRCAUTER': [], 'HREKG': ['ERRCAUTER', 'HR'], 'HRSAT': ['ERRCAUTER', 'HR'], 'ANAPHYLAXIS': [], 'TPR': ['ANAPHYLAXIS'], 'ARTCO2': ['VENTALV'], 'EXPCO2': ['ARTCO2', 'VENTLUNG'], 'VENTLUNG': ['INTUBATION', 'KINKEDTUBE', 'VENTTUBE'], 'INTUBATION': [], 'MINVOL': ['INTUBATION', 'VENTLUNG'], 'FIO2': [], 'PVSAT': ['FIO2', 'VENTALV'], 'VENTALV': ['INTUBATION', 'VENTLUNG'], 'SAO2': ['PVSAT', 'SHUNT'], 'SHUNT': ['INTUBATION', 'PULMEMBOLUS'], 'PULMEMBOLUS': [], 'PAP': ['PULMEMBOLUS'], 'PRESS': ['INTUBATION', 'KINKEDTUBE', 'VENTTUBE'], 'KINKEDTUBE': [], 'VENTTUBE': ['DISCONNECT', 'VENTMACH'], 'MINVOLSET': [], 'VENTMACH': ['MINVOLSET'], 'DISCONNECT': [], 'CATECHOL': ['ARTCO2', 'INSUFFANESTH', 'SAO2', 'TPR'], 'INSUFFANESTH': [], 'CO': ['HR', 'STROKEVOLUME'], 'BP': ['CO', 'TPR']}
# reference_possible_parents = {'hepatotoxic': [], 'THepatitis': ['hepatotoxic', 'alcoholism'], 'alcoholism': [], 'gallstones': [], 'choledocholithotomy': ['gallstones'], 'hospital': [], 'injections': ['hospital', 'surgery', 'choledocholithotomy'], 'surgery': [], 'transfusion': ['hospital', 'surgery', 'choledocholithotomy'], 'ChHepatitis': ['transfusion', 'vh_amn', 'injections'], 'vh_amn': [], 'sex': [], 'PBC': ['sex', 'age'], 'age': [], 'fibrosis': ['ChHepatitis'], 'diabetes': [], 'obesity': ['diabetes'], 'Steatosis': ['obesity', 'alcoholism'], 'Cirrhosis': ['fibrosis', 'Steatosis'], 'Hyperbilirubinemia': ['age', 'sex'], 'triglycerides': ['Steatosis'], 'RHepatitis': ['hepatotoxic'], 'fatigue': ['ChHepatitis', 'THepatitis', 'RHepatitis'], 'bilirubin': ['Hyperbilirubinemia', 'PBC', 'Cirrhosis', 'gallstones', 'ChHepatitis'], 'itching': ['bilirubin'], 'upper_pain': ['gallstones'], 'fat': ['gallstones'], 'pain_ruq': ['Steatosis', 'Hyperbilirubinemia'], 'pressure_ruq': ['gallstones', 'PBC', 'ChHepatitis'], 'phosphatase': ['RHepatitis', 'THepatitis', 'Cirrhosis', 'ChHepatitis'], 'skin': ['bilirubin'], 'ama': ['PBC'], 'le_cells': ['PBC'], 'joints': ['PBC'], 'pain': ['PBC', 'joints'], 'proteins': ['Cirrhosis'], 'edema': ['Cirrhosis'], 'platelet': ['Cirrhosis', 'PBC'], 'inr': ['ChHepatitis', 'Cirrhosis', 'THepatitis', 'Hyperbilirubinemia'], 'bleeding': ['platelet', 'inr'], 'flatulence': ['gallstones'], 'alcohol': ['Cirrhosis'], 'encephalopathy': ['Cirrhosis', 'PBC'], 'urea': ['encephalopathy'], 'ascites': ['proteins'], 'hepatomegaly': ['RHepatitis', 'THepatitis', 'Steatosis', 'Hyperbilirubinemia'], 'hepatalgia': ['hepatomegaly'], 'density': ['encephalopathy'], 'ESR': ['PBC', 'ChHepatitis', 'Steatosis', 'Hyperbilirubinemia'], 'alt': ['ChHepatitis', 'RHepatitis', 'THepatitis', 'Steatosis', 'Cirrhosis'], 'ast': ['ChHepatitis', 'RHepatitis', 'THepatitis', 'Steatosis', 'Cirrhosis'], 'amylase': ['gallstones'], 'ggtp': ['PBC', 'THepatitis', 'RHepatitis', 'Steatosis', 'ChHepatitis', 'Hyperbilirubinemia'], 'cholesterol': ['PBC', 'Steatosis', 'ChHepatitis'], 'hbsag': ['vh_amn', 'ChHepatitis'], 'hbsag_anti': ['vh_amn', 'ChHepatitis', 'hbsag'], 'anorexia': ['RHepatitis', 'THepatitis'], 'nausea': ['RHepatitis', 'THepatitis'], 'spleen': ['Cirrhosis', 'RHepatitis', 'THepatitis'], 'consciousness': ['encephalopathy'], 'spiders': ['Cirrhosis'], 'jaundice': ['bilirubin'], 'albumin': ['Cirrhosis'], 'edge': ['Cirrhosis'], 'irregular_liver': ['Cirrhosis'], 'hbc_anti': ['vh_amn', 'ChHepatitis'], 'hcv_anti': ['vh_amn', 'ChHepatitis'], 'palms': ['Cirrhosis'], 'hbeag': ['vh_amn', 'ChHepatitis'], 'carcinoma': ['Cirrhosis', 'PBC']}


# order_sum__ce_h_a_x = ['asia','tub','lung','either','xray','smoke','bronc','dysp']
max = 8 #asia
# max=46 #alarm
# max = 123
# order = order_raport__entropy_minus_mi_supra_coloana_minus_fata

order = order_based_on_conditional_entropy_sum_on_line(get_asia_dataset())
# order.reverse()
print(order)

def get_ok_parents_number():
    count = 0
    not_found_parents = {}
    for i in range(len(order)):
        reference_parents = set(reference_possible_parents[order[i]])
        no_of_reference_parents_correct = len(reference_parents & set(order[:i]))
        count += no_of_reference_parents_correct
        if no_of_reference_parents_correct != len(reference_parents):
            not_found_parents[order[i]] = reference_parents - set(order[:i])
    return count*100/max, not_found_parents

print(get_ok_parents_number())

# r = []
# for i in range(100):
#     order = order_random(get_dataset())
#     r.append(get_ok_parents_number()[0])
# print(sum(r)/len(r))