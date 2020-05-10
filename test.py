import networkx as nx
import numpy as np

# G = nx.DiGraph()
# G.add_edges_from([('A', 'B'), ('B', 'A')])
# print(list(nx.simple_cycles(G)))

# {'asiada39a3ee5e6b4b0d3255bfef95601890afd80709': -761.0612,
#  'tub79be276817ebcd106b9fc3d227e18059faf7790e': -830.3555,
#  'smokeda39a3ee5e6b4b0d3255bfef95601890afd80709': -10006.0333,
#  'bronc5cb1e6240fb46e67aac7c760c4f5a0319bdb7fd4': -9263.6793,
#  'lung5cb1e6240fb46e67aac7c760c4f5a0319bdb7fd4': -2724.8952,
#  'eitherd4ad8a0f6a0fef64f310035c1c399b79918a238e': -26.5754,
#  'dyspe00795a4b5f03dfc851ea27a49796ba79a15b58e': -5676.7599,
#  'xrayfa98d8d4306934555ca2d70b0b303a4c03b37e8d': -2839.5018}
# best asia =-32128.861600000004


# best alarm: -152044.75600000002


# empty node -42848.3815
# tree structure = -32888.8309 asia root
# masurat si numarul de operatii pentru a vedea viteza

# a = [["hepatotoxic", "THepatitis"], ["alcoholism", "THepatitis"], ["gallstones", "choledocholithotomy"], ["hospital", "injections"], ["surgery", "injections"], ["choledocholithotomy", "injections"], ["hospital", "transfusion"], ["surgery", "transfusion"], ["choledocholithotomy", "transfusion"], ["transfusion", "ChHepatitis"], ["vh_amn", "ChHepatitis"], ["injections", "ChHepatitis"], ["sex", "PBC"], ["age", "PBC"], ["ChHepatitis", "fibrosis"], ["diabetes", "obesity"], ["obesity", "Steatosis"], ["alcoholism", "Steatosis"], ["fibrosis", "Cirrhosis"], ["Steatosis", "Cirrhosis"], ["age", "Hyperbilirubinemia"], ["sex", "Hyperbilirubinemia"], ["Steatosis", "triglycerides"], ["hepatotoxic", "RHepatitis"], ["ChHepatitis", "fatigue"], ["THepatitis", "fatigue"], ["RHepatitis", "fatigue"], ["Hyperbilirubinemia", "bilirubin"], ["PBC", "bilirubin"], ["Cirrhosis", "bilirubin"], ["gallstones", "bilirubin"], ["ChHepatitis", "bilirubin"], ["bilirubin", "itching"], ["gallstones", "upper_pain"], ["gallstones", "fat"], ["Steatosis", "pain_ruq"], ["Hyperbilirubinemia", "pain_ruq"], ["gallstones", "pressure_ruq"], ["PBC", "pressure_ruq"], ["ChHepatitis", "pressure_ruq"], ["RHepatitis", "phosphatase"], ["THepatitis", "phosphatase"], ["Cirrhosis", "phosphatase"], ["ChHepatitis", "phosphatase"], ["bilirubin", "skin"], ["PBC", "ama"], ["PBC", "le_cells"], ["PBC", "joints"], ["PBC", "pain"], ["joints", "pain"], ["Cirrhosis", "proteins"], ["Cirrhosis", "edema"], ["Cirrhosis", "platelet"], ["PBC", "platelet"], ["ChHepatitis", "inr"], ["Cirrhosis", "inr"], ["THepatitis", "inr"], ["Hyperbilirubinemia", "inr"], ["platelet", "bleeding"], ["inr", "bleeding"], ["gallstones", "flatulence"], ["Cirrhosis", "alcohol"], ["Cirrhosis", "encephalopathy"], ["PBC", "encephalopathy"], ["encephalopathy", "urea"], ["proteins", "ascites"], ["RHepatitis", "hepatomegaly"], ["THepatitis", "hepatomegaly"], ["Steatosis", "hepatomegaly"], ["Hyperbilirubinemia", "hepatomegaly"], ["hepatomegaly", "hepatalgia"], ["encephalopathy", "density"], ["PBC", "ESR"], ["ChHepatitis", "ESR"], ["Steatosis", "ESR"], ["Hyperbilirubinemia", "ESR"], ["ChHepatitis", "alt"], ["RHepatitis", "alt"], ["THepatitis", "alt"], ["Steatosis", "alt"], ["Cirrhosis", "alt"], ["ChHepatitis", "ast"], ["RHepatitis", "ast"], ["THepatitis", "ast"], ["Steatosis", "ast"], ["Cirrhosis", "ast"], ["gallstones", "amylase"], ["PBC", "ggtp"], ["THepatitis", "ggtp"], ["RHepatitis", "ggtp"], ["Steatosis", "ggtp"], ["ChHepatitis", "ggtp"], ["Hyperbilirubinemia", "ggtp"], ["PBC", "cholesterol"], ["Steatosis", "cholesterol"], ["ChHepatitis", "cholesterol"], ["vh_amn", "hbsag"], ["ChHepatitis", "hbsag"], ["vh_amn", "hbsag_anti"], ["ChHepatitis", "hbsag_anti"], ["hbsag", "hbsag_anti"], ["RHepatitis", "anorexia"], ["THepatitis", "anorexia"], ["RHepatitis", "nausea"], ["THepatitis", "nausea"], ["Cirrhosis", "spleen"], ["RHepatitis", "spleen"], ["THepatitis", "spleen"], ["encephalopathy", "consciousness"], ["Cirrhosis", "spiders"], ["bilirubin", "jaundice"], ["Cirrhosis", "albumin"], ["Cirrhosis", "edge"], ["Cirrhosis", "irregular_liver"], ["vh_amn", "hbc_anti"], ["ChHepatitis", "hbc_anti"], ["vh_amn", "hcv_anti"], ["ChHepatitis", "hcv_anti"], ["Cirrhosis", "palms"], ["vh_amn", "hbeag"], ["ChHepatitis", "hbeag"], ["Cirrhosis", "carcinoma"], ["PBC", "carcinoma"]]
# parents = {}
# for i in a:
#     if i[0] not in parents.keys():
#         parents[i[0]] = []
#     if i[1] not in parents.keys():
#         parents[i[1]] = []
#     parents[i[1]].append(i[0])
# print(parents)
a = [["LVFAILURE", "HISTORY"], ["LVEDVOLUME", "CVP"], ["LVEDVOLUME", "PCWP"], ["HYPOVOLEMIA", "LVEDVOLUME"], ["LVFAILURE", "LVEDVOLUME"], ["HYPOVOLEMIA", "STROKEVOLUME"], ["LVFAILURE", "STROKEVOLUME"], ["ERRLOWOUTPUT", "HRBP"], ["HR", "HRBP"], ["ERRCAUTER", "HREKG"], ["HR", "HREKG"], ["ERRCAUTER", "HRSAT"], ["HR", "HRSAT"], ["ANAPHYLAXIS", "TPR"], ["ARTCO2", "EXPCO2"], ["VENTLUNG", "EXPCO2"], ["INTUBATION", "MINVOL"], ["VENTLUNG", "MINVOL"], ["FIO2", "PVSAT"], ["VENTALV", "PVSAT"], ["PVSAT", "SAO2"], ["SHUNT", "SAO2"], ["PULMEMBOLUS", "PAP"], ["INTUBATION", "SHUNT"], ["PULMEMBOLUS", "SHUNT"], ["INTUBATION", "PRESS"], ["KINKEDTUBE", "PRESS"], ["VENTTUBE", "PRESS"], ["MINVOLSET", "VENTMACH"], ["DISCONNECT", "VENTTUBE"], ["VENTMACH", "VENTTUBE"], ["INTUBATION", "VENTLUNG"], ["KINKEDTUBE", "VENTLUNG"], ["VENTTUBE", "VENTLUNG"], ["INTUBATION", "VENTALV"], ["VENTLUNG", "VENTALV"], ["VENTALV", "ARTCO2"], ["ARTCO2", "CATECHOL"], ["INSUFFANESTH", "CATECHOL"], ["SAO2", "CATECHOL"], ["TPR", "CATECHOL"], ["CATECHOL", "HR"], ["HR", "CO"], ["STROKEVOLUME", "CO"], ["CO", "BP"], ["TPR", "BP"]]
G = nx.OrderedGraph()
G.add_edges_from(a)
print(list(G.nodes))


perfect
[["asia", "lung"], ["asia", "either"], ["smoke", "lung"], ["smoke", "either"], ["smoke", "bronc"], ["smoke", "xray"], ["tub", "lung"], ["tub", "either"], ["tub", "bronc"], ["tub", "xray"], ["tub", "dysp"], ["lung", "either"], ["lung", "bronc"], ["lung", "xray"], ["lung", "dysp"], ["either", "bronc"], ["either", "xray"], ["either", "dysp"], ["bronc", "dysp"]]
[["either", "lung"], ["either", "tub"], ["either", "xray"], ["either", "dysp"], ["either", "bronc"], ["either", "smoke"], ["lung", "tub"], ["lung", "xray"], ["lung", "dysp"], ["lung", "bronc"], ["lung", "smoke"], ["tub", "xray"], ["tub", "dysp"], ["tub", "bronc"], ["tub", "smoke"], ["asia", "dysp"], ["dysp", "bronc"], ["bronc", "smoke"]]
