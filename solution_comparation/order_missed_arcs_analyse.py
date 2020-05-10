data_asia = {
    'MI_SUM':{'xray': {'either'}, 'dysp': {'either'}},
    'MI_MEDIAN':{'bronc': {'smoke'}, 'xray': {'either'}, 'lung': {'smoke'}, 'dysp': {'either'}},
    'CE_SUM':{'lung': {'smoke'}},
    'CE_SUM_KNOWN':{'dysp': {'either'}, 'xray': {'either'}, 'either': {'tub', 'lung'}},
    'INTRINSIC':{'either': {'tub', 'lung'}, 'lung': {'smoke'}, 'tub': {'asia'}, 'dysp': {'bronc'}, 'bronc': {'smoke'}}
}

data_alarm = {
    'MI_SUM':{'PAP': {'PULMEMBOLUS'}, 'SHUNT': {'INTUBATION'}, 'HISTORY': {'LVFAILURE'}, 'PRESS': {'VENTTUBE'}, 'EXPCO2': {'VENTLUNG', 'ARTCO2'}, 'BP': {'CO'}, 'CATECHOL': {'ARTCO2', 'SAO2'}, 'CVP': {'LVEDVOLUME'}, 'PCWP': {'LVEDVOLUME'}, 'HRBP': {'HR'}, 'CO': {'HR'}, 'SAO2': {'PVSAT'}, 'PVSAT': {'VENTALV'}, 'ARTCO2': {'VENTALV'}},
    'MI_MEDIAN':{'HISTORY': {'LVFAILURE'}, 'SHUNT': {'INTUBATION'}, 'HR': {'CATECHOL'}, 'EXPCO2': {'VENTLUNG', 'ARTCO2'}, 'BP': {'CO'}, 'CATECHOL': {'SAO2', 'ARTCO2'}, 'PVSAT': {'VENTALV'}, 'ARTCO2': {'VENTALV'}},
    'CE_SUM':{'SHUNT': {'INTUBATION'}, 'HISTORY': {'LVFAILURE'}, 'EXPCO2': {'ARTCO2', 'VENTLUNG'}, 'CATECHOL': {'ARTCO2', 'SAO2', 'TPR'}, 'PRESS': {'VENTTUBE'}, 'BP': {'CO'}, 'CVP': {'LVEDVOLUME'}, 'PCWP': {'LVEDVOLUME'}, 'SAO2': {'PVSAT'}, 'PVSAT': {'VENTALV'}, 'ARTCO2': {'VENTALV'}},
    'CE_SUM_KNOWN':{'CATECHOL': {'TPR', 'SAO2', 'ARTCO2', 'INSUFFANESTH'}, 'PVSAT': {'VENTALV'}, 'EXPCO2': {'VENTLUNG', 'ARTCO2'}, 'ARTCO2': {'VENTALV'}, 'VENTLUNG': {'VENTTUBE'}, 'CVP': {'LVEDVOLUME'}, 'BP': {'TPR'}},
    'INTRINSIC':{'VENTALV': {'INTUBATION', 'VENTLUNG'}, 'MINVOL': {'INTUBATION', 'VENTLUNG'}, 'PVSAT': {'FIO2'}, 'VENTLUNG': {'KINKEDTUBE', 'VENTTUBE', 'INTUBATION'}, 'SAO2': {'SHUNT'}, 'VENTTUBE': {'DISCONNECT', 'VENTMACH'}, 'HR': {'CATECHOL'}, 'HREKG': {'ERRCAUTER'}, 'HRSAT': {'ERRCAUTER'}, 'HRBP': {'ERRLOWOUTPUT'}, 'LVEDVOLUME': {'HYPOVOLEMIA', 'LVFAILURE'}, 'CO': {'STROKEVOLUME'}, 'VENTMACH': {'MINVOLSET'}, 'CATECHOL': {'TPR', 'INSUFFANESTH'}, 'BP': {'TPR'}}
}
def get_stats(data):
    result = {}
    arcs = set([])
    arcs_per_alg = {i:[] for i in data.keys()} # (child,parent)
    for alg,case in data.items():
        for child,parents in case.items():
            for parent in parents:
                arcs.add((child,parent))
                arcs_per_alg[alg].append((child,parent))
    for arc in arcs:
        for alg in data.keys():
            if arc in arcs_per_alg[alg]:
                if arc in result.keys():
                    result[arc].append(alg)
                else:
                    result[arc] = [alg]
    indecs = sorted(result, key=lambda k: len(result[k]), reverse=True)
    result = {i:result[i] for i in indecs}
    print(result)

get_stats(data_alarm)