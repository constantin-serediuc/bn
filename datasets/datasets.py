import pandas

def get_dataset():
    return get_asia_dataset()

def get_asia_dataset():
    return pandas.read_csv('../ASIA10k.csv')


def get_hepar_dataset():
    return pandas.read_csv('../HEPARTWO10k.csv')

def get_alarm_dataset():
    return pandas.read_csv('../ALARM10k.csv')

