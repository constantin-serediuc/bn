import pandas

def get_dataset():
    return get_hepar_dataset()

def get_asia_dataset():
    return pandas.read_csv('../ASIA10k.csv')


def get_hepar_dataset():
    return pandas.read_csv('../HEPARTWO10k.csv')

