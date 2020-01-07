import pandas

def get_dataset():
    df = pandas.read_csv('../ASIA10k.csv')
    return df.applymap(lambda x: True if x =='yes' else False)