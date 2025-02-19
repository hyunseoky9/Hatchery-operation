import pandas as pd
def call_paramset(filename,id):
    paramdf = pd.read_csv(filename, header=None)
    paramdf = paramdf.T
    paramdf.columns = paramdf.iloc[0]
    paramdf = paramdf.drop(0)
    return paramdf