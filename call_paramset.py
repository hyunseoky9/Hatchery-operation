import pandas as pd
from itertools import product
def call_paramset(filename,id):
    # basic processing
    data = pd.read_csv(filename, header=None)
    data = data.T
    data.columns = data.iloc[0]
    data = data.drop(0)   
    paramdf = data.iloc[id].to_dict()
    paramdflist = []
    # if there are semicolons, separate them and make all combinations of paramdf
    keys = paramdf.keys()
    tunekeys = []
    tunekeyvals = []
    # get which keys have multiple values to try out for tuning
    for key in keys:
        if key=='notes' or key=='score':
            continue
        if ';' in paramdf[key]:
            tunekeys.append(key)
            vals = paramdf[key].split(';')
            tunekeyvals.append(vals)
    # Generate all combinations of tuning parameters
    for combination in product(*tunekeyvals):
        temp_paramdf = paramdf.copy()
        for i, key in enumerate(tunekeys):
            temp_paramdf[key] = combination[i]
        paramdflist.append(temp_paramdf)
    return paramdflist


from env1_0 import Env1_0
from env1_1 import Env1_1
from env1_2 import Env1_2
from env2_0 import Env2_0
from env2_1 import Env2_1
from env2_2 import Env2_2
from env2_3 import Env2_3
from env2_4 import Env2_4
from env2_5 import Env2_5
from env2_6 import Env2_6
from tiger import Tiger
def call_env(param):
    config = eval(param['envconfig'])
    if param['envid'] == 'Env1.0':
        return Env1_0(config['init'], config['paramset'], config['discretization'])
    elif param['envid'] == 'Env1.1':
        return Env1_1(config['init'], config['paramset'], config['discretization'])
    elif param['envid'] == 'Env1.2':
        return Env1_2(config['init'], config['paramset'], config['discretization'])
    elif param['envid'] == 'Env2.0':
        return Env2_0(config['init'], config['paramset'], config['discretization'])
    elif param['envid'] == 'Env2.1':
        return Env2_1(config['init'], config['paramset'], config['discretization'])
    elif param['envid'] == 'Env2.2':
        return Env2_2(config['init'], config['paramset'], config['discretization'])
    elif param['envid'] == 'Env2.3':
        return Env2_3(config['init'], config['paramset'], config['discretization'])
    elif param['envid'] == 'Env2.4':
        return Env2_4(config['init'], config['paramset'], config['discretization'])
    elif param['envid'] == 'Env2.5':
        return Env2_5(config['init'], config['paramset'], config['discretization'])
    elif param['envid'] == 'Env2.6':
        return Env2_6(config['init'], config['paramset'], config['discretization'])
    elif param['envid'] == 'Tiger':
        return Tiger()
    else:
        raise ValueError("Unknown environment ID")
        