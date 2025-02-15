import numpy as np
# making learning rate and epsilon schedule list for RL lib. 

def lrscheduling(init, rate, type):
    unit = 1000 # number of timestep before the linear rate changes
    endtimestep = 10**7
    schedule = []
    if type == 'exponential':
        for step in range(0, endtimestep, unit):
            schedule.append((step,init*(rate**(step))))
        return schedule
    elif type == 'constant':
        return init
        
    
def epsilonscheduling(init=0.1, rate = 0.01, type='constant'):
    unit = 1000 
    endtimestep = 10**7
    schedule = []
    if type == 'linear':
        for step in range(0, endtimestep, unit):
            schedule = [(0,init),(endtimestep,0.01)]
    elif type == 'exponential':
        for step in range(0, endtimestep, unit):
            schedule.append((step,init*(rate**(step))))
        return schedule
    elif type == 'constant':
        return init


