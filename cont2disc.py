from env1_1 import Env1_1
from env1_0 import Env1_0
import numpy as np

def cont2disc(env,envd,policy):
    """
    used in env1.1_performance_testing
    Convert a continuous state into a discrete one.
    Then, pick an action from the policy from the policy trained from the discrete environment.
    convert the action to the action state in the actions available in the continuous state and discrete action environment
    pick the action too
    """
    # compare the continuous state to the discretized state
    i = 0
    discretized_state = []
    for var in env.state:
        discrete_vars = list(envd.states.values())[i]
        closest_stateidx = int(np.argmin(np.abs(var - np.array(discrete_vars))))
        discretized_state.append(closest_stateidx)
        i += 1
    stateid = envd._flatten(discretized_state)
    actionid = int(policy[stateid])
    actiond = envd.actions["a"][actionid] # action in the discrete environment
    closest_actionidx = int(np.argmin(np.abs(actiond - np.array(env.actions["a"]) ))) # action in the continuous environment
    return closest_actionidx



