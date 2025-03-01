def absorbing(env,state):
    if env.envID in ['Env2.0','Env2.1','Env2.2','Env2.3','Env2.4','Env2.5','Env2.6','Env2.7','Env2.8','Env2.9','Env2.10']:
        if state[env.statevaridx['NW']] == 0:
            return True
        else:
            return False
    elif env.envID == 'tiger':
        if state[0] == 1:
            return True