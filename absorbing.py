def absorbing(env,state):
    if env.envID in ['Env2.0','Env2.1','Env2.2','Env2.3','Env2.4','Env2.5','Env2.6','tiger']:
        if state[0] == 0:
            return True
        else:
            return False