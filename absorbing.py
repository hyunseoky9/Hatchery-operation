def absorbing(env,state):
    if env.envID in ['Env2.0']:
        if state[0] == 0:
            return True
        else:
            return False