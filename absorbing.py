def absorbing(env):
    if env.envID in ['Env2.0']:
        if env.state[0] == 0:
            return True
        else:
            return False