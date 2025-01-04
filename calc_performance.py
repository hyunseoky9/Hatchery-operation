from choose_action import choose_action
def calc_performance(env, device, Q=None, policy=None, episodenum=1000):
    """
    calculate the performance of the agent in the environment.
    For DQN calculate performance with the Q network. (use Q variable in the function input)
    For Tabular Q learning and value iteration, calculate perofrmance using the policy table.
    For policy gradient methods, calculate performance using the policy network.
    """
    t_maxstep = 1000
    avgrewards = 0
    action_size = env.actionspace_dim[0]
    if Q is not None:
        distributional = Q.distributional
    for i in range(episodenum):
        rewards = 0
        if env.envID in ['Env1.0','Env1.1']:
            env.reset([-1,-1,-1,-1,-1,-1])
        done = False
        t = 0
        while done == False:
            if Q is not None:
                action = choose_action(env.state,Q,0,action_size,distributional,device)
            elif policy is not None:
                # fill this in later when you get policy gradient algorithms!
                foo = 0
            reward, done, _ = env.step(action)
            rewards += reward
            if t >= t_maxstep:
                done = True
            t += 1
        avgrewards += rewards
    
    return avgrewards/episodenum