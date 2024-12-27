from Rainbow import choose_action
def calc_performance(env,Q=None,policy=None,episodenum=1000):
    """
    calculate the performance of the agent in the environment.
    For DQN calculate performance with the Q network. (use Q variable in the function input)
    For Tabular Q learning and value iteration, calculate perofrmance using the policy table.
    For policy gradient methods, calculate performance using the policy network.
    """
    avgrewards = 0
    action_size = env.actionspace_dim[0]
    if Q is not None:
        distributional = Q.distributional
    for i in range(episodenum):
        rewards = 0
        if env.envID == 'Env1.0':
            env.reset([-1,-1,-1,-1,-1,-1])
        done = False
        while done == False:
            if Q is not None:
                action = choose_action(env.state,Q,0,action_size,distributional)
            elif policy is not None:
                # fill this in later when you get policy gradient algorithms!
                foo = 0
            reward, done, _ = env.step(action)
            rewards += reward
        avgrewards += rewards
    
    return avgrewards/episodenum