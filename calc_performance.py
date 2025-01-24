import torch
from choose_action import choose_action
from choose_action_a3c import choose_action_a3c
def calc_performance(env, device, Q=None, policy=None, episodenum=1000, t_maxstep = 1000, drqn=False):
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
        if env.envID in ['Env1.0','Env1.1']:
            env.reset([-1,-1,-1,-1,-1,-1])
            hx = None # for A3C + lstm
        elif env.envID in ['Env2.0']:
            env.reset([-1,-1,-1,-1,-1,-1])
            hx = None # for A3C + lstm and RDQN
            previous_action = 0
            
        done = False
        t = 0
        while done == False:
            if env.partial == False:
                state = env.state
            else:
                state = env.obs

            if Q is not None:
                if drqn == True:
                    action, hx = choose_action(state,Q,0,action_size,distributional,device, drqn, hx)
                else:
                    action = choose_action(state,Q,0,action_size,distributional,device, drqn, hx)
            elif policy is not None:
                # fill this in later when you get policy gradient algorithms!
                if policy.type == 'A3C':
                    action, hx = choose_action_a3c(state,policy,hx)
            reward, done, _ = env.step(action)
            rewards += reward
            if t >= (t_maxstep - 1):
                done = True
            t += 1
        avgrewards += rewards
    
    return avgrewards/episodenum