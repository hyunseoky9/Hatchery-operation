from IPython.display import display
import numpy as np
from absorbing import *

def pretrain(env, nq, memory, max_steps, batch_size, PrioritizedReplay, max_priority, postterm_len):
    # Make a bunch of random actions from a random state and store the experiences
    reset = True
    memadd = 0 # number of transitions added to memory
    n = nq.n
    while memadd < batch_size:
        if reset == True:
            if env.envID in ['Env1.0', 'Env1.1']:
                env.reset([-1,-1,-1,-1,-1,-1])
                state = env.state
                previous_action = 0
                reset = False
            elif env.envID in ['Env2.0']:
                env.reset([-1,-1,-1,-1,-1,-1])
                state = env.obs
                previous_action = 0
                reset = False
            t = 0
            termination_t = 0
        # Make a random action
        action = np.random.randint(0, env.actionspace_dim[0])
        true_state = env.state
        reward, done, _ = env.step(action)

        if env.episodic == False and env.absorbing_cut == True: # if continuous task and absorbing state is defined
            if absorbing(env,true_state) == True: # terminate shortly after the absorbing state is reached.
                termination_t += 1
                print(termination_t)
                if termination_t >= postterm_len: # run x steps once in absorbing state and then terminate
                    done = True
        if t >= max_steps:
            done = True
        t += 1
        if env.partial == False:
            next_state = env.state
        else:
            next_state = env.obs

        if done:
            nq.add(state, action, reward, next_state, done, previous_action, memory, PrioritizedReplay)
            reset = True
            memadd += n
        else:
            # increase memadd by 1 if nq is full
            if len(nq.queue) == n-1:
                memadd += 1
            nq.add(state, action, reward, next_state, done, previous_action, memory, PrioritizedReplay)
            state = next_state
            previous_action = action
        print(memory.rewards_buffer[0:10])
        print(memory.states_buffer[0:10])

    nq.queue = [] # clear the n-step queue
    nq.rqueue = [] 
    