import numpy as np
from absorbing import *

def pretrain(env, nq, memory, max_steps, batch_size, PrioritizedReplay, max_priority, postterm_len, fstack):
    # Make a bunch of random actions from a random state and store the experiences
    reset = True
    memadd = 0 # number of transitions added to memory
    n = nq.n
    while memadd < batch_size:
        if reset == True:
            if env.partial == False:
                env.reset()
                state = env.state
            else:
                env.reset()
                state = env.obs
            previous_action = 0
            reset = False
            stack = state*fstack
            t = 0
            termination_t = 0
        # Make a random action
        action = np.random.randint(0, env.actionspace_dim[0])
        true_state = env.state
        reward, done, _ = env.step(action)

        if env.episodic == False and env.absorbing_cut == True: # if continuous task and absorbing state is defined
            if absorbing(env,true_state) == True: # terminate shortly after the absorbing state is reached.
                termination_t += 1
                if termination_t >= postterm_len: # run x steps once in absorbing state and then terminate
                    done = True
        if t >= max_steps:
            done = True
        t += 1
        if env.partial == False:
            next_state = env.state
        else:
            next_state = env.obs
        next_stack = stack[len(next_state):] + next_state
        if done:
            nq.add(stack, action, reward, next_stack, done, previous_action, memory, PrioritizedReplay)
            reset = True
            memadd += n
        else:
            if env.episodic == False and memadd == (batch_size - 1): # if continuous task AND this is the last transition to be added on the memory AND it's not done, make it done for marking episode ends properly.
                done = True
            # increase memadd by 1 if nq is full
            if len(nq.queue) == n-1:
                memadd += 1
            nq.add(stack, action, reward, next_stack, done, previous_action, memory, PrioritizedReplay)
            state = next_state
            stack = next_stack
            previous_action = action

    nq.queue = [] # clear the n-step queue
    nq.rqueue = [] 
    