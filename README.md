# Hatchery operation project
## Code Scripts

### environment classes

Toy RGSM hatchery augmentation environments
- *env0_0.py*: mock model of env1.0 made for practicing tabular Q learning. Practically useless now.
- *env1_0.py*: first mock model for hatchery operation and population dynamics
- *env1_1.py*: continuous state space from env1.0
- *env1_2.py*: continuous state & action space from env1.0
- *env2_0.py*: partial observability added from env1.0
- *env2_1.py*: POMDP format but fully observable. (basically env1.0)
- *env2_2.py*: same as 2.1 but past population size is not observable.
- *env2_3.py*: same as env2.1 but observation but NWm1 is not observed AND NW is observed only in Fall.
- *env2_4.py*: same as Env2.0 but catch (y) is observed every season (both in fall and spring) (most accurate of 2.x)
- *env2_5.py*: same as Env2.0 but catch (y) is observed every season and is continuous (both in fall and spring)
Tiger POMDP (Chades et al. 2008)
- *tiger.py*: Tiger POMDP used for testing DRQN
  
*there are matlab versions (.m) of some of these environments*


### Algorithm running / plotting and analyzing outputs
- *env1.0_performance_testing.ipynb*: compute average reward for a given policy
- *env1.0_plotting_running.ipynb*: running optimization algorithms (value iteration, Tabular Q learning, DeepQN, etc.) and then analyzing/plotting the results for env1.0
- *env0.0_plotting_running.ipynb*: same as above but for env0.0

(environment name)_performance_testing.ipynb is doing the same thing for the named environment, and (environment name)_plotting_running.ipynb as well.

### Optimization algorithms
#### value iteration
- *value_iteration.py*: perform value iteration
  
#### tabular Q learning
- *td_learning_nonclass.py*: perform Q learning or sarsa on the model

#### Deep Q Network (Rainbow)
- *Rainbow.py*: main DQN script. Has all the features in Rainbow.
- *choose_action.py*: choose action from Q or randomly depending on epsilon 
- *distributionalRL.py*: distributional RL feature in rainbow
- *DuelQNN.py*: Dueling network
- *NoisyLinear.py*: Noisy network
- *nq.py*: Queue for N-step learning
- *PrioritizedMemory.py*: Prioritized memory buffer
- *QNN.py*: Vanilla Q network
- *pretrain.py*: fills up the replay buffer before starting the environment simulation.

#### DRQN
- *DRQN.py*: main DRQN script. shares some of the function friles with DQN
- *RQNN.py*: network used for DRQN. has LSTM in it. 
- *RQNN.py*: similar to RQNN but LSTM layers come before the FF layers. Don't use it much.


#### Bash scripts
*tiger_performance_printout_search.py*: goes through the .out files created from the HPC jobs and finds episodes that has high performance with certain policy attributes (e.g., policy has both managing and surveying). 
*env2_x_performance_printout_search.py*: same as above but for env2.x's.


### Others
Performance evaluation scripts
*calc_performance.py*: Calculates expected reward by environment simulation, given Q or policy.
*performance_tester.py*: Uses calc_performance.py to calculate performance, used in tandem with RL algorithms (e.g., Rainbow.py). RL algorithms usually run this in a separate process for efficiency.
*performance.m*: calculates avg perofrmance with a given environment (currently env2.0)

*decay curves.R*: made visualizing different types of decay curves to decide on decay curve parameters. Used for epislon in epsilon-greedy and learning rate.

Scripts made for practicing function approximation using neural net.
- *fn_approx_practice.ipynb*
- *fnapproxto.py*
- *function_approximation_datagen.R*

## Input Files
Parameterization of the environments 
- parameterization_env0.0.csv
- parameterization_env1.0.csv

## Output Files
### Q, V, policy outputs
State action function, Value function, and Policy results made by RL algorithms
- policy, Q, V function outputs (tabular ones)
format: 
{policy,Q, or V}_{model version}\_par{parameterization id}\_dis{discretization id}\_{optimization algorithm used}.pkl

- Q function in network.
QNetwork_{model version}\_par{parameterization id}\_dis{discretization id}\_{optimization algorithm used}.pkl

These files are in results folders
  - *td_results*: results from tabular Q learning
  - *value iter results*: results from value iteration
  - *deepQN results*: results from deep Q learning
  - *DRQN results*: results from DRQN, has parameter id and seed number used for the result.
    - *HPCoutput*: saved HPC printouts.
    - *env2_1_good_ones*: saved network outputs on env2.1.

### Auxillary outputs
These are outputs that aren't Q,V, or policy outputs but outputs made by the algorithm scripts that helps the algorithm calculate the policy or value functions efficiently. 

- initQopt2_{envID}_par{parID}_dis{discID}.pkl: initial Q for tabular Q learning with initialization option 2. Check the td learning portion of env1.0_plotting_running.ipynb for details on initialization

- initQopt3_{envID}_par{parID}_dis{discID}.pkl:initial Q for tabular Q learning with option 3.
- trp_{envID}_par{parID}_dis{discID}.pkl: saved transition probability for the environment used for value iteration. Took one day of simulating transitions to make this.
- {state var}prob(tau_)_{envID}_par{parID}_dis{discID}.pkl: saved transition probability for each state variable (e.g. H, NW) in different tau (season). Made for value iteration for env1.0, but not used because calculating transition probability by state variables did not do very well in getting expected transition probabilities.



## Environments.
- *Env0.0*: Second environment model made. Drastically simplified version of Env1.0. It was made to practice the tabular Q learning. Now that I've moved on from tabular Q learning, it is no longer relevant.
- *Env1.0*: First environmental model ever made. Has genetic and demographic component of the augmentation environment. See document in overleaf.
- *Env1.1*: 


## Hyperparameter sets

Collection of hyperparameters used to run Deep Q network and other network based algorithms (policy gradient). Shows performance of the set used as well.

- DQNbest: hyperparameter settings used for DQN (rainbow)