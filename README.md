# Hatchery operation project
## Code Scripts

### environment classes
- *env1_0.py*: first mock model for hatchery operation and population dynamics
- *env1.0_plotting_running.ipynb*: run the training algorithms (e.g. value iteration and td learning) and plot results


### Algorithm running / plotting and analyzing outputs
- *env1.0_performance_testing.ipynb*: compute average reward for a given policy
- *env1.0_plotting_running.ipynb*: running optimization algorithms (value iteration, Tabular Q learning, DeepQN, etc.) and then analyzing/plotting the results for env1.0
- *env0.0_plotting_running.ipynb*: same as above but for env0.0

### Optimization algorithms
#### value iteration
- *value_iteration.py*: perform value iteration
  
#### tabular Q learning
- *td_learning_nonclass.py*: perform Q learning or sarsa on the model

## Output Files
- policy, Q, V function outputs (tabular ones)
format: 
{policy,Q, or V}_{model version}_par{parameterization id}\_dis{discretization id}\_{optimization algorithm used}.pkl

- Q function in network.
QNetwork_{model version}_par{parameterization id}\_dis{discretization id}\_{optimization algorithm used}.pkl

These files are in results folders

### Folders
  - *td_results*: results from tabular Q learning
  - *value iter results*: results from value iteration
  - *deepQN results*: results from deep Q learning


## Environments.
- *Env0.0*: Second environment model made. Drastically simplified version of Env1.0. It was made to practice the tabular Q learning. Now that I've moved on from tabular Q learning, it is no longer relevant.
- *Env1.0*: First environmental model ever made. Has genetic and demographic component of the augmentation environment. See document in overleaf.
- *Env1.1*: 


## Hyperparameter sets

Collection of hyperparameters used to run Deep Q network and other network based algorithms (policy gradient). Shows performance of the set used as well.

- DQNbest: hyperparameter settings used for DQN (rainbow)