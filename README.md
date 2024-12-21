# Hatchery operation project
## Code Scripts

### environment classes
- *env1_0.py*: [first mock model for hatchery operation and population dynamics]
- *env1.0_plotting_running.ipynb*: [run the training algorithms (e.g. value iteration and td learning) and plot results]

### algorithm running and plotting
- *env1.0_performance_testing.ipynb*: [compute average reward for a given policy]
- *env1.0_plotting_running.ipynb*: [running optimization algorithms (value iteration, Tabular Q learning, DeepQN, etc.) and then analyzing/plotting the results for env1.0]
- *env0.0_plotting_running.ipynb*: [same as above but for env0.0]

### Optimization algorithms
#### value iteration
- *value_iteration.py*: [perform value iteration]
- *td_learning.py*: [perform Q learning or sarsa on the model]
- *td_learning_nonclass.py*: [perform Q learning or sarsa on the model]


### Data Files
policy, Q, V function outputs
format: 
{policy,Q, or V}_{model version}_par{parameterization id}_dis{discretization id}_{optimization algorithm used}.pkl
