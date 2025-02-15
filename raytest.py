# register on ray tune
import gymnasium as gym
from env2_1gym import Env2_1gym
import numpy as np
import random
from ray.rllib.connectors.env_to_module import FlattenObservations
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig

from ray.tune.registry import register_env
def env_creator(config):
    return Env2_1gym(config)  # Return a gymnasium.Env instance.

register_env("Env2_1", env_creator)

config = (
    DQNConfig()
    .environment("Env2_1",
                env_config={"initstate": [-1, -1, -1, -1, -1, -1], "parameterization_set": 2, "discretization_set": 0})
    .env_runners(num_env_runners=1,
                 env_to_module_connector=lambda env: FlattenObservations(),
                 rollout_fragment_length=1)
    .framework("torch")
    .training(dueling =False,
              lr = [(0, 0.01), (10000, 0.0001)],
              gamma = 0.99,
              replay_buffer_config={
                "type": "PrioritizedEpisodeReplayBuffer",
                "capacity": 1000,
                "alpha": 0,
                "beta": 0.5},
              train_batch_size=100,
              num_steps_sampled_before_learning_starts = 0, 
              training_intensity = 12,
              target_network_update_freq=10,
              td_error_loss_fn = 'mse',
              )
    .rl_module(
    # Use a non-default 32,32-stack with ReLU activations.
    model_config=DefaultModelConfig(
        fcnet_hiddens=[30,30],
        fcnet_activation="relu",
    ))
    
)
algo = config.build()
trainoutput = algo.train()

print(f'num_episodes: {trainoutput["env_runners"]["num_episodes"]}, episode_len_mean: {trainoutput["env_runners"]["episode_len_mean"]}, num_env_steps_sampled: {trainoutput["env_runners"]["num_env_steps_sampled"]}, num_target_updates: {trainoutput["learners"]["default_policy"]["num_target_updates"]}, rollout_fragment_length: {algo.config.get_rollout_fragment_length()}')
print(f'num_target_updates: {trainoutput["learners"]["default_policy"]["num_target_updates"]}, num_training_step_calls_per_iteration: {trainoutput["num_training_step_calls_per_iteration"]}, num_module_steps_trained {trainoutput['learners']['default_policy']['num_module_steps_trained']}')

