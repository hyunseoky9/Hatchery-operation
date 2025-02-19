# now let's try to make a DRQN or R2D2 and see if I can get similar performance on env2.1
# register env on ray tune
# let's see if I can recreate the trained DQN performance score on env2.1 (no partial observation)
import gymnasium as gym
from env2_1gym import Env2_1gym
import numpy as np
import random
from ray.rllib.connectors.env_to_module import FlattenObservations
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.utils.replay_buffers.replay_buffer import StorageUnit
from scheduler import lrscheduling, epsilonscheduling
import torch
from ray.tune.registry import register_env

# register environment
def env_creator(config):
    return Env2_1gym(config)  # Return a gymnasium.Env instance.
register_env("Env2_1", env_creator)

lrschedule = lrscheduling(init=0.01, rate = 0.99995,type='exponential')
epsilonschedule = epsilonscheduling(init=0.1, rate = 0.01, type='constant')

config = (
    DQNConfig()
    .environment("Env2_1",
                env_config={"initstate": [-1, -1, -1, -1, -1, -1], "parameterization_set": 2, "discretization_set": 0})
    .env_runners(num_env_runners=1,
                 rollout_fragment_length=1)
    .framework("torch")
    .training(dueling =False,
              lr = lrschedule, # [(0, 0.01), (1000, 0.0001)],
              epsilon = epsilonschedule, #[(0, 0.1), (1000, 0.01)],
              gamma = 0.99,
              replay_buffer_config={
                "type": "MultiAgentPrioritizedReplayBuffer",
                "capacity": 1000,
                "alpha": 0.5,
                "storage_unit": StorageUnit.SEQUENCES,
                #"replay_sequence_length": 1,
                "replay_zero_init_states": True,
                },
              train_batch_size=100,
              num_steps_sampled_before_learning_starts = 100, 
              training_intensity = 7,
              target_network_update_freq=15,
              td_error_loss_fn = 'mse',
              #model=dict(use_lstm=True, lstm_cell_size=64, max_seq_len=20)
              )
    .rl_module(
        model_config=DefaultModelConfig(
        fcnet_hiddens=[30],
        fcnet_activation="relu",
        use_lstm=True,
        max_seq_len=20,
        lstm_cell_size=64
    ))
    .resources(
        num_gpus = 0
    )
    .evaluation(
        evaluation_interval = 14,
        evaluation_duration = 1000,
        evaluation_num_env_runners = 4,
    )
    #.debugging(seed=12)
)
algo = config.build()
foo = algo.train()
#iternum = 200
#last_reward = 0
#runid = random.randint(0,100000)
#savepath = 'G:/My Drive/research/nmsu/hatchery operation/codes/dynamic programming1/rayrllib results/'
#print('training start!')
#for iter in range(iternum):
#    trainoutput = algo.train()
#    if trainoutput.get("evaluation", {}) != {}:
#        eval_reward = trainoutput['evaluation']['env_runners']['episode_return_mean']
#        episodelifetime = trainoutput['env_runners']['num_episodes_lifetime']
#        cur_lr = trainoutput['learners']['default_policy']['default_optimizer_learning_rate']
#        if last_reward != eval_reward:
#            print('-----------------------------------------------------')
#            print(f'Train iter {iter+1}, Episode {episodelifetime}, Learning Rate: {cur_lr}, Avg Performance: {eval_reward}')
#            last_reward = eval_reward
#            # checkpoint
#            if eval_reward >= 4600:
#                checkpath = savepath + f'dqn/runid{runid}/iter{iter}_reward{round(eval_reward)}'
#                algo.save_to_path(checkpath)
#                print(f"saved algo")
#
#
#
#print('---------------------------------------------------------')
#print(f'num_episodes: {trainoutput["env_runners"]["num_episodes"]}, episode_len_mean: {trainoutput["env_runners"]["episode_len_mean"]}, num_env_steps_sampled: {trainoutput["env_runners"]["num_env_steps_sampled"]}, num_target_updates: {trainoutput["learners"]["default_policy"]["num_target_updates"]}, rollout_fragment_length: {algo.config.get_rollout_fragment_length()}')
#print(f'num_target_updates: {trainoutput["learners"]["default_policy"]["num_target_updates"]}, num_training_step_calls_per_iteration: {trainoutput["num_training_step_calls_per_iteration"]}, num_module_steps_trained {trainoutput['learners']['default_policy']['num_module_steps_trained']}')
#
#algo.stop()