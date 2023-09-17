"""Runs the bots trained in self_play_train.py and renders in pygame.

You must provide experiment_state, expected to be
~/ray_results/PPO/experiment_state_YOUR_RUN_ID.json
"""

import argparse

import dm_env
from dmlab2d.ui_renderer import pygame
import numpy as np
from ray.rllib.algorithms.registry import get_trainer_class
from ray.tune.analysis.experiment_analysis import ExperimentAnalysis
from ray.tune.registry import register_env
import os

from examples.rllib import utils

def main():
  agent_algorithm = "PPO"
  episode_num = 10
  episode_len = 2000
  ###
  exp_name = 'PP3'
  checkpoint_dir = './MARL/PP_checkpoints/'
  num_seed = 3
  config_size = 'o'
  config_name = 'c'
  ###

  config_name_dict = {
    'c': 'chicken',
    'pc': 'pure_coordination',
    'pd': 'prisoners_dilemma',
    'sh': 'stag_hunt'
  }

  save_name_list = [f'./MARL/data_tournament/{config_name_dict[config_name]}_{config_size.upper()}_{exp_name}_2']
  experiment_state_list = [f'~/ray_results/PPO/experiment_state-{config_name_dict[config_name]}_{config_size.upper()}_5M.json']
  ego_name_list = [f'{config_name}{num_seed:d}{config_size}']
  opponent_name_list = [f'{config_name}_{config_size}']

  for _ in range(len(save_name_list)):
    save_name = save_name_list[_]
    experiment_state = experiment_state_list[_]
    ego_name = ego_name_list[_]
    ego_seed = num_seed  # CHANGE THIS
    opponent_name = opponent_name_list[_]
    opponent_seed = 1  # CHANGE THIS

    ego_checkpoint = []
    opponent_checkpoint = []

    for i in range(ego_seed):
      temp_dir = checkpoint_dir + ego_name + '/seed_' + str(i)
      gen = os.listdir(temp_dir)
      gen.sort()
      checkpoint = os.listdir(os.path.join(temp_dir, gen[-1]))
      ego_checkpoint.append(os.path.join(temp_dir, gen[-1], checkpoint[-1]))
    
    print(ego_checkpoint)
    
    for j in range(opponent_seed):
      temp_dir = './MARL/SP_eval_checkpoints/' + opponent_name + '/seed_' + str(j)
      # gen = ['gen_006/checkpoint_000875', 'gen_012/checkpoint_001625',
      #       'gen_018/checkpoint_002375', 'gen_024/checkpoint_003125']
      gen = ['gen_018/checkpoint_002375', 'gen_020/checkpoint_002625', 
             'gen_022/checkpoint_002875', 'gen_024/checkpoint_003125']
      for k in range(len(gen)):
        opponent_checkpoint.append(os.path.join(temp_dir, gen[k]))

    print(opponent_checkpoint)
    
    register_env("meltingpot", utils.env_creator)

    experiment = ExperimentAnalysis(
        experiment_state,
        default_metric="episode_reward_mean",
        default_mode="max")

    config = experiment.best_config

    # Create a new environment to visualise
    env = utils.env_creator(config["env_config"]).get_dmlab2d_env()

    # rewards
    rewards = np.empty((len(ego_checkpoint), len(opponent_checkpoint), episode_num)) # ego x opponent x episode

    for ego in range(len(ego_checkpoint)):
      for opponent in range(len(opponent_checkpoint)):
        ego_trainer = get_trainer_class(agent_algorithm)(config=config)
        ego_trainer.restore(ego_checkpoint[ego])
        opponent_trainer = get_trainer_class(agent_algorithm)(config=config)
        opponent_trainer.restore(opponent_checkpoint[opponent])
        trainer = [ego_trainer, opponent_trainer]

        bots = [
          utils.RayModelPolicy(trainer[i], f"agent_0")
          for i in range(len(config["env_config"]["roles"]))
        ]

        timestep = env.reset()
        states = [bot.initial_state() for bot in bots]
        actions = [0] * len(bots)

        for ep in range(episode_num):
          ep_rewards = np.zeros(2)
          for _ in range(episode_len):
            obs = timestep.observation[0]["WORLD.RGB"]
            obs = np.transpose(obs, (1, 0, 2))

            for i, bot in enumerate(bots):
              timestep_bot = dm_env.TimeStep(
                  step_type=timestep.step_type,
                  reward=timestep.reward[i],
                  discount=timestep.discount,
                  observation=timestep.observation[i])

              actions[i], states[i] = bot.step(timestep_bot, states[i])

            ep_rewards += timestep.reward
            timestep = env.step(actions)

          print('-'*50)
          print(f'Ego {ego:d}xOpp {opponent:d}\tEpisode {ep:d}\tEgo reward: {ep_rewards[0]:.3f}\tOpponent reward: {ep_rewards[1]:.3f}')

          # save rewards
          rewards[ego, opponent, ep] = ep_rewards[0]

          timestep = env.reset()
          states = [bot.initial_state() for bot in bots]
          actions = [0] * len(bots)
          ep_rewards = np.zeros(2)

    print('Final rewards:', rewards)
    np.savez_compressed(save_name, rewards=rewards)

if __name__ == "__main__":
  main()

