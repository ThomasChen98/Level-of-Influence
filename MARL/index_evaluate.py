"""Runs the bots trained in self_play_train.py and renders in pygame.

You must provide experiment_state, expected to be
~/ray_results/PPO/experiment_state_YOUR_RUN_ID.json
"""

import argparse
import os
import dm_env
from dmlab2d.ui_renderer import pygame
import numpy as np
from ray.rllib.algorithms.registry import get_trainer_class
from ray.tune.analysis.experiment_analysis import ExperimentAnalysis
from ray.tune.registry import register_env

from examples.rllib import utils


def main():
  agent_algorithm = "PPO"
  episode_num = 6
  episode_len = 2000

  ###
  exp_name = 'SP5'
  checkpoint_dir = './MARL/SP_checkpoints/'
  ego_seed_num = 3
  opp_seed_num = 3
  config_size = 'l'
  config_name = 'sh'
  ###

  config_name_dict = {
    'c': 'chicken',
    'pc': 'pure_coordination',
    'pd': 'prisoners_dilemma',
    'sh': 'stag_hunt'
  }

  save_name = f'./MARL/data_index_33/{config_name_dict[config_name]}_{config_size.upper()}_5M_2'
  experiment_state = f'~/ray_results/PPO/experiment_state-{config_name_dict[config_name]}_{config_size.upper()}_5M.json'
  ego_name = f'{config_name}_{config_size}'
  ego_name_2 = f'{config_name}5{config_size}'
  opp_name = f'{config_name}5{config_size}'

  ego_checkpoint = []
  for j in range(ego_seed_num):
      if j == 0: # use SP_eval
        temp_dir = './MARL/SP_eval_checkpoints/' + ego_name + '/seed_' + str(j)
        gen = ['gen_018/checkpoint_002375', 'gen_020/checkpoint_002625', 
              'gen_022/checkpoint_002875', 'gen_024/checkpoint_003125']
        for k in range(len(gen)):
          ego_checkpoint.append(os.path.join(temp_dir, gen[k]))
      else: # use SP5 seed 3 & 4
        temp_dir = './MARL/SP_checkpoints/' + ego_name_2 + '/seed_' + str(j+2)
        gen = ['gen_018/checkpoint_002375', 'gen_020/checkpoint_002625', 
              'gen_022/checkpoint_002875', 'gen_024/checkpoint_003125']
        for k in range(len(gen)):
          ego_checkpoint.append(os.path.join(temp_dir, gen[k]))

  # opponent_checkpoint_list = [20, 140, 280, 560, 600, 660, 1040, 1160, 1200, 1820, 2120]  # Chicken Large
  # opponent_checkpoint_list = [80, 160, 200, 580, 760, 980, 1120, 1460, 2300, 2740, 3020] # Chicken Obstacle
  # opponent_checkpoint_list = [20, 100, 160, 280, 360, 560, 700, 1240, 1280, 1340, 1360] # Chicken Medium
  # opponent_checkpoint_list = [60, 140, 200, 860, 1060, 1120, 1740, 1800, 2340, 2760, 3000] # Chicken Small
  # opponent_checkpoint_list = [20, 260, 360, 420, 660, 580, 820, 1120, 1700, 2480, 2940]  # Pure Coordination Large
  # opponent_checkpoint_list = [20, 300, 420, 540, 640, 840, 1020, 1320, 1980, 2360, 3125]  # Pure Coordination Obstacle
  # opponent_checkpoint_list = [20, 120, 180, 260, 420, 500, 560, 900, 1680, 1880, 1960]  # Pure Coordination Medium
  # opponent_checkpoint_list = [40, 220, 320, 400, 580, 700, 740, 860, 1660, 2500, 2700]  # Pure Coordination Small
  # opponent_checkpoint_list = [20, 180, 440, 500, 660, 1480, 1880, 1980, 2020, 2040, 2080]  # Prisoners Dilemma Large
  # opponent_checkpoint_list = [20, 320, 840, 1740, 2300, 2580, 2640, 2700, 2740, 2800, 2840] # Prisoners Dilemma Obstacle
  # opponent_checkpoint_list = [80, 180, 280, 340, 1380, 1820, 1860, 2520, 2780, 3000, 3125] # Prisoners Dilemma Medium
  # opponent_checkpoint_list = [20, 100, 300, 480, 520, 1260, 1700, 2380, 2780, 2820, 2880]  # Prisoners Dilemma Small
  # opponent_checkpoint_list = [20, 60, 220, 360, 460, 660, 700, 1900, 1960, 2560, 2640]  # Stag Hunt Large
  # opponent_checkpoint_list = [20, 80, 420, 500, 600, 800, 900, 1440, 1880, 2340, 2460]  # Stag Hunt Obstacle
  # opponent_checkpoint_list = [140, 180, 240, 280, 400, 660, 920, 1280, 1500, 2740, 3040]  # Stag Hunt Medium
  # opponent_checkpoint_list = [20, 140, 240, 300, 380, 500, 1540, 1780, 2260, 2580, 2600]  # Stag Hunt Small

  register_env("meltingpot", utils.env_creator)

  experiment = ExperimentAnalysis(
      experiment_state,
      default_metric="episode_reward_mean",
      default_mode="max")

  config = experiment.best_config

  # rewards
  gen_list = [0, 2, 4, 6, 8, 12, 16, 20, 24]
  rewards = np.empty((ego_seed_num, 4, opp_seed_num, len(gen_list), episode_num, 2)) # ego_seed x ego_gen x opp_seed x opp_gen x episode x player

  # Create a new environment to visualise
  env = utils.env_creator(config["env_config"]).get_dmlab2d_env()

  for ego_seed in range(ego_seed_num):
    for opp_seed in range(opp_seed_num):
      print(ego_checkpoint)

      temp_dir = checkpoint_dir + opp_name + '/seed_' + str(opp_seed)
      gen = os.listdir(temp_dir)
      gen.sort()
      checkpoint_list = [os.listdir(os.path.join(temp_dir, gen[_]))[-1] for _ in range(len(gen))]
      opponent_checkpoint = [os.path.join(temp_dir, gen[_], checkpoint_list[_]) for _ in gen_list]
      print(opponent_checkpoint)

      for ego_gen in range(4):
        for opp_gen in range(len(opponent_checkpoint)):
          ego_trainer = get_trainer_class(agent_algorithm)(config=config)
          ego_trainer.restore(ego_checkpoint[ego_seed * 4 + ego_gen])
          opponent_trainer = get_trainer_class(agent_algorithm)(config=config)
          opponent_trainer.restore(opponent_checkpoint[opp_gen])
          trainer = [ego_trainer, opponent_trainer]

          bots = [
            utils.RayModelPolicy(trainer[i], f"agent_{i}")
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
            print(f'Ego Seed {ego_seed:d} Gen {ego_gen:d} x Opp Seed {opp_seed:d} Gen {opp_gen:d}\t \
              Episode {ep:d}\tEgo reward: {ep_rewards[0]:.3f}\tOpponent reward: {ep_rewards[1]:.3f}')

            # save rewards
            rewards[ego_seed, ego_gen, opp_seed, opp_gen, ep, :] = ep_rewards

            timestep = env.reset()
            states = [bot.initial_state() for bot in bots]
            actions = [0] * len(bots)
            ep_rewards = np.zeros(2)

          print('Final rewards:', rewards)
          np.savez_compressed(save_name, rewards=rewards, checkpoints=gen_list)

if __name__ == "__main__":
  main()
