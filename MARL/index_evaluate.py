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

from examples.rllib import utils


def main():
  agent_algorithm = "PPO"
  episode_num = 50
  episode_len = 2000
  save_name = './MARL/data/pure_coordination_5M'
  experiment_state = "~/ray_results/PPO/experiment_state-Pure-Coordination-5M.json"

  # opponent_checkpoint_list = [20, 180, 440, 500, 660, 1480, 1880, 1980, 2020, 2040, 2080] # Prisoners Dilemma
  # opponent_checkpoint_list = [20, 140, 280, 560, 600, 660, 1040, 1160, 1200, 1820, 2120] # Chicken
  # opponent_checkpoint_list = [20, 60, 220, 360, 460, 660, 700, 1900, 1960, 2560, 2640]  # Stag Hunt
  opponent_checkpoint_list = [20, 140, 280, 340, 500, 600, 740, 840, 1360, 2300, 2740] # Pure Coordination
  ego_checkpoint = '/home/yuxin/ray_results/PPO/PPO_meltingpot_Pure_Coordination_5M/checkpoint_00' + str(
    opponent_checkpoint_list[-1]).zfill(4)
  opponent_checkpoint = []
  for i in range(len(opponent_checkpoint_list)):
    opponent_checkpoint.append('/home/yuxin/ray_results/PPO/PPO_meltingpot_Pure_Coordination_5M/checkpoint_00' + str(
      opponent_checkpoint_list[i]).zfill(4))

  register_env("meltingpot", utils.env_creator)

  experiment = ExperimentAnalysis(
      experiment_state,
      default_metric="episode_reward_mean",
      default_mode="max")

  config = experiment.best_config

  # rewards
  rewards = np.empty((len(opponent_checkpoint), episode_num, 2)) # checkpoint x episode x player

  for checkpoint in range(len(opponent_checkpoint)):
    ego_trainer = get_trainer_class(agent_algorithm)(config=config)
    ego_trainer.restore(ego_checkpoint)
    opponent_trainer = get_trainer_class(agent_algorithm)(config=config)
    opponent_trainer.restore(opponent_checkpoint[checkpoint])
    trainer = [ego_trainer, opponent_trainer]

    # Create a new environment to visualise
    env = utils.env_creator(config["env_config"]).get_dmlab2d_env()

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
      print(f'Checkpoint {checkpoint:d}\tEpisode {ep:d}\tEgo reward: {ep_rewards[0]:.3f}\tOpponent reward: {ep_rewards[1]:.3f}')

      # save rewards
      rewards[checkpoint,ep,:] = ep_rewards

      timestep = env.reset()
      states = [bot.initial_state() for bot in bots]
      actions = [0] * len(bots)
      ep_rewards = np.zeros(2)

  print('Final rewards:', rewards)
  np.savez_compressed(save_name, rewards=rewards, checkpoints=opponent_checkpoint_list)

if __name__ == "__main__":
  main()
