import glob
import os
import shutil
import ray
import numpy as np
from ray import tune
from ray.rllib.algorithms import ppo
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.algorithms.ppo import PPO
import random
from examples.rllib import utils
from meltingpot.python import substrate

def get_config(
    substrate_name: str = "chicken_in_the_matrix__repeated",
    num_rollout_workers: int = 2,
    rollout_fragment_length: int = 100,
    train_batch_size: int = 1600,
    fcnet_hiddens=(64, 64),
    post_fcnet_hiddens=(256,),
    lstm_cell_size: int = 256,
    sgd_minibatch_size: int = 128,
):
  """Get the configuration for running an agent on a substrate using RLLib.

  We need the following 2 pieces to run the training:

  Args:
    substrate_name: The name of the MeltingPot substrate, coming from
      `substrate.AVAILABLE_SUBSTRATES`.
    num_rollout_workers: The number of workers for playing games.
    rollout_fragment_length: Unroll time for learning.
    train_batch_size: Batch size (batch * rollout_fragment_length)
    fcnet_hiddens: Fully connected layers.
    post_fcnet_hiddens: Layer sizes after the fully connected torso.
    lstm_cell_size: Size of the LSTM.
    sgd_minibatch_size: Size of the mini-batch for learning.

  Returns:
    The configuration for running the experiment.
  """
  # Gets the default training configuration
  config = ppo.PPOConfig()
  # Number of arenas.
  # This is called num_rollout_workers in 2.2.0.
  config.num_workers = num_rollout_workers
  # This is to match our unroll lengths.
  config.rollout_fragment_length = rollout_fragment_length
  # Total (time x batch) timesteps on the learning update.
  config.train_batch_size = train_batch_size
  # Mini-batch size.
  config.sgd_minibatch_size = sgd_minibatch_size
  # Use the raw observations/actions as defined by the environment.
  config.preprocessor_pref = None
  # Use TensorFlow as the tensor framework.
  config = config.framework("tf")
  # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
  config.num_gpus = int(os.environ.get("RLLIB_NUM_GPUS", "0"))
  config.log_level = "DEBUG"

  # 2. Set environment config. This will be passed to
  # the env_creator function via the register env lambda below.
  player_roles = substrate.get_config(substrate_name).default_player_roles
  config.env_config = {"substrate": substrate_name, "roles": player_roles}

  config.env = "meltingpot"

  # 4. Extract space dimensions
  test_env = utils.env_creator(config.env_config)

  # Setup PPO with policies, one per entry in default player roles.
  policies = {}
  player_to_agent = {}
  for i in range(len(player_roles)):
    rgb_shape = test_env.observation_space[f"player_{i}"]["RGB"].shape
    sprite_x = rgb_shape[0] // 8
    sprite_y = rgb_shape[1] // 8

    policies[f"agent_{i}"] = PolicySpec(
      policy_class=None,  # use default policy
      observation_space=test_env.observation_space[f"player_{i}"],
      action_space=test_env.action_space[f"player_{i}"],
      config={
        "model": {
          "conv_filters": [[16, [8, 8], 8],
                           [128, [sprite_x, sprite_y], 1]],
        },
      })
    player_to_agent[f"player_{i}"] = f"agent_{i}"

  def policy_mapping_fn(agent_id, **kwargs):
    del kwargs
    return player_to_agent[agent_id]

  # 5. Configuration for multi-agent setup with one policy per role:
  config.multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn, policies_to_train=["agent_0"])

  # 6. Set the agent architecture.
  # Definition of the model architecture.
  # The strides of the first convolutional layer were chosen to perfectly line
  # up with the sprites, which are 8x8.
  # The final layer must be chosen specifically so that its output is
  # [B, 1, 1, X]. See the explanation in
  # https://docs.ray.io/en/latest/rllib-models.html#built-in-models. It is
  # because rllib is unable to flatten to a vector otherwise.
  # The acb models used as baselines in the meltingpot paper were not run using
  # rllib, so they used a different configuration for the second convolutional
  # layer. It was 32 channels, [4, 4] kernel shape, and stride = 1.
  config.model["fcnet_hiddens"] = fcnet_hiddens
  config.model["fcnet_activation"] = "relu"
  config.model["conv_activation"] = "relu"
  config.model["post_fcnet_hiddens"] = post_fcnet_hiddens
  config.model["post_fcnet_activation"] = "relu"
  config.model["use_lstm"] = True
  config.model["lstm_use_prev_action"] = True
  config.model["lstm_use_prev_reward"] = False
  config.model["lstm_cell_size"] = lstm_cell_size

  return config

def main():
  config = get_config()
  tune.register_env("meltingpot", utils.env_creator)

  # parameters
  save_path = './MARL/PP_logs/c3l'
  checkpoints_path = './MARL/PP_checkpoints/c3l'
  log_path = './MARL/PP_outputs/c3l.txt'
  checkpoint_freq = 125
  num_gens = 25
  seeds = [11,22,33]

  gen_len = checkpoint_freq * config.train_batch_size
  num_seeds = len(seeds)

  # clear output
  f = open(log_path, "w")
  f.close()

  # Initialize ray, train and save
  ray.init()

  # logging
  timesteps = [[] for j in range(num_seeds)]
  policy_reward_min = [[[] for i in range(2)] for j in range(num_seeds)]
  policy_reward_mean = [[[] for i in range(2)] for j in range(num_seeds)]
  policy_reward_max = [[[] for i in range(2)] for j in range(num_seeds)]

  # A dictionary of lists of checkpoints for each seed/population

  for f in os.scandir(checkpoints_path):
    if f.is_dir():
      shutil.rmtree(f)
  checkpoints_dict = {}
  for seed in range(num_seeds):
    checkpoints_dict[f'Seed_{seed}'] = []

  for gen in range(num_gens):
    for seed in range(num_seeds):
      # config.env_config["seed"] = seeds[seed]
      ppo = PPO(config=config.to_dict())

      with open(log_path, "a") as f:
        f.write(f'SEED {seed}/{num_seeds} GENERATION {gen}/{num_gens}\n')
        f.close()

      # Set Player 0's policy
      if (len(checkpoints_dict[f"Seed_{seed}"]) > 0):
        with open(log_path, "a") as f:
          f.write('Load player 0 policy from '+checkpoints_dict[f"Seed_{seed}"][-1]+'\n')
          f.close()
        ppo.restore(checkpoints_dict[f"Seed_{seed}"][-1])  # This would write both the weights of agent_0 and agent_1 from its own seed

      # Set Player 1's policy
      opp_seed = random.randint(0, num_seeds - 1)
      if (len(checkpoints_dict[f"Seed_{opp_seed}"]) > 0):
        with open(log_path, "a") as f:
          f.write('Load player 1 policy from ' + checkpoints_dict[f"Seed_{opp_seed}"][-1] + '\n')
          f.close()
        ppo_dummy = ppo
        ppo_dummy.restore(checkpoints_dict[f"Seed_{opp_seed}"][-1])
        opp_weights = ppo_dummy.get_policy("agent_0").get_weights()
        ppo.set_weights({"agent_1": opp_weights})  # This would overwrite the weights of agent_1 with agent_0's weight from the other seed

      # Train the agent for checkpoint_freq times before saving the checkpoint
      with open(log_path, "a") as f:
        f.write('['+'.'*50+']'+f' 0/{checkpoint_freq}\n')
        f.close()
      for j in range(checkpoint_freq):
        # train policy 0
        results = ppo.train()
        # log
        with open(log_path, "r") as f:
          lines = f.readlines()[:-1]
          stars = int(50*(j+1)/checkpoint_freq)
          lines.append('[' + '*' * stars + '.' * (50-stars) + ']' + f' {j+1}/{checkpoint_freq}\n')
          f.close()
        with open(log_path, "w") as f:
          f.writelines(lines)
        f.close()
        # save results
        timesteps[seed].append(gen * gen_len + results["timesteps_total"])
        policy_reward_min[seed][0].append(results["policy_reward_min"]["agent_0"] if results["policy_reward_min"] else float('NaN'))
        policy_reward_min[seed][1].append(results["policy_reward_min"]["agent_1"] if results["policy_reward_min"] else float('NaN'))
        policy_reward_mean[seed][0].append(results["policy_reward_mean"]["agent_0"] if results["policy_reward_mean"] else float('NaN'))
        policy_reward_mean[seed][1].append(results["policy_reward_mean"]["agent_1"] if results["policy_reward_mean"] else float('NaN'))
        policy_reward_max[seed][0].append(results["policy_reward_max"]["agent_0"] if results["policy_reward_max"] else float('NaN'))
        policy_reward_max[seed][1].append(results["policy_reward_max"]["agent_1"] if results["policy_reward_max"] else float('NaN'))

      # Save the checkpoint
      path_to_checkpoint = ppo.save(f"{checkpoints_path}/seed_{seed}/gen_{str(gen).zfill(3)}")
      checkpoints_dict[f"Seed_{seed}"].append(path_to_checkpoint)

      # Print results
      with open(log_path, "a") as f:
        # f.write(f'REWARD {policy_reward_mean[seed]}\n')
        f.write(f'TIMESTEP {(gen * gen_len + results["timesteps_total"])/1000}K\n')
        f.write(f'EPISODE REWARD MEAN {results["episode_reward_mean"]}\n')
        f.write(f'CHECKPOINT PATH {path_to_checkpoint}\n')
        if seed == num_seeds-1:
          f.write('#' * 100 + '\n')
        else:
          f.write('-' * 100 + '\n')
        f.close()

  # Save logging
  np.savez_compressed(save_path, timesteps=timesteps,
                      policy_reward_min=policy_reward_min,
                      policy_reward_mean=policy_reward_mean,
                      policy_reward_max=policy_reward_max)

if __name__ == "__main__":
  main()