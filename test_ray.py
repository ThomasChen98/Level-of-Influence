from ray.tune.logger import pretty_print
import faulthandler
faulthandler.enable()
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.algorithm import Algorithm

config = (DQNConfig().resources(num_gpus=0, num_cpus_per_worker=2, num_gpus_per_worker=0,)
          .rollouts(num_rollout_workers=4, num_envs_per_worker=1, create_env_on_local_worker=True,)
          .environment(env="CartPole-v1", render_env=True))
pretty_print(config.to_dict())

algo = config.build()

for i in range(10):
    result = algo.train()

print(pretty_print(result))

checkpoint = algo.save()
print(checkpoint)

evaluation = algo.evaluate()
print(pretty_print(evaluation))

algo.stop()
# checkpoint = '/home/yuxin/ray_results/DQN_CartPole-v1_2023-05-09_01-16-443zrtssr8/checkpoint_000010/checkpoint-10'
# restored_algo = Algorithm.from_checkpoint(checkpoint)
# algo = restored_algo

# policy = algo.get_policy()
# print(policy.get_weights())

# model = policy.model
# model.base_model.summary()