import gym
from gym.utils.env_checker import check_env
env = gym.make("CartPole-v1")

check_env(env)

# env.action_space.seed(42)

# observation, info = env.reset(seed=42, return_info=True)

# for _ in range(1000):
#     observation, reward, done, info = env.step(env.action_space.sample())

#     if done:
#         observation, info = env.reset(return_info=True)

# env.close()
