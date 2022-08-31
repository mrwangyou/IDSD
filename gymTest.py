from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random

class ShowerEnv(Env):
    def __init__(self) -> None:
        self.action_space = Discrete(3)
        self.observation_space = Box(low=np.array([0]), high=np.array([100]))
        self.state = 38 + random.randint(-3, 3)
        self.shower_length = 60

    def step(self, action) -> None:
        self.state += action - 1
        self.shower_length -= 1

        if self.state >= 37 and self.state <= 39:
            reward = 1
        else:
            reward = -1

        if self.shower_length <= 0:
            done = True
        else:
            done = False
        
        self.state += random.randint(-1, 1)

        info = {}

        return self.state, reward, done, info

    def render(self) -> None:
        pass

    def reset(self) -> None:
        self.state = 38 + random.randint(-3, 3)

        self.shower_length = 60
        return self.state

env = ShowerEnv()

# print(env.observation_space.sample())  

episodes = 10

for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score += reward
    
    print('Episode:{} Score:{}'.format(episode, score))
    



































