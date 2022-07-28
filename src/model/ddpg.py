import os
import sys
import time

import jsbsim
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# torch.set_num_threads(8)

sys.path.append(str(jsbsim.get_default_root_dir()) + '/pFCM/')

from src.simEnv.jsbsimEnv import DogfightEnv as Env
from src.reward import reward


class Actor(nn.Module):

    def __init__(
        self,
        status_dim=10,
        hidden_dim_1=64,
        hidden_dim_2=64,
        hidden_dim_3=64,
        num_of_actions=4,  # temporary
        dropout=.5,
    ):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(status_dim, hidden_dim_1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, hidden_dim_3)
        self.hidden2action = nn.Linear(hidden_dim_3, num_of_actions)
        self.tanh = nn.Tanh()
        
    def forward(self, status):

        out = self.fc1(status)
        out = self.relu(out)

        out = self.fc2(out)
        out = self.relu(out)

        out = self.fc3(out)
        out = self.relu(out)

        out = self.hidden2action(out)
        action = self.tanh(out)

        return action


class Critic(nn.Module):

    def __init__(
        self,
        status_dim=10,
        num_of_actions=4,  # temporary
        hidden_dim_1=64,
        hidden_dim_2=64,
        hidden_dim_3=64,
        dropout=.5,
    ):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(status_dim + num_of_actions, hidden_dim_1)
        self.fc11 = nn.Linear(status_dim, hidden_dim_1)
        self.fc12 = nn.Linear(num_of_actions, hidden_dim_1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)

        self.fc3 = nn.Linear(hidden_dim_2, hidden_dim_3)

        self.hidden2q = nn.Linear(hidden_dim_3, 1)
        
    def forward(self, status, action):

        out = self.fc11(status) + self.fc12(action)
        out = self.relu(out)

        out = self.fc2(out)
        out = self.relu(out)

        out = self.fc3(out)
        out = self.relu(out)

        q = self.hidden2q(out)

        return q


class DDPG():

    def __init__(self, cuda) -> None:
        device = torch.device("cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")
        self.model_actor = Actor().to(device)
        self.target_model_actor = Actor().to(device)
        self.model_critic = Critic().to(device)
        self.target_model_critic = Critic().to(device)

    def _critic_learn(
        self, 
        status, 
        action, 
        reward, 
        next_status, 
        terminate,
        step,
    ):
        self.model_critic.train()
        self.gamma = .9
        self.weight_decay = 1e-2
        self.optimizer = optim.SGD(self.model_critic.parameters(), lr = 1e-2, momentum=0.9, weight_decay=self.weight_decay)

        next_action = self.target_model_actor(next_status)
        next_q = self.target_model_critic(next_status, next_action)

        if terminate:
            target_q = reward
        else:
            target_q = reward + self.gamma * next_q

        # target_q = reward if terminate else reward + self.gamma * next_q
        target_q = target_q.detach()
        
        q = self.model_critic(status, action.detach())
        loss_fn = nn.MSELoss()
        loss = loss_fn(q, target_q)
        # print("/*/*{}".format(loss))
        self.model_critic.zero_grad()
        loss.backward()
        if step:
            self.optimizer.step()

        # return loss

    def _actor_learn(
        self,
        status,
        step,
    ):
        self.model_actor.train()
        self.weight_decay = 1e-2
        self.optimizer = optim.SGD(self.model_actor.parameters(), lr = 1e-2, momentum=0.9, weight_decay=self.weight_decay)
        
        action = self.model_actor(status)
        # device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
        # action = torch.rand([4]).to(device)
        q = self.model_critic(status, action)
        # for p in self.model_critic.parameters():
        #     p.requires_grad = False
        loss = -1.0 * q
        # loss.requires_grad_(True)
        self.model_actor.zero_grad()
        loss.backward()
        if step:
            self.optimizer.step()

        # return loss

    def sync_target(
        self,
        decay=.9
    ):

        for target_param, source_param in zip(self.target_model_actor.parameters(), self.model_actor.parameters()):
            target_param.data.copy_((1-decay) * target_param.data + decay * source_param.data)
        for target_param, source_param in zip(self.target_model_critic.parameters(), self.model_critic.parameters()):
            target_param.data.copy_((1-decay) * target_param.data + decay * source_param.data)


    def getStatus(
        self,
        env,
        id
    ):  
        # raise Exception("Hasn't finished yet.")
        return torch.rand([10])


    def getReward(
        self,
        env,
        id
    ):
        r_rp = reward.R_rp(env, id)
        r_closure = reward.R_closure(env, id)
        r_gunsnap = reward.R_gunsnap(env, id)
        r_deck = reward.R_deck(env, id)
        r_too_close = reward.R_too_close(env, id)

        return r_rp + r_closure + r_gunsnap + r_deck + r_too_close







    def episode(
        self,
        device,
    ):
        env = Env()
        print("**********Nof: {}**********".format(env.getNof()))
        
        pre_status_1 = torch.zeros([10])
        pre_action_1 = torch.zeros([4])
        pre_reward_1 = 0
        pre_status_2 = torch.zeros([10])
        pre_action_2 = torch.zeros([4])
        pre_reward_2 = 0
        pre_terminate = 0

        while True:
            terminate = env.step(playSpeed=0)
            if terminate != 0:
                break
            
            status_1 = self.getStatus(env, 1)
            status_2 = self.getStatus(env, 2)
            action_1 = self.model_actor(self.getStatus(env, 1).to(device))
            action_2 = self.model_actor(self.getStatus(env, 2).to(device))
            reward_1 = self.getReward(env, 1)  # 当前状态下的状态价值函数，可以理解为上一状态的动作价值函数
            reward_2 = self.getReward(env, 2)

            env.getFdm(1).sendAction(action_1.unsqueeze(0))
            env.getFdm(2).sendAction(action_2.unsqueeze(0))

            pre_status_1 = pre_status_1.to(device)
            pre_status_2 = pre_status_2.to(device)
            pre_action_1 = pre_action_1.to(device)
            pre_action_2 = pre_action_2.to(device)
            status_1 = status_1.to(device)
            status_2 = status_2.to(device)

            self._actor_learn(pre_status_1, 1)

            self._critic_learn(pre_status_1, pre_action_1, reward_1, status_1, pre_terminate, 1)

            self._actor_learn(pre_status_2, 1)

            self._critic_learn(pre_status_2, pre_action_2, reward_1, status_2, pre_terminate, 1)
            
            pre_status_1 = status_1
            pre_status_2 = status_2
            pre_action_1 = action_1
            pre_action_2 = action_2
            pre_reward_1 = reward_1
            pre_reward_2 = reward_2
            pre_terminate = env.terminate()

        self.sync_target()

    def train(
        self,
        epochs=20000,
        cuda='0',
    ):
        device = torch.device("cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")

        for _ in tqdm(range(epochs)):
            self.episode(device)





if __name__ == "__main__":
    ddpg = DDPG(cuda='0')
    ddpg.train(cuda='0')





