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

torch.set_num_threads(8)

sys.path.append(str(jsbsim.get_default_root_dir()) + '/pFCM/')

from src.simEnv.jsbsimEnv import DogfightEnv as Env

torch.set_num_threads(8)


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
        self.relu = nn.ReLU(True)
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
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)

        self.fc3 = nn.Linear(hidden_dim_2, hidden_dim_3)

        self.hidden2q = nn.Linear(hidden_dim_3, 1)
        
    def forward(self, status, action):

        out = self.fc1(torch.cat([status, action], dim=0))
        out = self.relu(out)

        out = self.fc2(out)
        out = self.relu(out)

        out = self.fc3(out)
        out = self.relu(out)

        q = self.hidden2q(out)

        return q


class ActorCritic():
    
    def __init__(self):
        self.actor = Actor()
        self.critic = Critic()
        
    def policy(self, status):  # Get action through Actor()
        return self.actor(status)
    
    def value(self, status, action):  # Get estimated Q through Critic()
        return self.critic(status, action)


class DDPG():

    def __init__(self) -> None:
        self.model = ActorCritic()
        self.target_model = ActorCritic()

    def _critic_learn(
        self, 
        status, 
        action, 
        reward, 
        next_status, 
        terminal,
    ):
        self.gamma = .9
        self.weight_decay = 1e-2
        self.optimizer = optim.SGD(self.model.parameters(), lr = 1e-2, momentum=0.9, weight_decay=self.weight_decay)
        
        next_action = self.target_model.policy(next_status)
        next_q = self.target_model.value(next_status, next_action)

        target_q = reward + (1.0 - terminal) * self.gamma * next_q
        
        q = self.model.value(status, action)
        loss = nn.MSELoss(q, target_q)
        self.model.zero_grad()
        loss.backward()
        self.optimizer.step()

        # return loss

    def _actor_learn(self, status):
        self.optimizer = optim.SGD(self.model.parameters(), lr = 1e-2, momentum=0.9, weight_decay=self.weight_decay)

        action = self.model.policy(status)
        q = self.model.value(status, action)
        loss = -1.0 * q
        self.model.zero_grad()
        loss.backward()
        self.optimizer.step()

        # return loss

    def sync_target(self, decay=0):
        pass


    

    
