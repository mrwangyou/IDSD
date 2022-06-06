import os
import sys

import jsbsim
import numpy as np
import torch
import torch.nn as nn

sys.path.append(str(jsbsim.get_default_root_dir()) + '/FCM/')

from myModule.representation_learning.getRl import get_rl


class PlaneState():
    def __init__(self, hp_ego, hp_oppo):
        self.hp_ego = hp_ego
        self.hp_oppo = hp_oppo

    def hpego(self):
        return self.hp_ego

    def hpoppo(self):
        return self.hp_oppo

    




class MonteCarloTreeSearchNode():
    def __init__(self, expanded):
        self.expanded = expanded
        self.child = []
        self.action = []
        self.reward = 0
        self.num = 0

    
    def kruct(self, state):
        if state.hpego() == 0:
            return -1, False
        if state.hpoppo() == 0:
            return 1, False
        self.expanded = False
        
        tmp = 0
        for i in self.action:
            pass

        
    

def model(rl=None) -> list:
    
    if rl is None:
        rl = get_rl(status=torch.rand([1, 9, 10, 50, 50]),
                    property=torch.rand(1), 
                    path='./bestModel/{}'.format(os.listdir('./bestModel/')[-1])
                    )

    root = MonteCarloTreeSearchNode(state = rl)
    selected_node = root.best_action()

    return selected_node


if __name__ == '__main__':
    pass



