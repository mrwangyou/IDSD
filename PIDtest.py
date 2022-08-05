import argparse
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

from src.reward import reward
from src.simEnv.jsbsimEnv import JsbsimEnv as Env

def parse_args():
    parser = argparse.ArgumentParser(description='123')
    parser.add_argument('--cuda', default='0', metavar='int', help='specifies the GPU to be used')
    parser.add_argument('--fgfs_1', action='store_true', help='specifies the rendering in FlightGear')
    parser.add_argument('--fgfs_2', action='store_true', help='specifies the rendering in FlightGear')
    parser.add_argument('--playSpeed', default=0, metavar='double', help='specifies to run in real world time')
    args = parser.parse_args()
    return args

def PIDControl():

    def __init__(
        self,
        env
    ):
        pass

    def poseControl(
        self,
        psi,
        theta,
        phi
    ):
        pass





if __name__ == '__main__':
    args = parse_args()
    env = Env(
        fdm_fgfs=args.fgfs_1
    )

    tmp = 1
    c = 1
    while True:
        # if c == 0:
        #     print('err')
        # else:
        #     print((env.getProperty('attitudeRad')[1] - tmp) / c)
        # c = env.getProperty('attitudeRad')[1] - tmp
        # tmp = env.getProperty('attitudeRad')[1]

        if 75 <= env.getProperty('attitudeDeg')[2] <= 85:
        # env.sendAction([[0, -.07, 0, 0]])
            if env.getNof() > 360:
                env.sendAction([[0, -1, 0, 1]])    
            else:
                env.sendAction([[0, -1, 0, 1]])
        # if env.getNof() == 120:
        #     env.sendAction([[0, -.07, 0, 0]])
        elif env.getProperty('attitudeDeg')[2] <= 75:
            env.sendAction([[1, -1, 0, 1]])
        else:
            env.sendAction([[-1, -1, 0, 1]])
        env.step(playSpeed=1)

        print("{0[0]}\t{0[1]}\t{0[2]}".format(env.getProperty('attitudeRad')))
        # print("{0[0]}\t{0[1]}\t{0[2]}".format(env.getProperty('velocity')))











































