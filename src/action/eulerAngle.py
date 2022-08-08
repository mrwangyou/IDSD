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

class PIDControl():

    def __init__(
        self,
        env
    ):
        self.env = env

    def poseControl(
        self,
        psi,
        theta,
        phi,
        flag=0,
        flag2=0
    ):  
        # print("Warning: {}".format(flag))
        psi_ego = self.env.getProperty('attitudeDeg')[0]
        if psi_ego >= 180:
            psi_ego -= 360
        if flag >= 1 or np.abs(psi - psi_ego) >= 10:  # 调整偏航角
            if (psi > psi_ego or (flag == 1 or flag == 3)) and flag != 4:
                flag = 1
                self.env.sendAction(  # 翻滚角
                    action=[np.tanh((80 - self.env.getProperty('attitudeDeg')[2]) / 30)],
                    actionType='fcs/aileron-cmd-norm'
                )
                if np.abs(80 - self.env.getProperty('attitudeDeg')[2]) <= 5:
                    flag = 3
                
            elif (psi <= psi_ego or (flag == 2 or flag == 3)) and flag != 4:
                flag = 2
                self.env.sendAction(  # 翻滚角
                    action=[np.tanh((-80 - self.env.getProperty('attitudeDeg')[2])/ 30)],
                    actionType='fcs/aileron-cmd-norm'
                )
                if np.abs(-80 - self.env.getProperty('attitudeDeg')[2]) <= 5:
                    flag = 3
            
            if flag == 3:
                self.env.sendAction(  # 俯仰角
                    action=[-np.tanh(-1 - self.env.getProperty('attitudeDeg')[1])],
                    actionType='fcs/elevator-cmd-norm'
                )

                if np.abs(psi - psi_ego) <= 5:
                    flag = 4
            if flag == 4:
                self.env.sendAction(  # 翻滚角
                    action=[np.tanh((0 - self.env.getProperty('attitudeDeg')[2]) / 30)],
                    actionType='fcs/aileron-cmd-norm'
                )
                if np.abs(self.env.getProperty('attitudeDeg')[2]) <= 3:
                    flag2 = flag2 + 1
                if flag2 > 30:
                    flag2 = 0
                    flag = 0
                if np.abs(psi - psi_ego) >= 25:
                    flag2 = 0
                    flag = 1


        else:  # 微调偏航角
            self.env.sendAction(  # 偏航角
                action=[np.tanh(psi_ego - psi) * 10],
                actionType='fcs/rudder-cmd-norm'
            )

            self.env.sendAction(  # 俯仰角
                action=[-np.tanh((theta - self.env.getProperty('attitudeDeg')[1]) * 1.5)],
                actionType='fcs/elevator-cmd-norm'
            )
            
            self.env.sendAction(  # 翻滚角
                action=[np.tanh((phi - self.env.getProperty('attitudeDeg')[2]) / 30)],
                actionType='fcs/aileron-cmd-norm'
            )
            # print("{}".format(-np.tanh(theta - self.env.getProperty('attitudeDeg')[1])))
        return flag, flag2

    def toAction(
        self,
    ):
        pass

if __name__ == '__main__':
    args = parse_args()
    env = Env(
        fdm_fgfs=args.fgfs_1
    )

    PID = PIDControl(env)



    tmp = 1
    c = 1
    flag = 0
    flag2 = 0
    while True:
        # if c == 0:
        #     print('err')
        # else:
        #     print((env.getProperty('attitudeRad')[1] - tmp) / c)
        # c = env.getProperty('attitudeRad')[1] - tmp
        # tmp = env.getProperty('attitudeRad')[1]
        flag, flag2 = PID.poseControl(0, 30, 30, flag, flag2)
        
        env.sendAction([1], 'fcs/throttle-cmd-norm')
        env.step(playSpeed=1)

        print("Yaw: {0[0]}\t\tPitch: {0[1]}\t\tRoll: {0[2]}".format(env.getProperty('attitudeDeg')))
        # print("{0[0]}\t{0[1]}\t{0[2]}".format(env.getProperty('velocity')))











































