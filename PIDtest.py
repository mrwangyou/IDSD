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











if __name__ == '__main__':
    env = Env(

    )

    while True:
        env.step()










































