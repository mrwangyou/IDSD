import math
import os
import random
import sys
import time

import jsbsim
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.append(str(jsbsim.get_default_root_dir()) + '/FCM/')

# from model import model
from myModule.representation_learning import \
    cnn as representation  # Use CNN as representation learning model
from myModule.representation_learning import getStatus
from myModule.representation_learning.getRl import get_rl


class DogfightDataset(Dataset):

    def __init__(self,
                 status,  # is of size [num_of_data, 9, 10, 50, 50]
                 property,  # is of size [num_of_data, num_of_property]
                 label):  # is of size [num_of data]
        self.status = status
        self.property = property 
        self.label = label

    def __len__(self):
        return self.status.size()[0]

    def __getitem__(self, index):
        data = {}
        data["status"] = self.status[index]
        data["property"] = self.property[index]
        data["label"] = self.label[index]
        return data


def train(epoch):
    # input[0] = torch.ones([num_of_data, 9, 10, 50, 50])  # num of features; timeline; size x; size y
    # input[1] = torch.ones([num_of data])  # num of properties
    # input[2] = torch.ones([num_of data])  # num of properties

    representation_len=7
    hidden_dim = 256
    dropout = .5

    EPOCH = epoch
    BATCH_SIZE = 1
    WEIGHT_DECAY = 1e-3
    SMALL_STEP_EPOCH = 5


    # fullDataset = DogfightDataset(status, property, label)
    # trSize = int(len(fullDataset) * 0.7)
    # devSize = int(len(fullDataset) * 0.2)
    # testSize = len(fullDataset) - trSize - devSize
    # trainDataset, devDataset, testDataset = torch.utils.data.random_split(fullDataset, [trSize, devSize, testSize])

    
    # trainLoader = DataLoader(dataset=trainDataset,
    #                          batch_size=BATCH_SIZE,
    #                          shuffle=True,
    #                          )
    try:
        model = torch.load('./bestModel/Epoch.pt', map_location=torch.device('cpu'))
    except:
        model = representation.Representation(representation_len=representation_len, 
                                              hidden_dim=hidden_dim,
                                              dropout=dropout
                                              )
        print("First time!")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device_ids = [3]
    # model = nn.DataParallel(model, device_ids=device_ids)
    model = model.to(device)
    loss_function = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)
    optimizer = optim.SGD(model.parameters(), lr = 1e-2, momentum=0.9, weight_decay = WEIGHT_DECAY)
    bestTillNow = 0
    for i in tqdm(range(EPOCH)):
        print("*\n*\n*\n*\n*\n*\n*\n*\n*\n*\n*\n*\n*\nNew epoch {}*\n*\n*\n*\n*\n*\n*\n*\n*\n*\n*\n*\n*\n*\n*\n*\n*\n".format(str(i)))
        train_epoch(model, loss_function, optimizer)
        # dev_acc = evaluate_epoch(model, devLoader)
        # test_acc = evaluate_epoch(model, testLoader)
        # if dev_acc > bestTillNow:
        #     bestTillNow = dev_acc
        #     torch.save(model, './bestModel/Epoch' + str(i) + 'acc' + str(test_acc) + '.pt')
        time.sleep(2)
        torch.save(model, './bestModel/Epoch.pt')
        # if i == SMALL_STEP_EPOCH:
        #     optimizer = optim.SGD(model.parameters(), lr = 1e-3, momentum=0.9, weight_decay = WEIGHT_DECAY)
    

def train_epoch(model,
                loss_function,
                optimizer
                ):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # batch_label = torch.tensor([])
    # batch_pred = torch.tensor([])
    # batch_label = batch_label.to(device)
    # batch_pred = batch_pred.to(device)

    sum_loss = 0

    file = open('./log.txt', 'w', encoding="UTF-8")


    wins_record1 = []
    wins_record2 = []
    input1 = []
    input2 = []
    inputp_1 = []
    inputp_2 = []
    wins = -1
    # Hyperparameter
    code_model = 1  # 0 for test, 1 for main


    # FDM Initialization
    fdm1 = jsbsim.FGFDMExec(None)
    fdm1.load_model('f16')  # Aircraft
    # fdm1.set_output_directive('./data_output/flightgear.xml')  # Visualization fgfs

    fdm2 = jsbsim.FGFDMExec(None)
    fdm2.load_model('f16')  # Aircraft
    # fdm2.set_output_directive('./data_output/flightgear2.xml')  # Visualization fgfs


    # Velocity Initialization
    fdm1['ic/vc-kts'] = 1000  # Calibrated Velocity (knots) https://skybrary.aero/articles/calibrated-airspeed-cas#:~:text=Definition,port%20caused%20by%20airflow%20disruption).

    # fdm2['ic/vc-kts'] = 750
    fdm2['ic/vc-kts'] = 1000


    # Position Initialization
    fdm1["ic/lat-gc-deg"] = 0  # Latitude (degree)
    fdm1["ic/long-gc-deg"] = 0  # Longitude (degree)
    fdm1["ic/h-sl-ft"] = 30005.5  # Height above sea level (feet)
    fdm1["ic/psi-true-deg"] = 0
    # fdm1["ic/theta-deg"] = 90
    fdm1["ic/phi-deg"] = 90

    # fdm2["ic/lat-gc-deg"] = 0.01  # Latitude (degree)
    fdm2["ic/long-gc-deg"] = 0.00005  # Longitude (degree)
    fdm2["ic/h-sl-ft"] = 30005.5  # Height above sea level (feet)
    # fdm2["ic/psi-true-deg"] = 180
    # fdm2["ic/theta-deg"] = 90
    fdm2["ic/psi-true-deg"] = 0
    fdm2["ic/phi-deg"] = 90

    # fdm2["ic/lat-gc-deg"] = random.random() / 100  # Latitude (degree)
    # fdm2["ic/long-gc-deg"] = random.random() / 5000  # Longitude (degree)
    # fdm2["ic/h-sl-ft"] = 30005.5  # Height above sea level (feet)
    # fdm2["ic/psi-true-deg"] = 180 * (random.random() * 2 - 1)

    ##########################
    ## Model Initialization ##
    fdm1.run_ic()           ##
    fdm2.run_ic()           ##
    ##########################


    # Engine Turning on
    fdm1["propulsion/starter_cmd"] = 1
    fdm2["propulsion/starter_cmd"] = 1


    # Refueling
    if code_model == 0:
        fdm1["propulsion/refuel"] = 1
        fdm2["propulsion/refuel"] = 1

    # print("fdm1:{}\t{}\t{}".format(fdm1["position/eci-x-ft"], fdm1["position/eci-y-ft"], fdm1["position/eci-z-ft"]))
    print("fdm1:{}\t{}\t{}".format(fdm1["attitude/phi-rad"], fdm1["attitude/theta-rad"], fdm1["attitude/psi-rad"]))

    # First but not Initial
    fdm1.run()
    fdm1["propulsion/cutoff_cmd"] = 0
    fdm1["fcs/throttle-cmd-norm[0]"] = 1

    fdm2.run()
    fdm2["propulsion/cutoff_cmd"] = 0
    fdm2["fcs/throttle-cmd-norm[0]"] = 1
    # print("fdm1:{}\t{}\t{}".format(fdm1["position/eci-x-ft"], fdm1["position/eci-y-ft"], fdm1["position/eci-z-ft"]))
    print("fdm1:{}\t{}\t{}".format(fdm1["attitude/phi-rad"], fdm1["attitude/theta-rad"], fdm1["attitude/psi-rad"]))

    fdm1_hp = 1
    fdm2_hp = 1

    cnt = 1  # num of frames
    # height_flag1 = 0
    # height_flag2 = 0
    while fdm1.run() and fdm2.run():
        fdms = [fdm1, fdm2]
        cnt += 1
        print("cnt: {}".format(cnt))
        if code_model == 0:
            try:
                if fdm1['simulation/sim-time-sec'] >= 5:
                    fdm1["fcs/elevator-cmd-norm"] = -0.1
                if fdm1['simulation/sim-time-sec'] >= 100:
                    fdm1["fcs/elevator-cmd-norm"] = 0
                    fdm1["do-simple-trim"] = 1
                if fdm1["propulsion/total-fuel-lbs"] > 5000:
                    fdm1["propulsion/refuel"] = 0
            except:
                raise Exception('Error!')
        elif code_model == 1:
            file.write("{} {} {} {} {} {}\n".format(fdm1["position/ecef-x-ft"], fdm1["position/ecef-y-ft"], fdm1["position/ecef-z-ft"], fdm2["position/ecef-x-ft"], fdm2["position/ecef-y-ft"], fdm2["position/ecef-z-ft"], ))
            state_vector = torch.zeros(0)  # public features
            for fdm in fdms:
                tmp = [fdm["velocities/v-north-fps"] / 2000,
                       fdm["velocities/v-east-fps"] / 20,
                       fdm["velocities/v-down-fps"] / 40,
                       fdm["position/h-sl-ft"] / 30000,
                       fdm["position/long-gc-deg"] / 3e-5,
                       fdm["position/lat-gc-deg"] / 0.001,
                       fdm["attitude/phi-rad"] / 1,
                       fdm["attitude/theta-rad"] / 1,
                       fdm["attitude/psi-rad"] / 1
                       ]
                # print("*    {}  *".format(tmp))
                state_vector = torch.tensor(tmp).unsqueeze(0) if state_vector.size() == torch.Size([0])\
                    else torch.cat([state_vector, torch.tensor(tmp).unsqueeze(0)], 0)  # of size torch.Size([2, 9])
            
            property_vector = torch.zeros(0)  # private features
            fdmflag = 1
            for fdm in fdms:
                if fdmflag == 1:
                    # tmp = [fdm["propulsion/total-fuel-lbs"] / 3000,
                    #     fdm1_hp,
                    #     fdm2_hp
                    #     ]
                    tmp = [
                        ((fdm1["position/eci-x-ft"] - fdm2["position/eci-x-ft"]) ** 2 +\
                        (fdm1["position/eci-y-ft"] - fdm2["position/eci-y-ft"]) ** 2 +\
                        (fdm1["position/eci-z-ft"] - fdm2["position/eci-z-ft"]) ** 2) / 50000,
                        fdm1_hp,
                        fdm2_hp
                    ]
                    fdmflag = 0
                else:
                    # tmp = [fdm["propulsion/total-fuel-lbs"] / 3000,
                    #     fdm2_hp,
                    #     fdm1_hp
                    #     ]
                    tmp = [
                        ((fdm1["position/eci-x-ft"] - fdm2["position/eci-x-ft"]) ** 2 +\
                        (fdm1["position/eci-y-ft"] - fdm2["position/eci-y-ft"]) ** 2 +\
                        (fdm1["position/eci-z-ft"] - fdm2["position/eci-z-ft"]) ** 2) / 50000,
                        fdm2_hp,
                        fdm1_hp
                    ]
                property_vector = torch.tensor(tmp).unsqueeze(0) if property_vector.size() == torch.Size([0])\
                    else torch.cat([property_vector, torch.tensor(tmp).unsqueeze(0)], 0)  # of size torch.Size([2, 1])
                    
            ftpg = 1000  # Feet per grid
            grid = torch.zeros([1, 9, 100, 50, 50])  # 1; num of features; timeline; size x; size y

            flagg = getStatus.setStatus(cnt, ftpg, fdm2["position/eci-y-ft"], fdm2["position/eci-z-ft"], fdm1["position/eci-y-ft"], fdm1["position/eci-z-ft"], state_vector[1], state_vector[0])
            if flagg == -1:
                wins = 2
                break
            # os.mkdir('./bestModel')
            rl1 = get_rl(getStatus.getStatus(cnt), property_vector[0].unsqueeze(0), model)
            rl1 = rl1 + torch.cat([(torch.rand([1, 6]) * 2 - 1) / 2, torch.Tensor([[1, 0, 1, 0]])], dim=1).to(device)
            # rl1 = rl1 + torch.cat([(torch.rand([1, 6]) * 3 - 1) / 2 + torch.Tensor([[(random.random() * 2 - 1) / 2, (random.random() * 2 - 1) / 2, 0, 0, 0, 0]]), torch.Tensor([[1, 0, 1, 0]])], dim=1).to(device)
            # rl1 = torch.rand([1, 7]).to(device)
            # rl1 = torch.Tensor([[-.5, 1, .5, 0, 0, 0]]).to(device)
            # rl1 = (torch.rand([1, 7]).to(device) * 2 - 1)
            # if cnt <= 120:
            #     rl1 = torch.Tensor([[0, -1, 0, 0, 1, 0]]).to(device)
            # elif cnt <= 480:
            #     rl1 = torch.Tensor([[-.5, 1, .5, 0, 0, 0]]).to(device)
            # else:
            #     rl1 = get_rl(getStatus.getStatus(cnt), property_vector[0].unsqueeze(0), model)


            if cnt % 12 == 0:
                if wins_record1 == []:
                    wins_record1 = torch.cat([rl1[0, 0:6], torch.ones(1).to(device)]).unsqueeze(0)
                    input1 = getStatus.getStatus(cnt)
                    inputp_1 = property_vector[0].unsqueeze(0)
                else:
                    # print(rl1[0, 0:6])
                    # print(torch.ones(1).to(device))
                    wins_record1 = torch.cat([wins_record1, torch.cat([rl1[0, 0:6], torch.ones(1).to(device)]).unsqueeze(0)], dim=0)
                    input1 = torch.cat([input1, getStatus.getStatus(cnt)], dim=0)
                    inputp_1 = torch.cat([inputp_1, property_vector[0].unsqueeze(0)], dim=0)
            

            grid = torch.zeros([1, 9, 100, 50, 50])
            flagg = getStatus.setStatus(cnt, ftpg, fdm1["position/eci-y-ft"], fdm1["position/eci-z-ft"], fdm2["position/eci-y-ft"], fdm2["position/eci-z-ft"], state_vector[0], state_vector[1])
            if flagg == -1:
                wins = 1
                break
            rl2 = get_rl(getStatus.getStatus(cnt), property_vector[1].unsqueeze(0), model)
            rl2 = rl2 + torch.cat([(torch.rand([1, 6]) * 2 - 1) / 2, torch.Tensor([[1, 0, 1, 0]])], dim=1).to(device)
            # rl2 = rl2 + torch.cat([(torch.rand([1, 6]) * 2 - 1) / 2 + torch.Tensor([[0, -random.random() / 2, 0, 0, 0, 0]]), torch.Tensor([[1, 0, 1, 0]])], dim=1).to(device)

            # rl2 = torch.Tensor([[0, .3, 0, 0, 0, 0]]).to(device)
            # rl2 = (torch.rand([1, 10]) / 2 + torch.ones([1, 10]) / 2).to(device)
            # rl1 = torch.rand([1, 10]).to(device)

            if cnt % 12 == 0:
                if wins_record2 == []:
                    wins_record2 = torch.cat([rl2[0, 0:6], torch.ones(1).to(device)]).unsqueeze(0)
                    input2 = getStatus.getStatus(cnt)
                    inputp_2 = property_vector[1].unsqueeze(0)
                else:
                    wins_record2 = torch.cat([wins_record2, torch.cat([rl2[0, 0:6], torch.ones(1).to(device)]).unsqueeze(0)], dim=0)
                    input2 = torch.cat([input2, getStatus.getStatus(cnt)], dim=0)
                    inputp_2 = torch.cat([inputp_2, property_vector[1].unsqueeze(0)], dim=0)

            fdm1["fcs/aileron-cmd-norm"]     = (rl1[0, 0].item()) * rl1[0, 6].item()  # 副翼
            fdm1["fcs/elevator-cmd-norm"]    = (rl1[0, 1].item()) * rl1[0, 6].item()  # 升降舵
            fdm1["fcs/rudder-cmd-norm"]      = (rl1[0, 2].item()) * rl1[0, 6].item()  # 方向舵
            fdm1["fcs/flap-cmd-norm"]        = (rl1[0, 3].item()) * rl1[0, 6].item()  # 襟翼
            fdm1["fcs/speedbrake-cmd-norm"]  = (rl1[0, 4].item()) * rl1[0, 6].item()  # 减速板
            fdm1["fcs/spoiler-cmd-norm"]     = (rl1[0, 5].item()) * rl1[0, 6].item()  # 扰流片

            fdm2["fcs/aileron-cmd-norm"]     = (rl2[0, 0].item()) * rl2[0, 6].item()  # 副翼
            fdm2["fcs/elevator-cmd-norm"]    = (rl2[0, 1].item()) * rl2[0, 6].item()  # 升降舵
            fdm2["fcs/rudder-cmd-norm"]      = (rl2[0, 2].item()) * rl2[0, 6].item()  # 方向舵
            fdm2["fcs/flap-cmd-norm"]        = (rl2[0, 3].item()) * rl2[0, 6].item()  # 襟翼
            fdm2["fcs/speedbrake-cmd-norm"]  = (rl2[0, 4].item()) * rl2[0, 6].item()  # 减速板
            fdm2["fcs/spoiler-cmd-norm"]     = (rl2[0, 5].item()) * rl2[0, 6].item()  # 扰流片

            # fdm1["fcs/aileron-cmd-norm"]     = 0
            # fdm1["fcs/elevator-cmd-norm"]    = 0
            # fdm1["fcs/rudder-cmd-norm"]      = 0
            # fdm1["fcs/flap-cmd-norm"]        = 0
            # fdm1["fcs/speedbrake-cmd-norm"]  = 0
            # fdm1["fcs/spoiler-cmd-norm"]     = 0

            # fdm2["fcs/aileron-cmd-norm"]     = 0
            # fdm2["fcs/elevator-cmd-norm"]    = 0
            # fdm2["fcs/rudder-cmd-norm"]      = 0
            # fdm2["fcs/flap-cmd-norm"]        = 0
            # fdm2["fcs/speedbrake-cmd-norm"]  = 0
            # fdm2["fcs/spoiler-cmd-norm"]     = 0

            # fdm1["fcs/aileron-cmd-norm"] = max(min(rl1[0, 0].item(), 1), -1)  # 副翼
            # fdm1["fcs/elevator-cmd-norm"] = max(min(rl1[0, 1].item(), 1), -1)  # 升降舵
            # fdm1["fcs/rudder-cmd-norm"] = max(min(rl1[0, 2].item(), 1), -1)  # 方向舵
            # fdm1["fcs/flap-cmd-norm"] = max(min(rl1[0, 3].item(), 1), 0)  # 襟翼
            # fdm1["fcs/speedbrake-cmd-norm"] = max(min(rl1[0, 4].item(), 1), 0)  # 减速板
            # fdm1["fcs/spoiler-cmd-norm"] = max(min(rl1[0, 5].item(), 1), 0)  # 扰流片

            # fdm2["fcs/aileron-cmd-norm"] = max(min(rl2[0, 0].item(), 1), -1)  # 副翼
            # fdm2["fcs/elevator-cmd-norm"] = max(min(rl2[0, 1].item(), 1), -1)  # 升降舵
            # fdm2["fcs/rudder-cmd-norm"] = max(min(rl2[0, 2].item(), 1), -1)  # 方向舵
            # fdm2["fcs/flap-cmd-norm"] = max(min(rl2[0, 3].item(), 1), 0)  # 襟翼
            # fdm2["fcs/speedbrake-cmd-norm"] = max(min(rl2[0, 4].item(), 1), 0)  # 减速板
            # fdm2["fcs/spoiler-cmd-norm"] = max(min(rl2[0, 5].item(), 1), 0)  # 扰流片

            print("rl1: {}\t{}\t{}\t{}\t{}\t{}\t".format(fdm1["fcs/aileron-cmd-norm"], fdm1["fcs/elevator-cmd-norm"], fdm1["fcs/rudder-cmd-norm"], fdm1["fcs/flap-cmd-norm"], fdm1["fcs/speedbrake-cmd-norm"], fdm1["fcs/spoiler-cmd-norm"]))
            # print("rl2: {}\n_________________".format(fdm2["fcs/rudder-cmd-norm"]))
                
                

            x = fdm1["position/eci-x-ft"] - fdm2["position/eci-x-ft"]
            y = fdm1["position/eci-y-ft"] - fdm2["position/eci-y-ft"]
            z = fdm1["position/eci-z-ft"] - fdm2["position/eci-z-ft"]
            f1 = -fdm1["velocities/v-down-fps"]
            f2 = fdm1["velocities/v-east-fps"]
            f3 = fdm1["velocities/v-north-fps"]
            if 500 <= (x ** 2 + y ** 2 + z ** 2) ** .5 and (x ** 2 + y ** 2 + z ** 2) ** .5 <= 3000:
                # r = np.arcsin((f2 * z + f3 * x + f1 * y - f2 * x - f3 * y - f1 * z) / (f1**2 + f2**2 + f3**2)**.5 / (x**2 + y**2 + z**2)**.5) / np.pi * 180

                tmp_x = f2 * z - f3 * y
                tmp_y = f3 * x - f1 * z
                tmp_z = f1 * y - f2 * x
                r = np.arcsin((tmp_x**2 + tmp_y**2 + tmp_z**2) ** .5 / (f1**2 + f2**2 + f3**2)**.5 / (x**2 + y**2 + z**2)**.5) / np.pi * 180


                if r <= 1 and r >= -1:
                    fdm1_hp -= (3000 - (x ** 2 + y ** 2 + z ** 2) ** .5) / 2500 / 120
            
            x = -x
            y = -y
            z = -z
            f1 = -fdm2["velocities/v-down-fps"]
            f2 = fdm2["velocities/v-east-fps"]
            f3 = fdm2["velocities/v-north-fps"]
            if 500 <= (x ** 2 + y ** 2 + z ** 2) ** .5 and (x ** 2 + y ** 2 + z ** 2) ** .5 <= 3000:
                # r = np.arcsin((f2 * z + f3 * x + f1 * y - f2 * x - f3 * y - f1 * z) / (f1**2 + f2**2 + f3**2)**.5 / (x**2 + y**2 + z**2)**.5) / np.pi * 180

                tmp_x = f2 * z - f3 * y
                tmp_y = f3 * x - f1 * z
                tmp_z = f1 * y - f2 * x
                r = np.arcsin((tmp_x**2 + tmp_y**2 + tmp_z**2) ** .5 / (f1**2 + f2**2 + f3**2)**.5 / (x**2 + y**2 + z**2)**.5) / np.pi * 180

                if r <= 1 and r >= -1:
                    fdm2_hp -= (3000 - (x ** 2 + y ** 2 + z ** 2) ** .5) / 2500 / 120
            print("Distance: {}".format((x ** 2 + y ** 2 + z ** 2) ** .5))
            print("health: \t{}\t{}".format(fdm1_hp, fdm2_hp))
            if fdm1_hp <= 0:
                print("2 wins")
                wins = 2
                break
            if fdm2_hp <= 0:
                print("1 wins")
                wins = 1
                break
            if cnt >= 1000:
                if fdm1_hp >= fdm2_hp:
                    wins = 1
                    break
                else:
                    wins = 2
                    break

        else:
            raise Exception("Code model error!", code_model)


        # Height ASL Protection
        if 50 < fdm1["position/h-sl-ft"] < 1000:
            # raise Exception('Too low!', fdm1["position/h-sl-ft"])
            print("1 hit ground")
            wins = 2
            break
        if 50 < fdm2["position/h-sl-ft"] < 1000:
            # raise Exception('Too low!', fdm1["position/h-sl-ft"])
            print("2 hit ground")
            wins = 1
            break
        # height_flag1 = 0 if fdm1["position/h-sl-ft"] < 50 else 1 if fdm1["position/h-sl-ft"] > 100 else 2

        # if cnt % 120 == 0:
            # print("fdm1:{}\t{}\t{}".format(fdm1["position/h-agl-ft"], fdm1["position/long-gc-deg"], fdm1["position/lat-gc-deg"]))
            # print("fdm1:{}\t{}\t{}".format(fdm1["position/eci-x-ft"], fdm1["position/eci-y-ft"], fdm1["position/eci-z-ft"]))
            # print("fdm1:{}\t{}\t{}".format(fdm1["attitude/phi-rad"], fdm1["attitude/theta-rad"], fdm1["attitude/psi-rad"]))
            # print("fdm1:{}\t{}\t{}".format(fdm1["position/ecef-x-ft"], fdm1["position/ecef-y-ft"], fdm1["position/ecef-z-ft"]))
            # print(fdm1['simulation/sim-time-sec'])

        # Real-time
        # time.sleep(fdm1.get_delta_t() / 1)

    if wins == 1:
        wins_data = wins_record1
        wins_input = input1
        wins_inputp = inputp_1
    else:
        wins_data = wins_record2
        wins_input = input2
        wins_inputp = inputp_2
    print("***")
    print(wins_data.size())
    print(wins_input.size())
    print(wins_inputp.size())
    print("health1: {}".format(fdm1_hp))
    print("health2: {}".format(fdm2_hp))
    print("***")
    file.close()

    label = wins_data
    

    # batch_label = batch_label.to(device)
    # batch_pred = batch_pred.to(device)
    # batch_label = torch.cat([batch_label, label])
    # batch_pred = torch.cat([batch_pred, pred])

    fullDataset = DogfightDataset(wins_input, wins_inputp, label)
    trainLoader = DataLoader(dataset=fullDataset,
                             batch_size=1,
                             shuffle=True,
                             )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # loss_function = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(1):
        for batch in trainLoader:
            model.train()
            status = batch['status']
            property = batch['property']
            label = batch['label']
            status = status.to(device)
            property = property.to(device)
            label = label.to(device)
            pred = model(status, property, 0)  # torch.Size([1, 7])
            pred = pred.to(device)

            # for i in range(0, 3):
            #     label[0, i] = min(max(label[0, i], 1), -1)

            # for i in range(3, 6):
            #     label[0, i] = min(max(label[0, i], 1), 0)
            
            # print(batch_pred.size())
            # pred = F.normalize(pred, dim=0)
            # label = F.normalize(pred, dim=0)
            print(pred.size())
            loss = 0
            for i in range(6):
                # loss = loss + np.log(np.abs(pred[0, i].item()) + .0001) * np.abs(label[0, i].item())
                loss = loss + (label[0, i].item() - pred[0, i].item()) ** 2
            loss = loss + (1 - pred[0, 6].item()) ** 2
            loss = loss + pred[0, 7].item() - 0
            loss = loss + (1 - pred[0, 8].item()) ** 2
            loss = loss + (0 - pred[0, 9].item()) ** 2
            # loss = loss_function(pred, label)
            print("pred: {}\nlabel: {}".format(pred, label))
            print("loss: {}".format(loss))
            loss = torch.tensor(loss, requires_grad=True)
            model.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()



"""
ubody (velocity, ft/sec)
vbody (velocity, ft/sec)
wbody (velocity, ft/sec)
vnorth (velocity, ft/sec)
veast (velocity, ft/sec)
vdown (velocity, ft/sec)
latitude (position, degrees)
longitude (position, degrees)
phi (orientation, degrees)
theta (orientation, degrees)
psi (orientation, degrees)
alpha (angle, degrees)
beta (angle, degrees)
gamma (angle, degrees)
roc (vertical velocity, ft/sec)
elevation (local terrain elevation, ft)
altitude (altitude AGL, ft)
altitudeAGL (altitude AGL, ft)
altitudeMSL (altitude MSL, ft)
winddir (wind from-angle, degrees)
vwind (magnitude wind speed, ft/sec)
hwind (headwind speed, knots)
xwind (crosswind speed, knots)
vc (calibrated airspeed, ft/sec)
mach (mach)
vground (ground speed, ft/sec)
trim (0 for no trim, 1 for ground trim, 'Longitudinal', 'Full', 'Ground', 'Pullup', 'Custom', 'Turn')
running (-1 for all engines, 0 ... n-1 for speceffic engines)

@property ic/vc-kts (read/write) Calibrated airspeed initial condition in knots
@property ic/ve-kts (read/write) Knots equivalent airspeed initial condition
@property ic/vg-kts (read/write) Ground speed initial condition in knots
@property ic/vt-kts (read/write) True airspeed initial condition in knots
@property ic/mach (read/write) Mach initial condition
@property ic/roc-fpm (read/write) Rate of climb initial condition in feet/minute
@property ic/gamma-deg (read/write) Flightpath angle initial condition in degrees
@property ic/alpha-deg (read/write) Angle of attack initial condition in degrees
@property ic/beta-deg (read/write) Angle of sideslip initial condition in degrees
@property ic/theta-deg (read/write) Pitch angle initial condition in degrees
@property ic/phi-deg (read/write) Roll angle initial condition in degrees
@property ic/psi-true-deg (read/write) Heading angle initial condition in degrees
@property ic/lat-gc-deg (read/write) Latitude initial condition in degrees
@property ic/long-gc-deg (read/write) Longitude initial condition in degrees
@property ic/h-sl-ft (read/write) Height above sea level initial condition in feet
@property ic/h-agl-ft (read/write) Height above ground level initial condition in feet
@property ic/sea-level-radius-ft (read/write) Radius of planet at sea level in feet
@property ic/terrain-elevation-ft (read/write) Terrain elevation above sea level in feet
@property ic/vg-fps (read/write) Ground speed initial condition in feet/second
@property ic/vt-fps (read/write) True airspeed initial condition in feet/second
@property ic/vw-bx-fps (read/write) Wind velocity initial condition in Body X frame in feet/second
@property ic/vw-by-fps (read/write) Wind velocity initial condition in Body Y frame in feet/second
@property ic/vw-bz-fps (read/write) Wind velocity initial condition in Body Z frame in feet/second
@property ic/vw-north-fps (read/write) Wind northward velocity initial condition in feet/second
@property ic/vw-east-fps (read/write) Wind eastward velocity initial condition in feet/second
@property ic/vw-down-fps (read/write) Wind downward velocity initial condition in feet/second
@property ic/vw-mag-fps (read/write) Wind velocity magnitude initial condition in feet/sec.
@property ic/vw-dir-deg (read/write) Wind direction initial condition, in degrees from north
@property ic/roc-fps (read/write) Rate of climb initial condition, in feet/second
@property ic/u-fps (read/write) Body frame x-axis velocity initial condition in feet/second
@property ic/v-fps (read/write) Body frame y-axis velocity initial condition in feet/second
@property ic/w-fps (read/write) Body frame z-axis velocity initial condition in feet/second
@property ic/vn-fps (read/write) Local frame x-axis (north) velocity initial condition in feet/second
@property ic/ve-fps (read/write) Local frame y-axis (east) velocity initial condition in feet/second
@property ic/vd-fps (read/write) Local frame z-axis (down) velocity initial condition in feet/second
@property ic/gamma-rad (read/write) Flight path angle initial condition in radians
@property ic/alpha-rad (read/write) Angle of attack initial condition in radians
@property ic/theta-rad (read/write) Pitch angle initial condition in radians
@property ic/beta-rad (read/write) Angle of sideslip initial condition in radians
@property ic/phi-rad (read/write) Roll angle initial condition in radians
@property ic/psi-true-rad (read/write) Heading angle initial condition in radians
@property ic/lat-gc-rad (read/write) Geocentric latitude initial condition in radians
@property ic/long-gc-rad (read/write) Longitude initial condition in radians
@property ic/p-rad_sec (read/write) Roll rate initial condition in radians/second
@property ic/q-rad_sec (read/write) Pitch rate initial condition in radians/second
@property ic/r-rad_sec (read/write) Yaw rate initial condition in radians/second

"""

if __name__ == "__main__":
    # num_of_data = 500
    
    train(epoch=1000)





