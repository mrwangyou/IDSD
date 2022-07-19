import os
import sys

import jsbsim
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

torch.set_num_threads(8)

sys.path.append(str(jsbsim.get_default_root_dir()) + '/oFCM/')

from repLearning.repLearning import Representation
from src.simEnv.jsbsimEnv import JsbsimEnv as Env


class DogfightDataset(Dataset):

    def __init__(self,
                 status,  # of size [num_of_data, 9, 10, 50, 50]
                 property,  # of size [num_of_data, num_of_property]
                 label):  # of size [num_of data]
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


class IDSD():

    def __init__(self) -> None:
        from repLearning.cnn import Model
        self.model = Model(

        )
        if not os.path.exists('./bestModel'):
            os.mkdir('./bestModel/')
        if os.listdir('./bestModel') != []:
            self.model.load_state_dict(torch.load('./bestModel/{}'.format(os.listdir('./bestModel')[:-1])))

    def episode(
        self,
        device,
        optimizer,
    ):
        env = Env()
        print(env.nof)
        wins_record1 = []
        wins_record2 = []
        
        while True:
            flag = env.step()
            if flag != 0:
                break
            rep1 = Representation(env)
            rep2 = Representation(env)
            rl1 = rep1.getRepresentation(env.nof, 'IDSD', self.model, device, 1)
            rl2 = rep2.getRepresentation(env.nof, 'IDSD', self.model, device, 2)
            print(rl1)
            # print(rl1)
            # print("**************")
            # print(rl2)
            if env.nof % 12 == 0:
                if wins_record1 == []:
                    wins_record1 = torch.cat([rl1, torch.ones([1, 1]).to(device)], dim=1).unsqueeze(0)
                    input1 = rep1.getStatus(env.nof)
                    inputp_1 = torch.Tensor(rep1.getProperty()).unsqueeze(0)

                    wins_record2 = torch.cat([rl2, torch.ones([1, 1]).to(device)], dim=1).unsqueeze(0)
                    input2 = rep2.getStatus(env.nof)
                    inputp_2 = torch.Tensor(rep2.getProperty()).unsqueeze(0)
                else:
                    wins_record1 = torch.cat([wins_record1, torch.cat([rl1, torch.ones(1, 1).to(device)], dim=1).unsqueeze(0)], dim=0)
                    input1 = torch.cat([input1, rep1.getStatus(env.nof)])
                    inputp_1 = torch.cat([inputp_1, torch.Tensor(rep1.getProperty()).unsqueeze(0)])

                    wins_record2 = torch.cat([wins_record2, torch.cat([rl2, torch.ones(1, 1).to(device)], dim=1).unsqueeze(0)], dim=0)
                    input2 = torch.cat([input2, rep2.getStatus(env.nof)])
                    inputp_2 = torch.cat([inputp_2, torch.Tensor(rep2.getProperty()).unsqueeze(0)])
            env.sendAction(rl1.tolist(), 1)
            env.sendAction(rl2.tolist(), 2)
        if flag == 1:
            wins_data = wins_record1
            wins_input = input1
            wins_inputp = inputp_1
        elif flag == 2:
            wins_data = wins_record2
            wins_input = input2
            wins_inputp = inputp_2
        elif flag == -1:
            return
        else:
            raise Exception("Return code error!", flag)

        label = wins_data

        fullDataset = DogfightDataset(wins_input, wins_inputp, label)
        trainLoader = DataLoader(dataset=fullDataset, batch_size=1, shuffle=True)
        for _ in range(1):
            for batch in trainLoader:
                self.model.train()
                status = batch['status']
                property = batch['property']
                label = batch['label']
                status = status.to(device)
                property = property.to(device)
                label = label.to(device)
                pred = self.model(status, property)  # torch.Size([1, 7])
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
                label = label.squeeze().unsqueeze(dim=0)
                print(label)
                for i in range(5):
                    # loss = loss + np.log(np.abs(pred[0, i].item()) + .0001) * np.abs(label[0, i].item())
                    loss = loss + (label[0, i].item() - pred[0, i].item()) ** 2
                # loss = loss_function(pred, label)
                print("pred: {}\nlabel: {}".format(pred, label))
                print("loss: {}".format(loss))
                loss = torch.tensor(loss, requires_grad=True)
                self.model.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

    def train(
        self,
        epochs=50,
        cuda='0',
        optimizer='SGD',
        lr=1e-2,
        momentum=0.9,
        weight_decay=1e-3
    ):
        device = torch.device("cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        if optimizer == 'SGD':
            optimizer = optim.SGD(self.model.parameters(), lr, momentum, weight_decay)
        elif optimizer == 'Adam':
            pass
        else:
            raise Exception("Optimizer {} doesn't exist".format(optimizer))
        for _ in tqdm(range(epochs)):
            self.episode(device, optimizer)
        

    


if __name__ == '__main__':
    model = IDSD()
    model.train()

    
