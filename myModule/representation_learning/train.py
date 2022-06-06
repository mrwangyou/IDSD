import math
import os
import random
import sys

import jsbsim
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.append(str(jsbsim.get_default_root_dir()) + '/FCM/')
# sys.path.append('/home/wnn/anaconda3/envs/jsbsim/share/JSBSim/FCM')

from myModule.representation_learning import \
    cnn as representation  # Use CNN as representation learning model


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


def train(status, property, label, epoch):
    # input[0] = torch.ones([num_of_data, 9, 10, 50, 50])  # num of features; timeline; size x; size y
    # input[1] = torch.ones([num_of data])  # num of properties
    # input[2] = torch.ones([num_of data])  # num of properties

    representation_len=10
    hidden_dim = 256
    # dropout = .5

    EPOCH = epoch
    BATCH_SIZE = 1
    WEIGHT_DECAY = 1e-2
    SMALL_STEP_EPOCH = 5


    fullDataset = DogfightDataset(status, property, label)
    trSize = int(len(fullDataset) * 0.7)
    devSize = int(len(fullDataset) * 0.2)
    testSize = len(fullDataset) - trSize - devSize
    trainDataset, devDataset, testDataset = torch.utils.data.random_split(fullDataset, [trSize, devSize, testSize])

    
    trainLoader = DataLoader(dataset=trainDataset,
                             batch_size=BATCH_SIZE,
                             shuffle=True,
                             )
    devLoader = DataLoader(dataset=devDataset,
                           batch_size=BATCH_SIZE,
                           shuffle=True
                           )
    testLoader = DataLoader(dataset=testDataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True
                            )

    model = representation.Representation(representation_len=representation_len, 
                                          hidden_dim=hidden_dim,
                                        #   dropout=dropout
                                          )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device_ids = [3]
    # model = nn.DataParallel(model, device_ids=device_ids)
    model = model.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # optimizer = optim.SGD(model.parameters(), lr = 1e-2, momentum=0.9, weight_decay = WEIGHT_DECAY)
    bestTillNow = 0
    # for i in tqdm(range(EPOCH)):
    train_epoch(model, trainLoader, loss_function, optimizer)
        # dev_acc = evaluate_epoch(model, devLoader)
        # test_acc = evaluate_epoch(model, testLoader)
        # if dev_acc > bestTillNow:
        #     bestTillNow = dev_acc
        #     torch.save(model, './bestModel/Epoch' + str(i) + 'acc' + str(test_acc) + '.pt')
    # torch.save(model, './bestModel/Epoch.pt')
        # if i == SMALL_STEP_EPOCH:
        #     optimizer = optim.SGD(model.parameters(), lr = 1e-3, momentum=0.9, weight_decay = WEIGHT_DECAY)


def train_epoch(model,
                train_data,
                loss_function,
                optimizer
                ):
    model.train()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_label = torch.tensor([])
    batch_pred = torch.tensor([])
    batch_label = batch_label.to(device)
    batch_pred = batch_pred.to(device)

    sum_loss = 0

    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for batch in tqdm(train_data):  # "status""property""label"
        status = batch['status']
        property = batch['property']
        label = batch['label']
        status = status.to(device)
        property = property.to(device)
        label = label.to(device)

        pred = model(status, property, 0)  # torch.Size([1, 7])
        pred = pred.to(device)

        batch_label = batch_label.to(device)
        batch_pred = batch_pred.to(device)
        batch_label = torch.cat([batch_label, label])
        batch_pred = torch.cat([batch_pred, pred])

        print(batch_pred.size())

        loss = loss_function(pred, label.long())

        sum_loss += loss

        model.zero_grad()
        loss.backward()
        optimizer.step()
        if pred[0][0] > pred[0][1] and label <= 1:
            TP = TP + 1
        if pred[0][0] > pred[0][1] and label >= 1:
            FP = FP + 1
        if pred[0][0] < pred[0][1] and label >= 1:
            TN = TN + 1
        if pred[0][0] < pred[0][1] and label <= 1:
            FN = FN + 1


    print("ACC: {}".format((TP + TN) / (FP + FN + TP + TN)))
    print("TP: {} TN: {} FP: {} FN: {}".format(TP, TN, FP, FN))
    print("train loss: {}".format(sum_loss))



# def evaluate_epoch(model,
#                    train_data
#                    ):
#     model.eval()
#     TP = 0
#     FP = 0
#     TN = 0
#     FN = 0
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#     for batch in tqdm(train_data):  # "status""property""label"
#         status = batch['status']
#         property = batch['property']
#         label = batch['label']
#         status = status.to(device)
#         property = property.to(device)
#         label = label.to(device)

#         # print("*************************")
#         # print(status.size())  # [1, 9, 10, 50, 50]
#         # print(property.size())  # [1]
#         pred = model(status, property, 0)
#         pred = pred.to(device)

#         model.zero_grad()
#         if pred[0][0] > pred[0][1] and label <= 1:
#             TP = TP + 1
#         if pred[0][0] > pred[0][1] and label >= 1:
#             FP = FP + 1
#         if pred[0][0] < pred[0][1] and label >= 1:
#             TN = TN + 1
#         if pred[0][0] < pred[0][1] and label <= 1:
#             FN = FN + 1


#     # print("Acc: {}".format(accuracy_score(batch_label, batch_pred)))
#     # print("F1: {}".format(f1_score(batch_label, batch_pred)))
#     # print("Precision: {}".format(precision_score(batch_label, batch_pred)))
#     # print("Recall: {}\n".format(recall_score(batch_label, batch_pred)))
#     print("ACC: {}".format((TP + TN) / (FP + FN + TP + TN)))
#     print("TP: {} TN: {} FP: {} FN: {}".format(TP, TN, FP, FN))
#     return (TP + TN) / (FP + FN + TP + TN)

if __name__ == "__main__":
    num_of_data = 500
    
    train(status=torch.rand([num_of_data, 9, 10, 50, 50]),
          property=torch.rand([num_of_data, 3]), 
          label=torch.rand(num_of_data) * 2,
          epoch=1
          )

    pass

