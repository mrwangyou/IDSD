import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import numpy as np

torch.set_num_threads(8)


class Representation(nn.Module):

    def __init__(self,
                 representation_len=6,
                 hidden_dim = 256,
                 dropout = .5,
                 num_of_label = 2,
                 property_len=3,
                 num_of_action=6
                 ):
        super(Representation, self).__init__()
        # self.input[0] = torch.ones(1, 9, 10, 50, 50)  # num of features; timeline; size x; size y
        # self.input[1] = torch.ones(7)  # num of properties
        self.conv1 = nn.Conv3d(9, 16, kernel_size=(3, 3, 3), padding=1)
        self.BN1 = nn.BatchNorm3d(16)
        self.relu = nn.ReLU(True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.padding = nn.ZeroPad2d((0, 1, 0, 1))

        self.conv2 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1)
        self.BN2 = nn.BatchNorm3d(32)

        self.fc1 = nn.Linear(32*10*13*13, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_dim, representation_len)
        self.sigmoid = nn.Sigmoid()

        self.hidden2label = nn.Linear(representation_len + property_len, num_of_action + 1)
        
        self.lstm = nn.LSTM(input_size=32*10*13*13, hidden_size=hidden_dim, num_layers=1)
        self.hidden_dim = hidden_dim
        self.hidden = self.init_hidden()
        

    def init_hidden(self):
        return (
            autograd.Variable(torch.zeros(1, self.hidden_dim).cuda()),
            autograd.Variable(torch.zeros(1, self.hidden_dim).cuda())
        )

    def forward(self, status, property, rl_type):
        # print("status size {} property size {}".format(status.size(), property.size()))
        out = self.conv1(status)  # of size (1, 16, 10, 50, 50)
        out = self.BN1(out)
        out = self.relu(out)
        out = out.view(-1, 50, 50)  # of size (1*16*10, 50, 50)
        out = self.pool(out)  # of size (1*16*10, 25, 25)
        out = out.view(-1, 16, 10, 25, 25)  # of size (1, 16, 10, 25, 25)

        out = self.conv2(out)  # of size (1, 32, 10, 25, 25)
        out = self.BN2(out)
        out = self.relu(out)
        out = out.view(-1, 25, 25)  # of size (1*32*10, 25, 25)
        out = self.padding(out)  # of size (1*32*10, 26, 26)
        out = self.pool(out)  # of size (1*32*10, 13, 13)
        out = out.view(1, -1)  # of size (1, 32*10*13*13)

        out = self.fc1(out)  # of size (1, hidden_dim) (1, 256)
        
        # out, self.hidden = self.lstm(out, self.hidden)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.relu(out)  # 6
        # out = self.sigmoid(out)
        # print(out.size())
        # print(property.unsqueeze(0).size())
        out = torch.cat([out, property], 1)  # concat properties  # 9
        # print(out.size())
        # print(out)
        # out = self.hidden2label(out)  # of size (1, 7)
        # out = F.normalize(out)

        # for i in range(3):
        #     out[0, i] = torch.arctan(out[0, i])
        # for i in range(3, 6):
        #     out[0, i] = (torch.arctan(out[0, i]) + 1) / 2

        if rl_type == 1:
            return out
        elif rl_type == 0:
            # log_probs = F.log_softmax(out, dim=-1)
            log_probs = out
            return log_probs
        else:
            raise Exception("RL type error!", self.rl_type)


if __name__ == "__main__":
    pass



