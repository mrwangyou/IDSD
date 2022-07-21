import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import numpy as np

torch.set_num_threads(8)


class Model(nn.Module):

    def __init__(self,
                 hidden_dim = 512,
                 representation_len=300,
                 dropout = .5,
                 property_len=3,
                 num_of_action=4
                 ):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(9, 16, kernel_size=(3, 3), padding=1)  
        self.BN1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  
        self.padding = nn.ZeroPad2d((0, 1, 0, 1))

        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1)
        self.BN2 = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(32*13*13, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_dim + property_len, representation_len)
        self.sigmoid = nn.Sigmoid()

        self.hidden2action = nn.Linear(representation_len, num_of_action + 1)
        
        self.tanh = nn.Tanh()
        

    def forward(self, status, prop):
        if status.size() != torch.Size([1, 9, 50, 50]):
            raise Exception("Status Size Error!", status.size())
        if prop.size() != torch.Size([1, 3]):
            raise Exception("Property Size Error!", prop.size())
        out = self.conv1(status)  # A Tensor of size [1, 16, 50, 50]
        out = self.BN1(out)
        out = self.relu(out)
        # out = out.view(-1, 50, 50)
        out = self.pool(out)  # A Tensor of size [1, 16, 25, 25]
        # out = out.view(-1, 16, 10, 25, 25)

        out = self.conv2(out)  # A Tensor of size [1, 32, 25, 25]
        out = self.BN2(out)
        out = self.relu(out)
        # out = out.view(-1, 25, 25)
        out = self.padding(out)  # A Tensor of size [1, 32, 26, 26]
        out = self.pool(out)  # A Tensor of size [1, 32, 13, 13]
        out = out.view(1, -1)  # A Tensor of size [1, 32*13*13]

        out = self.fc1(out)  # A Tensor of size [1, hidden_dim_1] (1, 512)
        out = self.relu(out)
        out = self.dropout(out)

        out = torch.cat([out, prop], 1)  # Concat properties
        
        out = self.fc2(out)  # A Tensor of size [1, hidden_dim_2] (1, 300)
        out = self.sigmoid(out)

        out = self.hidden2action(out)  # A Tensor of size [1, 5]
        out = self.tanh(out)

        return out

if __name__ == "__main__":
    pass



