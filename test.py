import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import numpy as np
import copy



# class Test():

#     def __init__(self) -> None:
#         self.cnt = 0
    
#     def add(self):
#         self.cnt += 1

#     def print(self):
#         print(self.cnt)

# class Add():

#     def __init__(self, test) -> None:
#         self.test = test
    
#     def add(self):
#         self.test.add()
    
#     def print(self):
#         self.test.print()

class Add():
    def __init__(self) -> None:
        self.list = None

    def add(self, b):
        self.list = b
    def print(self):
        print(self.list)

if __name__ == '__main__':
    
    b=[1,2,3]
    a = b
    b = [4]
    print(a)
