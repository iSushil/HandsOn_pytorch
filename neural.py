#!/home/zeus/miniconda2/envs/py2/bin/python
import torch
from torch import autograd
import torch.nn as nn
import torch.nn.functional as F

batchSize = 5
inputSize = 3
hiddenSize = 8
outputSize = 4

torch.manual_seed(99)
inputData = autograd.Variable(torch.rand(batchSize, inputSize))

print('input data:',inputData)


class Net(nn.Module):
    def __init__(self,inputSize,hiddenSize, outputSize):
        super(Net,self).__init__()
        self.h1 = nn.Linear(inputSize,hiddenSize)
        self.h2 = nn.Linear(hiddenSize,outputSize)

    def forward(self,x):
        x = self.h1(x)
        x = F.tanh(x)
        x = self.h2(x)
        return x


model = Net(inputSize=inputSize,hiddenSize=hiddenSize,outputSize=outputSize)
model.zero_grad() #make the gradients on the weights zero
model.parameters() #iterator of the parameters, registers with the Module

out = model(inputData)
print('output data',out)
